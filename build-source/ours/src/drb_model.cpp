#include "ours/drb_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace ours {
namespace {

constexpr std::array<char, 8> kModelMagic = {'D', 'R', 'B', 'P', 'Q', '0', '2', '\0'};
constexpr std::array<char, 8> kVectorModelMagic = {'V', 'D', 'R', 'B', 'P', 'Q', '0', '1'};
constexpr std::array<char, 8> kMixedModelMagic = {'V', 'D', 'R', 'B', 'M', 'I', 'X', '1'};
constexpr std::array<char, 8> kResidualPqModelMagic = {'R', 'P', 'Q', 'O', 'N', 'L', '0', '1'};
constexpr std::array<char, 8> kCodesMagic = {'D', 'R', 'B', 'C', 'O', 'D', '0', '2'};
constexpr std::array<char, 8> kBitsMagic = {'D', 'R', 'B', 'B', 'I', 'T', '0', '2'};
constexpr std::array<char, 8> kVectorBitsMagic = {'V', 'B', 'R', 'B', 'I', 'T', 'S', '1'};
constexpr std::array<char, 8> kMixedBitsMagic = {'V', 'B', 'R', 'M', 'I', 'X', '0', '1'};
constexpr std::array<char, 8> kResidualPqCodesMagic = {'R', 'P', 'Q', 'C', 'O', 'D', '0', '1'};
constexpr uint32_t kVersion = 2;
constexpr uint32_t kVectorVersion = 1;
constexpr uint32_t kResidualPqVersion = 3;
constexpr uint32_t kResidualPqCodesVersion = 1;
constexpr uint32_t kMetricInnerProduct = 1;
constexpr uint32_t kFlagHasTransform = 1;
constexpr uint32_t kFlagHasResidualTransform = 2;
constexpr uint32_t kFlagAdditiveResidualQuantizer = 4;
constexpr uint32_t kFlagPrimaryConditioned = 8;

template <typename T>
T ReadScalar(std::ifstream& input, const std::string& name) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error("truncated artifact while reading " + name);
    }
    return value;
}

void ReadMagic(std::ifstream& input,
               const std::array<char, 8>& expected,
               const std::string& name) {
    std::array<char, 8> actual{};
    input.read(actual.data(), static_cast<std::streamsize>(actual.size()));
    if (!input || actual != expected) {
        throw std::runtime_error("invalid " + name + " magic");
    }
}

template <typename T>
std::vector<T> ReadVector(std::ifstream& input, uint64_t count, const std::string& name) {
    if (count > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(T))) {
        throw std::runtime_error(name + " is too large for this process");
    }
    std::vector<T> values(static_cast<size_t>(count));
    if (!values.empty()) {
        input.read(reinterpret_cast<char*>(values.data()),
                   static_cast<std::streamsize>(values.size() * sizeof(T)));
        if (!input) {
            throw std::runtime_error("truncated artifact while reading " + name);
        }
    }
    return values;
}

void RequireEof(std::ifstream& input, const std::string& name) {
    char extra = 0;
    input.read(&extra, 1);
    if (input.gcount() != 0) {
        throw std::runtime_error(name + " has unexpected trailing bytes");
    }
}

float Dot(const float* a, const float* b, uint32_t size) {
    double value = 0.0;
    for (uint32_t i = 0; i < size; ++i) {
        value += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(value);
}

}  // namespace

const float* MatrixView::Row(uint64_t row) const {
    if (data == nullptr || row >= rows) {
        throw std::out_of_range("matrix row is out of range");
    }
    return data + row * static_cast<uint64_t>(dim);
}

DrbPqIndex::DrbPqIndex(DrbModelData data) : data_(std::move(data)) {
    if (data_.bits_per_refined.empty()) {
        data_.bits_per_refined.assign(data_.refined_subspaces,
                                      static_cast<uint8_t>(data_.bits_per_subspace));
    }
    if (data_.branches_per_refined.empty()) {
        data_.branches_per_refined.assign(data_.refined_subspaces,
                                          data_.branches_per_cell);
    }
    Validate();
    BuildRuntimeLayout();
}

DrbPqIndex DrbPqIndex::FromData(DrbModelData data) {
    return DrbPqIndex(std::move(data));
}

DrbPqIndex DrbPqIndex::Load(const std::string& model_path,
                            const std::string& codes_path,
                            const std::string& side_bits_path) {
    DrbModelData data;
    {
        std::ifstream input(model_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("cannot open DRB model: " + model_path);
        }
        std::array<char, 8> magic{};
        input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
        if (!input || (magic != kModelMagic && magic != kVectorModelMagic &&
                       magic != kMixedModelMagic && magic != kResidualPqModelMagic)) {
            throw std::runtime_error("invalid DRB/vector/residual-PQ model magic");
        }
        const bool is_residual_pq = magic == kResidualPqModelMagic;
        const bool is_mixed = magic == kMixedModelMagic;
        const bool is_vector = magic == kVectorModelMagic || is_mixed;
        const uint32_t version = ReadScalar<uint32_t>(input, "model version");
        const uint32_t metric = ReadScalar<uint32_t>(input, "model metric");
        if ((is_residual_pq && version != 1 && version != 2 &&
             version != kResidualPqVersion) ||
            (!is_residual_pq && !is_vector && version != kVersion) ||
            (is_vector && version != kVectorVersion)) {
            throw std::runtime_error("unsupported DRB/vector/residual-PQ model version");
        }
        if (metric != kMetricInnerProduct) {
            throw std::runtime_error("this DRB kernel currently supports inner product only");
        }
        data.node_count = ReadScalar<uint64_t>(input, "node count");
        data.dim = ReadScalar<uint32_t>(input, "dimension");
        data.subspaces = ReadScalar<uint32_t>(input, "PQ subspaces");
        data.codewords = ReadScalar<uint32_t>(input, "PQ codewords");
        data.dsub = ReadScalar<uint32_t>(input, "PQ subspace dimension");
        if (is_residual_pq) {
            data.residual_groups = ReadScalar<uint32_t>(input, "residual-PQ groups");
            data.residual_codewords =
                ReadScalar<uint32_t>(input, "residual-PQ codewords");
            data.refined_subspaces = 0;
            data.refinement_kind = RefinementKind::kResidualPq;
        } else {
            data.refined_subspaces = ReadScalar<uint32_t>(input, "refined subspaces");
        }
        if (!is_residual_pq && is_vector && !is_mixed) {
            data.bits_per_subspace =
                ReadScalar<uint32_t>(input, "bits per refined subspace");
            data.branches_per_cell =
                ReadScalar<uint32_t>(input, "branches per PQ cell");
            data.refinement_kind = RefinementKind::kVector;
        } else if (!is_residual_pq && !is_vector) {
            data.bits_per_subspace = 1;
            data.branches_per_cell = 2;
            data.refinement_kind = RefinementKind::kScalar;
        } else if (!is_residual_pq) {
            data.bits_per_subspace = 0;
            data.branches_per_cell = 0;
            data.refinement_kind = RefinementKind::kVector;
        }
        const uint32_t flags = ReadScalar<uint32_t>(input, "model flags");
        data.artifact_id = ReadScalar<uint64_t>(input, "artifact id");
        if (is_residual_pq) {
            data.residual_encoding_id =
                ReadScalar<uint64_t>(input, "residual-PQ encoding id");
            data.metric_lambda = ReadScalar<double>(input, "residual-PQ metric lambda");
        }
        data.has_transform = (flags & kFlagHasTransform) != 0;
        data.has_residual_transform =
            is_residual_pq && (flags & kFlagHasResidualTransform) != 0;
        data.residual_additive =
            is_residual_pq && (flags & kFlagAdditiveResidualQuantizer) != 0;
        data.primary_conditioned =
            is_residual_pq && (flags & kFlagPrimaryConditioned) != 0;
        if ((data.residual_additive || data.primary_conditioned) && version < 3) {
            throw std::runtime_error(
                "additive/conditioned residual models require online version 3");
        }

        const uint64_t d = data.dim;
        const uint64_t m = data.subspaces;
        const uint64_t k = data.codewords;
        const uint64_t ds = data.dsub;
        const uint64_t r = data.refined_subspaces;
        data.transform = ReadVector<float>(input, d * d, "transform");
        data.transform_bias = ReadVector<float>(input, d, "transform bias");
        data.centroids = ReadVector<float>(input, m * k * ds, "centroids");
        if (is_residual_pq) {
            if (data.residual_groups == 0 || d % data.residual_groups != 0) {
                throw std::runtime_error("invalid residual-PQ group layout");
            }
            data.residual_group_dim = data.residual_additive
                ? static_cast<uint32_t>(d)
                : static_cast<uint32_t>(d / data.residual_groups);
            data.residual_pq_prototypes = ReadVector<float>(
                input,
                static_cast<uint64_t>(data.residual_groups) *
                    data.residual_codewords * data.residual_group_dim,
                "residual-PQ prototypes");
            if (data.has_residual_transform) {
                if (version < 2) {
                    throw std::runtime_error(
                        "residual transform requires online model version 2");
                }
                data.residual_inverse_transform = ReadVector<float>(
                    input, d * d, "residual-PQ inverse transform");
            }
            if (data.primary_conditioned) {
                data.conditional_rank =
                    ReadScalar<uint32_t>(input, "conditional rank");
                data.conditional_basis = ReadVector<float>(
                    input, d * data.conditional_rank, "conditional basis");
                data.conditional_embeddings = ReadVector<float>(
                    input, m * k * data.conditional_rank,
                    "primary-code conditional embeddings");
            }
            RequireEof(input, "residual-PQ online model");
        } else {
        data.selected_subspaces = ReadVector<uint32_t>(input, r, "selected subspaces");
        if (is_vector) {
            uint64_t total_branches = 0;
            if (is_mixed) {
                data.bits_per_refined = ReadVector<uint8_t>(input, r, "mixed side bits");
                data.branches_per_refined.resize(static_cast<size_t>(r));
                for (uint64_t ridx = 0; ridx < r; ++ridx) {
                    const uint8_t bits = data.bits_per_refined[ridx];
                    if (bits != 1 && bits != 2 && bits != 4) {
                        throw std::runtime_error("mixed model contains unsupported side bits");
                    }
                    data.branches_per_refined[ridx] = 1U << bits;
                    total_branches += data.branches_per_refined[ridx];
                }
            } else {
                data.bits_per_refined.assign(r,
                    static_cast<uint8_t>(data.bits_per_subspace));
                data.branches_per_refined.assign(r, data.branches_per_cell);
                total_branches = r * data.branches_per_cell;
            }
            data.residual_prototypes = ReadVector<float>(
                input, total_branches * k * ds,
                "vector residual prototypes");
        } else {
            data.bits_per_refined.assign(r, 1);
            data.branches_per_refined.assign(r, 2);
            data.directions = ReadVector<float>(input, r * ds, "directions");
            data.thresholds = ReadVector<float>(input, r * k, "thresholds");
            data.beta = ReadVector<float>(input, r * k * 2, "beta");
        }
            RequireEof(input, is_vector ? "vector DRB model" : "DRB model");
        }
    }
    {
        std::ifstream input(codes_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("cannot open PQ codes: " + codes_path);
        }
        ReadMagic(input, kCodesMagic, "PQ codes");
        const uint32_t version = ReadScalar<uint32_t>(input, "codes version");
        const uint64_t nodes = ReadScalar<uint64_t>(input, "codes node count");
        const uint32_t subspaces = ReadScalar<uint32_t>(input, "codes subspaces");
        const uint64_t artifact_id = ReadScalar<uint64_t>(input, "codes artifact id");
        if (version != kVersion || nodes != data.node_count || subspaces != data.subspaces ||
            artifact_id != data.artifact_id) {
            throw std::runtime_error("PQ codes header does not match DRB model");
        }
        data.codes = ReadVector<uint8_t>(input, nodes * subspaces, "PQ codes");
        RequireEof(input, "PQ codes");
    }
    if (data.refinement_kind == RefinementKind::kResidualPq) {
        if (side_bits_path.empty()) {
            throw std::runtime_error("residual-PQ model requires a side-code file");
        }
        std::ifstream input(side_bits_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("cannot open residual-PQ side codes: " + side_bits_path);
        }
        ReadMagic(input, kResidualPqCodesMagic, "residual-PQ side codes");
        const uint32_t version = ReadScalar<uint32_t>(input, "side-code version");
        const uint64_t nodes = ReadScalar<uint64_t>(input, "side-code node count");
        const uint32_t dim = ReadScalar<uint32_t>(input, "side-code dimension");
        const uint32_t groups = ReadScalar<uint32_t>(input, "side-code groups");
        const uint32_t codewords = ReadScalar<uint32_t>(input, "side-code codewords");
        const uint64_t artifact_id = ReadScalar<uint64_t>(input, "side-code artifact id");
        const uint64_t encoding_id = ReadScalar<uint64_t>(input, "side-code encoding id");
        if (version != kResidualPqCodesVersion || nodes != data.node_count ||
            dim != data.dim || groups != data.residual_groups ||
            codewords != data.residual_codewords || artifact_id != data.artifact_id ||
            encoding_id != data.residual_encoding_id) {
            throw std::runtime_error("residual-PQ side-code header does not match model");
        }
        data.side_bits = ReadVector<uint8_t>(
            input, nodes * groups, "residual-PQ side codes");
        RequireEof(input, "residual-PQ side codes");
        return DrbPqIndex(std::move(data));
    }
    if (data.refined_subspaces > 0) {
        if (side_bits_path.empty()) {
            throw std::runtime_error("DRB model requires a side-bit file");
        }
        std::ifstream input(side_bits_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("cannot open side bits: " + side_bits_path);
        }
        std::array<char, 8> side_magic{};
        input.read(side_magic.data(), static_cast<std::streamsize>(side_magic.size()));
        if (!input) throw std::runtime_error("truncated side-code magic");
        const uint32_t version = ReadScalar<uint32_t>(input, "side-code version");
        const uint64_t nodes = ReadScalar<uint64_t>(input, "side-code node count");
        const uint32_t refined = ReadScalar<uint32_t>(input, "side-code refined count");
        uint32_t bits = 1;
        uint32_t branches = 2;
        uint32_t width = 0;
        if (side_magic == kMixedBitsMagic) {
            if (data.refinement_kind != RefinementKind::kVector) {
                throw std::runtime_error("mixed side codes require a vector model");
            }
            width = ReadScalar<uint32_t>(input, "mixed side-code width");
            const uint64_t side_artifact_id =
                ReadScalar<uint64_t>(input, "mixed side-code artifact id");
            const std::vector<uint8_t> stored_bits =
                ReadVector<uint8_t>(input, refined, "mixed side-code bits");
            uint64_t total_bits = 0;
            for (uint8_t value : data.bits_per_refined) total_bits += value;
            const uint32_t expected_width = static_cast<uint32_t>((total_bits + 7U) / 8U);
            if (version != kVectorVersion || nodes != data.node_count ||
                refined != data.refined_subspaces || width != expected_width ||
                side_artifact_id != data.artifact_id || stored_bits != data.bits_per_refined) {
                throw std::runtime_error("mixed side-code header does not match model");
            }
            data.side_bits = ReadVector<uint8_t>(input, nodes * width, "mixed side codes");
            RequireEof(input, "mixed side codes");
            return DrbPqIndex(std::move(data));
        } else if (data.refinement_kind == RefinementKind::kVector) {
            if (side_magic != kVectorBitsMagic) {
                throw std::runtime_error("vector model requires vector side codes");
            }
            bits = ReadScalar<uint32_t>(input, "side-code bits per subspace");
            width = ReadScalar<uint32_t>(input, "side-code width");
            branches = ReadScalar<uint32_t>(input, "side-code branches");
        } else {
            if (side_magic != kBitsMagic) {
                throw std::runtime_error("scalar DRB model requires scalar side bits");
            }
            width = ReadScalar<uint32_t>(input, "side-bit width");
        }
        const uint64_t artifact_id = ReadScalar<uint64_t>(input, "side-code artifact id");
        uint64_t total_bits = 0;
        for (uint8_t value : data.bits_per_refined) total_bits += value;
        const uint32_t expected_width = static_cast<uint32_t>((total_bits + 7U) / 8U);
        const uint32_t expected_version =
            data.refinement_kind == RefinementKind::kVector ? kVectorVersion : kVersion;
        if (version != expected_version || nodes != data.node_count ||
            refined != data.refined_subspaces || bits != data.bits_per_subspace ||
            branches != data.branches_per_cell || width != expected_width ||
            artifact_id != data.artifact_id) {
            throw std::runtime_error("side-code header does not match DRB/vector model");
        }
        data.side_bits = ReadVector<uint8_t>(input, nodes * width, "side bits");
        RequireEof(input, "side codes");
    }
    return DrbPqIndex(std::move(data));
}

void DrbPqIndex::Validate() const {
    const uint64_t n = data_.node_count;
    const uint64_t d = data_.dim;
    const uint64_t m = data_.subspaces;
    const uint64_t k = data_.codewords;
    const uint64_t ds = data_.dsub;
    const uint64_t r = data_.refined_subspaces;
    if (data_.artifact_id == 0 || n == 0 || d == 0 || m == 0 || k == 0 || ds == 0 ||
        m * ds != d) {
        throw std::runtime_error("invalid DRB/PQ dimensions");
    }
    if (k > 256 || r > m) {
        throw std::runtime_error("the byte-code DRB artifact requires K<=256 and R<=M");
    }
    if (data_.transform.size() != d * d || data_.transform_bias.size() != d ||
        data_.centroids.size() != m * k * ds || data_.codes.size() != n * m ||
        data_.selected_subspaces.size() != r || data_.bits_per_refined.size() != r ||
        data_.branches_per_refined.size() != r) {
        throw std::runtime_error("DRB artifact array shape mismatch");
    }
    if (data_.refinement_kind == RefinementKind::kResidualPq) {
        if (r != 0 || data_.residual_groups == 0 ||
            data_.residual_codewords != 256 ||
            data_.residual_group_dim == 0 ||
            ((!data_.residual_additive &&
              static_cast<uint64_t>(data_.residual_groups) *
                      data_.residual_group_dim != d) ||
             (data_.residual_additive && data_.residual_group_dim != d)) ||
            data_.residual_encoding_id == 0 ||
            data_.residual_pq_prototypes.size() !=
                static_cast<uint64_t>(data_.residual_groups) *
                    data_.residual_codewords * data_.residual_group_dim ||
            (data_.has_residual_transform &&
             data_.residual_inverse_transform.size() != d * d) ||
            (!data_.has_residual_transform &&
             !data_.residual_inverse_transform.empty()) ||
            (data_.primary_conditioned &&
             (data_.conditional_rank == 0 ||
              data_.conditional_basis.size() != d * data_.conditional_rank ||
              data_.conditional_embeddings.size() !=
                  m * k * data_.conditional_rank)) ||
            (!data_.primary_conditioned &&
             (data_.conditional_rank != 0 || !data_.conditional_basis.empty() ||
              !data_.conditional_embeddings.empty())) ||
            !data_.directions.empty() || !data_.thresholds.empty() ||
            !data_.beta.empty() || !data_.residual_prototypes.empty() ||
            data_.side_bits.size() != n * data_.residual_groups) {
            throw std::runtime_error("residual-PQ artifact array shape mismatch");
        }
        return;
    }
    if (!data_.residual_pq_prototypes.empty() || data_.residual_groups != 0 ||
        data_.residual_codewords != 0 || data_.residual_group_dim != 0 ||
        data_.residual_encoding_id != 0) {
        throw std::runtime_error("DRB artifact unexpectedly contains residual-PQ data");
    }
    uint64_t total_bits = 0;
    uint64_t total_branches = 0;
    for (uint64_t ridx = 0; ridx < r; ++ridx) {
        const uint8_t bits = data_.bits_per_refined[ridx];
        const uint32_t branches = data_.branches_per_refined[ridx];
        if ((bits != 1 && bits != 2 && bits != 4) || branches != (1U << bits)) {
            throw std::runtime_error("DRB branch count does not match side-code bits");
        }
        total_bits += bits;
        total_branches += branches;
    }
    if (data_.refinement_kind == RefinementKind::kScalar) {
        if (std::any_of(data_.bits_per_refined.begin(), data_.bits_per_refined.end(),
                        [](uint8_t value) { return value != 1; }) ||
            data_.directions.size() != r * ds || data_.thresholds.size() != r * k ||
            data_.beta.size() != r * k * 2 || !data_.residual_prototypes.empty()) {
            throw std::runtime_error("scalar DRB artifact array shape mismatch");
        }
    } else {
        if (!data_.directions.empty() || !data_.thresholds.empty() || !data_.beta.empty() ||
            data_.residual_prototypes.size() != total_branches * k * ds) {
            throw std::runtime_error("vector DRB artifact array shape mismatch");
        }
    }
    for (uint32_t selected : data_.selected_subspaces) {
        if (selected >= data_.subspaces) {
            throw std::runtime_error("selected DRB subspace is out of range");
        }
    }
    if (r > 0 && data_.side_bits.size() != n * ((total_bits + 7U) / 8U)) {
        throw std::runtime_error("side-bit payload shape mismatch");
    }
}

void DrbPqIndex::BuildRuntimeLayout() {
    if (data_.refinement_kind == RefinementKind::kResidualPq) {
        decoded_residual_pq_prototypes_.assign(
            static_cast<uint64_t>(data_.residual_groups) *
                data_.residual_codewords * data_.dim,
            0.0f);
        for (uint32_t group = 0; group < data_.residual_groups; ++group) {
            for (uint32_t code = 0; code < data_.residual_codewords; ++code) {
                const float* source = data_.residual_pq_prototypes.data() +
                    (static_cast<uint64_t>(group) *
                         data_.residual_codewords + code) *
                        data_.residual_group_dim;
                float* decoded = decoded_residual_pq_prototypes_.data() +
                    (static_cast<uint64_t>(group) *
                         data_.residual_codewords + code) * data_.dim;
                if (!data_.has_residual_transform) {
                    if (data_.residual_additive) {
                        std::copy(source, source + data_.dim, decoded);
                    } else {
                        std::copy(source, source + data_.residual_group_dim,
                                  decoded + static_cast<uint64_t>(group) *
                                                data_.residual_group_dim);
                    }
                    continue;
                }
                const uint32_t coded_begin = data_.residual_additive
                    ? 0U
                    : group * data_.residual_group_dim;
                for (uint32_t local = 0;
                     local < data_.residual_group_dim; ++local) {
                    const float value = source[local];
                    const float* decoder_row =
                        data_.residual_inverse_transform.data() +
                        static_cast<uint64_t>(coded_begin + local) * data_.dim;
                    for (uint32_t coordinate = 0; coordinate < data_.dim;
                         ++coordinate) {
                        decoded[coordinate] += value * decoder_row[coordinate];
                    }
                }
            }
        }
        if (data_.primary_conditioned) {
            decoded_conditional_basis_.assign(
                static_cast<uint64_t>(data_.conditional_rank) * data_.dim,
                0.0f);
            for (uint32_t rank = 0; rank < data_.conditional_rank; ++rank) {
                float* decoded = decoded_conditional_basis_.data() +
                    static_cast<uint64_t>(rank) * data_.dim;
                for (uint32_t coded = 0; coded < data_.dim; ++coded) {
                    const float value = data_.conditional_basis[
                        static_cast<uint64_t>(coded) * data_.conditional_rank +
                        rank];
                    if (data_.has_residual_transform) {
                        const float* decoder =
                            data_.residual_inverse_transform.data() +
                            static_cast<uint64_t>(coded) * data_.dim;
                        for (uint32_t coordinate = 0; coordinate < data_.dim;
                             ++coordinate) {
                            decoded[coordinate] += value * decoder[coordinate];
                        }
                    } else {
                        decoded[coded] += value;
                    }
                }
            }
        }
        side_bit_width_ = data_.residual_groups;
        record_stride_ = data_.subspaces + side_bit_width_;
        const uint64_t bytes =
            data_.node_count * static_cast<uint64_t>(record_stride_);
        if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::runtime_error("interleaved PQ/residual-code records are too large");
        }
        node_records_.resize(static_cast<size_t>(bytes));
        for (uint64_t node = 0; node < data_.node_count; ++node) {
            uint8_t* record = node_records_.data() + node * record_stride_;
            std::memcpy(record,
                        data_.codes.data() + node * data_.subspaces,
                        data_.subspaces);
            std::memcpy(record + data_.subspaces,
                        data_.side_bits.data() + node * side_bit_width_,
                        side_bit_width_);
        }
        std::vector<uint8_t>().swap(data_.codes);
        std::vector<uint8_t>().swap(data_.side_bits);
        return;
    }
    side_bit_offsets_.resize(static_cast<size_t>(data_.refined_subspaces) + 1U, 0);
    side_byte_offsets_.resize(data_.refined_subspaces, 0);
    side_shifts_.resize(data_.refined_subspaces, 0);
    side_masks_.resize(data_.refined_subspaces, 0);
    side_crosses_byte_.resize(data_.refined_subspaces, 0);
    branch_lut_offsets_.resize(static_cast<size_t>(data_.refined_subspaces) + 1U, 0);
    for (uint32_t ridx = 0; ridx < data_.refined_subspaces; ++ridx) {
        const uint32_t bit_offset = side_bit_offsets_[ridx];
        side_byte_offsets_[ridx] = bit_offset / 8U;
        side_shifts_[ridx] = static_cast<uint8_t>(bit_offset % 8U);
        side_masks_[ridx] = static_cast<uint16_t>(
            data_.branches_per_refined[ridx] - 1U);
        side_crosses_byte_[ridx] = static_cast<uint8_t>(
            side_shifts_[ridx] + data_.bits_per_refined[ridx] > 8U);
        side_bit_offsets_[ridx + 1U] =
            bit_offset + data_.bits_per_refined[ridx];
        branch_lut_offsets_[ridx + 1U] =
            branch_lut_offsets_[ridx] +
            static_cast<uint64_t>(data_.codewords) * data_.branches_per_refined[ridx];
    }
    side_bit_width_ = (side_bit_offsets_.back() + 7U) / 8U;
    record_stride_ = data_.subspaces + side_bit_width_;

    std::vector<uint8_t> is_refined(data_.subspaces, 0);
    for (uint32_t subspace : data_.selected_subspaces) {
        if (is_refined[subspace] != 0) {
            throw std::runtime_error("selected DRB subspaces must be unique");
        }
        is_refined[subspace] = 1;
    }
    unrefined_subspaces_.reserve(data_.subspaces - data_.refined_subspaces);
    for (uint32_t subspace = 0; subspace < data_.subspaces; ++subspace) {
        if (is_refined[subspace] == 0) unrefined_subspaces_.push_back(subspace);
    }

    if (data_.refinement_kind == RefinementKind::kVector) {
        effective_centroids_.resize(branch_lut_offsets_.back() * data_.dsub);
        for (uint32_t ridx = 0; ridx < data_.refined_subspaces; ++ridx) {
            const uint32_t subspace = data_.selected_subspaces[ridx];
            const uint32_t branches = data_.branches_per_refined[ridx];
            for (uint32_t code = 0; code < data_.codewords; ++code) {
                const float* centroid = data_.centroids.data() +
                    (static_cast<uint64_t>(subspace) * data_.codewords + code) * data_.dsub;
                for (uint32_t branch = 0; branch < branches; ++branch) {
                    const uint64_t entry = branch_lut_offsets_[ridx] +
                                           static_cast<uint64_t>(code) * branches + branch;
                    const float* prototype =
                        data_.residual_prototypes.data() + entry * data_.dsub;
                    float* effective = effective_centroids_.data() + entry * data_.dsub;
                    for (uint32_t dim = 0; dim < data_.dsub; ++dim) {
                        effective[dim] = centroid[dim] + prototype[dim];
                    }
                }
            }
        }
    }

    if (side_bit_width_ == 0) {
        node_records_ = std::move(data_.codes);
        return;
    }

    const uint64_t bytes = data_.node_count * static_cast<uint64_t>(record_stride_);
    if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("interleaved PQ/side-bit records are too large");
    }
    node_records_.resize(static_cast<size_t>(bytes));
    for (uint64_t node = 0; node < data_.node_count; ++node) {
        uint8_t* record = node_records_.data() + node * record_stride_;
        std::memcpy(record,
                    data_.codes.data() + node * data_.subspaces,
                    data_.subspaces);
        std::memcpy(record + data_.subspaces,
                    data_.side_bits.data() + node * side_bit_width_,
                    side_bit_width_);
    }
    // Do not retain a second 1 GB copy of the online payload after interleaving.
    std::vector<uint8_t>().swap(data_.codes);
    std::vector<uint8_t>().swap(data_.side_bits);
}

const uint8_t* DrbPqIndex::NodeRecord(uint32_t node_id) const {
    return node_records_.data() + static_cast<uint64_t>(node_id) * record_stride_;
}

void DrbPqIndex::PrepareResidualPqBasePairLookup() {
    if (data_.refinement_kind != RefinementKind::kResidualPq) {
        throw std::invalid_argument(
            "base-pair lookup requires a full-coverage residual-PQ model");
    }
    if (base_pair_lookup_ready_) return;

    // A global residual transform makes every residual codeword dense.  The
    // prepared-anchor path remains lookup based; rare direct comparisons use
    // exact reconstruction instead of allocating O(M G K^2) dense pair tables.
    if (data_.has_residual_transform || data_.residual_additive ||
        data_.primary_conditioned) {
        base_pair_lookup_ready_ = true;
        return;
    }

    const uint64_t m = data_.subspaces;
    const uint64_t k = data_.codewords;
    const uint64_t groups = data_.residual_groups;
    const uint64_t rk = data_.residual_codewords;
    primary_pair_lut_.resize(m * k * k);
    for (uint32_t subspace = 0; subspace < data_.subspaces; ++subspace) {
        for (uint32_t lhs = 0; lhs < data_.codewords; ++lhs) {
            const float* lhs_centroid = data_.centroids.data() +
                (static_cast<uint64_t>(subspace) * data_.codewords + lhs) *
                    data_.dsub;
            for (uint32_t rhs = 0; rhs < data_.codewords; ++rhs) {
                const float* rhs_centroid = data_.centroids.data() +
                    (static_cast<uint64_t>(subspace) * data_.codewords + rhs) *
                        data_.dsub;
                primary_pair_lut_[
                    (static_cast<uint64_t>(subspace) * data_.codewords + lhs) *
                        data_.codewords + rhs] =
                    Dot(lhs_centroid, rhs_centroid, data_.dsub);
            }
        }
    }

    residual_pair_lut_.resize(groups * rk * rk);
    for (uint32_t group = 0; group < data_.residual_groups; ++group) {
        for (uint32_t lhs = 0; lhs < data_.residual_codewords; ++lhs) {
            const float* lhs_prototype = data_.residual_pq_prototypes.data() +
                (static_cast<uint64_t>(group) * data_.residual_codewords + lhs) *
                    data_.residual_group_dim;
            for (uint32_t rhs = 0; rhs < data_.residual_codewords; ++rhs) {
                const float* rhs_prototype = data_.residual_pq_prototypes.data() +
                    (static_cast<uint64_t>(group) * data_.residual_codewords + rhs) *
                        data_.residual_group_dim;
                residual_pair_lut_[
                    (static_cast<uint64_t>(group) * data_.residual_codewords + lhs) *
                        data_.residual_codewords + rhs] =
                    Dot(lhs_prototype, rhs_prototype,
                        data_.residual_group_dim);
            }
        }
    }

    // A residual group can overlap one or two primary-PQ subspaces.  Store one
    // 256x256 cross table for every non-empty overlap.  Pair scoring then adds
    // primary(lhs)-residual(rhs) and its symmetric counterpart using the same
    // table, so no decoded 200D node is needed in the hot build path.
    cross_pair_blocks_.clear();
    for (uint32_t subspace = 0; subspace < data_.subspaces; ++subspace) {
        const uint32_t subspace_begin = subspace * data_.dsub;
        const uint32_t subspace_end = subspace_begin + data_.dsub;
        for (uint32_t group = 0; group < data_.residual_groups; ++group) {
            const uint32_t group_begin = group * data_.residual_group_dim;
            const uint32_t group_end = group_begin + data_.residual_group_dim;
            const uint32_t overlap_begin = std::max(subspace_begin, group_begin);
            const uint32_t overlap_end = std::min(subspace_end, group_end);
            if (overlap_begin >= overlap_end) continue;
            CrossPairBlock block;
            block.subspace = subspace;
            block.group = group;
            block.subspace_offset = overlap_begin - subspace_begin;
            block.group_offset = overlap_begin - group_begin;
            block.length = overlap_end - overlap_begin;
            block.table_offset = primary_residual_pair_lut_.size();
            primary_residual_pair_lut_.resize(
                primary_residual_pair_lut_.size() + k * rk);
            for (uint32_t primary_code = 0;
                 primary_code < data_.codewords; ++primary_code) {
                const float* centroid = data_.centroids.data() +
                    (static_cast<uint64_t>(subspace) * data_.codewords +
                     primary_code) * data_.dsub + block.subspace_offset;
                for (uint32_t residual_code = 0;
                     residual_code < data_.residual_codewords; ++residual_code) {
                    const float* prototype = data_.residual_pq_prototypes.data() +
                        (static_cast<uint64_t>(group) *
                             data_.residual_codewords + residual_code) *
                            data_.residual_group_dim + block.group_offset;
                    primary_residual_pair_lut_[
                        block.table_offset +
                        static_cast<uint64_t>(primary_code) *
                            data_.residual_codewords + residual_code] =
                        Dot(centroid, prototype, block.length);
                }
            }
            cross_pair_blocks_.push_back(block);
        }
    }
    base_pair_lookup_ready_ = true;
}

float DrbPqIndex::ResidualPqBasePairScore(uint32_t lhs,
                                         uint32_t rhs) const {
    if (!base_pair_lookup_ready_ || lhs >= data_.node_count ||
        rhs >= data_.node_count) {
        throw std::out_of_range(
            "residual-PQ base-pair lookup is not prepared or node id is invalid");
    }
    if (data_.has_residual_transform || data_.residual_additive ||
        data_.primary_conditioned) {
        thread_local std::vector<float> lhs_reconstruction;
        thread_local std::vector<float> rhs_reconstruction;
        lhs_reconstruction.resize(data_.dim);
        rhs_reconstruction.resize(data_.dim);
        ReconstructResidualPqNode(lhs, lhs_reconstruction.data(), data_.dim);
        ReconstructResidualPqNode(rhs, rhs_reconstruction.data(), data_.dim);
        return Dot(lhs_reconstruction.data(), rhs_reconstruction.data(),
                   data_.dim);
    }
    const uint8_t* lhs_record = NodeRecord(lhs);
    const uint8_t* rhs_record = NodeRecord(rhs);
    double score = 0.0;
    for (uint32_t subspace = 0; subspace < data_.subspaces; ++subspace) {
        score += primary_pair_lut_[
            (static_cast<uint64_t>(subspace) * data_.codewords +
             lhs_record[subspace]) * data_.codewords + rhs_record[subspace]];
    }
    const uint8_t* lhs_residual = lhs_record + data_.subspaces;
    const uint8_t* rhs_residual = rhs_record + data_.subspaces;
    for (uint32_t group = 0; group < data_.residual_groups; ++group) {
        score += residual_pair_lut_[
            (static_cast<uint64_t>(group) * data_.residual_codewords +
             lhs_residual[group]) * data_.residual_codewords +
            rhs_residual[group]];
    }
    for (const CrossPairBlock& block : cross_pair_blocks_) {
        score += primary_residual_pair_lut_[
            block.table_offset +
            static_cast<uint64_t>(lhs_record[block.subspace]) *
                data_.residual_codewords + rhs_residual[block.group]];
        score += primary_residual_pair_lut_[
            block.table_offset +
            static_cast<uint64_t>(rhs_record[block.subspace]) *
                data_.residual_codewords + lhs_residual[block.group]];
    }
    return static_cast<float>(score);
}

float DrbPqIndex::ResidualPqBasePairDistance(uint32_t lhs,
                                            uint32_t rhs) const {
    return -ResidualPqBasePairScore(lhs, rhs);
}

void DrbPqIndex::PrepareResidualPqBaseAnchor(
    uint32_t anchor_id, ResidualPqBaseAnchorContext* context) const {
    if (data_.refinement_kind != RefinementKind::kResidualPq ||
        anchor_id >= data_.node_count || context == nullptr) {
        throw std::invalid_argument(
            "invalid residual-PQ prepared-anchor request");
    }

    context->reconstruction.resize(data_.dim);
    ReconstructResidualPqNode(anchor_id, context->reconstruction.data(),
                              data_.dim);
    context->query.adc_lut.resize(
        static_cast<uint64_t>(data_.subspaces) * data_.codewords);
    context->query.residual_lut.resize(
        static_cast<uint64_t>(data_.residual_groups) *
        data_.residual_codewords);

    for (uint32_t subspace = 0; subspace < data_.subspaces; ++subspace) {
        const float* anchor = context->reconstruction.data() +
            static_cast<uint64_t>(subspace) * data_.dsub;
        float* lut = context->query.adc_lut.data() +
            static_cast<uint64_t>(subspace) * data_.codewords;
        const float* centroids = data_.centroids.data() +
            static_cast<uint64_t>(subspace) * data_.codewords * data_.dsub;
        for (uint32_t code = 0; code < data_.codewords; ++code) {
            lut[code] = Dot(anchor,
                            centroids + static_cast<uint64_t>(code) *
                                            data_.dsub,
                            data_.dsub);
        }
    }

    std::vector<float> residual_query(data_.dim);
    if (data_.has_residual_transform) {
        for (uint32_t coordinate = 0; coordinate < data_.dim; ++coordinate) {
            residual_query[coordinate] = Dot(
                data_.residual_inverse_transform.data() +
                    static_cast<uint64_t>(coordinate) * data_.dim,
                context->reconstruction.data(), data_.dim);
        }
    } else {
        residual_query = context->reconstruction;
    }
    if (data_.primary_conditioned) {
        std::vector<float> projection(data_.conditional_rank, 0.0f);
        for (uint32_t rank = 0; rank < data_.conditional_rank; ++rank) {
            double value = 0.0;
            for (uint32_t coordinate = 0; coordinate < data_.dim; ++coordinate) {
                value += static_cast<double>(residual_query[coordinate]) *
                    data_.conditional_basis[
                        static_cast<uint64_t>(coordinate) *
                            data_.conditional_rank + rank];
            }
            projection[rank] = static_cast<float>(value);
        }
        for (uint32_t subspace = 0; subspace < data_.subspaces; ++subspace) {
            float* lut = context->query.adc_lut.data() +
                static_cast<uint64_t>(subspace) * data_.codewords;
            for (uint32_t code = 0; code < data_.codewords; ++code) {
                lut[code] += Dot(
                    data_.conditional_embeddings.data() +
                        (static_cast<uint64_t>(subspace) * data_.codewords +
                         code) * data_.conditional_rank,
                    projection.data(), data_.conditional_rank);
            }
        }
    }
    for (uint32_t group = 0; group < data_.residual_groups; ++group) {
        const float* anchor = residual_query.data() +
            (data_.residual_additive
                 ? 0
                 : static_cast<uint64_t>(group) *
                       data_.residual_group_dim);
        float* lut = context->query.residual_lut.data() +
            static_cast<uint64_t>(group) * data_.residual_codewords;
        const float* prototypes = data_.residual_pq_prototypes.data() +
            static_cast<uint64_t>(group) * data_.residual_codewords *
                data_.residual_group_dim;
        for (uint32_t code = 0; code < data_.residual_codewords; ++code) {
            lut[code] = Dot(anchor,
                            prototypes + static_cast<uint64_t>(code) *
                                             data_.residual_group_dim,
                            data_.residual_group_dim);
        }
    }
    context->anchor_id = anchor_id;
    context->ready = true;
}

float DrbPqIndex::ResidualPqPreparedBaseAnchorScore(
    const ResidualPqBaseAnchorContext& context,
    uint32_t candidate_id) const {
    if (!context.ready || candidate_id >= data_.node_count ||
        context.query.adc_lut.size() !=
            static_cast<uint64_t>(data_.subspaces) * data_.codewords ||
        context.query.residual_lut.size() !=
            static_cast<uint64_t>(data_.residual_groups) *
                data_.residual_codewords) {
        throw std::out_of_range(
            "residual-PQ prepared-anchor context is invalid");
    }
    return ResidualPqScore(context.query, candidate_id);
}

float DrbPqIndex::ResidualPqPreparedBaseAnchorDistance(
    const ResidualPqBaseAnchorContext& context,
    uint32_t candidate_id) const {
    return -ResidualPqPreparedBaseAnchorScore(context, candidate_id);
}

void DrbPqIndex::ResidualPqPreparedBaseAnchorDistanceBatch(
    const ResidualPqBaseAnchorContext& context,
    const uint32_t* candidate_ids,
    size_t count,
    float* distances) const {
    if (!context.ready) {
        throw std::invalid_argument(
            "residual-PQ prepared-anchor context is not ready");
    }
    ScoreBatch(context.query, candidate_ids, count, ScoreMode::kResidualPq,
               MatrixView{}, distances);
    for (size_t i = 0; i < count; ++i) distances[i] = -distances[i];
}

void DrbPqIndex::ReconstructResidualPqNode(uint32_t node_id, float* output,
                                           uint32_t dim) const {
    if (data_.refinement_kind != RefinementKind::kResidualPq ||
        node_id >= data_.node_count || output == nullptr || dim != data_.dim) {
        throw std::invalid_argument("invalid residual-PQ reconstruction request");
    }
    const uint8_t* record = NodeRecord(node_id);
    for (uint32_t subspace = 0; subspace < data_.subspaces; ++subspace) {
        const float* centroid = data_.centroids.data() +
            (static_cast<uint64_t>(subspace) * data_.codewords +
             record[subspace]) * data_.dsub;
        std::copy(centroid, centroid + data_.dsub,
                  output + static_cast<uint64_t>(subspace) * data_.dsub);
    }
    const uint8_t* residual_codes = record + data_.subspaces;
    if (data_.primary_conditioned) {
        std::vector<float> context(data_.conditional_rank, 0.0f);
        for (uint32_t subspace = 0; subspace < data_.subspaces; ++subspace) {
            const float* embedding = data_.conditional_embeddings.data() +
                (static_cast<uint64_t>(subspace) * data_.codewords +
                 record[subspace]) * data_.conditional_rank;
            for (uint32_t rank = 0; rank < data_.conditional_rank; ++rank) {
                context[rank] += embedding[rank];
            }
        }
        for (uint32_t rank = 0; rank < data_.conditional_rank; ++rank) {
            const float* basis = decoded_conditional_basis_.data() +
                static_cast<uint64_t>(rank) * data_.dim;
            for (uint32_t coordinate = 0; coordinate < data_.dim; ++coordinate) {
                output[coordinate] += context[rank] * basis[coordinate];
            }
        }
    }
    for (uint32_t group = 0; group < data_.residual_groups; ++group) {
        const float* decoded = decoded_residual_pq_prototypes_.data() +
            (static_cast<uint64_t>(group) * data_.residual_codewords +
             residual_codes[group]) * data_.dim;
        for (uint32_t coordinate = 0; coordinate < data_.dim; ++coordinate) {
            output[coordinate] += decoded[coordinate];
        }
    }
}

std::vector<float> DrbPqIndex::MeanResidualPqReconstruction() const {
    if (data_.refinement_kind != RefinementKind::kResidualPq) {
        throw std::invalid_argument(
            "mean reconstruction requires a residual-PQ model");
    }
    std::vector<uint64_t> primary_histogram(
        static_cast<uint64_t>(data_.subspaces) * data_.codewords, 0);
    std::vector<uint64_t> residual_histogram(
        static_cast<uint64_t>(data_.residual_groups) *
            data_.residual_codewords, 0);
    for (uint64_t node = 0; node < data_.node_count; ++node) {
        const uint8_t* record = NodeRecord(static_cast<uint32_t>(node));
        for (uint32_t subspace = 0; subspace < data_.subspaces; ++subspace) {
            ++primary_histogram[
                static_cast<uint64_t>(subspace) * data_.codewords +
                record[subspace]];
        }
        const uint8_t* residual_codes = record + data_.subspaces;
        for (uint32_t group = 0; group < data_.residual_groups; ++group) {
            ++residual_histogram[
                static_cast<uint64_t>(group) * data_.residual_codewords +
                residual_codes[group]];
        }
    }

    std::vector<double> sum(data_.dim, 0.0);
    for (uint32_t subspace = 0; subspace < data_.subspaces; ++subspace) {
        for (uint32_t code = 0; code < data_.codewords; ++code) {
            const double weight = static_cast<double>(primary_histogram[
                static_cast<uint64_t>(subspace) * data_.codewords + code]);
            const float* centroid = data_.centroids.data() +
                (static_cast<uint64_t>(subspace) * data_.codewords + code) *
                    data_.dsub;
            for (uint32_t coordinate = 0; coordinate < data_.dsub; ++coordinate) {
                sum[static_cast<uint64_t>(subspace) * data_.dsub + coordinate] +=
                    weight * centroid[coordinate];
            }
        }
    }
    if (data_.primary_conditioned) {
        std::vector<double> context_sum(data_.conditional_rank, 0.0);
        for (uint32_t subspace = 0; subspace < data_.subspaces; ++subspace) {
            for (uint32_t code = 0; code < data_.codewords; ++code) {
                const double weight = static_cast<double>(primary_histogram[
                    static_cast<uint64_t>(subspace) * data_.codewords + code]);
                const float* embedding = data_.conditional_embeddings.data() +
                    (static_cast<uint64_t>(subspace) * data_.codewords + code) *
                        data_.conditional_rank;
                for (uint32_t rank = 0; rank < data_.conditional_rank; ++rank) {
                    context_sum[rank] += weight * embedding[rank];
                }
            }
        }
        for (uint32_t rank = 0; rank < data_.conditional_rank; ++rank) {
            const float* basis = decoded_conditional_basis_.data() +
                static_cast<uint64_t>(rank) * data_.dim;
            for (uint32_t coordinate = 0; coordinate < data_.dim; ++coordinate) {
                sum[coordinate] += context_sum[rank] * basis[coordinate];
            }
        }
    }
    for (uint32_t group = 0; group < data_.residual_groups; ++group) {
        for (uint32_t code = 0; code < data_.residual_codewords; ++code) {
            const double weight = static_cast<double>(residual_histogram[
                static_cast<uint64_t>(group) * data_.residual_codewords + code]);
            const float* prototype = decoded_residual_pq_prototypes_.data() +
                (static_cast<uint64_t>(group) * data_.residual_codewords + code) *
                    data_.dim;
            for (uint32_t coordinate = 0; coordinate < data_.dim; ++coordinate) {
                sum[coordinate] +=
                    weight * prototype[coordinate];
            }
        }
    }
    std::vector<float> mean(data_.dim);
    const double inverse_nodes = 1.0 / static_cast<double>(data_.node_count);
    for (uint32_t coordinate = 0; coordinate < data_.dim; ++coordinate) {
        mean[coordinate] = static_cast<float>(sum[coordinate] * inverse_nodes);
    }
    return mean;
}

uint64_t DrbPqIndex::BasePairLookupBytes() const {
    return static_cast<uint64_t>(primary_pair_lut_.size() +
                                 residual_pair_lut_.size() +
                                 primary_residual_pair_lut_.size()) *
               sizeof(float) +
           static_cast<uint64_t>(cross_pair_blocks_.size()) *
               sizeof(CrossPairBlock);
}

uint32_t DrbPqIndex::DecodeBranch(const uint8_t* side_codes, uint32_t ridx) const {
    const uint32_t byte_offset = side_byte_offsets_[ridx];
    uint32_t word = side_codes[byte_offset];
    if (side_crosses_byte_[ridx] != 0) {
        word |= static_cast<uint32_t>(side_codes[byte_offset + 1U]) << 8U;
    }
    return (word >> side_shifts_[ridx]) & side_masks_[ridx];
}

QueryContext DrbPqIndex::PrepareQuery(const float* query,
                                     uint32_t dim,
                                     bool prepare_exact,
                                     ScoreMode mode) const {
    if (query == nullptr || dim != data_.dim) {
        throw std::invalid_argument("query dimension does not match DRB model");
    }
    QueryContext context;
    context.transformed_query.resize(dim);
    if (data_.has_transform) {
        for (uint32_t row = 0; row < dim; ++row) {
            context.transformed_query[row] =
                Dot(data_.transform.data() + static_cast<uint64_t>(row) * dim, query, dim) +
                data_.transform_bias[row];
        }
    } else {
        std::copy(query, query + dim, context.transformed_query.begin());
    }

    if (prepare_exact) {
        context.exact_linear_query.resize(dim);
        if (data_.has_transform) {
            for (uint32_t col = 0; col < dim; ++col) {
                double value = 0.0;
                for (uint32_t row = 0; row < dim; ++row) {
                    value += static_cast<double>(
                                 data_.transform[static_cast<uint64_t>(row) * dim + col]) *
                             context.transformed_query[row];
                }
                context.exact_linear_query[col] = static_cast<float>(value);
            }
            context.exact_bias =
                Dot(context.transformed_query.data(), data_.transform_bias.data(), dim);
        } else {
            context.exact_linear_query = context.transformed_query;
            context.exact_bias = 0.0f;
        }
    }

    context.adc_lut.resize(static_cast<uint64_t>(data_.subspaces) * data_.codewords, 0.0f);
    const bool refined_mode = mode == ScoreMode::kDrb || mode == ScoreMode::kVectorDrb;
    const bool build_all_adc = mode == ScoreMode::kAdc || mode == ScoreMode::kExact ||
                               prepare_exact || !refined_mode;
    const auto build_adc_subspace = [&](uint32_t subspace) {
        const float* q = context.transformed_query.data() +
                         static_cast<uint64_t>(subspace) * data_.dsub;
        for (uint32_t code = 0; code < data_.codewords; ++code) {
            const float* centroid = data_.centroids.data() +
                (static_cast<uint64_t>(subspace) * data_.codewords + code) * data_.dsub;
            context.adc_lut[static_cast<uint64_t>(subspace) * data_.codewords + code] =
                Dot(q, centroid, data_.dsub);
        }
    };
    if (build_all_adc) {
        for (uint32_t subspace = 0; subspace < data_.subspaces; ++subspace) {
            build_adc_subspace(subspace);
        }
    } else {
        for (uint32_t subspace : unrefined_subspaces_) build_adc_subspace(subspace);
    }

    if (mode == ScoreMode::kResidualPq) {
        context.residual_lut.resize(
            static_cast<uint64_t>(data_.residual_groups) *
            data_.residual_codewords);
        std::vector<float> residual_query(data_.dim);
        if (data_.has_residual_transform) {
            for (uint32_t coordinate = 0; coordinate < data_.dim; ++coordinate) {
                residual_query[coordinate] = Dot(
                    data_.residual_inverse_transform.data() +
                        static_cast<uint64_t>(coordinate) * data_.dim,
                    context.transformed_query.data(), data_.dim);
            }
        } else {
            residual_query = context.transformed_query;
        }
        if (data_.primary_conditioned) {
            std::vector<float> projection(data_.conditional_rank, 0.0f);
            for (uint32_t rank = 0; rank < data_.conditional_rank; ++rank) {
                double value = 0.0;
                for (uint32_t coordinate = 0; coordinate < data_.dim;
                     ++coordinate) {
                    value += static_cast<double>(residual_query[coordinate]) *
                        data_.conditional_basis[
                            static_cast<uint64_t>(coordinate) *
                                data_.conditional_rank + rank];
                }
                projection[rank] = static_cast<float>(value);
            }
            for (uint32_t subspace = 0; subspace < data_.subspaces;
                 ++subspace) {
                float* lut = context.adc_lut.data() +
                    static_cast<uint64_t>(subspace) * data_.codewords;
                for (uint32_t code = 0; code < data_.codewords; ++code) {
                    lut[code] += Dot(
                        data_.conditional_embeddings.data() +
                            (static_cast<uint64_t>(subspace) *
                                 data_.codewords + code) *
                                data_.conditional_rank,
                        projection.data(), data_.conditional_rank);
                }
            }
        }
        for (uint32_t group = 0; group < data_.residual_groups; ++group) {
            const float* q = residual_query.data() +
                (data_.residual_additive
                     ? 0
                     : static_cast<uint64_t>(group) *
                           data_.residual_group_dim);
            for (uint32_t code = 0; code < data_.residual_codewords; ++code) {
                const float* prototype = data_.residual_pq_prototypes.data() +
                    (static_cast<uint64_t>(group) * data_.residual_codewords + code) *
                        data_.residual_group_dim;
                context.residual_lut[
                    static_cast<uint64_t>(group) * data_.residual_codewords + code] =
                    Dot(q, prototype, data_.residual_group_dim);
            }
        }
        return context;
    }
    if (!refined_mode) return context;
    context.drb_fused_lut.resize(branch_lut_offsets_.back());
    for (uint32_t ridx = 0; ridx < data_.refined_subspaces; ++ridx) {
        const uint32_t subspace = data_.selected_subspaces[ridx];
        const uint32_t branches = data_.branches_per_refined[ridx];
        const float* q = context.transformed_query.data() +
                         static_cast<uint64_t>(subspace) * data_.dsub;
        float scalar_h = 0.0f;
        if (data_.refinement_kind == RefinementKind::kScalar) {
            const float* direction = data_.directions.data() +
                                     static_cast<uint64_t>(ridx) * data_.dsub;
            scalar_h = Dot(q, direction, data_.dsub);
        }
        for (uint32_t code = 0; code < data_.codewords; ++code) {
            for (uint32_t branch = 0; branch < branches; ++branch) {
                const uint64_t index = branch_lut_offsets_[ridx] +
                                       static_cast<uint64_t>(code) * branches + branch;
                if (data_.refinement_kind == RefinementKind::kScalar) {
                    float adc = context.adc_lut[
                        static_cast<uint64_t>(subspace) * data_.codewords + code];
                    if (!build_all_adc) {
                        const float* centroid = data_.centroids.data() +
                            (static_cast<uint64_t>(subspace) * data_.codewords + code) *
                                data_.dsub;
                        adc = Dot(q, centroid, data_.dsub);
                    }
                    context.drb_fused_lut[index] = adc + scalar_h * data_.beta[index];
                } else {
                    const float* effective =
                        effective_centroids_.data() + index * data_.dsub;
                    // OPQ200/PQ100 has dsub=2.  Keeping this hot LUT build in
                    // float avoids two float-to-double conversions per entry
                    // and lets the compiler vectorize consecutive prototypes.
                    if (data_.dsub == 2) {
                        context.drb_fused_lut[index] =
                            q[0] * effective[0] + q[1] * effective[1];
                    } else {
                        float value = 0.0f;
                        for (uint32_t dim_index = 0;
                             dim_index < data_.dsub; ++dim_index) {
                            value += q[dim_index] * effective[dim_index];
                        }
                        context.drb_fused_lut[index] = value;
                    }
                }
            }
        }
    }
    return context;
}

float DrbPqIndex::ExactScore(const QueryContext& query,
                             uint32_t node_id,
                             const MatrixView& base) const {
    if (node_id >= data_.node_count || base.rows < data_.node_count || base.dim != data_.dim ||
        query.exact_linear_query.size() != data_.dim) {
        throw std::out_of_range("exact-score node/base shape mismatch");
    }
    return Dot(query.exact_linear_query.data(), base.Row(node_id), data_.dim) + query.exact_bias;
}

float DrbPqIndex::AdcScore(const QueryContext& query, uint32_t node_id) const {
    if (node_id >= data_.node_count) {
        throw std::out_of_range("ADC node id is out of range");
    }
    const uint8_t* codes = NodeRecord(node_id);
    double score = 0.0;
    for (uint32_t subspace = 0; subspace < data_.subspaces; ++subspace) {
        score += query.adc_lut[static_cast<uint64_t>(subspace) * data_.codewords + codes[subspace]];
    }
    return static_cast<float>(score);
}

float DrbPqIndex::DrbScore(const QueryContext& query, uint32_t node_id) const {
    if (data_.refinement_kind != RefinementKind::kScalar) {
        throw std::invalid_argument("scalar DRB score requires a scalar DRB model");
    }
    return RefinedScore(query, node_id);
}

float DrbPqIndex::VectorDrbScore(const QueryContext& query, uint32_t node_id) const {
    if (data_.refinement_kind != RefinementKind::kVector) {
        throw std::invalid_argument("vector DRB score requires a vector DRB model");
    }
    return RefinedScore(query, node_id);
}

float DrbPqIndex::ResidualPqScore(const QueryContext& query,
                                  uint32_t node_id) const {
    if (data_.refinement_kind != RefinementKind::kResidualPq) {
        throw std::invalid_argument("RPQ score requires a residual-PQ model");
    }
    if (node_id >= data_.node_count ||
        query.residual_lut.size() !=
            static_cast<uint64_t>(data_.residual_groups) *
                data_.residual_codewords) {
        throw std::out_of_range("residual-PQ node/query shape mismatch");
    }
    const uint8_t* record = NodeRecord(node_id);
    double adc = 0.0;
    for (uint32_t subspace = 0; subspace < data_.subspaces; ++subspace) {
        adc += query.adc_lut[
            static_cast<uint64_t>(subspace) * data_.codewords + record[subspace]];
    }
    const uint8_t* residual_codes = record + data_.subspaces;
    float correction = 0.0f;
    for (uint32_t group = 0; group < data_.residual_groups; ++group) {
        correction += query.residual_lut[
            static_cast<uint64_t>(group) * data_.residual_codewords +
            residual_codes[group]];
    }
    return static_cast<float>(adc) + correction;
}

float DrbPqIndex::RefinedScore(const QueryContext& query, uint32_t node_id) const {
    if (node_id >= data_.node_count) {
        throw std::out_of_range("DRB node id is out of range");
    }
    if (data_.refined_subspaces == 0) return AdcScore(query, node_id);

    const uint8_t* codes = NodeRecord(node_id);
    const uint8_t* bits = codes + data_.subspaces;
    double score = 0.0;
    for (uint32_t subspace : unrefined_subspaces_) {
        score += query.adc_lut[
            static_cast<uint64_t>(subspace) * data_.codewords + codes[subspace]];
    }
    for (uint32_t ridx = 0; ridx < data_.refined_subspaces; ++ridx) {
        const uint32_t branch = DecodeBranch(bits, ridx);
        const uint32_t code = codes[data_.selected_subspaces[ridx]];
        const uint64_t index = branch_lut_offsets_[ridx] +
                               static_cast<uint64_t>(code) *
                                   data_.branches_per_refined[ridx] +
                               branch;
        score += query.drb_fused_lut[index];
    }
    return static_cast<float>(score);
}

void DrbPqIndex::ScoreBatch(const QueryContext& query,
                            const uint32_t* node_ids,
                            size_t count,
                            ScoreMode mode,
                            const MatrixView& base,
                            float* scores) const {
    if ((count != 0 && (node_ids == nullptr || scores == nullptr))) {
        throw std::invalid_argument("batch score received a null buffer");
    }
    constexpr size_t kPrefetchDistance = 8;

    size_t i = 0;
#if defined(__AVX2__)
    // Score eight unrelated graph neighbors together. AVX2 gathers the eight
    // LUT entries for one subspace, while double accumulators preserve the
    // scalar score's summation order for every node.
    if (mode == ScoreMode::kAdc || mode == ScoreMode::kDrb ||
        mode == ScoreMode::kVectorDrb || mode == ScoreMode::kResidualPq) {
        for (; i + 8 <= count; i += 8) {
            const uint8_t* records[8];
            for (size_t lane = 0; lane < 8; ++lane) {
                if (node_ids[i + lane] >= data_.node_count) {
                    throw std::out_of_range("batch score node id is out of range");
                }
                records[lane] = NodeRecord(node_ids[i + lane]);
            }
            if (i + 16 <= count) {
                for (size_t lane = 0; lane < 8; ++lane) {
                    const uint32_t future = node_ids[i + 8 + lane];
                    if (future < data_.node_count) {
                        __builtin_prefetch(NodeRecord(future), 0, 1);
                    }
                }
            }

            // Vector DRB performs 100 independent float LUT lookups per node.
            // Accumulating the eight lanes as float vectors avoids two
            // cvtps_pd operations and two double additions for every lookup.
            // ADC and scalar DRB retain the historical double accumulator.
            if (mode == ScoreMode::kResidualPq) {
                __m256d adc_low = _mm256_setzero_pd();
                __m256d adc_high = _mm256_setzero_pd();
                for (uint32_t subspace = 0; subspace < data_.subspaces; ++subspace) {
                    const __m256i indices = _mm256_setr_epi32(
                        records[0][subspace], records[1][subspace],
                        records[2][subspace], records[3][subspace],
                        records[4][subspace], records[5][subspace],
                        records[6][subspace], records[7][subspace]);
                    const __m256 values = _mm256_i32gather_ps(
                        query.adc_lut.data() +
                            static_cast<uint64_t>(subspace) * data_.codewords,
                        indices, 4);
                    adc_low = _mm256_add_pd(
                        adc_low, _mm256_cvtps_pd(_mm256_castps256_ps128(values)));
                    adc_high = _mm256_add_pd(
                        adc_high, _mm256_cvtps_pd(_mm256_extractf128_ps(values, 1)));
                }
                alignas(32) double adc_scores[8];
                _mm256_store_pd(adc_scores, adc_low);
                _mm256_store_pd(adc_scores + 4, adc_high);
                const __m256 adc_vector = _mm256_setr_ps(
                    static_cast<float>(adc_scores[0]),
                    static_cast<float>(adc_scores[1]),
                    static_cast<float>(adc_scores[2]),
                    static_cast<float>(adc_scores[3]),
                    static_cast<float>(adc_scores[4]),
                    static_cast<float>(adc_scores[5]),
                    static_cast<float>(adc_scores[6]),
                    static_cast<float>(adc_scores[7]));
                __m256 correction = _mm256_setzero_ps();
                for (uint32_t group = 0; group < data_.residual_groups; ++group) {
                    const uint32_t offset = data_.subspaces + group;
                    const __m256i indices = _mm256_setr_epi32(
                        records[0][offset], records[1][offset],
                        records[2][offset], records[3][offset],
                        records[4][offset], records[5][offset],
                        records[6][offset], records[7][offset]);
                    correction = _mm256_add_ps(
                        correction,
                        _mm256_i32gather_ps(
                            query.residual_lut.data() +
                                static_cast<uint64_t>(group) *
                                    data_.residual_codewords,
                            indices, 4));
                }
                _mm256_storeu_ps(scores + i,
                                 _mm256_add_ps(adc_vector, correction));
                continue;
            }

            if (mode == ScoreMode::kVectorDrb && data_.refined_subspaces != 0) {
                __m256 vector_scores = _mm256_setzero_ps();
                const auto accumulate_float = [&](const float* lut,
                                                  const __m256i indices) {
                    vector_scores = _mm256_add_ps(
                        vector_scores, _mm256_i32gather_ps(lut, indices, 4));
                };
                for (uint32_t subspace : unrefined_subspaces_) {
                    const __m256i indices = _mm256_setr_epi32(
                        records[0][subspace], records[1][subspace],
                        records[2][subspace], records[3][subspace],
                        records[4][subspace], records[5][subspace],
                        records[6][subspace], records[7][subspace]);
                    accumulate_float(query.adc_lut.data() +
                                         static_cast<uint64_t>(subspace) *
                                             data_.codewords,
                                     indices);
                }
                for (uint32_t ridx = 0; ridx < data_.refined_subspaces; ++ridx) {
                    const uint32_t subspace = data_.selected_subspaces[ridx];
                    const uint32_t byte_offset = side_byte_offsets_[ridx];
                    const uint32_t shift = side_shifts_[ridx];
                    const uint32_t mask = side_masks_[ridx];
                    const bool crosses_byte = side_crosses_byte_[ridx] != 0;
                    const uint8_t* side0 = records[0] + data_.subspaces;
                    const uint8_t* side1 = records[1] + data_.subspaces;
                    const uint8_t* side2 = records[2] + data_.subspaces;
                    const uint8_t* side3 = records[3] + data_.subspaces;
                    const uint8_t* side4 = records[4] + data_.subspaces;
                    const uint8_t* side5 = records[5] + data_.subspaces;
                    const uint8_t* side6 = records[6] + data_.subspaces;
                    const uint8_t* side7 = records[7] + data_.subspaces;
                    const auto side_word = [&](const uint8_t* side) {
                        uint32_t word = side[byte_offset];
                        if (crosses_byte) {
                            word |= static_cast<uint32_t>(side[byte_offset + 1U]) << 8U;
                        }
                        return static_cast<int>(word);
                    };
                    const __m256i words = _mm256_setr_epi32(
                        side_word(side0), side_word(side1), side_word(side2),
                        side_word(side3), side_word(side4), side_word(side5),
                        side_word(side6), side_word(side7));
                    const __m256i branch_values = _mm256_and_si256(
                        _mm256_srlv_epi32(words, _mm256_set1_epi32(shift)),
                        _mm256_set1_epi32(mask));
                    const __m256i code_values = _mm256_setr_epi32(
                        records[0][subspace], records[1][subspace],
                        records[2][subspace], records[3][subspace],
                        records[4][subspace], records[5][subspace],
                        records[6][subspace], records[7][subspace]);
                    const __m256i indices = _mm256_or_si256(
                        _mm256_sllv_epi32(
                            code_values,
                            _mm256_set1_epi32(data_.bits_per_refined[ridx])),
                        branch_values);
                    accumulate_float(query.drb_fused_lut.data() +
                                         branch_lut_offsets_[ridx],
                                     indices);
                }
                _mm256_storeu_ps(scores + i, vector_scores);
                continue;
            }

            __m256d score_low = _mm256_setzero_pd();
            __m256d score_high = _mm256_setzero_pd();
            const auto accumulate = [&](const float* lut, const __m256i indices) {
                const __m256 values = _mm256_i32gather_ps(lut, indices, 4);
                score_low = _mm256_add_pd(
                    score_low, _mm256_cvtps_pd(_mm256_castps256_ps128(values)));
                score_high = _mm256_add_pd(
                    score_high, _mm256_cvtps_pd(_mm256_extractf128_ps(values, 1)));
            };

            if (mode == ScoreMode::kAdc || data_.refined_subspaces == 0) {
                for (uint32_t subspace = 0; subspace < data_.subspaces; ++subspace) {
                    const __m256i indices = _mm256_setr_epi32(
                        records[0][subspace], records[1][subspace],
                        records[2][subspace], records[3][subspace],
                        records[4][subspace], records[5][subspace],
                        records[6][subspace], records[7][subspace]);
                    accumulate(query.adc_lut.data() +
                                   static_cast<uint64_t>(subspace) * data_.codewords,
                               indices);
                }
            } else {
                for (uint32_t subspace : unrefined_subspaces_) {
                    const __m256i indices = _mm256_setr_epi32(
                        records[0][subspace], records[1][subspace],
                        records[2][subspace], records[3][subspace],
                        records[4][subspace], records[5][subspace],
                        records[6][subspace], records[7][subspace]);
                    accumulate(query.adc_lut.data() +
                                   static_cast<uint64_t>(subspace) * data_.codewords,
                               indices);
                }
                for (uint32_t ridx = 0; ridx < data_.refined_subspaces; ++ridx) {
                    const uint32_t subspace = data_.selected_subspaces[ridx];
                    const uint32_t branches = data_.branches_per_refined[ridx];
                    const uint32_t byte_offset = side_byte_offsets_[ridx];
                    const uint32_t shift = side_shifts_[ridx];
                    const uint32_t mask = side_masks_[ridx];
                    const bool crosses_byte = side_crosses_byte_[ridx] != 0;
                    int indices_array[8];
                    for (size_t lane = 0; lane < 8; ++lane) {
                        const uint8_t* bits = records[lane] + data_.subspaces;
                        uint32_t word = bits[byte_offset];
                        if (crosses_byte) {
                            word |= static_cast<uint32_t>(bits[byte_offset + 1U]) << 8U;
                        }
                        const uint32_t branch = (word >> shift) & mask;
                        indices_array[lane] = branches * records[lane][subspace] + branch;
                    }
                    const __m256i indices = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(indices_array));
                    accumulate(query.drb_fused_lut.data() +
                                   branch_lut_offsets_[ridx],
                               indices);
                }
            }

            alignas(32) double lane_scores[8];
            _mm256_store_pd(lane_scores, score_low);
            _mm256_store_pd(lane_scores + 4, score_high);
            for (size_t lane = 0; lane < 8; ++lane) {
                scores[i + lane] = static_cast<float>(lane_scores[lane]);
            }
        }
    }
#endif

    for (; i < count; ++i) {
        if (i + kPrefetchDistance < count) {
            const uint32_t future = node_ids[i + kPrefetchDistance];
            if (future < data_.node_count) {
                if (mode == ScoreMode::kExact) {
                    __builtin_prefetch(base.data +
                        static_cast<uint64_t>(future) * base.dim, 0, 1);
                } else {
                    __builtin_prefetch(NodeRecord(future), 0, 1);
                }
            }
        }
        scores[i] = Score(query, node_ids[i], mode, base);
    }
}

float DrbPqIndex::Score(const QueryContext& query,
                        uint32_t node_id,
                        ScoreMode mode,
                        const MatrixView& base) const {
    switch (mode) {
        case ScoreMode::kExact:
            return ExactScore(query, node_id, base);
        case ScoreMode::kAdc:
            return AdcScore(query, node_id);
        case ScoreMode::kDrb:
            return DrbScore(query, node_id);
        case ScoreMode::kVectorDrb:
            return VectorDrbScore(query, node_id);
        case ScoreMode::kResidualPq:
            return ResidualPqScore(query, node_id);
    }
    throw std::logic_error("unknown score mode");
}

ScoreMode ParseScoreMode(const std::string& value) {
    if (value == "exact") return ScoreMode::kExact;
    if (value == "adc") return ScoreMode::kAdc;
    if (value == "drb") return ScoreMode::kDrb;
    if (value == "vdrb") return ScoreMode::kVectorDrb;
    if (value == "rpq") return ScoreMode::kResidualPq;
    throw std::invalid_argument(
        "score mode must be exact, adc, drb, vdrb, or rpq");
}

const char* ScoreModeName(ScoreMode mode) {
    switch (mode) {
        case ScoreMode::kExact:
            return "exact";
        case ScoreMode::kAdc:
            return "adc";
        case ScoreMode::kDrb:
            return "drb";
        case ScoreMode::kVectorDrb:
            return "vdrb";
        case ScoreMode::kResidualPq:
            return "rpq";
    }
    return "unknown";
}

}  // namespace ours
