#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

using data_type = rabitqlib::RowMajorArray<float>;
using id_type = rabitqlib::RowMajorArray<uint32_t>;

namespace {
struct QuantFileHeader {
    char magic[8];
    uint32_t version;
    uint32_t metric_type;
    uint32_t rotator_type;
    uint32_t total_bits;
    uint64_t num_points;
    uint64_t dim;
    uint64_t padded_dim;
    uint64_t bin_bytes_per_vec;
    uint64_t ex_bytes_per_vec;
    uint64_t rotator_bytes;
};

struct QuantStore {
    QuantFileHeader header{};
    std::vector<char> bin_all;
    std::vector<char> ex_all;
};

void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <quant_file> <query_file> <candidate_ids_file> <topk> <output_ibin>\n"
              << "query_file: .fvecs or .fbin\n"
              << "candidate_ids_file: .ivecs or .ibin (shape: nq x ncand)\n";
}

bool load_quant_store(const char* filename, QuantStore& qs) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        std::cerr << "Cannot open quant file: " << filename << '\n';
        return false;
    }

    input.read(reinterpret_cast<char*>(&qs.header), sizeof(QuantFileHeader));
    if (std::memcmp(qs.header.magic, "RBQROAR1", 8) != 0 || qs.header.version != 1) {
        std::cerr << "Bad quant file magic/version\n";
        return false;
    }

    rabitqlib::RotatorType rot_type =
        static_cast<rabitqlib::RotatorType>(qs.header.rotator_type);
    auto* rotator = rabitqlib::choose_rotator<float>(
        static_cast<size_t>(qs.header.dim), rot_type, static_cast<size_t>(qs.header.padded_dim)
    );
    rotator->load(input);

    const size_t num_points = static_cast<size_t>(qs.header.num_points);
    const size_t bin_bytes = static_cast<size_t>(qs.header.bin_bytes_per_vec);
    const size_t ex_bytes = static_cast<size_t>(qs.header.ex_bytes_per_vec);

    qs.bin_all.resize(num_points * bin_bytes);
    input.read(qs.bin_all.data(), static_cast<long>(qs.bin_all.size()));

    if (ex_bytes > 0) {
        qs.ex_all.resize(num_points * ex_bytes);
        input.read(qs.ex_all.data(), static_cast<long>(qs.ex_all.size()));
    }

    input.close();
    delete rotator;
    return true;
}

void save_ibin(const char* filename, const std::vector<uint32_t>& ids, uint32_t rows, uint32_t cols) {
    std::ofstream output(filename, std::ios::binary);
    if (!output.is_open()) {
        std::cerr << "Cannot open output file: " << filename << '\n';
        std::exit(1);
    }
    output.write(reinterpret_cast<const char*>(&rows), sizeof(uint32_t));
    output.write(reinterpret_cast<const char*>(&cols), sizeof(uint32_t));
    output.write(reinterpret_cast<const char*>(ids.data()), static_cast<long>(ids.size() * sizeof(uint32_t)));
    output.close();
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        usage(argv[0]);
        return 1;
    }

    const char* quant_file = argv[1];
    const char* query_file = argv[2];
    const char* candidate_file = argv[3];
    const size_t topk = static_cast<size_t>(std::atoi(argv[4]));
    const char* output_file = argv[5];

    if (topk == 0) {
        std::cerr << "topk must be > 0\n";
        return 1;
    }

    QuantStore qs;
    if (!load_quant_store(quant_file, qs)) {
        return 1;
    }

    const size_t num_points = static_cast<size_t>(qs.header.num_points);
    const size_t dim = static_cast<size_t>(qs.header.dim);
    const size_t padded_dim = static_cast<size_t>(qs.header.padded_dim);
    const size_t total_bits = static_cast<size_t>(qs.header.total_bits);
    const size_t ex_bits = total_bits - 1;
    const auto metric_type = static_cast<rabitqlib::MetricType>(qs.header.metric_type);
    const size_t bin_bytes = static_cast<size_t>(qs.header.bin_bytes_per_vec);
    const size_t ex_bytes = static_cast<size_t>(qs.header.ex_bytes_per_vec);
    const auto ip_func = rabitqlib::select_excode_ipfunc(ex_bits);

    rabitqlib::RotatorType rot_type =
        static_cast<rabitqlib::RotatorType>(qs.header.rotator_type);
    auto* rotator = rabitqlib::choose_rotator<float>(dim, rot_type, padded_dim);
    {
        std::ifstream input(quant_file, std::ios::binary);
        input.seekg(static_cast<std::streamoff>(sizeof(QuantFileHeader)), std::ios::beg);
        rotator->load(input);
    }

    data_type queries;
    id_type candidates;
    rabitqlib::load_data<float, data_type>(query_file, queries);
    rabitqlib::load_data<uint32_t, id_type>(candidate_file, candidates);

    const size_t nq = static_cast<size_t>(queries.rows());
    const size_t qdim = static_cast<size_t>(queries.cols());
    const size_t ncand = static_cast<size_t>(candidates.cols());

    if (queries.rows() != candidates.rows()) {
        std::cerr << "query rows and candidate rows mismatch\n";
        delete rotator;
        return 1;
    }
    if (qdim != dim) {
        std::cerr << "query dim does not match quant file dim\n";
        delete rotator;
        return 1;
    }
    if (ncand < topk) {
        std::cerr << "candidate columns < topk\n";
        delete rotator;
        return 1;
    }

    std::vector<uint32_t> output_ids(nq * topk, 0);
    std::vector<float> rotated_query(padded_dim, 0.F);
    rabitqlib::quant::RabitqConfig query_config =
        rabitqlib::quant::faster_config(padded_dim, total_bits);

    for (size_t qi = 0; qi < nq; ++qi) {
        rotator->rotate(&queries(qi, 0), rotated_query.data());
        rabitqlib::SplitSingleQuery<float> q_obj(
            rotated_query.data(), padded_dim, ex_bits, query_config, metric_type
        );

        float q_norm = std::sqrt(rabitqlib::dot_product(rotated_query.data(), rotated_query.data(), padded_dim));
        float g_add = (metric_type == rabitqlib::METRIC_L2) ? (q_norm * q_norm) : 0.F;
        float g_error = q_norm;

        std::vector<std::pair<float, uint32_t>> scored;
        scored.reserve(ncand);
        for (size_t ci = 0; ci < ncand; ++ci) {
            uint32_t pid = candidates(qi, ci);
            if (pid >= num_points) {
                continue;
            }

            const char* bin_ptr = qs.bin_all.data() + (static_cast<size_t>(pid) * bin_bytes);
            float ip_x0_qr = 0.F;
            float est_dist = 0.F;
            float low_dist = 0.F;
            rabitqlib::split_single_estdist(
                bin_ptr, q_obj, padded_dim, ip_x0_qr, est_dist, low_dist, g_add, g_error
            );

            if (ex_bits > 0) {
                const char* ex_ptr = qs.ex_all.data() + (static_cast<size_t>(pid) * ex_bytes);
                est_dist = rabitqlib::split_distance_boosting(
                    ex_ptr, ip_func, q_obj, padded_dim, ex_bits, ip_x0_qr
                );
            }
            scored.emplace_back(est_dist, pid);
        }

        if (scored.size() < topk) {
            std::cerr << "query " << qi << " has fewer valid candidates than topk\n";
            delete rotator;
            return 1;
        }

        std::partial_sort(
            scored.begin(),
            scored.begin() + static_cast<long>(topk),
            scored.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; }
        );

        for (size_t k = 0; k < topk; ++k) {
            output_ids[qi * topk + k] = scored[k].second;
        }
    }

    save_ibin(
        output_file,
        output_ids,
        static_cast<uint32_t>(nq),
        static_cast<uint32_t>(topk)
    );
    std::cout << "Saved reranked ids to " << output_file << '\n';

    delete rotator;
    return 0;
}
