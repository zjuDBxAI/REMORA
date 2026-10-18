#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/quantization/rabitq_impl.hpp"
#include "rabitqlib/utils/rotator.hpp"

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

class Mapping {
  public:
    Mapping(const std::string &path, bool writable, size_t create_size = 0) {
        const int flags = writable ? (O_RDWR | O_CREAT | O_TRUNC) : O_RDONLY;
        fd_ = open(path.c_str(), flags, 0644);
        if (fd_ < 0) throw std::runtime_error("cannot open: " + path);
        if (writable) {
            size_ = create_size;
            if (ftruncate(fd_, static_cast<off_t>(size_)) != 0)
                throw std::runtime_error("cannot resize: " + path);
        } else {
            struct stat st {};
            if (fstat(fd_, &st) != 0) throw std::runtime_error("cannot stat: " + path);
            size_ = static_cast<size_t>(st.st_size);
        }
        const int prot = writable ? (PROT_READ | PROT_WRITE) : PROT_READ;
        data_ = mmap(nullptr, size_, prot, MAP_SHARED, fd_, 0);
        if (data_ == MAP_FAILED) throw std::runtime_error("cannot mmap: " + path);
        writable_ = writable;
    }
    ~Mapping() {
        if (data_ != MAP_FAILED) {
            if (writable_) msync(data_, size_, MS_SYNC);
            munmap(data_, size_);
        }
        if (fd_ >= 0) close(fd_);
    }
    void *data() { return data_; }
    const void *data() const { return data_; }
    size_t size() const { return size_; }

  private:
    int fd_ = -1;
    void *data_ = MAP_FAILED;
    size_t size_ = 0;
    bool writable_ = false;
};

struct FbinView {
    explicit FbinView(const std::string &path) : mapping(path, false) {
        if (mapping.size() < 8) throw std::runtime_error("truncated fbin: " + path);
        const auto *header = static_cast<const uint32_t *>(mapping.data());
        rows = header[0];
        dim = header[1];
        if (mapping.size() != 8ULL + static_cast<uint64_t>(rows) * dim * sizeof(float))
            throw std::runtime_error("fbin size mismatch: " + path);
        values = reinterpret_cast<const float *>(static_cast<const char *>(mapping.data()) + 8);
    }
    const float *row(uint64_t i) const { return values + i * dim; }
    Mapping mapping;
    uint32_t rows = 0;
    uint32_t dim = 0;
    const float *values = nullptr;
};

struct FbinOutput {
    FbinOutput(const std::string &path, uint32_t rows_value, uint32_t dim_value)
        : rows(rows_value), dim(dim_value),
          mapping(path, true, 8ULL + static_cast<size_t>(rows) * dim * sizeof(float)) {
        auto *header = static_cast<uint32_t *>(mapping.data());
        header[0] = rows;
        header[1] = dim;
        values = reinterpret_cast<float *>(static_cast<char *>(mapping.data()) + 8);
    }
    float *row(uint64_t i) { return values + i * dim; }
    uint32_t rows;
    uint32_t dim;
    Mapping mapping;
    float *values = nullptr;
};

void write_header(std::ofstream &output, uint32_t rows, uint32_t cols) {
    output.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    output.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
}

void usage(const char *program) {
    std::cerr << "Usage: " << program
              << " QUANT.roarq BASE.fbin QUERY.fbin OUT_DIR [base_limit=0]"
                 " [query_limit=20000] [threads=64]\n";
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 5 || argc > 8) {
        usage(argv[0]);
        return 2;
    }
    const std::string quant_path = argv[1];
    const std::string base_path = argv[2];
    const std::string query_path = argv[3];
    const std::string out_dir = argv[4];
    const uint64_t base_limit = argc > 5 ? std::stoull(argv[5]) : 0;
    const uint64_t query_limit = argc > 6 ? std::stoull(argv[6]) : 20000;
    const int threads = argc > 7 ? std::stoi(argv[7]) : 64;
    if (query_limit == 0 || threads <= 0) throw std::runtime_error("invalid limits/threads");
    std::filesystem::create_directories(out_dir);

    Mapping quant_mapping(quant_path, false);
    if (quant_mapping.size() < sizeof(QuantFileHeader)) throw std::runtime_error("truncated RaBitQ file");
    QuantFileHeader header {};
    std::memcpy(&header, quant_mapping.data(), sizeof(header));
    if (std::memcmp(header.magic, "RBQROAR1", 8) != 0 || header.version != 1)
        throw std::runtime_error("unsupported RaBitQ file");
    if (header.metric_type != rabitqlib::METRIC_IP || header.total_bits < 2)
        throw std::runtime_error("adapter requires multi-bit IP RaBitQ codes");
    const uint64_t expected = sizeof(header) + header.rotator_bytes +
                              header.num_points * header.bin_bytes_per_vec +
                              header.num_points * header.ex_bytes_per_vec;
    if (quant_mapping.size() != expected) throw std::runtime_error("RaBitQ file-size mismatch");

    FbinView base(base_path), queries(query_path);
    if (base.rows != header.num_points || base.dim != header.dim || queries.dim != header.dim)
        throw std::runtime_error("RaBitQ/base/query metadata mismatch");
    const uint64_t rows = base_limit ? std::min<uint64_t>(base.rows, base_limit) : base.rows;
    const uint64_t query_rows = std::min<uint64_t>(queries.rows, query_limit);
    if (rows > std::numeric_limits<uint32_t>::max() || query_rows > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("fbin row count exceeds uint32");

    auto *rotator = rabitqlib::choose_rotator<float>(
        header.dim, static_cast<rabitqlib::RotatorType>(header.rotator_type), header.padded_dim);
    rotator->load(static_cast<const char *>(quant_mapping.data()) + sizeof(header));
    const char *bin_begin = static_cast<const char *>(quant_mapping.data()) + sizeof(header) + header.rotator_bytes;
    const char *ex_begin = bin_begin + header.num_points * header.bin_bytes_per_vec;
    const size_t ex_bits = header.total_bits - 1;

    // Detect which official RaBitQ+ training configuration generated the file.
    std::vector<float> rotated(header.padded_dim), centroid(header.padded_dim, 0.0f);
    rotator->rotate(base.row(0), rotated.data());
    std::vector<char> bin_probe(header.bin_bytes_per_vec), ex_probe(header.ex_bytes_per_vec);
    std::string quantizer_config;
    rabitqlib::quant::RabitqConfig selected_config;
    for (const bool faster : {false, true}) {
        auto config = faster ? rabitqlib::quant::faster_config(header.padded_dim, header.total_bits)
                             : rabitqlib::quant::RabitqConfig();
        rabitqlib::quant::quantize_split_single(
            rotated.data(), centroid.data(), header.padded_dim, ex_bits,
            bin_probe.data(), ex_probe.data(), rabitqlib::METRIC_IP, config);
        if (std::memcmp(bin_probe.data(), bin_begin, header.bin_bytes_per_vec) == 0 &&
            std::memcmp(ex_probe.data(), ex_begin, header.ex_bytes_per_vec) == 0) {
            selected_config = config;
            quantizer_config = faster ? "faster" : "exact";
            break;
        }
    }
    if (quantizer_config.empty())
        throw std::runtime_error("cannot reproduce stored RaBitQ code with official configurations");

    const std::string transformed_base_path = out_dir + "/base.transformed.fbin";
    const std::string reconstruction_path = out_dir + "/primary.reconstruction.fbin";
    const std::string transformed_query_path = out_dir + "/query.transformed.fbin";
    FbinOutput transformed_base(transformed_base_path, rows, header.padded_dim);
    FbinOutput reconstruction(reconstruction_path, rows, header.padded_dim);
    omp_set_num_threads(threads);
#pragma omp parallel
    {
        std::vector<float> rotated_local(header.padded_dim);
        std::vector<float> decoded(header.padded_dim);
        std::vector<uint8_t> raw_code(header.padded_dim);
#pragma omp for schedule(static)
        for (uint64_t i = 0; i < rows; ++i) {
            rotator->rotate(base.row(i), rotated_local.data());
            float f_add = 0.0f, f_rescale = 0.0f, f_error = 0.0f;
            rabitqlib::quant::rabitq_impl::ex_bits::ex_bits_code_with_factor(
                rotated_local.data(), centroid.data(), header.padded_dim, ex_bits,
                raw_code.data(), f_add, f_rescale, f_error,
                rabitqlib::METRIC_IP, selected_config.t_const);
            const float scale = -f_rescale;
            const int sign_bit = 1 << ex_bits;
            const float center = -static_cast<float>(sign_bit) + 0.5f;
            for (size_t j = 0; j < header.padded_dim; ++j) {
                const int total_code = static_cast<int>(raw_code[j]) +
                                       (rotated_local[j] >= 0.0f ? sign_bit : 0);
                decoded[j] = scale * (static_cast<float>(total_code) + center);
            }
            std::memcpy(transformed_base.row(i), rotated_local.data(), header.padded_dim * sizeof(float));
            std::memcpy(reconstruction.row(i), decoded.data(), header.padded_dim * sizeof(float));
        }
    }

    FbinOutput transformed_queries(transformed_query_path, query_rows, header.padded_dim);
#pragma omp parallel
    {
        std::vector<float> rotated_local(header.padded_dim);
#pragma omp for schedule(static)
        for (uint64_t i = 0; i < query_rows; ++i) {
            rotator->rotate(queries.row(i), rotated_local.data());
            std::memcpy(transformed_queries.row(i), rotated_local.data(), header.padded_dim * sizeof(float));
        }
    }

    constexpr uint32_t parity_queries = 16;
    constexpr uint32_t parity_candidates = 256;
    const uint32_t parity_q = std::min<uint64_t>(parity_queries, query_rows);
    const uint32_t parity_c = std::min<uint64_t>(parity_candidates, rows);
    std::mt19937_64 rng(20260831);
    std::vector<uint32_t> ids(rows);
    std::iota(ids.begin(), ids.end(), 0u);
    std::shuffle(ids.begin(), ids.end(), rng);
    ids.resize(parity_c);
    std::sort(ids.begin(), ids.end());
    const std::string ids_path = out_dir + "/parity.candidate_ids.ibin";
    const std::string score_path = out_dir + "/parity.official_scores.fbin";
    std::ofstream ids_out(ids_path, std::ios::binary), score_out(score_path, std::ios::binary);
    write_header(ids_out, parity_q, parity_c);
    write_header(score_out, parity_q, parity_c);
    std::vector<float> rotated_query(header.padded_dim);
    const auto ip_func = rabitqlib::select_excode_ipfunc(ex_bits);
    for (uint32_t qi = 0; qi < parity_q; ++qi) {
        rotator->rotate(queries.row(qi), rotated_query.data());
        rabitqlib::SplitSingleQuery<float> query(
            rotated_query.data(), header.padded_dim, ex_bits, selected_config,
            rabitqlib::METRIC_IP);
        for (uint32_t id : ids) {
            float distance = 0.0f, lower = 0.0f, binary_ip = 0.0f;
            rabitqlib::split_single_fulldist(
                bin_begin + static_cast<uint64_t>(id) * header.bin_bytes_per_vec,
                ex_begin + static_cast<uint64_t>(id) * header.ex_bytes_per_vec,
                ip_func, query, header.padded_dim, ex_bits, distance, lower,
                binary_ip, 0.0f, 0.0f);
            const float similarity = 1.0f - distance;
            ids_out.write(reinterpret_cast<const char *>(&id), sizeof(id));
            score_out.write(reinterpret_cast<const char *>(&similarity), sizeof(similarity));
        }
    }

    const std::string summary_path = out_dir + "/export_summary.json";
    std::ofstream summary(summary_path);
    summary << std::setprecision(12)
            << "{\n  \"version\": 1,\n  \"implementation\": \"official-extended-rabitq-full-ip-adapter\",\n"
            << "  \"base_rows\": " << rows << ",\n  \"original_dim\": " << header.dim
            << ",\n  \"scorer_dim\": " << header.padded_dim << ",\n"
            << "  \"bits_per_dimension\": " << header.total_bits << ",\n"
            // Extended-RaBitQ actually scores/stores the padded coordinates.
            // Calling total_bits * original_dim the code budget hid both the
            // padding and the per-vector calibration metadata.  Use the real
            // coordinate payload as the nominal algorithmic rate, and report
            // the complete serialized footprint separately.
            << "  \"nominal_code_bits_per_node\": "
            << header.total_bits * header.padded_dim << ",\n"
            << "  \"original_dimension_nominal_bits_per_node\": "
            << header.total_bits * header.dim << ",\n"
            << "  \"coordinate_code_bits_per_node\": "
            << header.total_bits * header.padded_dim << ",\n"
            << "  \"physical_bytes_per_node\": "
            << header.bin_bytes_per_vec + header.ex_bytes_per_vec << ",\n"
            << "  \"physical_bits_per_node\": "
            << 8 * (header.bin_bytes_per_vec + header.ex_bytes_per_vec) << ",\n"
            << "  \"metadata_and_layout_overhead_bits_per_node\": "
            << 8 * (header.bin_bytes_per_vec + header.ex_bytes_per_vec) -
                   header.total_bits * header.padded_dim
            << ",\n"
            << "  \"quantizer_config\": \"" << quantizer_config << "\",\n"
            << "  \"official_score\": \"1 - split_single_fulldist(IP)\",\n"
            << "  \"files\": {\n"
            << "    \"quantized_codes\": \"" << std::filesystem::absolute(quant_path).string() << "\",\n"
            << "    \"base_transformed\": \"" << std::filesystem::absolute(transformed_base_path).string() << "\",\n"
            << "    \"reconstruction\": \"" << std::filesystem::absolute(reconstruction_path).string() << "\",\n"
            << "    \"query_transformed\": \"" << std::filesystem::absolute(transformed_query_path).string() << "\",\n"
            << "    \"parity_candidate_ids\": \"" << std::filesystem::absolute(ids_path).string() << "\",\n"
            << "    \"parity_official_scores\": \"" << std::filesystem::absolute(score_path).string() << "\"\n"
            << "  }\n}\n";
    delete rotator;
    std::cout << "Extended-RaBitQ primary export complete: " << summary_path << std::endl;
    return 0;
}
