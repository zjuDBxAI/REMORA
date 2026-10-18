#include <fcntl.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
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
#include <unordered_map>
#include <vector>

#include "quantization/saq_data.hpp"
#include "quantization/saq_estimator.hpp"
#include "quantization/saq_quantizer.hpp"
#include "quantization/single_data.hpp"
#include "utils/memory.hpp"

DEFINE_string(base_fbin, "", "input base vectors in fbin format");
DEFINE_string(query_fbin, "", "input queries in fbin format");
DEFINE_string(out_dir, "", "output directory");
DEFINE_double(avg_bits, 4.0, "SAQ average bits per padded dimension");
DEFINE_int32(threads, 64, "OpenMP worker count");
DEFINE_uint64(base_limit, 0, "optional base-row pilot limit");
DEFINE_uint64(query_limit, 20000, "number of transformed queries to export");
DEFINE_uint64(parity_queries, 16, "queries in scorer parity evidence");
DEFINE_uint64(parity_candidates, 256, "candidates per parity query");
DEFINE_uint64(seed, 20260831, "deterministic adapter seed");
DEFINE_bool(rand_rotate, true, "use the official SAQ per-segment rotation");
DEFINE_bool(export_full_codes, true, "export every code for code-native graph search");
DEFINE_int32(caq_adj_rd_lmt, 6, "official CAQ adjustment-round limit");
DEFINE_double(caq_adj_eps, 1e-8, "official CAQ adjustment epsilon");

namespace {

class MappedInput {
  public:
    explicit MappedInput(const std::string &path) {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open input: " + path);
        struct stat st {};
        if (fstat(fd_, &st) != 0 || st.st_size < 8) throw std::runtime_error("bad fbin: " + path);
        size_ = static_cast<size_t>(st.st_size);
        mapping_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (mapping_ == MAP_FAILED) throw std::runtime_error("cannot mmap input: " + path);
        const auto *header = static_cast<const uint32_t *>(mapping_);
        rows_ = header[0];
        dim_ = header[1];
        if (size_ != 8ULL + rows_ * static_cast<uint64_t>(dim_) * sizeof(float))
            throw std::runtime_error("fbin size mismatch: " + path);
        data_ = reinterpret_cast<const float *>(static_cast<const char *>(mapping_) + 8);
    }
    ~MappedInput() {
        if (mapping_ != MAP_FAILED) munmap(mapping_, size_);
        if (fd_ >= 0) close(fd_);
    }
    const float *row(uint64_t i) const { return data_ + i * dim_; }
    uint64_t rows() const { return rows_; }
    uint32_t dim() const { return dim_; }

  private:
    int fd_ = -1;
    void *mapping_ = MAP_FAILED;
    size_t size_ = 0;
    uint64_t rows_ = 0;
    uint32_t dim_ = 0;
    const float *data_ = nullptr;
};

class MappedOutput {
  public:
    MappedOutput(const std::string &path, uint32_t rows, uint32_t dim)
        : rows_(rows), dim_(dim), size_(8ULL + static_cast<size_t>(rows) * dim * sizeof(float)) {
        fd_ = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd_ < 0 || ftruncate(fd_, static_cast<off_t>(size_)) != 0)
            throw std::runtime_error("cannot create output: " + path);
        mapping_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapping_ == MAP_FAILED) throw std::runtime_error("cannot mmap output: " + path);
        auto *header = static_cast<uint32_t *>(mapping_);
        header[0] = rows;
        header[1] = dim;
        data_ = reinterpret_cast<float *>(static_cast<char *>(mapping_) + 8);
    }
    ~MappedOutput() {
        if (mapping_ != MAP_FAILED) {
            msync(mapping_, size_, MS_SYNC);
            munmap(mapping_, size_);
        }
        if (fd_ >= 0) close(fd_);
    }
    float *row(uint64_t i) { return data_ + i * dim_; }

  private:
    int fd_ = -1;
    void *mapping_ = MAP_FAILED;
    uint32_t rows_ = 0;
    uint32_t dim_ = 0;
    size_t size_ = 0;
    float *data_ = nullptr;
};

#pragma pack(push, 1)
struct SaqCodeHeader {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint64_t rows;
    uint64_t record_bytes;
    uint64_t model_bytes;
    uint64_t reserved[3];
};
#pragma pack(pop)
static_assert(sizeof(SaqCodeHeader) == 64);

class MappedCodeOutput {
  public:
    MappedCodeOutput(const std::string &path, uint64_t rows, uint64_t record_bytes,
                     uint64_t model_bytes)
        : rows_(rows), record_bytes_(record_bytes),
          size_(sizeof(SaqCodeHeader) + rows * record_bytes) {
        fd_ = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd_ < 0 || ftruncate(fd_, static_cast<off_t>(size_)) != 0)
            throw std::runtime_error("cannot create SAQ code output: " + path);
        mapping_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapping_ == MAP_FAILED)
            throw std::runtime_error("cannot mmap SAQ code output: " + path);
        SaqCodeHeader header{};
        std::memcpy(header.magic, "SAQCODE1", 8);
        header.version = 1;
        header.header_bytes = sizeof(SaqCodeHeader);
        header.rows = rows;
        header.record_bytes = record_bytes;
        header.model_bytes = model_bytes;
        std::memcpy(mapping_, &header, sizeof(header));
        data_ = static_cast<uint8_t *>(mapping_) + sizeof(header);
    }
    ~MappedCodeOutput() {
        if (mapping_ != MAP_FAILED) {
            msync(mapping_, size_, MS_SYNC);
            munmap(mapping_, size_);
        }
        if (fd_ >= 0) close(fd_);
    }
    uint8_t *row(uint64_t i) { return data_ + i * record_bytes_; }

  private:
    int fd_ = -1;
    void *mapping_ = MAP_FAILED;
    uint64_t rows_ = 0;
    uint64_t record_bytes_ = 0;
    size_t size_ = 0;
    uint8_t *data_ = nullptr;
};

void write_matrix_header(std::ofstream &out, uint32_t rows, uint32_t cols) {
    out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
}

std::string json_escape(const std::string &value) {
    std::string result;
    for (char ch : value) {
        if (ch == '\\' || ch == '"') result.push_back('\\');
        result.push_back(ch);
    }
    return result;
}

}  // namespace

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    if (FLAGS_base_fbin.empty() || FLAGS_query_fbin.empty() || FLAGS_out_dir.empty())
        throw std::runtime_error("--base-fbin, --query-fbin and --out-dir are required");
    if (FLAGS_avg_bits <= 0 || FLAGS_threads <= 0 || FLAGS_query_limit == 0 ||
        FLAGS_parity_queries == 0 || FLAGS_parity_candidates == 0)
        throw std::runtime_error("invalid SAQ export configuration");

    std::filesystem::create_directories(FLAGS_out_dir);
    MappedInput base(FLAGS_base_fbin);
    MappedInput queries(FLAGS_query_fbin);
    if (base.dim() != queries.dim()) throw std::runtime_error("base/query dimensions differ");
    const uint64_t rows = FLAGS_base_limit ? std::min<uint64_t>(base.rows(), FLAGS_base_limit) : base.rows();
    const uint64_t query_rows = std::min<uint64_t>(queries.rows(), FLAGS_query_limit);
    if (rows > std::numeric_limits<uint32_t>::max() || query_rows > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("fbin row count exceeds uint32 format");

    const size_t dim = base.dim();
    const size_t padded_dim = saqlib::utils::rd_up_to_multiple_of(dim, saqlib::kDimPaddingSize);
    std::vector<double> sum(dim, 0.0), sumsq(dim, 0.0);
    omp_set_num_threads(FLAGS_threads);
#pragma omp parallel
    {
        std::vector<double> local_sum(dim, 0.0), local_sumsq(dim, 0.0);
#pragma omp for schedule(static)
        for (uint64_t i = 0; i < rows; ++i) {
            const float *value = base.row(i);
            for (size_t j = 0; j < dim; ++j) {
                local_sum[j] += value[j];
                local_sumsq[j] += static_cast<double>(value[j]) * value[j];
            }
        }
#pragma omp critical
        {
            for (size_t j = 0; j < dim; ++j) {
                sum[j] += local_sum[j];
                sumsq[j] += local_sumsq[j];
            }
        }
    }
    saqlib::FloatVec variance = saqlib::FloatVec::Zero(padded_dim);
    for (size_t j = 0; j < dim; ++j) {
        const double mean = sum[j] / static_cast<double>(rows);
        variance[j] = static_cast<float>(std::max(0.0, sumsq[j] / rows - mean * mean));
    }

    std::srand(static_cast<unsigned>(FLAGS_seed));
    saqlib::QuantizeConfig cfg;
    cfg.avg_bits = static_cast<float>(FLAGS_avg_bits);
    cfg.enable_segmentation = true;
    cfg.single.random_rotation = FLAGS_rand_rotate;
    cfg.single.use_fastscan = false;
    cfg.single.quant_type = saqlib::BaseQuantType::CAQ;
    cfg.single.caq_adj_rd_lmt = FLAGS_caq_adj_rd_lmt;
    cfg.single.caq_adj_eps = static_cast<float>(FLAGS_caq_adj_eps);
    saqlib::SaqDataMaker maker(cfg, dim);
    maker.set_variance(std::move(variance));
    auto data = maker.return_data();
    const auto &plan = data->quant_plan;
    const size_t record_bytes = saqlib::SaqSingleDataWrapper::calculate_memory_size(plan);

    const std::string model_path = FLAGS_out_dir + "/saq_model.bin";
    {
        std::ofstream model(model_path, std::ios::binary);
        data->save(model);
    }
    const std::string transformed_base_path = FLAGS_out_dir + "/base.transformed.fbin";
    const std::string reconstruction_path = FLAGS_out_dir + "/primary.reconstruction.fbin";
    const std::string transformed_query_path = FLAGS_out_dir + "/query.transformed.fbin";
    const std::string codes_path = FLAGS_out_dir + "/saq_codes.bin";
    std::unique_ptr<MappedCodeOutput> full_codes;
    if (FLAGS_export_full_codes) {
        full_codes = std::make_unique<MappedCodeOutput>(
            codes_path, rows, record_bytes, std::filesystem::file_size(model_path));
    }
    MappedOutput transformed_base(transformed_base_path, static_cast<uint32_t>(rows), padded_dim);
    MappedOutput reconstruction(reconstruction_path, static_cast<uint32_t>(rows), padded_dim);

    const size_t parity_candidate_count = std::min<uint64_t>(rows, FLAGS_parity_candidates);
    std::mt19937_64 rng(FLAGS_seed);
    // Sample only the audit IDs, rather than allocating/shuffling 1B IDs.
    std::unordered_map<uint32_t, int32_t> parity_slot;
    std::uniform_int_distribution<uint32_t> pick(0, rows - 1);
    while (parity_slot.size() < parity_candidate_count) parity_slot.emplace(pick(rng), 0);
    std::vector<uint32_t> parity_ids;
    for (const auto& entry : parity_slot) parity_ids.push_back(entry.first);
    std::sort(parity_ids.begin(), parity_ids.end());
    for (size_t i = 0; i < parity_ids.size(); ++i) parity_slot[parity_ids[i]] = static_cast<int32_t>(i);
    const auto& parity_lookup = parity_slot;
    std::vector<uint8_t> parity_codes(parity_ids.size() * record_bytes);

#pragma omp parallel
    {
        saqlib::SAQuantizerSingle quantizer(data.get());
        saqlib::SaqSingleDataWrapper wrapper(plan);
        auto memory = saqlib::memory::make_unique_array<uint8_t>(record_bytes, 64);
        wrapper.set_memory_base(memory.get());
        saqlib::FloatVec original(dim), exact_transformed, decoded_transformed;
#pragma omp for schedule(static)
        for (uint64_t i = 0; i < rows; ++i) {
            std::memcpy(original.data(), base.row(i), dim * sizeof(float));
            wrapper.clear();
            quantizer.quantize_and_reconstruct(
                original, &wrapper, &exact_transformed, &decoded_transformed);
            if (full_codes) {
                std::memcpy(full_codes->row(i), wrapper.memory_base(), record_bytes);
            }
            std::memcpy(transformed_base.row(i), exact_transformed.data(), padded_dim * sizeof(float));
            std::memcpy(reconstruction.row(i), decoded_transformed.data(), padded_dim * sizeof(float));
            const auto found = parity_lookup.find(static_cast<uint32_t>(i));
            if (found != parity_lookup.end()) {
                const int32_t slot = found->second;
                std::memcpy(parity_codes.data() + static_cast<size_t>(slot) * record_bytes,
                            wrapper.memory_base(), record_bytes);
            }
        }
    }

    MappedOutput transformed_queries(transformed_query_path, static_cast<uint32_t>(query_rows), padded_dim);
#pragma omp parallel
    {
        saqlib::SAQuantizerSingle transformer(data.get());
        saqlib::FloatVec original(dim), transformed;
#pragma omp for schedule(static)
        for (uint64_t i = 0; i < query_rows; ++i) {
            std::memcpy(original.data(), queries.row(i), dim * sizeof(float));
            transformer.transform(original, &transformed);
            std::memcpy(transformed_queries.row(i), transformed.data(), padded_dim * sizeof(float));
        }
    }

    const size_t parity_query_count = std::min<uint64_t>(query_rows, FLAGS_parity_queries);
    const std::string ids_path = FLAGS_out_dir + "/parity.candidate_ids.ibin";
    const std::string values_path = FLAGS_out_dir + "/parity.official_scores.fbin";
    std::ofstream ids_out(ids_path, std::ios::binary);
    std::ofstream values_out(values_path, std::ios::binary);
    write_matrix_header(ids_out, parity_query_count, parity_ids.size());
    write_matrix_header(values_out, parity_query_count, parity_ids.size());
    for (size_t qi = 0; qi < parity_query_count; ++qi) {
        saqlib::FloatVec query(dim);
        std::memcpy(query.data(), queries.row(qi), dim * sizeof(float));
        saqlib::SearcherConfig search_cfg;
        search_cfg.dist_type = saqlib::DistType::IP;
        saqlib::SaqSingleEstimator<saqlib::DistType::IP> estimator(*data, search_cfg, query);
        saqlib::SaqSingleDataWrapper wrapper(plan);
        auto memory = saqlib::memory::make_unique_array<uint8_t>(record_bytes, 64);
        wrapper.set_memory_base(memory.get());
        for (size_t ci = 0; ci < parity_ids.size(); ++ci) {
            std::memcpy(wrapper.memory_base(), parity_codes.data() + ci * record_bytes, record_bytes);
            const float score = estimator.compAccurateDist(wrapper);
            ids_out.write(reinterpret_cast<const char *>(&parity_ids[ci]), sizeof(uint32_t));
            values_out.write(reinterpret_cast<const char *>(&score), sizeof(float));
        }
    }

    size_t nominal_code_bits = 0;
    for (const auto &[segment_dim, bits] : plan) nominal_code_bits += segment_dim * bits;
    const std::string metadata_path = FLAGS_out_dir + "/export_summary.json";
    std::ofstream metadata(metadata_path);
    metadata << std::setprecision(12)
             << "{\n  \"version\": 1,\n  \"implementation\": \"official-saq-adapter\",\n"
             << "  \"base_rows\": " << rows << ",\n  \"original_dim\": " << dim
             << ",\n  \"scorer_dim\": " << padded_dim << ",\n"
             << "  \"avg_bits\": " << FLAGS_avg_bits << ",\n"
             << "  \"nominal_code_bits_per_node\": " << nominal_code_bits << ",\n"
             << "  \"physical_bytes_per_node\": " << record_bytes << ",\n"
             << "  \"query_rows\": " << query_rows << ",\n"
             << "  \"quantization_plan\": [";
    for (size_t i = 0; i < plan.size(); ++i) {
        if (i) metadata << ", ";
        metadata << "[" << plan[i].first << ", " << plan[i].second << "]";
    }
    metadata << "],\n  \"files\": {\n"
             << "    \"model\": \"" << json_escape(std::filesystem::absolute(model_path).string()) << "\",\n"
             << "    \"quantized_codes\": "
             << (FLAGS_export_full_codes
                     ? "\"" + json_escape(std::filesystem::absolute(codes_path).string()) + "\""
                     : "null")
             << ",\n"
             << "    \"base_transformed\": \"" << json_escape(std::filesystem::absolute(transformed_base_path).string()) << "\",\n"
             << "    \"reconstruction\": \"" << json_escape(std::filesystem::absolute(reconstruction_path).string()) << "\",\n"
             << "    \"query_transformed\": \"" << json_escape(std::filesystem::absolute(transformed_query_path).string()) << "\",\n"
             << "    \"parity_candidate_ids\": \"" << json_escape(std::filesystem::absolute(ids_path).string()) << "\",\n"
             << "    \"parity_official_scores\": \"" << json_escape(std::filesystem::absolute(values_path).string()) << "\"\n"
             << "  }\n}\n";
    std::cout << "SAQ primary export complete: " << metadata_path << std::endl;
    return 0;
}
