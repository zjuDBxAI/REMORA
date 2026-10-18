#include <fcntl.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#include "quantization/saq_data.hpp"
#include "quantization/saq_quantizer.hpp"
#include "quantization/single_data.hpp"
#include "utils/memory.hpp"

DEFINE_string(base_fbin, "", "input base vectors in fbin format");
DEFINE_string(model, "", "existing official SAQ model");
DEFINE_string(out_codes, "", "output full code file");
DEFINE_int32(threads, 64, "OpenMP worker count");
DEFINE_uint64(base_limit, 0, "optional row limit");

namespace {

class Mapping {
  public:
    Mapping(const std::string &path, bool writable, size_t create_size = 0) {
        const int flags = writable ? (O_RDWR | O_CREAT | O_TRUNC) : O_RDONLY;
        fd_ = open(path.c_str(), flags, 0644);
        if (fd_ < 0)
            throw std::runtime_error("cannot open: " + path);
        if (writable) {
            size_ = create_size;
            if (ftruncate(fd_, static_cast<off_t>(size_)) != 0)
                throw std::runtime_error("cannot resize: " + path);
        } else {
            struct stat status {};
            if (fstat(fd_, &status) != 0)
                throw std::runtime_error("cannot stat: " + path);
            size_ = static_cast<size_t>(status.st_size);
        }
        data_ = mmap(nullptr, size_, writable ? PROT_READ | PROT_WRITE : PROT_READ,
                     MAP_SHARED, fd_, 0);
        if (data_ == MAP_FAILED)
            throw std::runtime_error("cannot mmap: " + path);
        writable_ = writable;
    }
    ~Mapping() {
        if (data_ != MAP_FAILED) {
            if (writable_)
                msync(data_, size_, MS_SYNC);
            munmap(data_, size_);
        }
        if (fd_ >= 0)
            close(fd_);
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

} // namespace

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    if (FLAGS_base_fbin.empty() || FLAGS_model.empty() || FLAGS_out_codes.empty() ||
        FLAGS_threads <= 0)
        throw std::runtime_error("base-fbin, model, out-codes and threads are required");

    Mapping base(FLAGS_base_fbin, false);
    if (base.size() < 8)
        throw std::runtime_error("truncated base fbin");
    const auto *fbin_header = static_cast<const uint32_t *>(base.data());
    const uint64_t all_rows = fbin_header[0];
    const uint32_t dim = fbin_header[1];
    if (base.size() != 8 + all_rows * static_cast<uint64_t>(dim) * sizeof(float))
        throw std::runtime_error("base fbin size mismatch");
    const uint64_t rows = FLAGS_base_limit ? std::min(FLAGS_base_limit, all_rows) : all_rows;
    const float *values = reinterpret_cast<const float *>(
        static_cast<const char *>(base.data()) + 8);

    saqlib::SaqData data;
    {
        std::ifstream model(FLAGS_model, std::ios::binary);
        if (!model)
            throw std::runtime_error("cannot open SAQ model");
        data.load(model);
    }
    if (data.num_dim != dim)
        throw std::runtime_error("SAQ model/base dimension mismatch");
    const size_t record_bytes = saqlib::SaqSingleDataWrapper::calculate_memory_size(data.quant_plan);
    Mapping output(FLAGS_out_codes, true, sizeof(SaqCodeHeader) + rows * record_bytes);
    SaqCodeHeader header{};
    std::memcpy(header.magic, "SAQCODE1", 8);
    header.version = 1;
    header.header_bytes = sizeof(SaqCodeHeader);
    header.rows = rows;
    header.record_bytes = record_bytes;
    header.model_bytes = std::filesystem::file_size(FLAGS_model);
    std::memcpy(output.data(), &header, sizeof(header));
    auto *code_begin = static_cast<uint8_t *>(output.data()) + sizeof(header);

    omp_set_num_threads(FLAGS_threads);
#pragma omp parallel
    {
        saqlib::SAQuantizerSingle quantizer(&data);
        saqlib::SaqSingleDataWrapper wrapper(data.quant_plan);
        auto memory = saqlib::memory::make_unique_array<uint8_t>(record_bytes, 64);
        wrapper.set_memory_base(memory.get());
        saqlib::FloatVec original(dim);
#pragma omp for schedule(static)
        for (uint64_t row = 0; row < rows; ++row) {
            std::memcpy(original.data(), values + row * dim, dim * sizeof(float));
            wrapper.clear();
            quantizer.quantize(original, &wrapper);
            std::memcpy(code_begin + row * record_bytes, wrapper.memory_base(), record_bytes);
        }
    }
    std::cout << "SAQ full-code export complete: " << FLAGS_out_codes
              << " (" << rows << " x " << record_bytes << " bytes)\n";
    return 0;
}
