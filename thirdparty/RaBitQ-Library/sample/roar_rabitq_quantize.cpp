#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/stopw.hpp"

using data_type = rabitqlib::RowMajorArray<float>;

namespace {
struct QuantFileHeader {
    char magic[8];  // "RBQROAR1"
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

rabitqlib::MetricType parse_metric(const std::string& metric) {
    if (metric == "ip" || metric == "IP") {
        return rabitqlib::METRIC_IP;
    }
    return rabitqlib::METRIC_L2;
}

void usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <base_data> <output_quant_file> <total_bits> [metric=l2|ip] "
                 "[faster=false|true]\n"
              << "base_data: .fvecs or .fbin\n";
}
}  // namespace

// The 1B input must never be copied into an N x D heap allocation.  Accept
// fbin download names such as *.fbin.4curl as well as the standard suffix.
class FbinMap {
 public:
    explicit FbinMap(const char* path) {
        fd_ = open(path, O_RDONLY);
        struct stat st{};
        if (fd_ < 0 || fstat(fd_, &st) || st.st_size < 8)
            throw std::runtime_error("cannot open fbin");
        bytes_ = st.st_size;
        mapping_ = mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapping_ == MAP_FAILED) throw std::runtime_error("cannot mmap fbin");
        std::memcpy(&rows_, mapping_, 4);
        std::memcpy(&dim_, static_cast<const char*>(mapping_) + 4, 4);
        if (!rows_ || !dim_ || bytes_ != 8 + uint64_t(rows_) * dim_ * 4)
            throw std::runtime_error("truncated or invalid fbin");
        madvise(mapping_, bytes_, MADV_SEQUENTIAL);
    }
    ~FbinMap() { if (mapping_ != MAP_FAILED) munmap(mapping_, bytes_); if (fd_ >= 0) close(fd_); }
    size_t rows() const { return rows_; }
    size_t dim() const { return dim_; }
    const float* row(size_t i) const {
        return reinterpret_cast<const float*>(static_cast<const char*>(mapping_) + 8) + i * dim_;
    }
 private:
    int fd_ = -1; void* mapping_ = MAP_FAILED; size_t bytes_ = 0;
    uint32_t rows_ = 0, dim_ = 0;
};

void write_at(int fd, const char* data, size_t bytes, uint64_t offset) {
    while (bytes) {
        const auto n = pwrite(fd, data, bytes, offset);
        if (n <= 0) throw std::runtime_error("quantization output write failed");
        data += n; bytes -= n; offset += n;
    }
}

int main(int argc, char** argv) {
  try {
    if (argc < 4) {
        usage(argv[0]);
        return 1;
    }

    const char* data_file = argv[1];
    const char* output_file = argv[2];
    const size_t total_bits = static_cast<size_t>(std::atoi(argv[3]));

    if (total_bits == 0 || total_bits > 8) {
        std::cerr << "total_bits must be in [1, 8]\n";
        return 1;
    }

    rabitqlib::MetricType metric_type = rabitqlib::METRIC_L2;
    if (argc > 4) {
        metric_type = parse_metric(argv[4]);
    }

    bool faster_quant = false;
    if (argc > 5) {
        std::string faster_str(argv[5]);
        faster_quant = (faster_str == "true");
    }

    data_type data;
    std::unique_ptr<FbinMap> mapped;
    const std::string input_name(data_file);
    if (input_name.size() >= 6 && input_name.substr(input_name.size() - 6) == ".fvecs")
        rabitqlib::load_data<float, data_type>(data_file, data);
    else mapped = std::make_unique<FbinMap>(data_file);
    const size_t num_points = mapped ? mapped->rows() : data.rows();
    const size_t dim = mapped ? mapped->dim() : data.cols();
    const size_t padded_dim = rabitqlib::round_up_to_multiple(dim, 64);
    const size_t ex_bits = total_bits - 1;
    const size_t bin_bytes = rabitqlib::BinDataMap<float>::data_bytes(padded_dim);
    const size_t ex_bytes = rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits);

    auto* rotator = rabitqlib::choose_rotator<float>(
        dim, rabitqlib::RotatorType::FhtKacRotator, padded_dim
    );

    rabitqlib::quant::RabitqConfig config;
    if (faster_quant) {
        config = rabitqlib::quant::faster_config(padded_dim, total_bits);
    }

    std::vector<float> centroid(padded_dim, 0.F);

    std::ofstream output(output_file, std::ios::binary);
    if (!output.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << '\n';
        delete rotator;
        return 1;
    }

    QuantFileHeader header{};
    std::memcpy(header.magic, "RBQROAR1", 8);
    header.version = 1;
    header.metric_type = static_cast<uint32_t>(metric_type);
    header.rotator_type = static_cast<uint32_t>(rabitqlib::RotatorType::FhtKacRotator);
    header.total_bits = static_cast<uint32_t>(total_bits);
    header.num_points = static_cast<uint64_t>(num_points);
    header.dim = static_cast<uint64_t>(dim);
    header.padded_dim = static_cast<uint64_t>(padded_dim);
    header.bin_bytes_per_vec = static_cast<uint64_t>(bin_bytes);
    header.ex_bytes_per_vec = static_cast<uint64_t>(ex_bytes);
    header.rotator_bytes = static_cast<uint64_t>(rotator->dump_bytes());

    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    rotator->save(output);
    const uint64_t payload = output.tellp();
    output.close();
    const int out_fd = open(output_file, O_WRONLY);
    if (out_fd < 0 || ftruncate(out_fd, payload + num_points * (bin_bytes + ex_bytes)))
        throw std::runtime_error("cannot size quantization output");
    rabitqlib::StopW stopw;
    constexpr size_t batch_size = 65536;
    std::vector<char> bins(batch_size * bin_bytes), extras(batch_size * ex_bytes);
    for (size_t start = 0; start < num_points; start += batch_size) {
        const size_t count = std::min(batch_size, num_points - start);
#pragma omp parallel
        {
            std::vector<float> rotated(padded_dim, 0.F);
#pragma omp for schedule(static)
            for (size_t j = 0; j < count; ++j) {
                const float* row = mapped ? mapped->row(start + j) : &data(start + j, 0);
                rotator->rotate(row, rotated.data());
                rabitqlib::quant::quantize_split_single(rotated.data(), centroid.data(),
                    padded_dim, ex_bits, bins.data() + j * bin_bytes,
                    ex_bytes ? extras.data() + j * ex_bytes : nullptr, metric_type, config);
            }
        }
        write_at(out_fd, bins.data(), count * bin_bytes, payload + start * bin_bytes);
        if (ex_bytes) write_at(out_fd, extras.data(), count * ex_bytes,
            payload + num_points * bin_bytes + start * ex_bytes);
        if (start == 0 || start + count == num_points || start % (batch_size * 64) == 0)
            std::cout << "[quantize] " << start + count << '/' << num_points << std::endl;
    }
    if (fsync(out_fd)) throw std::runtime_error("cannot sync quantization output");
    close(out_fd);
    const float secs = stopw.get_elapsed_mili() / 1000.F;

    std::cout << "RaBitQ export done\n";
    std::cout << "output: " << output_file << '\n';
    std::cout << "N=" << num_points << ", dim=" << dim << ", padded_dim=" << padded_dim
              << ", total_bits=" << total_bits << '\n';
    std::cout << "quantization_time=" << secs << " sec\n";

    delete rotator;
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
