#include "ours/exact_vector_store.h"
#include "ours/fp32_inner_product.h"

#include "ours/mapped_fbin.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ours {
namespace {

using Clock = std::chrono::steady_clock;

double Milliseconds(Clock::duration duration) {
  return 1000.0 * std::chrono::duration<double>(duration).count();
}

class WarmMmapStore final : public ExactVectorStore {
public:
  explicit WarmMmapStore(const std::string &path)
      : vectors_(path), file_bytes_(FileBytes(path)) {
    RequireFp32Avx512();
  }

  uint64_t rows() const override { return vectors_.rows(); }
  uint32_t dim() const override { return vectors_.dim(); }
  uint64_t file_bytes() const override { return file_bytes_; }
  const char *kind() const override { return "mmap"; }
  const char *scoring_kernel() const override { return Fp32InnerProductKernel(); }

  void ScoreBatch(const float *query, const uint32_t *ids, size_t count,
                  float *scores, ExactScoreStats *stats) const override {
    const auto started = Clock::now();
    for (size_t index = 0; index < count; ++index)
      scores[index] = Fp32InnerProduct(query, vectors_.Row(ids[index]), vectors_.dim());
    if (stats) {
      stats->logical_bytes +=
          count * static_cast<uint64_t>(vectors_.dim()) * sizeof(float);
      // mmap page faults are intentionally not reported as physical I/O:
      // Linux does not expose reliable per-call page-cache misses here.
      stats->compute_ms += Milliseconds(Clock::now() - started);
    }
  }

private:
  static uint64_t FileBytes(const std::string &path) {
    struct stat status {};
    if (stat(path.c_str(), &status) != 0 || status.st_size < 0)
      throw std::runtime_error("cannot stat exact-vector file: " + path);
    return static_cast<uint64_t>(status.st_size);
  }

  MappedFbin vectors_;
  uint64_t file_bytes_ = 0;
};

#pragma pack(push, 1)
struct ColdVectorHeader {
  char magic[8];
  uint32_t version;
  uint32_t header_bytes;
  uint64_t rows;
  uint32_t dim;
  uint32_t scalar_bytes;
  uint32_t vector_bytes;
  uint32_t vectors_per_block;
  uint32_t block_bytes;
  uint64_t data_offset;
  uint64_t data_blocks;
};
#pragma pack(pop)
static_assert(sizeof(ColdVectorHeader) == 60);

void ReadFully(int fd, void *destination, size_t bytes, uint64_t offset,
               const std::string &path) {
  auto *cursor = static_cast<unsigned char *>(destination);
  size_t completed = 0;
  while (completed < bytes) {
    const ssize_t result =
        pread(fd, cursor + completed, bytes - completed,
              static_cast<off_t>(offset + completed));
    if (result < 0 && errno == EINTR)
      continue;
    if (result < 0)
      throw std::runtime_error("pread failed for " + path + ": " +
                               std::strerror(errno));
    if (result == 0)
      throw std::runtime_error("unexpected EOF in cold-vector file: " + path);
    completed += static_cast<size_t>(result);
  }
}

class AlignedBuffer {
public:
  AlignedBuffer(size_t alignment, size_t bytes) : bytes_(bytes) {
    if (posix_memalign(&data_, alignment, bytes) != 0)
      throw std::bad_alloc();
  }
  ~AlignedBuffer() { std::free(data_); }
  AlignedBuffer(const AlignedBuffer &) = delete;
  AlignedBuffer &operator=(const AlignedBuffer &) = delete;
  void *data() { return data_; }
  size_t bytes() const { return bytes_; }

private:
  void *data_ = nullptr;
  size_t bytes_ = 0;
};

class ColdDirectStore final : public ExactVectorStore {
public:
  explicit ColdDirectStore(const std::string &path) : path_(path) {
    RequireFp32Avx512();
    const int metadata_fd = open(path.c_str(), O_RDONLY);
    if (metadata_fd < 0)
      throw std::runtime_error("cannot open cold-vector file: " + path);
    try {
      ReadFully(metadata_fd, &header_, sizeof(header_), 0, path_);
      struct stat status {};
      if (fstat(metadata_fd, &status) != 0 || status.st_size < 0)
        throw std::runtime_error("cannot stat cold-vector file: " + path);
      file_bytes_ = static_cast<uint64_t>(status.st_size);
      close(metadata_fd);
    } catch (...) {
      close(metadata_fd);
      throw;
    }
    Validate();
#if defined(O_DIRECT)
    fd_ = open(path.c_str(), O_RDONLY | O_DIRECT);
#else
    throw std::runtime_error("O_DIRECT is unavailable on this platform");
#endif
    if (fd_ < 0)
      throw std::runtime_error("cannot open cold-vector file with O_DIRECT: " +
                               path + ": " + std::strerror(errno));
  }

  ~ColdDirectStore() override {
    if (fd_ >= 0)
      close(fd_);
  }

  uint64_t rows() const override { return header_.rows; }
  uint32_t dim() const override { return header_.dim; }
  uint64_t file_bytes() const override { return file_bytes_; }
  const char *kind() const override { return "direct"; }
  const char *scoring_kernel() const override { return Fp32InnerProductKernel(); }

  void ScoreBatch(const float *query, const uint32_t *ids, size_t count,
                  float *scores, ExactScoreStats *stats) const override {
    struct Request {
      uint64_t block = 0;
      uint32_t slot = 0;
      size_t output = 0;
    };
    std::vector<Request> requests;
    requests.reserve(count);
    for (size_t index = 0; index < count; ++index) {
      if (ids[index] >= header_.rows)
        throw std::out_of_range("cold-vector id is out of range");
      requests.push_back(Request{
          ids[index] / header_.vectors_per_block,
          ids[index] % header_.vectors_per_block, index});
    }
    std::sort(requests.begin(), requests.end(),
              [](const Request &left, const Request &right) {
                return left.block < right.block ||
                       (left.block == right.block && left.slot < right.slot);
              });

    AlignedBuffer buffer(header_.block_bytes, header_.block_bytes);
    if (stats)
      stats->workspace_peak_bytes = std::max<uint64_t>(stats->workspace_peak_bytes,
          requests.capacity() * sizeof(Request) + buffer.bytes());
    size_t begin = 0;
    while (begin < requests.size()) {
      size_t end = begin + 1;
      while (end < requests.size() &&
             requests[end].block == requests[begin].block)
        ++end;
      const uint64_t offset =
          header_.data_offset + requests[begin].block * header_.block_bytes;
      const auto io_started = Clock::now();
      ReadFully(fd_, buffer.data(), buffer.bytes(), offset, path_);
      const auto io_finished = Clock::now();
      if (stats) {
        stats->physical_bytes += header_.block_bytes;
        ++stats->read_ops;
        stats->io_ms += Milliseconds(io_finished - io_started);
      }
      const auto compute_started = Clock::now();
      const auto *bytes = static_cast<const unsigned char *>(buffer.data());
      for (size_t index = begin; index < end; ++index) {
        const float *vector = reinterpret_cast<const float *>(
            bytes + requests[index].slot * header_.vector_bytes);
        scores[requests[index].output] = Fp32InnerProduct(query, vector, header_.dim);
      }
      if (stats)
        stats->compute_ms += Milliseconds(Clock::now() - compute_started);
      begin = end;
    }
    if (stats)
      stats->logical_bytes +=
          count * static_cast<uint64_t>(header_.vector_bytes);
  }

private:
  void Validate() const {
    if (std::memcmp(header_.magic, "CDVEC001", 8) != 0 ||
        header_.version != 1 || header_.header_bytes < sizeof(header_) ||
        header_.rows == 0 || header_.dim == 0 ||
        header_.scalar_bytes != sizeof(float) ||
        header_.vector_bytes != header_.dim * sizeof(float) ||
        header_.vectors_per_block == 0 || header_.block_bytes < 512 ||
        (header_.block_bytes & (header_.block_bytes - 1)) != 0 ||
        header_.data_offset != header_.header_bytes ||
        header_.data_offset % header_.block_bytes != 0 ||
        header_.vector_bytes * header_.vectors_per_block >
            header_.block_bytes ||
        header_.data_blocks !=
            (header_.rows + header_.vectors_per_block - 1) /
                header_.vectors_per_block ||
        file_bytes_ !=
            header_.data_offset + header_.data_blocks * header_.block_bytes)
      throw std::runtime_error("invalid cold-vector header/layout: " + path_);
    if (header_.data_offset >
        static_cast<uint64_t>(std::numeric_limits<off_t>::max()))
      throw std::runtime_error("cold-vector offsets exceed off_t");
  }

  std::string path_;
  int fd_ = -1;
  ColdVectorHeader header_{};
  uint64_t file_bytes_ = 0;
};

} // namespace

std::unique_ptr<ExactVectorStore>
OpenExactVectorStore(const std::string &path, const std::string &mode) {
  if (mode == "mmap")
    return std::make_unique<WarmMmapStore>(path);
  if (mode == "direct")
    return std::make_unique<ColdDirectStore>(path);
  throw std::invalid_argument("unknown rerank storage mode: " + mode);
}

} // namespace ours
