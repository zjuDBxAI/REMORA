#include "ours/mapped_fbin.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ours {

MappedFbin::MappedFbin(const std::string& path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("cannot open fbin: " + path);
    }
    struct stat info {};
    if (fstat(fd_, &info) != 0 || info.st_size < 8) {
        Close();
        throw std::runtime_error("invalid fbin file: " + path);
    }
    mapping_size_ = static_cast<size_t>(info.st_size);
    mapping_ = mmap(nullptr, mapping_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mapping_ == MAP_FAILED) {
        mapping_ = nullptr;
        Close();
        throw std::runtime_error("cannot mmap fbin: " + path);
    }
    uint32_t header[2]{};
    std::memcpy(header, mapping_, sizeof(header));
    rows_ = header[0];
    dim_ = header[1];
    const uint64_t expected = 8ULL + rows_ * static_cast<uint64_t>(dim_) * sizeof(float);
    if (rows_ == 0 || dim_ == 0 || expected != mapping_size_) {
        Close();
        throw std::runtime_error("fbin header/file-size mismatch: " + path);
    }
    data_ = reinterpret_cast<const float*>(static_cast<const char*>(mapping_) + 8);
}

MappedFbin::~MappedFbin() { Close(); }

MappedFbin::MappedFbin(MappedFbin&& other) noexcept { *this = std::move(other); }

MappedFbin& MappedFbin::operator=(MappedFbin&& other) noexcept {
    if (this != &other) {
        Close();
        fd_ = other.fd_;
        mapping_ = other.mapping_;
        mapping_size_ = other.mapping_size_;
        data_ = other.data_;
        rows_ = other.rows_;
        dim_ = other.dim_;
        other.fd_ = -1;
        other.mapping_ = nullptr;
        other.mapping_size_ = 0;
        other.data_ = nullptr;
        other.rows_ = 0;
        other.dim_ = 0;
    }
    return *this;
}

void MappedFbin::Close() noexcept {
    if (mapping_ != nullptr) {
        munmap(mapping_, mapping_size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
    fd_ = -1;
    mapping_ = nullptr;
    mapping_size_ = 0;
    data_ = nullptr;
    rows_ = 0;
    dim_ = 0;
}

const float* MappedFbin::Row(uint64_t row) const {
    if (row >= rows_) {
        throw std::out_of_range("fbin row is out of range");
    }
    return data_ + row * static_cast<uint64_t>(dim_);
}

}  // namespace ours
