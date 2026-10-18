#ifndef OURS_MAPPED_FBIN_H_
#define OURS_MAPPED_FBIN_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "ours/drb_model.h"

namespace ours {

class MappedFbin {
   public:
    explicit MappedFbin(const std::string& path);
    ~MappedFbin();

    MappedFbin(const MappedFbin&) = delete;
    MappedFbin& operator=(const MappedFbin&) = delete;
    MappedFbin(MappedFbin&& other) noexcept;
    MappedFbin& operator=(MappedFbin&& other) noexcept;

    uint64_t rows() const { return rows_; }
    uint32_t dim() const { return dim_; }
    const float* Row(uint64_t row) const;
    MatrixView view() const { return MatrixView{data_, rows_, dim_}; }

   private:
    void Close() noexcept;

    int fd_ = -1;
    void* mapping_ = nullptr;
    size_t mapping_size_ = 0;
    const float* data_ = nullptr;
    uint64_t rows_ = 0;
    uint32_t dim_ = 0;
};

}  // namespace ours

#endif  // OURS_MAPPED_FBIN_H_
