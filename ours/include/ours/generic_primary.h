#ifndef OURS_GENERIC_PRIMARY_H_
#define OURS_GENERIC_PRIMARY_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ours {

// Common code-native representation used by the reproducible OPQ-PQ,
// QAQ-PQ, and LeanVec-OOD adapters.  The model stores the query projection;
// database records are either byte PQ indices or packed locally affine scalar
// values (4-bit or 8-bit).
// Both graph construction and search consume these records directly.
class GenericPrimaryIndex {
 public:
  enum class Encoding : uint32_t {
    kCodebook8 = 1,
    kAffine8 = 2,
    kAffine4 = 3,
  };

  struct QueryContext {
    std::vector<float> transformed;
    std::vector<float> lut;
    // Query-only affine term, accumulated in the legacy coordinate order.
    float query_sum = 0.0f;
  };

  static GenericPrimaryIndex Load(const std::string& model_path,
                                  const std::string& codes_path);
  GenericPrimaryIndex(GenericPrimaryIndex&&) noexcept;
  GenericPrimaryIndex& operator=(GenericPrimaryIndex&&) noexcept;
  ~GenericPrimaryIndex();

  GenericPrimaryIndex(const GenericPrimaryIndex&) = delete;
  GenericPrimaryIndex& operator=(const GenericPrimaryIndex&) = delete;

  uint64_t nodes() const;
  uint32_t original_dim() const;
  uint32_t scorer_dim() const;
  uint32_t groups() const;
  uint64_t storage_bytes() const;
  Encoding encoding() const;
  const char* scoring_kernel() const;

  QueryContext Prepare(const float* query, uint32_t dim) const;
  QueryContext PrepareTransformed(const float* query, uint32_t dim) const;
  float Score(const QueryContext& context, uint32_t id) const;
  void ScoreBatch(const QueryContext& context, const uint32_t* ids,
                  size_t count, float* scores) const;
  void Reconstruct(uint32_t id, float* output, uint32_t dim) const;

 private:
  class Impl;
  explicit GenericPrimaryIndex(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace ours

#endif  // OURS_GENERIC_PRIMARY_H_
