#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace ours {
// Read-only adapter for the migrated QVRR records (not official-library files).
class QvrrPrimaryIndex {
 public:
  struct QueryContext { std::vector<float> transformed, lut; float bias=0, transformed_sum=0; };
  QvrrPrimaryIndex(const std::string& model, const std::string& codes);
  ~QvrrPrimaryIndex();
  uint64_t nodes() const;
  uint32_t original_dim() const;
  uint32_t scorer_dim() const;
  uint64_t storage_bytes() const;
  const char* scoring_kernel() const;
  QueryContext Prepare(const float*, uint32_t) const;
  float Score(const QueryContext&, uint32_t) const;
  void ScoreBatch(const QueryContext&, const uint32_t*, size_t, float*) const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
