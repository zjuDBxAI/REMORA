#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace ours {

struct ExactScoreStats {
  uint64_t logical_bytes = 0;
  uint64_t physical_bytes = 0;
  uint64_t read_ops = 0;
  double io_ms = 0.0;
  double compute_ms = 0.0;
  uint64_t workspace_peak_bytes = 0;
};

// Storage backend used only for terminal exact reranking.  Approximate graph
// traversal never consults this interface.
class ExactVectorStore {
public:
  virtual ~ExactVectorStore() = default;

  virtual uint64_t rows() const = 0;
  virtual uint32_t dim() const = 0;
  virtual uint64_t file_bytes() const = 0;
  virtual const char *kind() const = 0;
  virtual const char *scoring_kernel() const = 0;

  // Scores ids[0:count] against one FP32 query.  Implementations must be safe
  // for concurrent calls from independent search threads.
  virtual void ScoreBatch(const float *query, const uint32_t *ids,
                          size_t count, float *scores,
                          ExactScoreStats *stats) const = 0;
};

// mode="mmap" opens a normal .fbin through the Linux page cache.
// mode="direct" opens a blocked cold-vector file with O_DIRECT.
std::unique_ptr<ExactVectorStore>
OpenExactVectorStore(const std::string &path, const std::string &mode);

} // namespace ours
