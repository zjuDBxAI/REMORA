#ifndef OURS_JI_ONLINE_H_
#define OURS_JI_ONLINE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ours {

struct JiQueryContext {
  float mean_bias = 0.0f;
  // A conservative upper bound on Correction(*this, node_id), valid for
  // every node.  PrepareQuery builds it with upward rounding after every
  // group so it remains safe under the scorer's float accumulation order.
  float max_correction_upper = 0.0f;
  // Query-specific group-local lookup tables.  The deployment path uses the
  // learned 8x8-bit model directly: one small 256-entry table per group.
  std::vector<float> lookup;
};

struct JiPrepareTimings {
  double projection_ms = 0.0;
  double lut_ms = 0.0;
  double upper_bound_ms = 0.0;
};

// Owned by one search worker. No mutable query state is stored in the index.
struct JiQueryWorkspace {
  JiQueryContext context;
  std::vector<float> padded;
  std::vector<float> phi;
};

class JiOnlineIndex {
public:
  static JiOnlineIndex Load(const std::string &model_path,
                            const std::string &side_codes_path);
  ~JiOnlineIndex();

  JiOnlineIndex(const JiOnlineIndex &) = delete;
  JiOnlineIndex &operator=(const JiOnlineIndex &) = delete;
  JiOnlineIndex(JiOnlineIndex &&other) noexcept;
  JiOnlineIndex &operator=(JiOnlineIndex &&other) noexcept;

  uint64_t node_count() const { return node_count_; }
  uint32_t dim() const { return dim_; }
  uint32_t groups() const { return declared_groups_; }
  uint32_t budget_bits() const { return budget_bits_; }
  uint64_t artifact_id() const { return artifact_id_; }
  uint64_t model_id() const { return model_id_; }
  uint64_t storage_bytes() const { return model_bytes_ + side_bytes_; }
  uint32_t format_version() const { return format_version_; }
  uint32_t active_groups() const {
    return static_cast<uint32_t>(groups_.size());
  }
  uint32_t score_groups() const {
    return static_cast<uint32_t>(groups_.size());
  }
  const char *scoring_kernel() const;
  uint64_t runtime_cache_bytes() const {
    return (transform_columns_.size() + prototype_columns_.size()) * sizeof(float);
  }

  JiQueryContext PrepareQuery(const float *transformed_query,
                              uint32_t dim, bool vectorized = true,
                              JiPrepareTimings *timings = nullptr) const;
  // Explicit padding for terminal decoders trained in a common padded space.
  // Does NOT rotate/project the query or infer which coordinate system it uses.
  JiQueryContext PrepareQueryZeroPadded(const float *query, uint32_t dim,
                                       bool vectorized = true,
                                       JiPrepareTimings *timings = nullptr) const;
  // Same LUT arithmetic; reuse worker buffers and omit the unused lazy bound.
  // Returned reference remains valid until the next call with this workspace.
  const JiQueryContext &PrepareTerminalQuery(const float *query, uint32_t dim,
                                            JiQueryWorkspace &workspace,
                                            JiPrepareTimings *timings = nullptr) const;
  // Reuses worker-local storage. Only lazy traversal requests a bound.
  const JiQueryContext &PrepareQueryReused(const float *query, uint32_t dim,
                                          JiQueryWorkspace &workspace, bool build_upper_bound,
                                          JiPrepareTimings *timings = nullptr) const;
  float Correction(const JiQueryContext &query, uint32_t node_id) const;
  // Computes JI-only corrections.  Keeping corrections separate from the
  // primary scores preserves the scalar scorer's arithmetic order; the caller
  // can then fuse the final vector addition with its batched primary kernel.
  void ComputeCorrectionsBatch(const JiQueryContext &query,
                               const uint32_t *node_ids, size_t count,
                               float *corrections) const;
  // Exactly primary + Correction(), not a reassociated running sum.
  void AddCorrectionsBatch(const JiQueryContext &query, const uint32_t *node_ids,
                           size_t count, float *primary_scores) const;
  // Decode one residual refinement directly from the packed side code.  This
  // is intended for offline code-native graph construction: it never
  // materializes or caches an N x D reconstruction matrix.
  void ReconstructNode(uint32_t node_id, float *output, uint32_t dim) const;

private:
  JiOnlineIndex() = default;

  struct Group {
    uint32_t bit_offset = 0;
    uint32_t bits = 0;
    uint32_t codewords = 0;
    uint32_t group_dim = 0;
    uint64_t prototype_offset = 0;
    uint64_t index_offset = 0;
    uint64_t lut_offset = 0;
    uint64_t mask = 0;
  };

  void Close() noexcept;
  uint32_t CodeAt(uint32_t node_id, const Group &group) const;
  template<bool AddPrimary>
  void CorrectionsBatchImpl(const JiQueryContext &query, const uint32_t *node_ids,
                            size_t count, float *output) const;
  void PrepareQueryInto(const float *query, uint32_t dim, bool vectorized,
                        JiQueryContext &result, std::vector<float> &phi,
                        bool build_upper_bound, JiPrepareTimings *timings) const;

  uint32_t dim_ = 0;
  uint32_t format_version_ = 0;
  uint32_t declared_groups_ = 0;
  uint32_t budget_bits_ = 0;
  uint64_t node_count_ = 0;
  uint64_t artifact_id_ = 0;
  uint64_t model_id_ = 0;
  uint64_t model_bytes_ = 0;
  uint64_t side_bytes_ = 0;
  std::vector<float> residual_mean_;
  // groups_ contains only groups with a non-zero bit width.  Zero-bit group
  // codewords are folded into JiQueryContext::mean_bias at query preparation.
  std::vector<Group> groups_;
  std::vector<Group> zero_bit_groups_;
  // v1 stores dense decoded prototypes.  v2 stores a dense inverse transform,
  // a partition of its rows, and group-local innovation prototypes.
  std::vector<float> dense_prototypes_;
  std::vector<float> inverse_transform_;
  std::vector<uint32_t> group_indices_;
  std::vector<float> local_prototypes_;
  // Read-only shared caches built at load, not per query. SIMD lanes span
  // output coordinates/codewords; no padded overreads or larger node codes.
  std::vector<float> transform_columns_;
  std::vector<float> prototype_columns_;

  int side_fd_ = -1;
  void *side_mapping_ = nullptr;
  size_t side_mapping_size_ = 0;
  uint32_t words_per_node_ = 1;
  uint32_t dense_bytes_per_node_ = 0; // side v3; zero retains legacy word packing
  const uint64_t *packed_codes_ = nullptr;
  // True only for the deployment representation: eight byte-aligned 8-bit
  // groups in one 64-bit word.  This enables one code load per candidate and
  // an unrolled direct-LUT scorer without changing the on-disk format.
  bool direct_8x8_ = false;
};

} // namespace ours

#endif // OURS_JI_ONLINE_H_
