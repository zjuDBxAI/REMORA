#ifndef OURS_DRB_MODEL_H_
#define OURS_DRB_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ours {

enum class ScoreMode { kExact, kAdc, kDrb, kVectorDrb, kResidualPq };
enum class RefinementKind { kScalar, kVector, kResidualPq };

struct MatrixView {
    const float* data = nullptr;
    uint64_t rows = 0;
    uint32_t dim = 0;

    const float* Row(uint64_t row) const;
};

struct DrbModelData {
    uint64_t artifact_id = 0;
    uint64_t node_count = 0;
    uint32_t dim = 0;
    uint32_t subspaces = 0;
    uint32_t codewords = 0;
    uint32_t dsub = 0;
    uint32_t refined_subspaces = 0;
    uint32_t bits_per_subspace = 1;
    uint32_t branches_per_cell = 2;
    uint32_t residual_groups = 0;
    uint32_t residual_codewords = 0;
    uint32_t residual_group_dim = 0;
    uint64_t residual_encoding_id = 0;
    double metric_lambda = 0.0;
    bool has_transform = false;
    bool has_residual_transform = false;
    bool residual_additive = false;
    bool primary_conditioned = false;
    uint32_t conditional_rank = 0;
    RefinementKind refinement_kind = RefinementKind::kScalar;

    // transformed = transform * original + transform_bias
    std::vector<float> transform;
    std::vector<float> transform_bias;
    std::vector<float> centroids;
    std::vector<uint32_t> selected_subspaces;
    std::vector<float> directions;
    std::vector<float> thresholds;
    std::vector<float> beta;
    // [refined subspace, codeword, branch, dsub]. Used by vector DRB.
    std::vector<float> residual_prototypes;
    // [residual group, codeword, group dimension]. Used by full-coverage RPQ.
    std::vector<float> residual_pq_prototypes;
    // Row-coordinate decoder: residual = coded_residual * inverse_transform.
    std::vector<float> residual_inverse_transform;
    // Code-coordinate predictor: U * sum_m E[m, primary_code_m].
    // Basis layout is [dimension, rank], embeddings [subspace, codeword, rank].
    std::vector<float> conditional_basis;
    std::vector<float> conditional_embeddings;
    // Empty means the uniform bits/branches fields above. Mixed vector models
    // store one entry per refined subspace.
    std::vector<uint8_t> bits_per_refined;
    std::vector<uint32_t> branches_per_refined;
    std::vector<uint8_t> codes;
    std::vector<uint8_t> side_bits;
};

struct QueryContext {
    std::vector<float> transformed_query;
    std::vector<float> exact_linear_query;
    std::vector<float> adc_lut;
    // For refined subspaces this already contains ADC + DRB correction.
    std::vector<float> drb_fused_lut;
    // Full-coverage residual-PQ correction LUT [group, codeword].
    std::vector<float> residual_lut;
    float exact_bias = 0.0f;
};

// Per-thread scratch space for repeated base--base comparisons against one
// fixed anchor during code-native graph construction.  The prepared tables
// have the same layout as an online PQ/RPQ query LUT and never contain a
// materialized database matrix.
struct ResidualPqBaseAnchorContext {
    uint32_t anchor_id = 0;
    bool ready = false;
    std::vector<float> reconstruction;
    QueryContext query;
};

class DrbPqIndex {
   public:
    static DrbPqIndex Load(const std::string& model_path,
                           const std::string& codes_path,
                           const std::string& side_bits_path = "");
    static DrbPqIndex FromData(DrbModelData data);

    QueryContext PrepareQuery(const float* query,
                              uint32_t dim,
                              bool prepare_exact = true,
                              ScoreMode mode = ScoreMode::kDrb) const;
    float Score(const QueryContext& query,
                uint32_t node_id,
                ScoreMode mode,
                const MatrixView& base) const;
    float ExactScore(const QueryContext& query,
                     uint32_t node_id,
                     const MatrixView& base) const;
    float AdcScore(const QueryContext& query, uint32_t node_id) const;
    float DrbScore(const QueryContext& query, uint32_t node_id) const;
    float VectorDrbScore(const QueryContext& query, uint32_t node_id) const;
    float ResidualPqScore(const QueryContext& query, uint32_t node_id) const;
    void ScoreBatch(const QueryContext& query,
                    const uint32_t* node_ids,
                    size_t count,
                    ScoreMode mode,
                    const MatrixView& base,
                    float* scores) const;

    // Offline code-native RoarGraph construction support.  These methods use
    // only PQ/RPQ codes and codebooks; they never require the original base
    // vectors or a materialized N x D reconstruction matrix.
    void PrepareResidualPqBasePairLookup();
    float ResidualPqBasePairScore(uint32_t lhs, uint32_t rhs) const;
    float ResidualPqBasePairDistance(uint32_t lhs, uint32_t rhs) const;
    void PrepareResidualPqBaseAnchor(
        uint32_t anchor_id, ResidualPqBaseAnchorContext* context) const;
    float ResidualPqPreparedBaseAnchorScore(
        const ResidualPqBaseAnchorContext& context,
        uint32_t candidate_id) const;
    float ResidualPqPreparedBaseAnchorDistance(
        const ResidualPqBaseAnchorContext& context,
        uint32_t candidate_id) const;
    void ResidualPqPreparedBaseAnchorDistanceBatch(
        const ResidualPqBaseAnchorContext& context,
        const uint32_t* candidate_ids,
        size_t count,
        float* distances) const;
    void ReconstructResidualPqNode(uint32_t node_id, float* output,
                                   uint32_t dim) const;
    std::vector<float> MeanResidualPqReconstruction() const;
    uint64_t BasePairLookupBytes() const;

    uint64_t node_count() const { return data_.node_count; }
    uint32_t dim() const { return data_.dim; }
    uint32_t refined_subspaces() const { return data_.refined_subspaces; }
    RefinementKind refinement_kind() const { return data_.refinement_kind; }
    uint32_t bits_per_subspace() const { return data_.bits_per_subspace; }
    uint32_t branches_per_cell() const { return data_.branches_per_cell; }
    uint32_t residual_groups() const { return data_.residual_groups; }
    uint32_t subspaces() const { return data_.subspaces; }
    uint32_t codewords() const { return data_.codewords; }
    uint32_t residual_codewords() const { return data_.residual_codewords; }
    uint32_t residual_group_dim() const { return data_.residual_group_dim; }

   private:
    explicit DrbPqIndex(DrbModelData data);
    void Validate() const;
    void BuildRuntimeLayout();
    float RefinedScore(const QueryContext& query, uint32_t node_id) const;
    uint32_t DecodeBranch(const uint8_t* side_codes, uint32_t ridx) const;
    const uint8_t* NodeRecord(uint32_t node_id) const;

    struct CrossPairBlock {
        uint32_t subspace = 0;
        uint32_t group = 0;
        uint32_t subspace_offset = 0;
        uint32_t group_offset = 0;
        uint32_t length = 0;
        uint64_t table_offset = 0;
    };

    DrbModelData data_;
    // Runtime record: [M PQ bytes][ceil(R/8) side-bit bytes]. The on-disk
    // artifacts remain backward compatible and are interleaved at load time.
    std::vector<uint8_t> node_records_;
    std::vector<uint32_t> unrefined_subspaces_;
    std::vector<uint32_t> side_bit_offsets_;
    std::vector<uint32_t> side_byte_offsets_;
    std::vector<uint8_t> side_shifts_;
    std::vector<uint16_t> side_masks_;
    std::vector<uint8_t> side_crosses_byte_;
    std::vector<uint64_t> branch_lut_offsets_;
    std::vector<float> effective_centroids_;
    // [residual group, codeword, full dimension] in primary-PQ coordinates.
    std::vector<float> decoded_residual_pq_prototypes_;
    // [rank, full dimension] in primary-PQ coordinates.
    std::vector<float> decoded_conditional_basis_;
    std::vector<float> primary_pair_lut_;
    std::vector<float> residual_pair_lut_;
    std::vector<float> primary_residual_pair_lut_;
    std::vector<CrossPairBlock> cross_pair_blocks_;
    bool base_pair_lookup_ready_ = false;
    uint32_t side_bit_width_ = 0;
    uint32_t record_stride_ = 0;
};

ScoreMode ParseScoreMode(const std::string& value);
const char* ScoreModeName(ScoreMode mode);

}  // namespace ours

#endif  // OURS_DRB_MODEL_H_
