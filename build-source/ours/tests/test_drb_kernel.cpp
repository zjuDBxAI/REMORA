#include "ours/drb_model.h"
#include "ours/roar_graph.h"
#include "ours/search_kernel.h"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void CheckNear(float actual, float expected, float tolerance, const char* message) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

ours::DrbPqIndex SyntheticModel() {
    ours::DrbModelData data;
    data.artifact_id = 7;
    data.node_count = 8;
    data.dim = 2;
    data.subspaces = 1;
    data.codewords = 2;
    data.dsub = 2;
    data.refined_subspaces = 1;
    data.has_transform = false;
    data.transform = {1.0f, 0.0f, 0.0f, 1.0f};
    data.transform_bias = {0.0f, 0.0f};
    data.centroids = {0.0f, 0.0f, 0.5f, 0.0f};
    data.selected_subspaces = {0};
    data.directions = {1.0f, 0.0f};
    data.thresholds = {0.0f, 0.0f};
    // [code 0 branch 0/1, code 1 branch 0/1]
    data.beta = {0.0f, 1.0f, 0.0f, 0.25f};
    data.codes = {0, 0, 1, 1, 0, 0, 1, 1};
    data.side_bits = {0, 1, 0, 1, 0, 1, 0, 1};
    return ours::DrbPqIndex::FromData(std::move(data));
}

ours::DrbPqIndex SyntheticVectorModel() {
    ours::DrbModelData data;
    data.artifact_id = 17;
    data.node_count = 8;
    data.dim = 2;
    data.subspaces = 1;
    data.codewords = 2;
    data.dsub = 2;
    data.refined_subspaces = 1;
    data.bits_per_subspace = 2;
    data.branches_per_cell = 4;
    data.refinement_kind = ours::RefinementKind::kVector;
    data.has_transform = false;
    data.transform = {1.0f, 0.0f, 0.0f, 1.0f};
    data.transform_bias = {0.0f, 0.0f};
    data.centroids = {0.0f, 0.0f, 0.5f, 0.0f};
    data.selected_subspaces = {0};
    // [code 0 branches 0..3, code 1 branches 0..3], each prototype is 2-D.
    data.residual_prototypes = {
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
        0.0f, 0.0f, 0.25f, 0.0f, 0.0f, 0.5f, 0.25f, 0.5f,
    };
    data.codes = {0, 0, 1, 1, 0, 0, 1, 1};
    data.side_bits = {0, 1, 2, 3, 0, 1, 2, 3};
    return ours::DrbPqIndex::FromData(std::move(data));
}

ours::DrbPqIndex SyntheticResidualPqModel(bool transformed = false) {
    ours::DrbModelData data;
    data.artifact_id = 27;
    data.residual_encoding_id = 31;
    data.node_count = 8;
    data.dim = 4;
    data.subspaces = 2;
    data.codewords = 2;
    data.dsub = 2;
    data.refinement_kind = ours::RefinementKind::kResidualPq;
    data.residual_groups = 2;
    data.residual_codewords = 256;
    data.residual_group_dim = 2;
    data.has_transform = false;
    data.has_residual_transform = transformed;
    data.transform = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    data.transform_bias.assign(4, 0.0f);
    data.centroids = {
        0.1f, -0.2f, 0.5f, 0.25f,
        -0.3f, 0.4f, 0.75f, -0.5f,
    };
    data.residual_pq_prototypes.assign(2 * 256 * 2, 0.0f);
    data.residual_pq_prototypes[(0 * 256 + 1) * 2 + 0] = 1.0f;
    data.residual_pq_prototypes[(1 * 256 + 1) * 2 + 1] = 1.0f;
    if (transformed) {
        // Swap the two 2D residual blocks when decoding coded coordinates.
        data.residual_inverse_transform = {
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
        };
    }
    data.codes = {
        0, 0, 0, 1, 1, 0, 1, 1,
        0, 0, 0, 1, 1, 0, 1, 1,
    };
    data.side_bits = {
        0, 0, 1, 0, 0, 1, 1, 1,
        0, 0, 1, 0, 0, 1, 1, 1,
    };
    return ours::DrbPqIndex::FromData(std::move(data));
}

ours::DrbPqIndex SyntheticAdditiveConditionalResidualModel() {
    ours::DrbModelData data;
    data.artifact_id = 37;
    data.residual_encoding_id = 41;
    data.node_count = 8;
    data.dim = 4;
    data.subspaces = 2;
    data.codewords = 2;
    data.dsub = 2;
    data.refinement_kind = ours::RefinementKind::kResidualPq;
    data.residual_groups = 2;
    data.residual_codewords = 256;
    data.residual_group_dim = 4;
    data.residual_additive = true;
    data.primary_conditioned = true;
    data.conditional_rank = 2;
    data.has_transform = false;
    data.transform = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    data.transform_bias.assign(4, 0.0f);
    data.centroids = {
        0.1f, -0.2f, 0.5f, 0.25f,
        -0.3f, 0.4f, 0.75f, -0.5f,
    };
    // Two full-dimensional additive dictionaries. Only code 1 is nonzero.
    data.residual_pq_prototypes.assign(2 * 256 * 4, 0.0f);
    data.residual_pq_prototypes[(0 * 256 + 1) * 4 + 0] = 1.0f;
    data.residual_pq_prototypes[(1 * 256 + 1) * 4 + 3] = 1.0f;
    // Basis is stored as [dimension, rank].
    data.conditional_basis = {
        0.2f, 0.0f,
        0.0f, 0.1f,
        0.05f, 0.0f,
        0.0f, 0.2f,
    };
    // Embeddings are [primary subspace, primary codeword, rank].
    data.conditional_embeddings = {
        0.0f, 0.0f, 1.0f, 0.5f,
        0.25f, 0.0f, 0.0f, 2.0f,
    };
    data.codes = {
        0, 0, 0, 1, 1, 0, 1, 1,
        0, 0, 0, 1, 1, 0, 1, 1,
    };
    data.side_bits = {
        0, 0, 1, 0, 0, 1, 1, 1,
        0, 0, 1, 0, 0, 1, 1, 1,
    };
    return ours::DrbPqIndex::FromData(std::move(data));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 4) {
            const ours::DrbPqIndex loaded =
                ours::DrbPqIndex::Load(argv[1], argv[2], argv[3]);
            std::vector<float> query(loaded.dim());
            for (uint32_t coordinate = 0; coordinate < loaded.dim(); ++coordinate) {
                query[coordinate] = 0.1f * static_cast<float>(coordinate + 1);
            }
            const ours::QueryContext context = loaded.PrepareQuery(
                query.data(), loaded.dim(), false, ours::ScoreMode::kResidualPq);
            std::cout << std::setprecision(9);
            const uint32_t count = static_cast<uint32_t>(
                std::min<uint64_t>(loaded.node_count(), 8));
            for (uint32_t node = 0; node < count; ++node) {
                std::cout << loaded.ResidualPqScore(context, node) << '\n';
            }
            return 0;
        }
        if (argc != 1) {
            throw std::invalid_argument(
                "usage: roar_drb_tests [MODEL CODES SIDE_CODES]");
        }
        const ours::DrbPqIndex model = SyntheticModel();
        const std::vector<float> base_values = {
            0.0f, 0.0f,
            1.0f, 0.0f,
            0.5f, 0.0f,
            0.75f, 0.0f,
            0.0f, 0.0f,
            1.0f, 0.0f,
            0.5f, 0.0f,
            0.75f, 0.0f,
        };
        const ours::MatrixView base{base_values.data(), 8, 2};
        const float query[2] = {1.0f, 0.0f};
        const ours::QueryContext context = model.PrepareQuery(query, 2);
        CheckNear(model.AdcScore(context, 1), 0.0f, 1e-6f, "unexpected ADC score");
        CheckNear(model.DrbScore(context, 1), 1.0f, 1e-6f, "unexpected DRB correction");
        CheckNear(model.ExactScore(context, 3, base), 0.75f, 1e-6f, "unexpected exact score");
        const uint32_t batch_ids[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        float batch_scores[8] = {};
        model.ScoreBatch(context, batch_ids, 8, ours::ScoreMode::kDrb,
                         base, batch_scores);
        for (uint32_t i = 0; i < 8; ++i) {
            CheckNear(batch_scores[i], model.DrbScore(context, i), 1e-6f,
                      "SIMD batch DRB score differs from scalar score");
        }

        const ours::DrbPqIndex vector_model = SyntheticVectorModel();
        const float vector_query[2] = {1.0f, 2.0f};
        const ours::QueryContext vector_context =
            vector_model.PrepareQuery(vector_query, 2, false,
                                      ours::ScoreMode::kVectorDrb);
        CheckNear(vector_model.VectorDrbScore(vector_context, 1), 1.0f, 1e-6f,
                  "unexpected vector branch-1 score");
        CheckNear(vector_model.VectorDrbScore(vector_context, 2), 1.5f, 1e-6f,
                  "unexpected vector branch-2 score");
        float vector_batch_scores[8] = {};
        vector_model.ScoreBatch(vector_context, batch_ids, 8,
                                ours::ScoreMode::kVectorDrb, base,
                                vector_batch_scores);
        for (uint32_t i = 0; i < 8; ++i) {
            CheckNear(vector_batch_scores[i],
                      vector_model.VectorDrbScore(vector_context, i), 1e-6f,
                      "SIMD vector DRB score differs from scalar score");
        }
        Check(ours::ParseScoreMode("vdrb") == ours::ScoreMode::kVectorDrb,
              "vdrb mode parsing failed");

        ours::DrbPqIndex residual_model = SyntheticResidualPqModel();
        const float residual_query[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        const ours::QueryContext residual_context = residual_model.PrepareQuery(
            residual_query, 4, false, ours::ScoreMode::kResidualPq);
        const std::vector<float> residual_base_values(8 * 4, 0.0f);
        const ours::MatrixView residual_base{
            residual_base_values.data(), 8, 4};
        CheckNear(residual_model.ResidualPqScore(residual_context, 3),
                  6.25f, 1e-6f, "unexpected residual-PQ score");
        float residual_batch_scores[8] = {};
        residual_model.ScoreBatch(residual_context, batch_ids, 8,
                                  ours::ScoreMode::kResidualPq,
                                  residual_base, residual_batch_scores);
        for (uint32_t i = 0; i < 8; ++i) {
            CheckNear(residual_batch_scores[i],
                      residual_model.ResidualPqScore(residual_context, i),
                      1e-6f,
                      "SIMD residual-PQ score differs from scalar score");
        }
        Check(ours::ParseScoreMode("rpq") == ours::ScoreMode::kResidualPq,
              "rpq mode parsing failed");

        ours::DrbPqIndex transformed_residual_model =
            SyntheticResidualPqModel(true);
        const float transformed_residual_query[4] = {1.0f, 2.0f, 5.0f, 9.0f};
        const ours::QueryContext transformed_residual_context =
            transformed_residual_model.PrepareQuery(
                transformed_residual_query, 4, false,
                ours::ScoreMode::kResidualPq);
        std::vector<float> transformed_reconstruction(4);
        transformed_residual_model.ReconstructResidualPqNode(
            3, transformed_reconstruction.data(), 4);
        float transformed_expected_score = 0.0f;
        for (uint32_t coordinate = 0; coordinate < 4; ++coordinate) {
            transformed_expected_score += transformed_residual_query[coordinate] *
                                          transformed_reconstruction[coordinate];
        }
        CheckNear(
            transformed_residual_model.ResidualPqScore(
                transformed_residual_context, 3),
            transformed_expected_score, 1e-6f,
            "transformed residual-PQ score differs from decoded reconstruction");
        transformed_residual_model.PrepareResidualPqBasePairLookup();
        ours::ResidualPqBaseAnchorContext transformed_prepared_anchor;
        transformed_residual_model.PrepareResidualPqBaseAnchor(
            3, &transformed_prepared_anchor);
        for (uint32_t node = 0; node < 8; ++node) {
            std::vector<float> candidate_reconstruction(4);
            transformed_residual_model.ReconstructResidualPqNode(
                node, candidate_reconstruction.data(), 4);
            float expected = 0.0f;
            for (uint32_t coordinate = 0; coordinate < 4; ++coordinate) {
                expected += transformed_reconstruction[coordinate] *
                            candidate_reconstruction[coordinate];
            }
            CheckNear(
                transformed_residual_model.ResidualPqBasePairScore(3, node),
                expected, 1e-6f,
                "transformed direct pair score differs from reconstruction");
            CheckNear(
                transformed_residual_model.ResidualPqPreparedBaseAnchorScore(
                    transformed_prepared_anchor, node),
                expected, 1e-6f,
                "transformed prepared pair score differs from reconstruction");
        }

        // The offline builder's direct code-pair lookup must be exactly the
        // inner product of the two PQ+RPQ reconstructions.  This also checks
        // the primary/residual cross terms, which cannot be omitted.
        residual_model.PrepareResidualPqBasePairLookup();
        ours::ResidualPqBaseAnchorContext prepared_anchor;
        std::vector<float> lhs_reconstruction(4);
        std::vector<float> rhs_reconstruction(4);
        for (uint32_t lhs = 0; lhs < 8; ++lhs) {
            residual_model.ReconstructResidualPqNode(
                lhs, lhs_reconstruction.data(), 4);
            for (uint32_t rhs = 0; rhs < 8; ++rhs) {
                residual_model.ReconstructResidualPqNode(
                    rhs, rhs_reconstruction.data(), 4);
                float expected_pair_score = 0.0f;
                for (uint32_t coordinate = 0; coordinate < 4; ++coordinate) {
                    expected_pair_score += lhs_reconstruction[coordinate] *
                                           rhs_reconstruction[coordinate];
                }
                CheckNear(
                    residual_model.ResidualPqBasePairScore(lhs, rhs),
                    expected_pair_score, 1e-6f,
                    "direct PQ/RPQ base-pair lookup differs from reconstruction");
                CheckNear(
                    residual_model.ResidualPqBasePairDistance(lhs, rhs),
                    -expected_pair_score, 1e-6f,
                    "direct PQ/RPQ base-pair distance has the wrong sign");
                residual_model.PrepareResidualPqBaseAnchor(
                    rhs, &prepared_anchor);
                CheckNear(
                    residual_model.ResidualPqPreparedBaseAnchorScore(
                        prepared_anchor, lhs),
                    expected_pair_score, 1e-6f,
                    "prepared PQ/RPQ anchor score differs from reconstruction");
                CheckNear(
                    residual_model.ResidualPqPreparedBaseAnchorDistance(
                        prepared_anchor, lhs),
                    -expected_pair_score, 1e-6f,
                    "prepared PQ/RPQ anchor distance has the wrong sign");
            }
        }
        residual_model.PrepareResidualPqBaseAnchor(3, &prepared_anchor);
        float prepared_batch_distances[8] = {};
        residual_model.ResidualPqPreparedBaseAnchorDistanceBatch(
            prepared_anchor, batch_ids, 8, prepared_batch_distances);
        for (uint32_t node = 0; node < 8; ++node) {
            CheckNear(
                prepared_batch_distances[node],
                residual_model.ResidualPqPreparedBaseAnchorDistance(
                    prepared_anchor, node),
                1e-6f,
                "prepared PQ/RPQ batch distance differs from scalar distance");
        }
        const std::vector<float> mean_reconstruction =
            residual_model.MeanResidualPqReconstruction();
        std::vector<float> explicit_mean(4, 0.0f);
        for (uint32_t node = 0; node < 8; ++node) {
            residual_model.ReconstructResidualPqNode(
                node, lhs_reconstruction.data(), 4);
            for (uint32_t coordinate = 0; coordinate < 4; ++coordinate) {
                explicit_mean[coordinate] += lhs_reconstruction[coordinate] / 8.0f;
            }
        }
        for (uint32_t coordinate = 0; coordinate < 4; ++coordinate) {
            CheckNear(mean_reconstruction[coordinate], explicit_mean[coordinate],
                      1e-6f,
                      "histogram-derived reconstruction mean is incorrect");
        }

        ours::DrbPqIndex additive_model =
            SyntheticAdditiveConditionalResidualModel();
        const float additive_query[4] = {1.0f, 2.0f, 5.0f, 9.0f};
        const ours::QueryContext additive_context = additive_model.PrepareQuery(
            additive_query, 4, false, ours::ScoreMode::kResidualPq);
        std::vector<float> additive_anchor(4);
        additive_model.ReconstructResidualPqNode(
            3, additive_anchor.data(), 4);
        float additive_expected = 0.0f;
        for (uint32_t coordinate = 0; coordinate < 4; ++coordinate) {
            additive_expected +=
                additive_query[coordinate] * additive_anchor[coordinate];
        }
        CheckNear(additive_model.ResidualPqScore(additive_context, 3),
                  additive_expected, 1e-6f,
                  "additive conditional query score differs from reconstruction");
        additive_model.PrepareResidualPqBasePairLookup();
        ours::ResidualPqBaseAnchorContext additive_prepared_anchor;
        additive_model.PrepareResidualPqBaseAnchor(
            3, &additive_prepared_anchor);
        for (uint32_t node = 0; node < 8; ++node) {
            std::vector<float> candidate(4);
            additive_model.ReconstructResidualPqNode(node, candidate.data(), 4);
            float pair_expected = 0.0f;
            for (uint32_t coordinate = 0; coordinate < 4; ++coordinate) {
                pair_expected += additive_anchor[coordinate] * candidate[coordinate];
            }
            CheckNear(additive_model.ResidualPqBasePairScore(3, node),
                      pair_expected, 1e-6f,
                      "additive conditional pair score differs from reconstruction");
            CheckNear(
                additive_model.ResidualPqPreparedBaseAnchorScore(
                    additive_prepared_anchor, node),
                pair_expected, 1e-6f,
                "prepared additive conditional score differs from reconstruction");
        }
        const std::vector<float> additive_mean =
            additive_model.MeanResidualPqReconstruction();
        std::vector<float> additive_explicit_mean(4, 0.0f);
        for (uint32_t node = 0; node < 8; ++node) {
            additive_model.ReconstructResidualPqNode(
                node, lhs_reconstruction.data(), 4);
            for (uint32_t coordinate = 0; coordinate < 4; ++coordinate) {
                additive_explicit_mean[coordinate] +=
                    lhs_reconstruction[coordinate] / 8.0f;
            }
        }
        for (uint32_t coordinate = 0; coordinate < 4; ++coordinate) {
            CheckNear(additive_mean[coordinate],
                      additive_explicit_mean[coordinate], 1e-6f,
                      "additive conditional reconstruction mean is incorrect");
        }

        const ours::RoarGraph graph = ours::RoarGraph::FromAdjacency(
            0, {{2, 1}, {3}, {}, {}, {}, {}, {}, {}});
        const ours::DrbRoarSearch kernel(graph, model, base);
        ours::SearchOptions options;
        options.topk = 2;
        options.beam_width = 2;
        options.mode = ours::ScoreMode::kAdc;
        options.collect_visited = true;
        options.trace.enabled = true;
        options.trace.protect_adc_margin = 1.0f;
        const ours::SearchResult adc = kernel.Search(query, 2, 7, options);
        Check(adc.ids.size() == 2 && adc.ids[0] == 2 && adc.ids[1] == 0,
              "ADC did not control graph traversal as expected");
        Check(adc.visited_ids == std::vector<uint32_t>({0, 1, 2}),
              "ADC visited-candidate collection is incorrect");
        bool queue_inversion = false;
        bool stop_inversion = false;
        for (const auto& event : adc.events) {
            queue_inversion |= event.source == ours::EventSource::kQueue &&
                               event.kind == ours::EventKind::kInversion;
            stop_inversion |= event.source == ours::EventSource::kStop &&
                              event.kind == ours::EventKind::kInversion;
        }
        Check(queue_inversion, "missing queue inversion event");
        Check(stop_inversion, "missing stop inversion event");

        options.mode = ours::ScoreMode::kDrb;
        options.trace.enabled = false;
        const ours::SearchResult drb = kernel.Search(query, 2, 7, options);
        Check(drb.ids.size() == 2 && drb.ids[0] == 1 && drb.ids[1] == 3,
              "DRB score did not change expansion/queue evolution");

        // A production ADC/DRB query with tracing disabled does not require a
        // full-precision base payload. Online scores come from codes/LUT/bits.
        const ours::DrbRoarSearch code_only_kernel(
            graph, model, ours::MatrixView{});
        const ours::SearchResult code_only =
            code_only_kernel.Search(query, 2, 8, options);
        Check(code_only.ids == drb.ids, "code-only DRB search changed results");
        std::cout << "all DRB-PQ kernel tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
