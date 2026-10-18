#ifndef OURS_SEARCH_KERNEL_H_
#define OURS_SEARCH_KERNEL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "ours/drb_model.h"
#include "ours/roar_graph.h"

namespace ours {

enum class EventSource : uint8_t { kExpansion = 0, kQueue = 1, kStop = 2 };
enum class EventKind : uint8_t { kProtect = 0, kInversion = 1 };

struct DecisionEvent {
    uint64_t query_id = 0;
    uint64_t step = 0;
    EventSource source = EventSource::kExpansion;
    EventKind kind = EventKind::kProtect;
    uint32_t u_id = 0;
    uint32_t v_id = 0;
    float exact_gap = 0.0f;
    float adc_gap = 0.0f;
    float trajectory_gap = 0.0f;
};

struct TraceOptions {
    bool enabled = false;
    float min_exact_gap = 1e-8f;
    float protect_adc_margin = 0.02f;
    uint32_t max_inversions_per_source = 32;
    uint32_t max_protect_per_source = 32;
};

struct SearchOptions {
    uint32_t topk = 10;
    uint32_t beam_width = 100;
    ScoreMode mode = ScoreMode::kDrb;
    bool collect_visited = false;
    TraceOptions trace;
};

struct SearchResult {
    std::vector<uint32_t> ids;
    std::vector<float> scores;
    std::vector<DecisionEvent> events;
    std::vector<uint32_t> visited_ids;
    uint32_t comparisons = 0;
    uint32_t expansions = 0;
    uint32_t visited_nodes = 0;
    uint32_t final_candidate_pool = 0;
};

class DrbRoarSearch {
   public:
    DrbRoarSearch(const RoarGraph& graph,
                  const DrbPqIndex& model,
                  MatrixView base);

    SearchResult Search(const float* query,
                        uint32_t query_dim,
                        uint64_t query_id,
                        const SearchOptions& options) const;

   private:
    const RoarGraph& graph_;
    const DrbPqIndex& model_;
    MatrixView base_;
};

const char* EventSourceName(EventSource source);
const char* EventKindName(EventKind kind);

}  // namespace ours

#endif  // OURS_SEARCH_KERNEL_H_
