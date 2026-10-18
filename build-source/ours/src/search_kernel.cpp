#include "ours/search_kernel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace ours {
namespace {

struct Candidate {
    uint32_t id = 0;
    float score = 0.0f;
    bool expanded = false;
};

bool Better(const Candidate& left, const Candidate& right) {
    return left.score > right.score || (left.score == right.score && left.id < right.id);
}

class CandidatePool {
   public:
    struct InsertResult {
        bool inserted = false;
        bool duplicate = false;
        std::optional<Candidate> rejected;
    };

    explicit CandidatePool(uint32_t capacity) : capacity_(capacity) {
        values_.reserve(capacity + 1U);
    }

    bool full() const { return values_.size() >= capacity_; }
    bool empty() const { return values_.empty(); }
    size_t size() const { return values_.size(); }
    const Candidate& worst() const { return values_.back(); }
    const std::vector<Candidate>& values() const { return values_; }

    std::vector<uint32_t> UnexpandedIds() const {
        std::vector<uint32_t> ids;
        ids.reserve(values_.size());
        for (const Candidate& value : values_) {
            if (!value.expanded) ids.push_back(value.id);
        }
        return ids;
    }

    std::optional<Candidate> PopBestUnexpanded() {
        for (Candidate& value : values_) {
            if (!value.expanded) {
                value.expanded = true;
                return value;
            }
        }
        return std::nullopt;
    }

    InsertResult Insert(Candidate candidate) {
        if (capacity_ == 0) {
            throw std::logic_error("candidate pool has zero capacity");
        }
        for (const Candidate& value : values_) {
            if (value.id == candidate.id) {
                return InsertResult{false, true, std::nullopt};
            }
        }
        const auto position = std::lower_bound(values_.begin(), values_.end(), candidate, Better);
        if (values_.size() == capacity_ && position == values_.end()) {
            return InsertResult{false, false, candidate};
        }
        values_.insert(position, candidate);
        InsertResult result;
        result.inserted = true;
        if (values_.size() > capacity_) {
            result.rejected = values_.back();
            values_.pop_back();
        }
        return result;
    }

   private:
    uint32_t capacity_;
    std::vector<Candidate> values_;
};

class TraceBuilder {
   public:
    TraceBuilder(const TraceOptions& options,
                 uint64_t query_id,
                 const DrbPqIndex& model,
                 const QueryContext& query,
                 ScoreMode trajectory_mode,
                 MatrixView base)
        : options_(options),
          query_id_(query_id),
          model_(model),
          query_(query),
          trajectory_mode_(trajectory_mode),
          base_(base) {}

    void AddPair(EventSource source, uint64_t step, uint32_t first, uint32_t second) {
        if (!options_.enabled || first == second) return;
        const float exact_first = model_.ExactScore(query_, first, base_);
        const float exact_second = model_.ExactScore(query_, second, base_);
        if (!std::isfinite(exact_first) || !std::isfinite(exact_second) ||
            exact_first == exact_second) {
            return;
        }
        const uint32_t u = exact_first > exact_second ? first : second;
        const uint32_t v = exact_first > exact_second ? second : first;
        const float exact_gap = std::fabs(exact_first - exact_second);
        if (exact_gap <= options_.min_exact_gap) return;
        const float adc_gap = model_.AdcScore(query_, u) - model_.AdcScore(query_, v);
        const EventKind kind = adc_gap <= 0.0f ? EventKind::kInversion : EventKind::kProtect;
        if (kind == EventKind::kProtect && adc_gap > options_.protect_adc_margin) return;
        const size_t source_index = static_cast<size_t>(source);
        const size_t kind_index = static_cast<size_t>(kind);
        const uint32_t cap = kind == EventKind::kInversion
                                 ? options_.max_inversions_per_source
                                 : options_.max_protect_per_source;
        if (counts_[source_index][kind_index] >= cap) return;
        counts_[source_index][kind_index]++;
        const float trajectory_gap =
            model_.Score(query_, u, trajectory_mode_, base_) -
            model_.Score(query_, v, trajectory_mode_, base_);
        events_.push_back(DecisionEvent{query_id_, step, source, kind, u, v,
                                        exact_gap, adc_gap, trajectory_gap});
    }

    std::vector<DecisionEvent> Take() { return std::move(events_); }

   private:
    TraceOptions options_;
    uint64_t query_id_;
    const DrbPqIndex& model_;
    const QueryContext& query_;
    ScoreMode trajectory_mode_;
    MatrixView base_;
    std::array<std::array<uint32_t, 2>, 3> counts_{};
    std::vector<DecisionEvent> events_;
};

uint32_t ExactBest(const std::vector<uint32_t>& ids,
                   const DrbPqIndex& model,
                   const QueryContext& query,
                   MatrixView base) {
    if (ids.empty()) throw std::logic_error("cannot choose from an empty id list");
    uint32_t best = ids.front();
    float best_score = model.ExactScore(query, best, base);
    for (size_t i = 1; i < ids.size(); ++i) {
        const float score = model.ExactScore(query, ids[i], base);
        if (score > best_score || (score == best_score && ids[i] < best)) {
            best = ids[i];
            best_score = score;
        }
    }
    return best;
}

uint32_t ExactWorst(const std::vector<Candidate>& values,
                    const DrbPqIndex& model,
                    const QueryContext& query,
                    MatrixView base) {
    if (values.empty()) throw std::logic_error("cannot choose from an empty candidate pool");
    uint32_t worst = values.front().id;
    float worst_score = model.ExactScore(query, worst, base);
    for (size_t i = 1; i < values.size(); ++i) {
        const float score = model.ExactScore(query, values[i].id, base);
        if (score < worst_score || (score == worst_score && values[i].id > worst)) {
            worst = values[i].id;
            worst_score = score;
        }
    }
    return worst;
}

}  // namespace

DrbRoarSearch::DrbRoarSearch(const RoarGraph& graph,
                             const DrbPqIndex& model,
                             MatrixView base)
    : graph_(graph), model_(model), base_(base) {
    if (graph_.size() != model_.node_count()) {
        throw std::invalid_argument("graph/model shapes do not match");
    }
    const bool has_base = base_.data != nullptr || base_.rows != 0 || base_.dim != 0;
    if (has_base &&
        (base_.data == nullptr || base_.rows < model_.node_count() ||
         base_.dim != model_.dim())) {
        throw std::invalid_argument("base/model shapes do not match");
    }
}

SearchResult DrbRoarSearch::Search(const float* query,
                                   uint32_t query_dim,
                                   uint64_t query_id,
                                   const SearchOptions& options) const {
    if (options.topk == 0 || options.beam_width < options.topk) {
        throw std::invalid_argument("search requires 0 < topk <= beam_width");
    }
    if (options.mode == ScoreMode::kDrb && model_.refined_subspaces() == 0) {
        throw std::invalid_argument("DRB mode requires a fitted model with side bits");
    }
    if (options.mode == ScoreMode::kDrb &&
        model_.refinement_kind() != RefinementKind::kScalar) {
        throw std::invalid_argument("DRB mode requires a scalar DRB model");
    }
    if (options.mode == ScoreMode::kVectorDrb &&
        (model_.refined_subspaces() == 0 ||
         model_.refinement_kind() != RefinementKind::kVector)) {
        throw std::invalid_argument("vdrb mode requires a fitted vector DRB model");
    }
    if (options.mode == ScoreMode::kResidualPq &&
        (model_.residual_groups() == 0 ||
         model_.refinement_kind() != RefinementKind::kResidualPq)) {
        throw std::invalid_argument("rpq mode requires a residual-PQ online model");
    }
    const bool needs_exact = options.mode == ScoreMode::kExact || options.trace.enabled;
    if (needs_exact && base_.data == nullptr) {
        throw std::invalid_argument(
            "full-precision base vectors are required for exact scoring or tracing");
    }
    const QueryContext context =
        model_.PrepareQuery(query, query_dim, needs_exact, options.mode);
    CandidatePool pool(options.beam_width);
    std::unordered_set<uint32_t> visited;
    visited.reserve(static_cast<size_t>(options.beam_width) * 32U);
    TraceBuilder trace(options.trace, query_id, model_, context, options.mode, base_);

    const uint32_t entry = graph_.entry_point();
    visited.insert(entry);
    pool.Insert(Candidate{entry, model_.Score(context, entry, options.mode, base_), false});

    SearchResult result;
    std::vector<uint32_t> pending_ids;
    std::vector<float> pending_scores;
    pending_ids.reserve(128);
    pending_scores.reserve(128);
    std::optional<uint32_t> best_rejected;
    float best_rejected_exact = -std::numeric_limits<float>::infinity();
    uint64_t step = 0;

    const auto remember_rejected = [&](const Candidate& rejected) {
        if (!options.trace.enabled) return;
        if (rejected.expanded) return;
        const float exact = model_.ExactScore(context, rejected.id, base_);
        if (!best_rejected.has_value() || exact > best_rejected_exact ||
            (exact == best_rejected_exact && rejected.id < *best_rejected)) {
            best_rejected = rejected.id;
            best_rejected_exact = exact;
        }
    };

    while (true) {
        if (options.trace.enabled) {
            const std::vector<uint32_t> unexpanded = pool.UnexpandedIds();
            if (unexpanded.empty()) break;
            const uint32_t trajectory_best = unexpanded.front();
            const uint32_t teacher_best = ExactBest(unexpanded, model_, context, base_);
            if (teacher_best != trajectory_best) {
                trace.AddPair(EventSource::kExpansion, step, teacher_best, trajectory_best);
            } else if (unexpanded.size() > 1) {
                std::vector<uint32_t> competitors;
                competitors.reserve(unexpanded.size() - 1);
                for (uint32_t id : unexpanded) {
                    if (id != trajectory_best) competitors.push_back(id);
                }
                trace.AddPair(EventSource::kExpansion, step, trajectory_best,
                              ExactBest(competitors, model_, context, base_));
            }
        }

        const std::optional<Candidate> current = pool.PopBestUnexpanded();
        if (!current.has_value()) break;
        result.expansions++;

        pending_ids.clear();
        for (uint32_t neighbor : graph_.Neighbors(current->id)) {
            if (!visited.insert(neighbor).second) continue;
            pending_ids.push_back(neighbor);
        }
        result.comparisons += static_cast<uint32_t>(pending_ids.size());
        pending_scores.resize(pending_ids.size());
        model_.ScoreBatch(context, pending_ids.data(), pending_ids.size(),
                          options.mode, base_, pending_scores.data());

        for (size_t neighbor_index = 0; neighbor_index < pending_ids.size(); ++neighbor_index) {
            const uint32_t neighbor = pending_ids[neighbor_index];
            const float score = pending_scores[neighbor_index];
            if (pool.full()) {
                trace.AddPair(EventSource::kQueue, step, neighbor, pool.worst().id);
            }
            const CandidatePool::InsertResult inserted =
                pool.Insert(Candidate{neighbor, score, false});
            if (inserted.rejected.has_value()) {
                remember_rejected(*inserted.rejected);
            }
        }
        ++step;
    }

    if (best_rejected.has_value() && !pool.empty()) {
        trace.AddPair(EventSource::kStop, step, *best_rejected,
                      ExactWorst(pool.values(), model_, context, base_));
    }
    if (pool.size() < options.topk) {
        throw std::runtime_error("RoarGraph search returned fewer than topk nodes");
    }
    result.ids.reserve(options.topk);
    result.scores.reserve(options.topk);
    for (uint32_t i = 0; i < options.topk; ++i) {
        result.ids.push_back(pool.values()[i].id);
        result.scores.push_back(pool.values()[i].score);
    }
    result.events = trace.Take();
    result.visited_nodes = static_cast<uint32_t>(visited.size());
    result.final_candidate_pool = static_cast<uint32_t>(pool.size());
    if (options.collect_visited) {
        result.visited_ids.assign(visited.begin(), visited.end());
        std::sort(result.visited_ids.begin(), result.visited_ids.end());
    }
    return result;
}

const char* EventSourceName(EventSource source) {
    switch (source) {
        case EventSource::kExpansion:
            return "exp";
        case EventSource::kQueue:
            return "queue";
        case EventSource::kStop:
            return "stop";
    }
    return "unknown";
}

const char* EventKindName(EventKind kind) {
    return kind == EventKind::kInversion ? "inversion" : "protect";
}

}  // namespace ours
