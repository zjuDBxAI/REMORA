#ifndef OURS_ROAR_GRAPH_H_
#define OURS_ROAR_GRAPH_H_

#include <cstdint>
#include <string>
#include <vector>

namespace ours {

class RoarGraph {
   public:
    static RoarGraph Load(const std::string& path);
    // CSR keeps the same edges and ordering without a vector allocation per node.
    static RoarGraph LoadPacked(const std::string& path);
    static RoarGraph FromAdjacency(uint32_t entry_point,
                                   std::vector<std::vector<uint32_t>> adjacency);

    uint32_t entry_point() const { return entry_point_; }
    uint32_t size() const {
        return offsets_.empty() ? static_cast<uint32_t>(adjacency_.size())
                                : static_cast<uint32_t>(offsets_.size() - 1);
    }
    const std::vector<uint32_t>& Neighbors(uint32_t node_id) const;
    struct NeighborRange {
        const uint32_t* first;
        const uint32_t* last;
        const uint32_t* begin() const { return first; }
        const uint32_t* end() const { return last; }
    };
    NeighborRange NeighborIds(uint32_t node_id) const;
    uint64_t memory_bytes() const;

   private:
    uint32_t entry_point_ = 0;
    std::vector<std::vector<uint32_t>> adjacency_;
    std::vector<uint64_t> offsets_;
    std::vector<uint32_t> edges_;
};

}  // namespace ours

#endif  // OURS_ROAR_GRAPH_H_
