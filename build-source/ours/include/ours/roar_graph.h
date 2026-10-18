#ifndef OURS_ROAR_GRAPH_H_
#define OURS_ROAR_GRAPH_H_

#include <cstdint>
#include <string>
#include <vector>

namespace ours {

class RoarGraph {
   public:
    static RoarGraph Load(const std::string& path);
    static RoarGraph FromAdjacency(uint32_t entry_point,
                                   std::vector<std::vector<uint32_t>> adjacency);

    uint32_t entry_point() const { return entry_point_; }
    uint32_t size() const { return static_cast<uint32_t>(adjacency_.size()); }
    const std::vector<uint32_t>& Neighbors(uint32_t node_id) const;

   private:
    uint32_t entry_point_ = 0;
    std::vector<std::vector<uint32_t>> adjacency_;
};

}  // namespace ours

#endif  // OURS_ROAR_GRAPH_H_
