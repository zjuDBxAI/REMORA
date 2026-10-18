#include "ours/roar_graph.h"

#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ours {
namespace {

uint32_t ReadU32(std::ifstream& input, const std::string& name) {
    uint32_t value = 0;
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) {
        throw std::runtime_error("truncated RoarGraph while reading " + name);
    }
    return value;
}

}  // namespace

RoarGraph RoarGraph::Load(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open RoarGraph index: " + path);
    }
    RoarGraph graph;
    graph.entry_point_ = ReadU32(input, "entry point");
    const uint32_t node_count = ReadU32(input, "node count");
    if (node_count == 0 || graph.entry_point_ >= node_count) {
        throw std::runtime_error("invalid RoarGraph header");
    }
    graph.adjacency_.resize(node_count);
    for (uint32_t node = 0; node < node_count; ++node) {
        const uint32_t degree = ReadU32(input, "node degree");
        if (degree > node_count) {
            throw std::runtime_error("invalid RoarGraph degree");
        }
        auto& neighbors = graph.adjacency_[node];
        neighbors.resize(degree);
        if (degree > 0) {
            input.read(reinterpret_cast<char*>(neighbors.data()),
                       static_cast<std::streamsize>(degree) * sizeof(uint32_t));
            if (!input) {
                throw std::runtime_error("truncated RoarGraph adjacency payload");
            }
        }
        for (uint32_t neighbor : neighbors) {
            if (neighbor >= node_count) {
                throw std::runtime_error("RoarGraph neighbor id is out of range");
            }
        }
    }
    char extra = 0;
    input.read(&extra, 1);
    if (input.gcount() != 0) {
        throw std::runtime_error("RoarGraph index has trailing bytes");
    }
    return graph;
}

RoarGraph RoarGraph::FromAdjacency(
    uint32_t entry_point, std::vector<std::vector<uint32_t>> adjacency) {
    if (adjacency.empty() || entry_point >= adjacency.size()) {
        throw std::invalid_argument("invalid in-memory RoarGraph header");
    }
    for (const auto& neighbors : adjacency) {
        for (uint32_t neighbor : neighbors) {
            if (neighbor >= adjacency.size()) {
                throw std::invalid_argument("in-memory RoarGraph neighbor is out of range");
            }
        }
    }
    RoarGraph graph;
    graph.entry_point_ = entry_point;
    graph.adjacency_ = std::move(adjacency);
    return graph;
}

const std::vector<uint32_t>& RoarGraph::Neighbors(uint32_t node_id) const {
    if (node_id >= adjacency_.size()) {
        throw std::out_of_range("RoarGraph node id is out of range");
    }
    return adjacency_[node_id];
}

}  // namespace ours
