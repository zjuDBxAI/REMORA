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

RoarGraph RoarGraph::LoadPacked(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open RoarGraph index: " + path);
    RoarGraph graph;
    graph.entry_point_ = ReadU32(input, "entry point");
    const uint32_t count = ReadU32(input, "node count");
    if (!count || graph.entry_point_ >= count)
        throw std::runtime_error("invalid RoarGraph header");
    input.seekg(0, std::ios::end);
    const auto file_bytes = input.tellg();
    input.seekg(8);
    graph.offsets_.resize(static_cast<size_t>(count) + 1, 0);
    uint64_t consumed = 8;
    for (uint32_t i = 0; i < count; ++i) {
        const auto degree = ReadU32(input, "node degree");
        if (degree > count) throw std::runtime_error("invalid RoarGraph degree");
        graph.offsets_[i + 1] = graph.offsets_[i] + degree;
        consumed += 4 + static_cast<uint64_t>(degree) * 4;
        if (consumed > static_cast<uint64_t>(file_bytes))
            throw std::runtime_error("truncated RoarGraph adjacency payload");
        // Buffered sequential scan: one seek/tell per node is prohibitive
        // for 1B nodes and destroys filesystem readahead.
        input.ignore(static_cast<std::streamsize>(degree) * 4);
        if (!input) throw std::runtime_error("truncated RoarGraph adjacency payload");
    }
    if (consumed != static_cast<uint64_t>(file_bytes))
        throw std::runtime_error("RoarGraph index has trailing bytes");
    graph.edges_.resize(graph.offsets_.back());
    input.seekg(8);
    for (uint32_t i = 0; i < count; ++i) {
        const auto degree = ReadU32(input, "node degree");
        if (degree != graph.offsets_[i + 1] - graph.offsets_[i])
            throw std::runtime_error("RoarGraph changed while loading");
        if (degree) {
            auto* output = graph.edges_.data() + graph.offsets_[i];
            input.read(reinterpret_cast<char*>(output), static_cast<std::streamsize>(degree) * 4);
            if (!input) throw std::runtime_error("truncated RoarGraph adjacency payload");
            for (uint32_t j = 0; j < degree; ++j)
                if (output[j] >= count) throw std::runtime_error("RoarGraph neighbor id is out of range");
        }
    }
    return graph;
}

RoarGraph::NeighborRange RoarGraph::NeighborIds(uint32_t node_id) const {
    if (node_id >= size()) throw std::out_of_range("RoarGraph node id is out of range");
    if (offsets_.empty()) {
        const auto& row = adjacency_[node_id];
        return {row.data(), row.empty() ? row.data() : row.data() + row.size()};
    }
    const auto* data = edges_.data();
    if (edges_.empty()) return {data, data};
    return {data + offsets_[node_id], data + offsets_[node_id + 1]};
}

uint64_t RoarGraph::memory_bytes() const {
    uint64_t bytes = offsets_.capacity() * sizeof(uint64_t) + edges_.capacity() * sizeof(uint32_t);
    bytes += adjacency_.capacity() * sizeof(std::vector<uint32_t>);
    for (const auto& row : adjacency_) bytes += row.capacity() * sizeof(uint32_t);
    return bytes;
}

const std::vector<uint32_t>& RoarGraph::Neighbors(uint32_t node_id) const {
    if (node_id >= adjacency_.size()) {
        throw std::out_of_range("RoarGraph node id is out of range");
    }
    return adjacency_[node_id];
}

}  // namespace ours
