#include "ours/roar_graph.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <unistd.h>

void Check(bool ok) { if (!ok) throw std::runtime_error("graph storage check failed"); }
int main() {
  auto path = std::filesystem::temp_directory_path() /
      ("remora-graph-test-" + std::to_string(getpid()));
  const std::vector<uint32_t> encoded{0, 4, 2, 1, 2, 1, 3, 0, 2, 0, 1};
  { std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(encoded.data()), encoded.size() * 4); }
  auto legacy = ours::RoarGraph::Load(path.string());
  auto packed = ours::RoarGraph::LoadPacked(path.string());
  Check(legacy.size() == packed.size() && legacy.entry_point() == packed.entry_point());
  for (uint32_t i = 0; i < legacy.size(); ++i) {
    std::vector<uint32_t> ids;
    for (auto id : packed.NeighborIds(i)) ids.push_back(id);
    Check(ids == legacy.Neighbors(i));
  }
  Check(packed.memory_bytes() == 5 * 8 + 5 * 4);
  Check(packed.memory_bytes() < legacy.memory_bytes());
  { std::ofstream out(path, std::ios::binary | std::ios::app); out.put('x'); }
  bool rejected = false;
  try { auto invalid = ours::RoarGraph::LoadPacked(path.string()); }
  catch (const std::runtime_error&) { rejected = true; }
  Check(rejected);
  std::filesystem::remove(path);
}
