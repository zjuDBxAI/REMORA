#include "ours/ji_online.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

void Require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
bool Same(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  auto index = ours::JiOnlineIndex::Load(argv[1], argv[2]);
  std::mt19937 rng(34117);
  std::normal_distribution<float> normal;
  ours::JiQueryWorkspace workspace;
  for (unsigned trial=0; trial<12; ++trial) {
    const uint32_t dim = trial % 2 ? index.dim() : index.dim()-1;
    std::vector<float> query(dim);
    for (auto &v : query) v = trial ? normal(rng) : 0;
    auto reference = index.PrepareQueryZeroPadded(query.data(), dim);
    for (bool bound : {true, false, true, false}) {
      ours::JiPrepareTimings timings;
      const auto &prepared = index.PrepareQueryReused(query.data(), dim, workspace, bound, &timings);
      Require(Same(reference.mean_bias, prepared.mean_bias), "bias mismatch");
      Require(reference.lookup.size() == prepared.lookup.size() &&
              std::memcmp(reference.lookup.data(), prepared.lookup.data(), reference.lookup.size()*sizeof(float)) == 0,
              "LUT mismatch");
      Require(bound ? Same(reference.max_correction_upper, prepared.max_correction_upper)
                    : (std::isinf(prepared.max_correction_upper) && timings.upper_bound_ms == 0), "bound mismatch");
      for (size_t count : {0,1,7,8,9,15,16,17,31,32,33,63,64,65,100}) {
        std::vector<uint32_t> ids(count);
        std::vector<float> corrections(count+2, 12345), scores(count+2, 12345), expected(count);
        for (size_t i=0; i<count; ++i) {
          ids[i] = i % 3 == 0 ? index.node_count()-1 : rng()%index.node_count();
          scores[i+1] = trial ? normal(rng) : -0.0f;
          expected[i] = scores[i+1] + index.Correction(reference, ids[i]);
        }
        index.ComputeCorrectionsBatch(prepared, ids.data(), count, corrections.data()+1);
        index.AddCorrectionsBatch(prepared, ids.data(), count, scores.data()+1);
        for (size_t i=0; i<count; ++i) {
          Require(Same(corrections[i+1], index.Correction(reference, ids[i])), "correction mismatch");
          Require(Same(scores[i+1], expected[i]), "in-place addition changed rounding");
          if (bound) Require(corrections[i+1] <= prepared.max_correction_upper, "unsafe bound");
        }
        Require(corrections.front()==12345 && corrections.back()==12345 && scores.front()==12345 && scores.back()==12345,
                "output overwritten");
      }
    }
  }
  uint32_t bad = index.node_count(); float value = 2;
  bool rejected = false;
  try { index.AddCorrectionsBatch(workspace.context, &bad, 1, &value); }
  catch (const std::out_of_range &) { rejected = true; }
  Require(rejected && value==2, "invalid ID must fail before modifying output");
  std::cout << "PASS: " << index.scoring_kernel() << " b" << index.budget_bits()
            << " v" << index.format_version() << "; ordered add, batch tails, padding, workspace/bound reuse\n";
}
