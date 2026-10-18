#include "ours/terminal_selection.h"
#include <iostream>
#include <random>
#include <utility>

int main() {
  using Value = std::pair<float, unsigned>;
  const auto better = [](const Value &a, const Value &b) {
    return a.first > b.first || (a.first == b.first && a.second < b.second);
  };
  std::mt19937 random(20260927);
  for (unsigned n : {1, 10, 17, 50, 100, 200, 1000}) {
    std::vector<Value> initial;
    for (unsigned i = 0; i < n; ++i)
      initial.emplace_back(static_cast<int>(random() % 11) - 5.f, i);
    std::shuffle(initial.begin(), initial.end(), random);
    auto reference = initial;
    std::sort(reference.begin(), reference.end(), better);
    for (unsigned keep = 1; keep <= n; ++keep) {
      auto actual = initial;
      const auto capacity = actual.capacity();
      ours::SelectSortedPrefix(actual, keep, better);
      if (actual.size() != keep || actual.capacity() != capacity ||
          !std::equal(actual.begin(), actual.end(), reference.begin()))
        throw std::runtime_error("sorted prefix/tie/capacity mismatch");
    }
    for (unsigned bad : {0u, n + 1}) {
      bool rejected = false;
      try { ours::SelectSortedPrefix(initial, bad, better); }
      catch (const std::invalid_argument &) { rejected = true; }
      if (!rejected) throw std::runtime_error("invalid prefix accepted");
    }
  }
  std::cout << "terminal selection tests passed (ties, every prefix, no reallocation)\n";
}
