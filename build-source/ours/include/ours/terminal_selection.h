#ifndef OURS_TERMINAL_SELECTION_H_
#define OURS_TERMINAL_SELECTION_H_

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace ours {
// Exactly the same sorted prefix as a complete sort with a total comparator.
// Keeping the prefix sorted also preserves the subsequent exact-fetch order.
template <class T, class Compare>
void SelectSortedPrefix(std::vector<T> &values, size_t keep, Compare better) {
  if (keep == 0 || keep > values.size())
    throw std::invalid_argument("invalid terminal selection size");
  if (keep < values.size()) {
    std::partial_sort(values.begin(), values.begin() + keep, values.end(), better);
    values.resize(keep);
  } else {
    std::sort(values.begin(), values.end(), better);
  }
}
} // namespace ours
#endif
