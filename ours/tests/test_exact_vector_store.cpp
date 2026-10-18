#include "ours/exact_vector_store.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::invalid_argument(
          "usage: roar_exact_store_tests BASE.fbin BASE.cold.bin");
    const auto warm = ours::OpenExactVectorStore(argv[1], "mmap");
    const auto cold = ours::OpenExactVectorStore(argv[2], "direct");
    if (warm->rows() != cold->rows() || warm->dim() != cold->dim())
      throw std::runtime_error("warm/cold shape mismatch");
    std::vector<float> query(warm->dim());
    for (uint32_t coordinate = 0; coordinate < warm->dim(); ++coordinate)
      query[coordinate] = 0.25f * static_cast<float>(coordinate + 1);
    const uint32_t last = static_cast<uint32_t>(warm->rows() - 1);
    const std::vector<uint32_t> ids{last, 0, 7, 1, 7, last / 2};
    std::vector<float> warm_scores(ids.size());
    std::vector<float> cold_scores(ids.size());
    ours::ExactScoreStats warm_stats;
    ours::ExactScoreStats cold_stats;
    warm->ScoreBatch(query.data(), ids.data(), ids.size(), warm_scores.data(),
                     &warm_stats);
    cold->ScoreBatch(query.data(), ids.data(), ids.size(), cold_scores.data(),
                     &cold_stats);
    for (size_t index = 0; index < ids.size(); ++index) {
      if (std::fabs(warm_scores[index] - cold_scores[index]) > 1e-6f)
        throw std::runtime_error("warm/direct exact score mismatch");
    }
    if (cold_stats.read_ops == 0 || cold_stats.physical_bytes == 0 ||
        cold_stats.logical_bytes !=
            ids.size() * static_cast<uint64_t>(warm->dim()) * sizeof(float))
      throw std::runtime_error("invalid direct-I/O accounting");
    std::cout << "exact-vector stores agree; direct read_ops="
              << cold_stats.read_ops
              << " physical_bytes=" << cold_stats.physical_bytes << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
