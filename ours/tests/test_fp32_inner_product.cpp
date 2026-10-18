#include "ours/fp32_inner_product.h"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// Both inputs end exactly at a protected page: a full-vector tail load would
// fault. These starts also exercise non-64-byte-aligned .fbin-style inputs.
class GuardedVector {
public:
  explicit GuardedVector(uint32_t dim) {
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t bytes = static_cast<size_t>(dim) * sizeof(float);
    const size_t data_bytes = std::max(page, ((bytes + page - 1) / page) * page);
    size_ = data_bytes + page;
    mapping_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping_ == MAP_FAILED) throw std::runtime_error("guard mmap failed");
    char *end = static_cast<char *>(mapping_) + data_bytes;
    if (mprotect(end, page, PROT_NONE) != 0) {
      munmap(mapping_, size_);
      throw std::runtime_error("guard mprotect failed");
    }
    data_ = reinterpret_cast<float *>(end - bytes);
  }
  ~GuardedVector() { munmap(mapping_, size_); }
  float *data() { return data_; }
private:
  void *mapping_ = nullptr;
  size_t size_ = 0;
  float *data_ = nullptr;
};

void CheckScore(const float *q, const float *x, uint32_t dim) {
  double expected = 0.0, absolute_products = 0.0;
  for (uint32_t i = 0; i < dim; ++i) {
    const double product = static_cast<double>(q[i]) * x[i];
    expected += product;
    absolute_products += std::fabs(product);
  }
  const float actual = ours::Fp32InnerProduct(q, x, dim);
  const double tolerance = 8.0 * std::numeric_limits<float>::epsilon() *
                           absolute_products + 1e-7;
  if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance)
    throw std::runtime_error("AVX-512/reference mismatch at dim=" + std::to_string(dim));
}
} // namespace

int main() {
  try {
    ours::RequireFp32Avx512();
    std::cout << "check,status,detail\n"
              << "kernel,PASS," << ours::Fp32InnerProductKernel() << '\n';
    if (ours::Fp32InnerProduct(nullptr, nullptr, 0) != 0.0f)
      throw std::runtime_error("empty dot product is not zero");
    std::mt19937 rng(127);
    std::uniform_real_distribution<float> random(-2.0f, 2.0f);
    for (uint32_t dim : {1U, 7U, 8U, 15U, 16U, 17U, 31U, 32U, 33U,
                        63U, 64U, 65U, 127U, 128U, 129U, 191U, 192U,
                        193U, 200U, 255U, 256U, 257U, 1023U, 1024U, 1025U}) {
      GuardedVector q(dim), x(dim);
      for (unsigned repeat = 0; repeat < 12; ++repeat) {
        for (uint32_t i = 0; i < dim; ++i) {
          q.data()[i] = random(rng);
          x.data()[i] = random(rng);
        }
        CheckScore(q.data(), x.data(), dim);
      }
      std::fill(q.data(), q.data() + dim, 1.0f);
      for (uint32_t i = 0; i < dim; ++i)
        x.data()[i] = (i % 2 ? -1.0f : 1.0f) * (i % 3 ? 1.0f : 1e6f);
      CheckScore(q.data(), x.data(), dim);
      std::cout << "dim-" << dim << ",PASS,random scores and cancellation; protected-page tails\n";
    }
    // Well-separated scores must retain their exact ranking after the new
    // reduction order. Duplicate vectors must yield exactly equal scores.
    std::vector<float> q(200), x(200);
    for (float &v : q) v = random(rng);
    float previous = -std::numeric_limits<float>::infinity();
    for (int scale = -64; scale <= 64; ++scale) {
      for (size_t i = 0; i < q.size(); ++i) x[i] = q[i] * scale;
      const float score = ours::Fp32InnerProduct(q.data(), x.data(), q.size());
      if (!(score > previous) || score != ours::Fp32InnerProduct(q.data(), x.data(), q.size()))
        throw std::runtime_error("ranking/repeatability mismatch");
      previous = score;
    }
    std::cout << "ranking-200d,PASS,129 separated scores and deterministic repeats\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
