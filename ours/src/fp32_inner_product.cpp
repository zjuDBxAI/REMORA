#include "ours/fp32_inner_product.h"

#include <stdexcept>

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define OURS_FP32_AVX512_COMPILED 1
#define OURS_FP32_AVX512_TARGET __attribute__((target("avx512f,fma")))
#else
#define OURS_FP32_AVX512_COMPILED 0
#define OURS_FP32_AVX512_TARGET
#endif

namespace ours {

void RequireFp32Avx512() {
#if OURS_FP32_AVX512_COMPILED
  if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma"))
    return;
#endif
  throw std::runtime_error("FP32 search/rerank requires AVX-512F and FMA; "
                           "no scalar fallback is enabled");
}

const char *Fp32InnerProductKernel() {
#if OURS_FP32_AVX512_COMPILED
  return "avx512-fma-fp32x16-4acc-masked-tail";
#else
  return "unavailable-avx512";
#endif
}

OURS_FP32_AVX512_TARGET
float Fp32InnerProduct(const float *query, const float *base, uint32_t dim) {
#if OURS_FP32_AVX512_COMPILED
  __m512 a = _mm512_setzero_ps(), b = _mm512_setzero_ps();
  __m512 c = _mm512_setzero_ps(), d = _mm512_setzero_ps();
  uint32_t i = 0;
  for (; dim - i >= 64; i += 64) {
    a = _mm512_fmadd_ps(_mm512_loadu_ps(query + i),
                        _mm512_loadu_ps(base + i), a);
    b = _mm512_fmadd_ps(_mm512_loadu_ps(query + i + 16),
                        _mm512_loadu_ps(base + i + 16), b);
    c = _mm512_fmadd_ps(_mm512_loadu_ps(query + i + 32),
                        _mm512_loadu_ps(base + i + 32), c);
    d = _mm512_fmadd_ps(_mm512_loadu_ps(query + i + 48),
                        _mm512_loadu_ps(base + i + 48), d);
  }
  __m512 sum = _mm512_add_ps(_mm512_add_ps(a, b), _mm512_add_ps(c, d));
  for (; dim - i >= 16; i += 16)
    sum = _mm512_fmadd_ps(_mm512_loadu_ps(query + i),
                          _mm512_loadu_ps(base + i), sum);
  if (i < dim) {
    const __mmask16 mask = static_cast<__mmask16>((1U << (dim - i)) - 1U);
    sum = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, query + i),
                          _mm512_maskz_loadu_ps(mask, base + i), sum);
  }
  return _mm512_reduce_add_ps(sum);
#else
  (void)query; (void)base; (void)dim;
  RequireFp32Avx512();
  return 0.0f;
#endif
}

} // namespace ours
