#pragma once

#include <cstdint>

namespace ours {

// Call once when opening an FP32 scorer/store. Unsupported CPUs fail explicitly;
// there is no scalar fallback under the AVX-512 experiment protocol.
void RequireFp32Avx512();
const char *Fp32InnerProductKernel();

// Unaligned inputs and arbitrary dimensions, including zero, are supported.
// FP32 FMA accumulation uses four independent 16-lane accumulators and a
// masked tail. Its reduction order differs from a sequential scalar sum.
float Fp32InnerProduct(const float *query, const float *base, uint32_t dim);

} // namespace ours
