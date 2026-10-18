#include "ours/generic_primary.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <unistd.h>

namespace {
template <class T> void Write(std::ofstream &f, T value) {
  f.write(reinterpret_cast<const char *>(&value), sizeof(value));
}
void Require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
bool Same(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }
float Madd(float a, float b, float c) {
#if defined(__FMA__)
  return std::fma(a, b, c);
#else
  return a * b + c;
#endif
}
struct Temporary {
  std::filesystem::path path;
  Temporary() {
    char name[] = "/tmp/generic-primary-test-XXXXXX";
    char *p = mkdtemp(name);
    if (!p) throw std::runtime_error("mkdtemp failed");
    path = p;
  }
  ~Temporary() { std::filesystem::remove_all(path); }
};

void Check(uint32_t encoding, uint32_t dim, uint32_t groups, std::mt19937 &rng) {
  constexpr uint32_t nodes = 73, original_dim = 13;
  const bool pq = encoding == 1, u8 = encoding == 2;
  const uint32_t coordinate_bytes = u8 ? dim : (dim + 1) / 2;
  const uint32_t record_bytes = pq ? groups : coordinate_bytes + 8;
  Temporary tmp;
  std::vector<float> transform(dim * original_dim), centroids(256 * dim);
  for (float &x : transform) x = (int(rng() % 1001) - 500) * 0.00023f;
  for (float &x : centroids) x = (int(rng() % 1001) - 500) * 0.013f;
  std::vector<uint32_t> offsets(groups + 1);
  if (pq) for (uint32_t i = 0; i <= groups; ++i) offsets[i] = i * dim / groups;
  std::vector<uint8_t> records(nodes * record_bytes);
  for (auto &b : records) b = rng() & 255;
  if (!pq) for (uint32_t i = 0; i < nodes; ++i) {
    // Include zero scales, negative offsets, and an odd final nibble.
    const float minimum = (int(rng() % 1001) - 500) * 0.017f;
    const float scale = i % 11 == 0 ? 0.0f : (rng() % 999 + 1) * 0.00017f;
    std::memcpy(records.data() + i * record_bytes + coordinate_bytes, &minimum, 4);
    std::memcpy(records.data() + i * record_bytes + coordinate_bytes + 4, &scale, 4);
  }
  const auto model = tmp.path / "model.bin", codes = tmp.path / "codes.bin";
  {
    std::ofstream f(model, std::ios::binary);
    f.write("GPRIM01\0", 8);
    Write(f, uint32_t(1)); Write(f, encoding); Write(f, uint64_t(nodes));
    for (uint32_t x : {original_dim, dim, groups, pq ? 256U : 0U, record_bytes, 0U}) Write(f, x);
    Write(f, uint64_t(123));
    f.write(reinterpret_cast<const char *>(transform.data()), transform.size() * 4);
    if (pq) {
      f.write(reinterpret_cast<const char *>(offsets.data()), offsets.size() * 4);
      f.write(reinterpret_cast<const char *>(centroids.data()), centroids.size() * 4);
    }
  }
  {
    std::ofstream f(codes, std::ios::binary);
    f.write("GPCOD01\0", 8);
    Write(f, uint32_t(1)); Write(f, uint32_t(40)); Write(f, uint64_t(nodes));
    Write(f, record_bytes); Write(f, uint32_t(0)); Write(f, uint64_t(123));
    f.write(reinterpret_cast<const char *>(records.data()), records.size());
  }
  auto index = ours::GenericPrimaryIndex::Load(model.string(), codes.string());
  for (unsigned qi = 0; qi < 6; ++qi) {
    std::vector<float> query(original_dim);
    for (float &x : query) x = qi ? (int(rng() % 10001) - 5000) * 0.031f : 0.0f;
    const auto context = index.Prepare(query.data(), original_dim);
    const auto transformed = index.PrepareTransformed(context.transformed.data(), dim);
    Require(context.lut == transformed.lut && Same(context.query_sum, transformed.query_sum), "PrepareTransformed mismatch");
    float sum = 0;
    for (float x : context.transformed) sum += x;
    if (!pq) Require(Same(sum, context.query_sum), "cached query sum mismatch");
    std::vector<float> reference(nodes);
    for (uint32_t id = 0; id < nodes; ++id) {
      const auto *record = records.data() + id * record_bytes;
      float value = 0;
      if (pq) {
        for (uint32_t g = 0; g < groups; ++g) value += context.lut[g * 256 + record[g]];
      } else {
        float minimum, scale;
        std::memcpy(&minimum, record + coordinate_bytes, 4);
        std::memcpy(&scale, record + coordinate_bytes + 4, 4);
        for (uint32_t j = 0; j < dim; ++j) {
          const unsigned code = u8 ? record[j] : (record[j / 2] >> (4 * (j & 1))) & 15;
          value = Madd(context.transformed[j], static_cast<float>(code), value);
        }
        value = Madd(minimum, sum, scale * value);
      }
      reference[id] = value;
      Require(Same(value, index.Score(context, id)), "scalar/reference mismatch");
    }
    for (size_t count : {0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 73, 100}) {
      std::vector<uint32_t> ids(count);
      for (size_t i = 0; i < count; ++i) ids[i] = i % 3 == 0 ? nodes - 1 : rng() % nodes;
      std::vector<float> output(count + 1, -1234567.0f);
      index.ScoreBatch(context, ids.data(), count, output.data());
      Require(output.back() == -1234567.0f, "batch tail overwrote guard");
      for (size_t i = 0; i < count; ++i) Require(Same(output[i], reference[ids[i]]), "SIMD/tail/reference mismatch");
    }
    bool rejected = false;
    const uint32_t bad[] = {0, nodes};
    float scores[2] = {};
    try { index.ScoreBatch(context, bad, 2, scores); }
    catch (const std::out_of_range &) { rejected = true; }
    Require(rejected, "invalid ID accepted");
    auto malformed = context;
    malformed.transformed.pop_back();
    rejected = false;
    try { index.ScoreBatch(malformed, bad, 1, scores); }
    catch (const std::invalid_argument &) { rejected = true; }
    Require(rejected, "invalid context accepted");
  }
}
}  // namespace

int main() {
  std::mt19937 rng(20260927);
  for (uint32_t dim : {1, 3, 7, 8, 9, 17, 191, 192, 193}) {
    Check(2, dim, 0, rng);
    Check(3, dim, 0, rng);
  }
  for (uint32_t groups : {1, 3, 4, 7, 8, 17, 96}) Check(1, groups * 2 + 1, groups, rng);
  std::cout << "PASS: generic PQ/affine4/affine8, 25 layouts, cached query sum, scalar/batch bitwise parity, "
               "partial batches, odd dimensions/groups, last record, zero queries/scales, invalid inputs\n";
}
