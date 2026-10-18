#include "ours/generic_primary.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#if defined(__AVX512F__) && defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace ours {
namespace {

constexpr std::array<char, 8> kModelMagic = {'G','P','R','I','M','0','1','\0'};
constexpr std::array<char, 8> kCodesMagic = {'G','P','C','O','D','0','1','\0'};
constexpr uint32_t kVersion = 1;

#pragma pack(push, 1)
struct ModelHeader {
  char magic[8];
  uint32_t version;
  uint32_t encoding;
  uint64_t nodes;
  uint32_t original_dim;
  uint32_t scorer_dim;
  uint32_t groups;
  uint32_t codewords;
  uint32_t record_bytes;
  uint32_t reserved;
  uint64_t artifact_id;
};
struct CodesHeader {
  char magic[8];
  uint32_t version;
  uint32_t header_bytes;
  uint64_t nodes;
  uint32_t record_bytes;
  uint32_t reserved;
  uint64_t artifact_id;
};
#pragma pack(pop)
static_assert(sizeof(ModelHeader) == 56);
static_assert(sizeof(CodesHeader) == 40);

float Dot(const float* a, const float* b, uint32_t dim) {
  float value = 0.0f;
  for (uint32_t i = 0; i < dim; ++i) value += a[i] * b[i];
  return value;
}

// Keep the old left-to-right accumulation, including native-build FMA
// contraction. SIMD lanes below span nodes, NOT coordinates, so neither
// coordinate nor PQ group reductions are reassociated.
inline float AffineFinish(float minimum, float sum, float scale, float dot) {
  const float scaled = scale * dot;
#if defined(__FMA__)
  return std::fma(minimum, sum, scaled);
#else
  return minimum * sum + scaled;
#endif
}

template <unsigned Bits>
inline float ScoreAffine(const uint8_t* record, const float* query,
                         uint32_t dim, float query_sum) {
  constexpr uint32_t mask = (1U << Bits) - 1U;
  const uint32_t coordinate_bytes = Bits == 8 ? dim : (dim + 1U) / 2U;
  float minimum, scale;
  std::memcpy(&minimum, record + coordinate_bytes, sizeof(float));
  std::memcpy(&scale, record + coordinate_bytes + sizeof(float), sizeof(float));
  float dot = 0.0f;
  for (uint32_t j = 0; j < dim; ++j) {
    const uint32_t code = Bits == 8 ? record[j] : (record[j / 2U] >> (4U * (j & 1U))) & mask;
#if defined(__FMA__)
    dot = std::fma(query[j], static_cast<float>(code), dot);
#else
    dot += query[j] * code;
#endif
  }
  return AffineFinish(minimum, query_sum, scale, dot);
}

inline float ScorePQ(const uint8_t* record, const float* lut, uint32_t groups) {
  float value = 0.0f;
  for (uint32_t g = 0; g < groups; ++g)
    value += lut[static_cast<size_t>(g) * 256 + record[g]];
  return value;
}

#if defined(__AVX512F__) && defined(__AVX2__) && defined(__FMA__)
// Eight 64-bit byte offsets avoid the signed-32-bit addressing limit for
// large code arrays. Only 4 bytes per node are read by each packed gather.
template <unsigned Bits>
inline void ScoreAffine8Nodes(const uint8_t* records, __m512i offsets,
                              const float* query, uint32_t dim,
                              float query_sum, float* scores) {
  constexpr unsigned per_word = 32 / Bits;
  const auto mask = _mm256_set1_epi32((1U << Bits) - 1U);
  auto dot = _mm256_setzero_ps();
  for (uint32_t j = 0; j < dim; j += per_word) {
    const auto at = _mm512_add_epi64(offsets, _mm512_set1_epi64(j * uint64_t(Bits) / 8));
    auto packed = _mm512_i64gather_epi32(at, records, 1);
    // A partial last word can overlap the record's 8-byte metadata, never
    // the end of the record. Unused nibbles/bytes are not scored.
    const uint32_t limit = std::min<uint32_t>(per_word, dim - j);
    for (uint32_t k = 0; k < limit; ++k) {
      const auto code = _mm256_cvtepi32_ps(_mm256_and_si256(packed, mask));
      dot = _mm256_fmadd_ps(_mm256_set1_ps(query[j + k]), code, dot);
      packed = _mm256_srli_epi32(packed, Bits);
    }
  }
  const uint32_t coordinate_bytes = Bits == 8 ? dim : (dim + 1U) / 2U;
  const auto at = _mm512_add_epi64(offsets, _mm512_set1_epi64(coordinate_bytes));
  const auto minimum = _mm512_i64gather_ps(at, records, 1);
  const auto scale = _mm512_i64gather_ps(_mm512_add_epi64(at, _mm512_set1_epi64(4)), records, 1);
  const auto scaled = _mm256_mul_ps(scale, dot);
  _mm256_storeu_ps(scores, _mm256_fmadd_ps(minimum, _mm256_set1_ps(query_sum), scaled));
}

inline void ScorePQ8Nodes(const uint8_t* records, const uint64_t* row_offsets,
                         __m512i offsets, const float* lut, uint32_t groups,
                         float* scores) {
  auto values = _mm256_setzero_ps();
  const auto mask = _mm256_set1_epi32(255);
  uint32_t g = 0;
  for (; g + 4 <= groups; g += 4) {
    auto packed = _mm512_i64gather_epi32(_mm512_add_epi64(offsets, _mm512_set1_epi64(g)), records, 1);
    for (uint32_t k = 0; k < 4; ++k) {
      const auto code = _mm256_and_si256(packed, mask);
      const auto entry = _mm256_i32gather_ps(lut + static_cast<size_t>(g + k) * 256, code, 4);
      values = _mm256_add_ps(values, entry);
      packed = _mm256_srli_epi32(packed, 8);
    }
  }
  // PQ has no trailing metadata. Load individual bytes for odd group counts;
  // no word gather may overread the last node's record/mapped file.
  for (; g < groups; ++g) {
    alignas(32) int32_t codes[8];
    for (unsigned lane = 0; lane < 8; ++lane) codes[lane] = records[row_offsets[lane] + g];
    const auto code = _mm256_load_si256(reinterpret_cast<const __m256i*>(codes));
    values = _mm256_add_ps(values, _mm256_i32gather_ps(lut + static_cast<size_t>(g) * 256, code, 4));
  }
  _mm256_storeu_ps(scores, values);
}
#endif

class Mapping {
 public:
  explicit Mapping(const std::string& path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open generic codes: " + path);
    struct stat status {};
    if (fstat(fd_, &status) != 0 || status.st_size <= 0)
      throw std::runtime_error("cannot stat generic codes: " + path);
    size_ = static_cast<size_t>(status.st_size);
    data_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (data_ == MAP_FAILED) {
      data_ = nullptr;
      throw std::runtime_error("cannot mmap generic codes: " + path);
    }
  }
  ~Mapping() {
    if (data_) munmap(data_, size_);
    if (fd_ >= 0) close(fd_);
  }
  const void* data() const { return data_; }
  size_t size() const { return size_; }
 private:
  int fd_ = -1;
  void* data_ = nullptr;
  size_t size_ = 0;
};

template <typename T>
void ReadVector(std::ifstream& input, std::vector<T>* output, size_t count,
                const char* name) {
  output->resize(count);
  input.read(reinterpret_cast<char*>(output->data()),
             static_cast<std::streamsize>(count * sizeof(T)));
  if (!input) throw std::runtime_error(std::string("truncated generic model: ") + name);
}

}  // namespace

class GenericPrimaryIndex::Impl {
 public:
  Impl(const std::string& model_path, const std::string& codes_path)
      : codes(codes_path) {
    std::ifstream input(model_path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open generic model: " + model_path);
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input || !std::equal(kModelMagic.begin(), kModelMagic.end(), header.magic) ||
        header.version != kVersion || header.nodes == 0 ||
        header.original_dim == 0 || header.scorer_dim == 0 ||
        header.record_bytes == 0 || header.artifact_id == 0)
      throw std::runtime_error("invalid generic primary model header");
    if (header.encoding != static_cast<uint32_t>(Encoding::kCodebook8) &&
        header.encoding != static_cast<uint32_t>(Encoding::kAffine8) &&
        header.encoding != static_cast<uint32_t>(Encoding::kAffine4))
      throw std::runtime_error("unsupported generic primary encoding");
    ReadVector(input, &query_transform,
               static_cast<size_t>(header.scorer_dim) * header.original_dim,
               "query transform");
    if (encoding() == Encoding::kCodebook8) {
      if (header.groups == 0 || header.codewords != 256 ||
          header.record_bytes != header.groups)
        throw std::runtime_error("invalid codebook primary layout");
      ReadVector(input, &group_offsets, static_cast<size_t>(header.groups) + 1,
                 "group offsets");
      if (group_offsets.front() != 0 || group_offsets.back() != header.scorer_dim)
        throw std::runtime_error("generic primary groups do not cover scorer dimensions");
      for (uint32_t g = 0; g < header.groups; ++g)
        if (group_offsets[g] >= group_offsets[g + 1])
          throw std::runtime_error("generic primary has an empty group");
      ReadVector(input, &centroids,
                 static_cast<size_t>(header.codewords) * header.scorer_dim,
                 "centroids");
    } else {
      const uint32_t coordinate_bytes =
          encoding() == Encoding::kAffine8
              ? header.scorer_dim
              : (header.scorer_dim + 1U) / 2U;
      if (header.groups != 0 || header.codewords != 0 ||
          header.record_bytes != coordinate_bytes + 8U)
        throw std::runtime_error("invalid affine primary layout");
    }
    if (input.peek() != std::ifstream::traits_type::eof())
      throw std::runtime_error("generic primary model has trailing bytes");

    if (codes.size() < sizeof(CodesHeader))
      throw std::runtime_error("truncated generic primary codes");
    std::memcpy(&codes_header, codes.data(), sizeof(codes_header));
    if (!std::equal(kCodesMagic.begin(), kCodesMagic.end(), codes_header.magic) ||
        codes_header.version != kVersion ||
        codes_header.header_bytes != sizeof(CodesHeader) ||
        codes_header.nodes != header.nodes ||
        codes_header.record_bytes != header.record_bytes ||
        codes_header.artifact_id != header.artifact_id ||
        codes.size() != sizeof(CodesHeader) +
                            header.nodes * static_cast<uint64_t>(header.record_bytes))
      throw std::runtime_error("generic primary code/model mismatch");
    records = static_cast<const uint8_t*>(codes.data()) + sizeof(CodesHeader);
  }

  Encoding encoding() const { return static_cast<Encoding>(header.encoding); }
  const uint8_t* Record(uint32_t id) const {
    if (id >= header.nodes) throw std::out_of_range("generic primary node id");
    return records + static_cast<uint64_t>(id) * header.record_bytes;
  }

  ModelHeader header{};
  CodesHeader codes_header{};
  std::vector<float> query_transform;
  std::vector<uint32_t> group_offsets;
  // Layout is group-major, then codeword, then local coordinate.  Since the
  // group dimensions sum to scorer_dim, each group plane contains K*d_g
  // floats and all planes together contain K*scorer_dim floats.
  std::vector<float> centroids;
  Mapping codes;
  const uint8_t* records = nullptr;
};

GenericPrimaryIndex::GenericPrimaryIndex(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
GenericPrimaryIndex GenericPrimaryIndex::Load(const std::string& model_path,
                                              const std::string& codes_path) {
  return GenericPrimaryIndex(std::make_unique<Impl>(model_path, codes_path));
}
GenericPrimaryIndex::GenericPrimaryIndex(GenericPrimaryIndex&&) noexcept = default;
GenericPrimaryIndex& GenericPrimaryIndex::operator=(GenericPrimaryIndex&&) noexcept = default;
GenericPrimaryIndex::~GenericPrimaryIndex() = default;

uint64_t GenericPrimaryIndex::nodes() const { return impl_->header.nodes; }
uint32_t GenericPrimaryIndex::original_dim() const { return impl_->header.original_dim; }
uint32_t GenericPrimaryIndex::scorer_dim() const { return impl_->header.scorer_dim; }
uint32_t GenericPrimaryIndex::groups() const { return impl_->header.groups; }
uint64_t GenericPrimaryIndex::storage_bytes() const { return impl_->codes.size(); }
GenericPrimaryIndex::Encoding GenericPrimaryIndex::encoding() const {
  return impl_->encoding();
}

const char* GenericPrimaryIndex::scoring_kernel() const {
#if defined(__AVX512F__) && defined(__AVX2__) && defined(__FMA__)
  return "avx512-batch8-ordered-pq-affine";
#else
  return "portable-ordered-pq-affine";
#endif
}

GenericPrimaryIndex::QueryContext GenericPrimaryIndex::Prepare(
    const float* query, uint32_t dim) const {
  if (!query || dim != original_dim())
    throw std::invalid_argument("generic primary query dimension mismatch");
  QueryContext result;
  const uint32_t scorer_dimensions = impl_->header.scorer_dim;
  const uint32_t group_count = impl_->header.groups;
  result.transformed.resize(scorer_dimensions);
  for (uint32_t row = 0; row < scorer_dimensions; ++row)
    result.transformed[row] =
        Dot(impl_->query_transform.data() + static_cast<uint64_t>(row) * dim,
            query, dim);
  if (impl_->encoding() == Encoding::kCodebook8) {
    result.lut.resize(static_cast<size_t>(group_count) * 256U);
    uint64_t centroid_offset = 0;
    for (uint32_t group = 0; group < group_count; ++group) {
      const uint32_t begin = impl_->group_offsets[group];
      const uint32_t width = impl_->group_offsets[group + 1] - begin;
      for (uint32_t code = 0; code < 256; ++code) {
        result.lut[static_cast<size_t>(group) * 256 + code] = Dot(
            result.transformed.data() + begin,
            impl_->centroids.data() + centroid_offset +
                static_cast<uint64_t>(code) * width,
            width);
      }
      centroid_offset += static_cast<uint64_t>(256) * width;
    }
  } else {
    for (float value : result.transformed) result.query_sum += value;
  }
  return result;
}

GenericPrimaryIndex::QueryContext GenericPrimaryIndex::PrepareTransformed(
    const float* query, uint32_t dim) const {
  if (!query || dim != scorer_dim())
    throw std::invalid_argument("generic primary transformed-query mismatch");
  // Build the same LUT without applying the original-space projection.
  QueryContext result;
  result.transformed.assign(query, query + dim);
  const uint32_t group_count = impl_->header.groups;
  if (impl_->encoding() == Encoding::kCodebook8) {
    result.lut.resize(static_cast<size_t>(group_count) * 256U);
    uint64_t centroid_offset = 0;
    for (uint32_t group = 0; group < group_count; ++group) {
      const uint32_t begin = impl_->group_offsets[group];
      const uint32_t width = impl_->group_offsets[group + 1] - begin;
      for (uint32_t code = 0; code < 256; ++code)
        result.lut[static_cast<size_t>(group) * 256 + code] = Dot(
            query + begin,
            impl_->centroids.data() + centroid_offset +
                static_cast<uint64_t>(code) * width,
            width);
      centroid_offset += static_cast<uint64_t>(256) * width;
    }
  } else {
    for (float value : result.transformed) result.query_sum += value;
  }
  return result;
}

float GenericPrimaryIndex::Score(const QueryContext& context, uint32_t id) const {
  const auto& header = impl_->header;
  if (context.transformed.size() != header.scorer_dim)
    throw std::invalid_argument("generic primary context mismatch");
  const uint8_t* record = impl_->Record(id);
  if (impl_->encoding() == Encoding::kCodebook8) {
    if (context.lut.size() != static_cast<size_t>(header.groups) * 256)
      throw std::invalid_argument("generic primary LUT mismatch");
    return ScorePQ(record, context.lut.data(), header.groups);
  }
  if (impl_->encoding() == Encoding::kAffine8)
    return ScoreAffine<8>(record, context.transformed.data(), header.scorer_dim, context.query_sum);
  return ScoreAffine<4>(record, context.transformed.data(), header.scorer_dim, context.query_sum);
}

void GenericPrimaryIndex::ScoreBatch(const QueryContext& context,
                                     const uint32_t* ids, size_t count,
                                     float* scores) const {
  if (!count) return;
  const auto& h = impl_->header;
  const auto type = impl_->encoding();
  if (!ids || !scores || context.transformed.size() != h.scorer_dim ||
      (type == Encoding::kCodebook8 && context.lut.size() != static_cast<size_t>(h.groups) * 256))
    throw std::invalid_argument("generic primary batch/context mismatch");
  for (size_t i = 0; i < count; ++i)
    if (ids[i] >= h.nodes) throw std::out_of_range("generic primary node id");
  const uint8_t* records = impl_->records;
  size_t i = 0;
#if defined(__AVX512F__) && defined(__AVX2__) && defined(__FMA__)
  for (; i + 8 <= count; i += 8) {
    alignas(64) uint64_t offsets[8];
    for (unsigned lane = 0; lane < 8; ++lane) {
      offsets[lane] = uint64_t(ids[i + lane]) * h.record_bytes;
      if (i + 8 + lane < count)
        __builtin_prefetch(records + uint64_t(ids[i + 8 + lane]) * h.record_bytes, 0, 1);
    }
    const auto at = _mm512_load_si512(reinterpret_cast<const __m512i*>(offsets));
    if (type == Encoding::kCodebook8)
      ScorePQ8Nodes(records, offsets, at, context.lut.data(), h.groups, scores + i);
    else if (type == Encoding::kAffine8)
      ScoreAffine8Nodes<8>(records, at, context.transformed.data(), h.scorer_dim, context.query_sum, scores + i);
    else
      ScoreAffine8Nodes<4>(records, at, context.transformed.data(), h.scorer_dim, context.query_sum, scores + i);
  }
#endif
  for (; i < count; ++i) {
    const uint8_t* record = records + uint64_t(ids[i]) * h.record_bytes;
    if (type == Encoding::kCodebook8)
      scores[i] = ScorePQ(record, context.lut.data(), h.groups);
    else if (type == Encoding::kAffine8)
      scores[i] = ScoreAffine<8>(record, context.transformed.data(), h.scorer_dim, context.query_sum);
    else
      scores[i] = ScoreAffine<4>(record, context.transformed.data(), h.scorer_dim, context.query_sum);
  }
}

void GenericPrimaryIndex::Reconstruct(uint32_t id, float* output,
                                      uint32_t dim) const {
  if (!output || dim != scorer_dim())
    throw std::invalid_argument("generic primary reconstruction mismatch");
  const uint8_t* record = impl_->Record(id);
  if (encoding() == Encoding::kCodebook8) {
    uint64_t centroid_offset = 0;
    for (uint32_t group = 0; group < groups(); ++group) {
      const uint32_t begin = impl_->group_offsets[group];
      const uint32_t width = impl_->group_offsets[group + 1] - begin;
      const float* centroid = impl_->centroids.data() + centroid_offset +
                              static_cast<uint64_t>(record[group]) * width;
      std::copy(centroid, centroid + width, output + begin);
      centroid_offset += static_cast<uint64_t>(256) * width;
    }
    return;
  }
  const uint32_t coordinate_bytes =
      encoding() == Encoding::kAffine8 ? scorer_dim() : (scorer_dim() + 1U) / 2U;
  float minimum = 0.0f, scale = 0.0f;
  std::memcpy(&minimum, record + coordinate_bytes, sizeof(float));
  std::memcpy(&scale, record + coordinate_bytes + sizeof(float), sizeof(float));
  for (uint32_t coordinate = 0; coordinate < scorer_dim(); ++coordinate) {
    const uint8_t quantized =
        encoding() == Encoding::kAffine8
            ? record[coordinate]
            : static_cast<uint8_t>((record[coordinate / 2U] >>
                                    (4U * (coordinate & 1U))) & 0x0fU);
    output[coordinate] = minimum + scale * quantized;
  }
}

}  // namespace ours
