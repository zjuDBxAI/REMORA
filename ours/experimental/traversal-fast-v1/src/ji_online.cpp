#include "ours/ji_online.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ours {
namespace {

#pragma pack(push, 1)
struct OnlineHeader {
  char magic[8];
  uint32_t version;
  uint32_t dim;
  uint32_t groups;
  uint32_t budget_bits;
  uint64_t node_count;
  uint64_t artifact_id;
  uint64_t model_id;
  uint64_t total_codewords;
};

struct GroupRecord {
  uint32_t bit_offset;
  uint32_t bits;
  uint32_t codewords;
  uint32_t reserved0;
  uint64_t prototype_offset;
  uint64_t reserved1;
};

struct SideHeader {
  char magic[8];
  uint32_t version;
  uint64_t node_count;
  uint32_t dim;
  uint32_t budget_bits;
  uint32_t groups;
  uint64_t artifact_id;
  uint64_t model_id;
};
#pragma pack(pop)

static_assert(sizeof(OnlineHeader) == 56);
static_assert(sizeof(GroupRecord) == 32);
static_assert(sizeof(SideHeader) == 48);

template <typename T>
void ReadExact(std::ifstream &input, T *output, size_t count,
               const char *name) {
  input.read(reinterpret_cast<char *>(output),
             static_cast<std::streamsize>(count * sizeof(T)));
  if (!input)
    throw std::runtime_error(std::string("truncated JI online model: ") + name);
}

} // namespace

JiOnlineIndex JiOnlineIndex::Load(const std::string &model_path,
                                  const std::string &side_codes_path) {
  JiOnlineIndex result;
  std::ifstream model(model_path, std::ios::binary);
  if (!model)
    throw std::runtime_error("cannot open JI online model: " + model_path);
  OnlineHeader header{};
  ReadExact(model, &header, 1, "header");
  const bool version1 =
      std::memcmp(header.magic, "JIONL001", 8) == 0 && header.version == 1;
  const bool version2 =
      std::memcmp(header.magic, "JIONL002", 8) == 0 && header.version == 2;
  if ((!version1 && !version2) || header.dim == 0 || header.groups == 0 ||
      header.budget_bits == 0 || header.budget_bits > 256 ||
      header.node_count == 0 ||
      header.total_codewords == 0) {
    throw std::runtime_error("invalid JI online model header");
  }
  result.dim_ = header.dim;
  result.format_version_ = header.version;
  result.declared_groups_ = header.groups;
  result.budget_bits_ = header.budget_bits;
  result.node_count_ = header.node_count;
  result.artifact_id_ = header.artifact_id;
  result.model_id_ = header.model_id;
  result.residual_mean_.resize(header.dim);
  ReadExact(model, result.residual_mean_.data(), header.dim, "residual mean");
  if (version2) {
    if (header.dim > std::numeric_limits<size_t>::max() / header.dim)
      throw std::runtime_error("JI inverse transform is too large");
    result.inverse_transform_.resize(static_cast<size_t>(header.dim) *
                                     header.dim);
    ReadExact(model, result.inverse_transform_.data(),
              result.inverse_transform_.size(), "inverse transform");
  }

  std::vector<GroupRecord> records(header.groups);
  ReadExact(model, records.data(), records.size(), "groups");
  uint64_t expected_bit_offset = 0;
  uint64_t expected_codewords = 0;
  uint64_t expected_index_offset = 0;
  uint64_t expected_prototype_values = 0;
  uint64_t lut_offset = 0;
  for (const GroupRecord &source : records) {
    if (source.bit_offset != expected_bit_offset || source.bits > 16 ||
        source.codewords != (1U << source.bits)) {
      throw std::runtime_error("invalid JI online group table");
    }
    const uint32_t group_dim = version1 ? header.dim : source.reserved0;
    const uint64_t index_offset = version1 ? 0 : source.reserved1;
    const uint64_t prototype_offset =
        version1 ? source.prototype_offset * header.dim
                 : source.prototype_offset;
    if (group_dim == 0 || group_dim > header.dim ||
        (version1 && source.prototype_offset != expected_codewords) ||
        (version2 &&
         (index_offset != expected_index_offset ||
          prototype_offset != expected_prototype_values))) {
      throw std::runtime_error("invalid JI online group layout");
    }
    const uint64_t mask =
        source.bits == 64 ? ~uint64_t{0}
                          : ((uint64_t{1} << source.bits) - 1U);
    Group decoded{source.bit_offset, source.bits, source.codewords, group_dim,
                  prototype_offset, index_offset, lut_offset, mask};
    if (source.bits == 0) {
      result.zero_bit_groups_.push_back(decoded);
    } else {
      result.groups_.push_back(decoded);
      lut_offset += source.codewords;
    }
    expected_bit_offset += source.bits;
    expected_codewords += source.codewords;
    expected_index_offset += version2 ? group_dim : 0;
    expected_prototype_values +=
        version2 ? static_cast<uint64_t>(source.codewords) * group_dim : 0;
  }
  if (expected_bit_offset != header.budget_bits ||
      expected_codewords != header.total_codewords) {
    throw std::runtime_error("JI online group bits/codewords are inconsistent");
  }

  if (version1) {
    if (header.total_codewords >
        std::numeric_limits<size_t>::max() / header.dim) {
      throw std::runtime_error("JI online prototype table is too large");
    }
    result.dense_prototypes_.resize(
        static_cast<size_t>(header.total_codewords) * header.dim);
    ReadExact(model, result.dense_prototypes_.data(),
              result.dense_prototypes_.size(), "dense prototypes");
  } else {
    if (expected_index_offset != header.dim)
      throw std::runtime_error("JI online groups do not cover the dimension");
    result.group_indices_.resize(expected_index_offset);
    ReadExact(model, result.group_indices_.data(), result.group_indices_.size(),
              "group indices");
    std::vector<uint32_t> sorted = result.group_indices_;
    std::sort(sorted.begin(), sorted.end());
    for (uint32_t coordinate = 0; coordinate < header.dim; ++coordinate) {
      if (sorted[coordinate] != coordinate)
        throw std::runtime_error(
            "JI online groups are not a coordinate partition");
    }
    result.local_prototypes_.resize(expected_prototype_values);
    ReadExact(model, result.local_prototypes_.data(),
              result.local_prototypes_.size(), "group-local prototypes");
#if defined(__AVX512F__)
    result.transform_columns_.resize(result.inverse_transform_.size());
    for (uint32_t r = 0; r < header.dim; ++r)
      for (uint32_t c = 0; c < header.dim; ++c)
        result.transform_columns_[static_cast<size_t>(c) * header.dim + r] =
            result.inverse_transform_[static_cast<size_t>(r) * header.dim + c];
    result.prototype_columns_.resize(result.local_prototypes_.size());
    for (const Group &g : result.groups_)
      for (uint32_t k = 0; k < g.codewords; ++k)
        for (uint32_t c = 0; c < g.group_dim; ++c)
          result.prototype_columns_[g.prototype_offset + static_cast<size_t>(c) * g.codewords + k] =
              result.local_prototypes_[g.prototype_offset + static_cast<size_t>(k) * g.group_dim + c];
#endif
  }
  char trailing = 0;
  model.read(&trailing, 1);
  if (model.gcount() != 0)
    throw std::runtime_error("JI online model has trailing bytes");
  result.model_bytes_ = std::filesystem::file_size(model_path);

  result.side_fd_ = open(side_codes_path.c_str(), O_RDONLY);
  if (result.side_fd_ < 0)
    throw std::runtime_error("cannot open JI side codes: " + side_codes_path);
  struct stat status {};
  if (fstat(result.side_fd_, &status) != 0 ||
      status.st_size < static_cast<off_t>(sizeof(SideHeader))) {
    result.Close();
    throw std::runtime_error("cannot stat or truncated JI side codes");
  }
  result.side_mapping_size_ = static_cast<size_t>(status.st_size);
  result.side_mapping_ = mmap(nullptr, result.side_mapping_size_, PROT_READ,
                              MAP_SHARED, result.side_fd_, 0);
  if (result.side_mapping_ == MAP_FAILED) {
    result.side_mapping_ = nullptr;
    result.Close();
    throw std::runtime_error("cannot mmap JI side codes");
  }
  SideHeader side{};
  std::memcpy(&side, result.side_mapping_, sizeof(side));
  const uint32_t words_per_node = (header.budget_bits + 63U) / 64U;
  const uint32_t expected_side_version = words_per_node == 1 ? 1U : 2U;
  const bool dense_byte_codes = side.version == 3;
  if (dense_byte_codes) {
    if (header.budget_bits % 8 != 0 || header.budget_bits != header.groups*8)
      throw std::runtime_error("Dense byte side codes require 8-bit groups");
    for (const Group &g : result.groups_)
      if (g.bits != 8 || g.bit_offset % 8 != 0)
        throw std::runtime_error("Dense byte side group is not byte aligned");
  }
  const uint32_t stride = dense_byte_codes ? header.budget_bits/8 : words_per_node*sizeof(uint64_t);
  const uint64_t expected_side_size =
      sizeof(SideHeader) + header.node_count * stride;
  if (std::memcmp(side.magic, "JIRCODE1", 8) != 0 ||
      (!dense_byte_codes && side.version != expected_side_version) ||
      side.node_count != header.node_count || side.dim != header.dim ||
      side.budget_bits != header.budget_bits || side.groups != header.groups ||
      side.artifact_id != header.artifact_id ||
      side.model_id != header.model_id ||
      result.side_mapping_size_ != expected_side_size) {
    result.Close();
    throw std::runtime_error("JI side codes do not match online model");
  }
  result.packed_codes_ = reinterpret_cast<const uint64_t *>(
      static_cast<const char *>(result.side_mapping_) + sizeof(SideHeader));
  result.words_per_node_ = words_per_node;
  result.dense_bytes_per_node_ = dense_byte_codes ? stride : 0;
  result.direct_8x8_ = header.budget_bits == 64 && words_per_node == 1 &&
                       result.groups_.size() == 8;
  if (result.direct_8x8_) {
    for (uint32_t index = 0; index < 8; ++index) {
      const Group &group = result.groups_[index];
      if (group.bits != 8 || group.bit_offset != index * 8 ||
          group.codewords != 256) {
        result.direct_8x8_ = false;
        break;
      }
    }
  }
  result.side_bytes_ = expected_side_size;
  return result;
}

JiOnlineIndex::~JiOnlineIndex() { Close(); }

JiOnlineIndex::JiOnlineIndex(JiOnlineIndex &&other) noexcept {
  *this = std::move(other);
}

JiOnlineIndex &JiOnlineIndex::operator=(JiOnlineIndex &&other) noexcept {
  if (this == &other)
    return *this;
  Close();
  dim_ = other.dim_;
  format_version_ = other.format_version_;
  declared_groups_ = other.declared_groups_;
  budget_bits_ = other.budget_bits_;
  node_count_ = other.node_count_;
  artifact_id_ = other.artifact_id_;
  model_id_ = other.model_id_;
  model_bytes_ = other.model_bytes_;
  side_bytes_ = other.side_bytes_;
  residual_mean_ = std::move(other.residual_mean_);
  groups_ = std::move(other.groups_);
  zero_bit_groups_ = std::move(other.zero_bit_groups_);
  dense_prototypes_ = std::move(other.dense_prototypes_);
  inverse_transform_ = std::move(other.inverse_transform_);
  group_indices_ = std::move(other.group_indices_);
  local_prototypes_ = std::move(other.local_prototypes_);
  transform_columns_ = std::move(other.transform_columns_);
  prototype_columns_ = std::move(other.prototype_columns_);
  side_fd_ = std::exchange(other.side_fd_, -1);
  side_mapping_ = std::exchange(other.side_mapping_, nullptr);
  side_mapping_size_ = std::exchange(other.side_mapping_size_, 0);
  words_per_node_ = std::exchange(other.words_per_node_, 1);
  dense_bytes_per_node_ = std::exchange(other.dense_bytes_per_node_, 0);
  packed_codes_ = std::exchange(other.packed_codes_, nullptr);
  direct_8x8_ = std::exchange(other.direct_8x8_, false);
  return *this;
}

void JiOnlineIndex::Close() noexcept {
  if (side_mapping_)
    munmap(side_mapping_, side_mapping_size_);
  if (side_fd_ >= 0)
    close(side_fd_);
  side_mapping_ = nullptr;
  side_fd_ = -1;
  side_mapping_size_ = 0;
  words_per_node_ = 1;
  dense_bytes_per_node_ = 0;
  packed_codes_ = nullptr;
  direct_8x8_ = false;
}

uint32_t JiOnlineIndex::CodeAt(uint32_t node_id, const Group &group) const {
  if (group.bits == 0) return 0;
  if (dense_bytes_per_node_)
    return reinterpret_cast<const uint8_t*>(packed_codes_)[
        static_cast<uint64_t>(node_id)*dense_bytes_per_node_+group.bit_offset/8];
  const uint64_t *row =
      packed_codes_ + static_cast<uint64_t>(node_id) * words_per_node_;
  const uint32_t word = group.bit_offset / 64U;
  const uint32_t shift = group.bit_offset % 64U;
  uint64_t value = row[word] >> shift;
  if (shift + group.bits > 64U)
    value |= row[word + 1] << (64U - shift);
  return static_cast<uint32_t>(value & group.mask);
}

const char *JiOnlineIndex::scoring_kernel() const {
#if defined(__AVX512F__)
  if (direct_8x8_) return "avx512-direct8x8-register-decode";
  return "avx512-lane-parallel-lut-and-gather";
#else
  return "portable-direct-or-avx2";
#endif
}

JiQueryContext JiOnlineIndex::PrepareQueryZeroPadded(const float *query,
                                                   uint32_t dim,
                                                   bool vectorized,
                                                   JiPrepareTimings *timings) const {
  if (!query || dim == 0 || dim > dim_)
    throw std::invalid_argument("JI padded query dimension mismatch");
  if (dim == dim_) return PrepareQuery(query, dim, vectorized, timings);
  std::vector<float> padded(dim_, 0.0f);
  std::copy(query, query + dim, padded.begin());
  return PrepareQuery(padded.data(), dim_, vectorized, timings);
}

JiQueryContext JiOnlineIndex::PrepareQuery(const float *query,
                                           uint32_t dim, bool vectorized,
                                           JiPrepareTimings *timings) const {
  JiQueryContext result;
  std::vector<float> phi;
  PrepareQueryInto(query, dim, vectorized, result, phi, true, timings);
  return result;
}

const JiQueryContext &JiOnlineIndex::PrepareTerminalQuery(
    const float *query, uint32_t dim, JiQueryWorkspace &workspace,
    JiPrepareTimings *timings) const {
  return PrepareQueryReused(query, dim, workspace, false, timings);
}

const JiQueryContext &JiOnlineIndex::PrepareQueryReused(
    const float *query, uint32_t dim, JiQueryWorkspace &workspace,
    bool build_upper_bound, JiPrepareTimings *timings) const {
  if (!query || dim == 0 || dim > dim_)
    throw std::invalid_argument("JI terminal query dimension mismatch");
  if (dim < dim_) {
    workspace.padded.resize(dim_);
    std::copy(query, query + dim, workspace.padded.begin());
    std::fill(workspace.padded.begin() + dim, workspace.padded.end(), 0.0f);
    query = workspace.padded.data();
  }
  PrepareQueryInto(query, dim_, true, workspace.context, workspace.phi, build_upper_bound, timings);
  return workspace.context;
}

void JiOnlineIndex::PrepareQueryInto(const float *query, uint32_t dim,
                                     bool vectorized, JiQueryContext &result,
                                     std::vector<float> &phi,
                                     bool build_upper_bound,
                                     JiPrepareTimings *timings) const {
  if (!query || dim != dim_)
    throw std::invalid_argument("JI query dimension mismatch");
  using Clock = std::chrono::steady_clock;
  const auto milliseconds = [](Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
  };
  if (timings) *timings = {};
  result.mean_bias = 0.0f;
  // Infinite bound is safe if a terminal-only context is accidentally used
  // for pruning. No stale previous-query bound may survive buffer reuse.
  result.max_correction_upper = std::numeric_limits<float>::infinity();
  for (uint32_t coordinate = 0; coordinate < dim_; ++coordinate)
    result.mean_bias += query[coordinate] * residual_mean_[coordinate];
  size_t total = 0;
  for (const Group &group : groups_)
    total += group.codewords;
  result.lookup.resize(total);

  if (format_version_ == 1) {
    const auto lut_started = timings ? Clock::now() : Clock::time_point{};
    const auto prepare_group = [&](const Group &group, float *destination) {
      for (uint32_t code = 0; code < group.codewords; ++code) {
        const float *prototype = dense_prototypes_.data() +
                                 group.prototype_offset +
                                 static_cast<uint64_t>(code) * dim_;
        float score = 0.0f;
        for (uint32_t coordinate = 0; coordinate < dim_; ++coordinate)
          score += query[coordinate] * prototype[coordinate];
        destination[code] = score;
      }
    };
    for (const Group &group : groups_)
      prepare_group(group, result.lookup.data() + group.lut_offset);
    std::array<float, 1> constant{};
    for (const Group &group : zero_bit_groups_) {
      prepare_group(group, constant.data());
      result.mean_bias += constant[0];
    }
    if (timings) timings->lut_ms = milliseconds(Clock::now() - lut_started);
  } else {
    // Offline reconstruction is y @ inverse_transform.  Therefore
    // phi=inverse_transform @ q is exactly T^T q in column-vector notation.
    phi.resize(dim_);
    const auto projection_started = timings ? Clock::now() : Clock::time_point{};
    uint32_t row = 0;
#if defined(__AVX512F__)
    if (vectorized) {
      for (; row + 16 <= dim_; row += 16) {
        __m512 sums[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(),
                          _mm512_setzero_ps(), _mm512_setzero_ps()};
        uint32_t c = 0;
        for (; c + 4 <= dim_; c += 4)
          for (uint32_t t = 0; t < 4; ++t)
            sums[t] = _mm512_fmadd_ps(
                _mm512_loadu_ps(transform_columns_.data() + static_cast<size_t>(c+t)*dim_ + row),
                _mm512_set1_ps(query[c+t]), sums[t]);
        __m512 sum = _mm512_add_ps(_mm512_add_ps(sums[0], sums[1]),
                                   _mm512_add_ps(sums[2], sums[3]));
        for (; c < dim_; ++c)
          sum = _mm512_fmadd_ps(
              _mm512_loadu_ps(transform_columns_.data() + static_cast<size_t>(c)*dim_ + row),
              _mm512_set1_ps(query[c]), sum);
        _mm512_storeu_ps(phi.data() + row, sum);
      }
    }
#endif
    for (; row < dim_; ++row) {
      const float *transform =
          inverse_transform_.data() + static_cast<uint64_t>(row) * dim_;
      float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
      uint32_t coordinate = 0;
      for (; coordinate + 4 <= dim_; coordinate += 4) {
        sum0 += transform[coordinate] * query[coordinate];
        sum1 += transform[coordinate + 1] * query[coordinate + 1];
        sum2 += transform[coordinate + 2] * query[coordinate + 2];
        sum3 += transform[coordinate + 3] * query[coordinate + 3];
      }
      float score = (sum0 + sum1) + (sum2 + sum3);
      for (; coordinate < dim_; ++coordinate)
        score += transform[coordinate] * query[coordinate];
      phi[row] = score;
    }
    if (timings) timings->projection_ms = milliseconds(Clock::now() - projection_started);
    const auto lut_started = timings ? Clock::now() : Clock::time_point{};
    const auto prepare_group = [&](const Group &group, float *destination) {
      const uint32_t *indices = group_indices_.data() + group.index_offset;
      uint32_t code = 0;
#if defined(__AVX512F__)
      if (vectorized && group.bits != 0) {
        for (; code + 16 <= group.codewords; code += 16) {
          __m512 sum = _mm512_setzero_ps();
          for (uint32_t c = 0; c < group.group_dim; ++c)
            sum = _mm512_fmadd_ps(
                _mm512_loadu_ps(prototype_columns_.data() + group.prototype_offset +
                               static_cast<size_t>(c)*group.codewords + code),
                _mm512_set1_ps(phi[indices[c]]), sum);
          _mm512_storeu_ps(destination + code, sum);
        }
      }
#endif
      for (; code < group.codewords; ++code) {
        const float *prototype =
            local_prototypes_.data() + group.prototype_offset +
            static_cast<uint64_t>(code) * group.group_dim;
        float score = 0.0f;
        for (uint32_t coordinate = 0; coordinate < group.group_dim;
             ++coordinate)
          score += prototype[coordinate] * phi[indices[coordinate]];
        destination[code] = score;
      }
    };
    for (const Group &group : groups_)
      prepare_group(group, result.lookup.data() + group.lut_offset);
    std::array<float, 16> constant{};
    for (const Group &group : zero_bit_groups_) {
      prepare_group(group, constant.data());
      result.mean_bias += constant[0];
    }
    if (timings) timings->lut_ms = milliseconds(Clock::now() - lut_started);
  }

  if (!build_upper_bound) return;
  const auto bound_started = timings ? Clock::now() : Clock::time_point{};
  float upper = result.mean_bias;
  const float infinity = std::numeric_limits<float>::infinity();
  for (const Group &group : groups_) {
    const float *begin = result.lookup.data() + group.lut_offset;
    const float group_max =
        *std::max_element(begin, begin + group.codewords);
    upper = std::nextafter(upper + group_max, infinity);
  }
  result.max_correction_upper = upper;
  if (timings) timings->upper_bound_ms = milliseconds(Clock::now() - bound_started);
}

float JiOnlineIndex::Correction(const JiQueryContext &query,
                                uint32_t node_id) const {
  if (node_id >= node_count_)
    throw std::out_of_range("JI node id is out of range");
  if (direct_8x8_) {
    const uint64_t code = packed_codes_[node_id];
    const float *lookup = query.lookup.data();
    float result = query.mean_bias;
    result += lookup[groups_[0].lut_offset + static_cast<uint8_t>(code)];
    result += lookup[groups_[1].lut_offset + static_cast<uint8_t>(code >> 8)];
    result += lookup[groups_[2].lut_offset + static_cast<uint8_t>(code >> 16)];
    result += lookup[groups_[3].lut_offset + static_cast<uint8_t>(code >> 24)];
    result += lookup[groups_[4].lut_offset + static_cast<uint8_t>(code >> 32)];
    result += lookup[groups_[5].lut_offset + static_cast<uint8_t>(code >> 40)];
    result += lookup[groups_[6].lut_offset + static_cast<uint8_t>(code >> 48)];
    result += lookup[groups_[7].lut_offset + static_cast<uint8_t>(code >> 56)];
    return result;
  }
  float result = query.mean_bias;
  for (const Group &group : groups_) {
    const uint32_t code = CodeAt(node_id, group);
    result += query.lookup[group.lut_offset + code];
  }
  return result;
}

void JiOnlineIndex::ComputeCorrectionsBatch(const JiQueryContext &query,
                                            const uint32_t *node_ids,
                                            size_t count,
                                            float *corrections) const {
  CorrectionsBatchImpl<false>(query, node_ids, count, corrections);
}

void JiOnlineIndex::AddCorrectionsBatch(const JiQueryContext &query,
                                      const uint32_t *node_ids, size_t count,
                                      float *scores) const {
  CorrectionsBatchImpl<true>(query, node_ids, count, scores);
}

template<bool AddPrimary>
void JiOnlineIndex::CorrectionsBatchImpl(const JiQueryContext &query,
                                        const uint32_t *node_ids, size_t count,
                                        float *corrections) const {
  if (count != 0 && (!node_ids || !corrections))
    throw std::invalid_argument("JI batch correction received a null buffer");
  for (size_t index = 0; index < count; ++index) {
    if (node_ids[index] >= node_count_)
      throw std::out_of_range("JI batch node id is out of range");
  }
#if defined(__AVX512F__)
  if (direct_8x8_) {
    // Load each side-code word once. Lanes are candidates; ordered group sums
    // match Correction(). 64-bit indices avoid signed-32-bit addressing limits.
    size_t i = 0;
    const auto mask = _mm512_set1_epi64(255);
    for (; i + 8 <= count; i += 8) {
      const auto ids = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(node_ids+i));
      auto packed = _mm512_i64gather_epi64(_mm512_cvtepu32_epi64(ids), packed_codes_, 8);
      auto sum = _mm256_set1_ps(query.mean_bias);
      for (uint32_t group = 0; group < 8; ++group) {
        const auto code = _mm512_cvtepi64_epi32(_mm512_and_si512(packed, mask));
        sum = _mm256_add_ps(sum, _mm256_i32gather_ps(
            query.lookup.data()+groups_[group].lut_offset, code, 4));
        packed = _mm512_srli_epi64(packed, 8);
      }
      if constexpr (AddPrimary) sum = _mm256_add_ps(_mm256_loadu_ps(corrections+i), sum);
      _mm256_storeu_ps(corrections+i, sum);
    }
    for (; i < count; ++i) {
      const float value = Correction(query, node_ids[i]);
      if constexpr (AddPrimary) corrections[i] += value;
      else corrections[i] = value;
    }
    return;
  }
  // Independent candidates in 16 lanes, with identical group accumulation
  // order to Correction(). Includes 256-bit codes and non-byte-aligned formats.
  size_t batch = 0;
  for (; batch + 16 <= count; batch += 16) {
    __m512 sum = _mm512_set1_ps(query.mean_bias);
    alignas(64) int32_t codes[16];
    for (const Group &group : groups_) {
      for (size_t lane = 0; lane < 16; ++lane)
        codes[lane] = static_cast<int32_t>(CodeAt(node_ids[batch + lane], group));
      sum = _mm512_add_ps(sum, _mm512_i32gather_ps(
          _mm512_load_si512(codes), query.lookup.data() + group.lut_offset, 4));
    }
    if constexpr (AddPrimary) sum = _mm512_add_ps(_mm512_loadu_ps(corrections+batch), sum);
    _mm512_storeu_ps(corrections + batch, sum);
  }
  for (; batch < count; ++batch) {
    const float value = Correction(query, node_ids[batch]);
    if constexpr (AddPrimary) corrections[batch] += value;
    else corrections[batch] = value;
  }
  return;
#endif
  if (direct_8x8_) {
    const float *lookup = query.lookup.data();
    const float *lut0 = lookup + groups_[0].lut_offset;
    const float *lut1 = lookup + groups_[1].lut_offset;
    const float *lut2 = lookup + groups_[2].lut_offset;
    const float *lut3 = lookup + groups_[3].lut_offset;
    const float *lut4 = lookup + groups_[4].lut_offset;
    const float *lut5 = lookup + groups_[5].lut_offset;
    const float *lut6 = lookup + groups_[6].lut_offset;
    const float *lut7 = lookup + groups_[7].lut_offset;
    for (size_t index = 0; index < count; ++index) {
      const uint64_t code = packed_codes_[node_ids[index]];
      float result = query.mean_bias;
      result += lut0[static_cast<uint8_t>(code)];
      result += lut1[static_cast<uint8_t>(code >> 8)];
      result += lut2[static_cast<uint8_t>(code >> 16)];
      result += lut3[static_cast<uint8_t>(code >> 24)];
      result += lut4[static_cast<uint8_t>(code >> 32)];
      result += lut5[static_cast<uint8_t>(code >> 40)];
      result += lut6[static_cast<uint8_t>(code >> 48)];
      result += lut7[static_cast<uint8_t>(code >> 56)];
      if constexpr (AddPrimary) corrections[index] += result;
      else corrections[index] = result;
    }
    return;
  }
  if constexpr (AddPrimary) {
    // Preserve parentheses on the portable fallback as well.
    for (size_t i = 0; i < count; ++i) corrections[i] += Correction(query, node_ids[i]);
    return;
  }
  size_t index = 0;
#if defined(__AVX2__)
  const __m256 bias = _mm256_set1_ps(query.mean_bias);
  for (; index + 8 <= count; index += 8) {
    _mm256_storeu_ps(corrections + index, bias);
  }
#endif
  for (; index < count; ++index)
    corrections[index] = query.mean_bias;

  // Group-major accumulation keeps the current LUT hot.  AVX2 performs eight
  // independent LUT reads and additions together; side-code extraction stays
  // scalar because node ids form an irregular graph-neighbor stream.
  for (const Group &group : groups_) {
    const float *lookup = query.lookup.data() + group.lut_offset;
    size_t local = 0;
#if defined(__AVX2__)
    for (; local + 8 <= count; local += 8) {
      const auto code_at = [&](size_t lane) {
        return static_cast<int32_t>(CodeAt(node_ids[local + lane], group));
      };
      const __m256i codes = _mm256_set_epi32(
          code_at(7), code_at(6), code_at(5), code_at(4), code_at(3),
          code_at(2), code_at(1), code_at(0));
      const __m256 values = _mm256_i32gather_ps(lookup, codes, 4);
      _mm256_storeu_ps(
          corrections + local,
          _mm256_add_ps(_mm256_loadu_ps(corrections + local), values));
    }
#endif
    for (; local < count; ++local) {
      const uint32_t code = CodeAt(node_ids[local], group);
      corrections[local] += lookup[code];
    }
  }
}

void JiOnlineIndex::ReconstructNode(uint32_t node_id, float *output,
                                    uint32_t dim) const {
  if (!output || dim != dim_)
    throw std::invalid_argument("JI reconstruction dimension mismatch");
  if (node_id >= node_count_)
    throw std::out_of_range("JI reconstruction node id is out of range");

  std::copy(residual_mean_.begin(), residual_mean_.end(), output);
  const auto code_for = [&](const Group &group) {
    return CodeAt(node_id, group);
  };

  if (format_version_ == 1) {
    const auto add_group = [&](const Group &group) {
      const float *prototype =
          dense_prototypes_.data() + group.prototype_offset +
          static_cast<uint64_t>(code_for(group)) * dim_;
      for (uint32_t coordinate = 0; coordinate < dim_; ++coordinate)
        output[coordinate] += prototype[coordinate];
    };
    for (const Group &group : groups_)
      add_group(group);
    for (const Group &group : zero_bit_groups_)
      add_group(group);
    return;
  }

  // Offline materialization uses row-vector y @ inverse_transform.  Accumulate
  // only the selected group-local y coordinates into the decoded residual.
  const auto add_group = [&](const Group &group) {
    const uint32_t code = code_for(group);
    const uint32_t *indices = group_indices_.data() + group.index_offset;
    const float *prototype = local_prototypes_.data() + group.prototype_offset +
                             static_cast<uint64_t>(code) * group.group_dim;
    for (uint32_t local = 0; local < group.group_dim; ++local) {
      const float value = prototype[local];
      const float *transform = inverse_transform_.data() +
                               static_cast<uint64_t>(indices[local]) * dim_;
      for (uint32_t coordinate = 0; coordinate < dim_; ++coordinate)
        output[coordinate] += value * transform[coordinate];
    }
  };
  for (const Group &group : groups_)
    add_group(group);
  for (const Group &group : zero_bit_groups_)
    add_group(group);
}

} // namespace ours
