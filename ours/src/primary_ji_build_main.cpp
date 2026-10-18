#include "ours/ji_online.h"
#include "ours/generic_primary.h"
#include "ours/mapped_fbin.h"

#include <omp.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "efanna2e/parameters.h"
#include "index_bipartite.h"

#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/pack_excode.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"

#include "quantization/saq_data.hpp"
#include "quantization/saq_estimator.hpp"
#include "quantization/saq_quantizer.hpp"
#include "quantization/single_data.hpp"
#include "utils/code_helper.hpp"

namespace {

using Clock = std::chrono::steady_clock;

class Mapping {
 public:
  explicit Mapping(const std::string &path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open: " + path);
    struct stat status {};
    if (fstat(fd_, &status) != 0 || status.st_size <= 0)
      throw std::runtime_error("cannot stat or empty file: " + path);
    size_ = static_cast<size_t>(status.st_size);
    data_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (data_ == MAP_FAILED) {
      data_ = nullptr;
      throw std::runtime_error("cannot mmap: " + path);
    }
  }
  ~Mapping() {
    if (data_) munmap(data_, size_);
    if (fd_ >= 0) close(fd_);
  }
  Mapping(const Mapping &) = delete;
  Mapping &operator=(const Mapping &) = delete;
  const void *data() const { return data_; }
  size_t size() const { return size_; }

 private:
  int fd_ = -1;
  void *data_ = nullptr;
  size_t size_ = 0;
};

#pragma pack(push, 1)
struct RabitqHeader {
  char magic[8];
  uint32_t version;
  uint32_t metric_type;
  uint32_t rotator_type;
  uint32_t total_bits;
  uint64_t num_points;
  uint64_t dim;
  uint64_t padded_dim;
  uint64_t bin_bytes_per_vec;
  uint64_t ex_bytes_per_vec;
  uint64_t rotator_bytes;
};

struct SaqCodeHeader {
  char magic[8];
  uint32_t version;
  uint32_t header_bytes;
  uint64_t rows;
  uint64_t record_bytes;
  uint64_t model_bytes;
  uint64_t reserved[3];
};
#pragma pack(pop)
static_assert(sizeof(RabitqHeader) == 72);
static_assert(sizeof(SaqCodeHeader) == 64);

struct BitLocation {
  uint32_t byte = 0;
  uint8_t mask = 0;
};

template <class Pack>
std::vector<BitLocation> BuildBitLocations(uint32_t dim, uint32_t bits,
                                           size_t packed_bytes, Pack pack) {
  std::vector<BitLocation> result(static_cast<size_t>(dim) * bits);
  if (bits == 0) return result;
  std::vector<uint16_t> raw(dim, 0);
  std::vector<uint8_t> packed(packed_bytes, 0);
  for (uint32_t coordinate = 0; coordinate < dim; ++coordinate) {
    for (uint32_t bit = 0; bit < bits; ++bit) {
      std::fill(raw.begin(), raw.end(), 0);
      std::fill(packed.begin(), packed.end(), 0);
      raw[coordinate] = static_cast<uint16_t>(1U << bit);
      pack(raw.data(), packed.data());
      BitLocation location;
      uint32_t found = 0;
      for (uint32_t byte = 0; byte < packed.size(); ++byte) {
        const uint8_t value = packed[byte];
        for (uint32_t packed_bit = 0; packed_bit < 8; ++packed_bit) {
          const uint8_t mask = static_cast<uint8_t>(1U << packed_bit);
          if (value & mask) {
            location = BitLocation{byte, mask};
            ++found;
          }
        }
      }
      if (found != 1)
        throw std::runtime_error("compressed code layout is not a bit permutation");
      result[static_cast<size_t>(coordinate) * bits + bit] = location;
    }
  }
  return result;
}

uint32_t DecodeBits(const uint8_t *packed,
                    const std::vector<BitLocation> &locations,
                    uint32_t coordinate, uint32_t bits) {
  uint32_t value = 0;
  for (uint32_t bit = 0; bit < bits; ++bit) {
    const BitLocation location =
        locations[static_cast<size_t>(coordinate) * bits + bit];
    if (packed[location.byte] & location.mask) value |= 1U << bit;
  }
  return value;
}

class RabitqPrimary {
 public:
  struct QueryContext {
    std::vector<float> transformed;
    std::unique_ptr<rabitqlib::SplitSingleQuery<float>> query;
  };
  using PairContext = QueryContext;
  struct Scratch {};

  RabitqPrimary(const std::string &path, const std::string &config_name)
      : mapping_(path) {
    if (mapping_.size() < sizeof(header_))
      throw std::runtime_error("truncated RaBitQ file");
    std::memcpy(&header_, mapping_.data(), sizeof(header_));
    if (std::memcmp(header_.magic, "RBQROAR1", 8) != 0 ||
        header_.version != 1 || header_.metric_type != rabitqlib::METRIC_IP ||
        header_.total_bits < 2 || header_.total_bits > 9)
      throw std::runtime_error("unsupported Extended-RaBitQ file");
    const uint64_t expected = sizeof(header_) + header_.rotator_bytes +
                              header_.num_points *
                                  (header_.bin_bytes_per_vec +
                                   header_.ex_bytes_per_vec);
    if (mapping_.size() != expected)
      throw std::runtime_error("RaBitQ file-size mismatch");
    rotator_.reset(rabitqlib::choose_rotator<float>(
        header_.dim,
        static_cast<rabitqlib::RotatorType>(header_.rotator_type),
        header_.padded_dim));
    rotator_->load(static_cast<const char *>(mapping_.data()) + sizeof(header_));
    bin_begin_ = static_cast<const char *>(mapping_.data()) + sizeof(header_) +
                 header_.rotator_bytes;
    ex_begin_ = bin_begin_ + header_.num_points * header_.bin_bytes_per_vec;
    ex_bits_ = header_.total_bits - 1;
    config_ = config_name == "faster"
                  ? rabitqlib::quant::faster_config(header_.padded_dim,
                                                    header_.total_bits)
                  : rabitqlib::quant::RabitqConfig();
    if (config_name != "faster" && config_name != "exact")
      throw std::invalid_argument("rabitq-config must be faster or exact");
    ip_func_ = rabitqlib::select_excode_ipfunc(ex_bits_);
    const size_t packed_bytes = header_.padded_dim * ex_bits_ / 8;
    ex_locations_ = BuildBitLocations(
        static_cast<uint32_t>(header_.padded_dim),
        static_cast<uint32_t>(ex_bits_), packed_bytes,
        [&](const uint16_t *raw16, uint8_t *packed) {
          std::vector<uint8_t> raw(header_.padded_dim);
          for (uint32_t i = 0; i < header_.padded_dim; ++i)
            raw[i] = static_cast<uint8_t>(raw16[i]);
          rabitqlib::quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
              raw.data(), packed, header_.padded_dim, ex_bits_);
        });
  }

  uint64_t nodes() const { return header_.num_points; }
  uint32_t original_dim() const { return static_cast<uint32_t>(header_.dim); }
  uint32_t scorer_dim() const {
    return static_cast<uint32_t>(header_.padded_dim);
  }
  uint64_t storage_bytes() const { return mapping_.size(); }
  Scratch MakeScratch() const { return Scratch{}; }

  QueryContext Prepare(const float *query, uint32_t dim) const {
    if (dim != original_dim())
      throw std::invalid_argument("RaBitQ query dimension mismatch");
    std::vector<float> transformed(scorer_dim());
    rotator_->rotate(query, transformed.data());
    return PrepareTransformed(std::move(transformed));
  }
  PairContext PreparePair(const float *query) const {
    return PrepareTransformed(
        std::vector<float>(query, query + scorer_dim()));
  }
  const float *Transformed(const QueryContext &context) const {
    return context.transformed.data();
  }
  float Score(const QueryContext &context, uint32_t id) const {
    float distance = 0.0f, lower = 0.0f, binary_ip = 0.0f;
    rabitqlib::split_single_fulldist(
        Bin(id), Ex(id), ip_func_, *context.query, scorer_dim(), ex_bits_,
        distance, lower, binary_ip, 0.0f, 0.0f);
    return 1.0f - distance;
  }
  void ScoreBatch(const QueryContext &context, const uint32_t *ids,
                  size_t count, float *scores) const {
    for (size_t i = 0; i < count; ++i) scores[i] = Score(context, ids[i]);
  }
  float PairScore(PairContext &context, Scratch &, uint32_t id) const {
    return Score(context, id);
  }
  void PairScoreBatch(PairContext &context, Scratch &scratch,
                      const uint32_t *ids, size_t count, float *scores) const {
    (void)scratch;
    ScoreBatch(context, ids, count, scores);
  }
  void Reconstruct(uint32_t id, float *output, Scratch &) const {
    if (id >= nodes()) throw std::out_of_range("RaBitQ node id");
    auto bin = rabitqlib::ConstBinDataMap<float>(Bin(id), scorer_dim());
    auto ex = rabitqlib::ConstExDataMap<float>(Ex(id), scorer_dim(), ex_bits_);
    const uint64_t *sign = bin.bin_code();
    const uint8_t *extra = ex.ex_code();
    const float scale = -ex.f_rescale_ex();
    const uint32_t sign_bit = 1U << ex_bits_;
    const float center = -static_cast<float>(sign_bit) + 0.5f;
    for (uint32_t coordinate = 0; coordinate < scorer_dim(); ++coordinate) {
      const uint32_t positive =
          (sign[coordinate / 64] &
           (uint64_t{1} << (63U - coordinate % 64U)))
              ? 1U
              : 0U;
      const uint32_t low = DecodeBits(extra, ex_locations_, coordinate,
                                      static_cast<uint32_t>(ex_bits_));
      output[coordinate] =
          scale * (static_cast<float>(low + positive * sign_bit) + center);
    }
  }

 private:
  QueryContext PrepareTransformed(std::vector<float> transformed) const {
    QueryContext result;
    result.transformed = std::move(transformed);
    result.query = std::make_unique<rabitqlib::SplitSingleQuery<float>>(
        result.transformed.data(), scorer_dim(), ex_bits_, config_,
        rabitqlib::METRIC_IP);
    return result;
  }
  const char *Bin(uint32_t id) const {
    return bin_begin_ + static_cast<uint64_t>(id) * header_.bin_bytes_per_vec;
  }
  const char *Ex(uint32_t id) const {
    return ex_begin_ + static_cast<uint64_t>(id) * header_.ex_bytes_per_vec;
  }

  Mapping mapping_;
  RabitqHeader header_{};
  std::unique_ptr<rabitqlib::Rotator<float>> rotator_;
  const char *bin_begin_ = nullptr;
  const char *ex_begin_ = nullptr;
  size_t ex_bits_ = 0;
  rabitqlib::quant::RabitqConfig config_;
  decltype(rabitqlib::select_excode_ipfunc(1)) ip_func_;
  std::vector<BitLocation> ex_locations_;
};

struct AlignedFree {
  void operator()(uint8_t *value) const { std::free(value); }
};

class SaqPrimary {
 public:
  struct Scratch {
    saqlib::SaqSingleDataWrapper wrapper;
    std::unique_ptr<uint8_t, AlignedFree> memory;
    Scratch(const saqlib::SaqData &data, size_t bytes)
        : wrapper(data.quant_plan) {
      void *raw = nullptr;
      if (posix_memalign(&raw, 64, ((bytes + 63) / 64) * 64) != 0)
        throw std::bad_alloc();
      memory.reset(static_cast<uint8_t *>(raw));
      wrapper.set_memory_base(memory.get());
    }
    Scratch(Scratch &&) = default;
    Scratch &operator=(Scratch &&) = default;
    Scratch(const Scratch &) = delete;
    Scratch &operator=(const Scratch &) = delete;
  };
  struct QueryContext {
    saqlib::FloatVec original;
    saqlib::FloatVec transformed;
    Scratch scratch;
    std::unique_ptr<saqlib::SaqSingleEstimator<saqlib::DistType::IP>> estimator;
    QueryContext(const saqlib::SaqData &data, const float *query, uint32_t dim,
                 size_t bytes)
        : original(dim), scratch(data, bytes) {
      std::memcpy(original.data(), query, dim * sizeof(float));
      saqlib::SAQuantizerSingle transformer(&data);
      transformer.transform(original, &transformed);
      saqlib::SearcherConfig config;
      config.dist_type = saqlib::DistType::IP;
      estimator =
          std::make_unique<saqlib::SaqSingleEstimator<saqlib::DistType::IP>>(
              data, config, original);
    }
  };
  struct PairContext {
    std::vector<float> transformed;
  };

  SaqPrimary(const std::string &model_path, const std::string &codes_path)
      : codes_(codes_path) {
    std::ifstream input(model_path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open SAQ model: " + model_path);
    data_.load(input);
    if (codes_.size() < sizeof(header_))
      throw std::runtime_error("truncated SAQ codes");
    std::memcpy(&header_, codes_.data(), sizeof(header_));
    if (std::memcmp(header_.magic, "SAQCODE1", 8) != 0 ||
        header_.version != 1 || header_.header_bytes != sizeof(SaqCodeHeader) ||
        header_.record_bytes == 0 ||
        header_.model_bytes != std::filesystem::file_size(model_path) ||
        codes_.size() != sizeof(SaqCodeHeader) +
                             header_.rows * header_.record_bytes)
      throw std::runtime_error("invalid SAQ code file");
    code_begin_ = static_cast<const uint8_t *>(codes_.data()) + sizeof(header_);
    uint32_t offset = 0;
    for (const auto &plan : data_.quant_plan) {
      Segment segment;
      segment.dim = static_cast<uint32_t>(plan.first);
      segment.bits = static_cast<uint32_t>(plan.second);
      segment.offset = offset;
      if (segment.bits > 1) {
        const uint32_t low_bits = segment.bits - 1;
        const size_t packed_bytes =
            static_cast<size_t>(segment.dim) * low_bits / 8;
        auto pack = saqlib::utils::get_compacted_code16_func(low_bits);
        segment.low_locations = BuildBitLocations(
            segment.dim, low_bits, packed_bytes,
            [=](const uint16_t *raw, uint8_t *packed) {
              pack(packed, raw, segment.dim);
            });
      }
      segments_.push_back(std::move(segment));
      offset += static_cast<uint32_t>(plan.first);
    }
    if (offset != scorer_dim())
      throw std::runtime_error("SAQ segment dimensions do not match scorer dim");
  }

  uint64_t nodes() const { return header_.rows; }
  uint32_t original_dim() const { return static_cast<uint32_t>(data_.num_dim); }
  uint32_t scorer_dim() const {
    return static_cast<uint32_t>(data_.data_variance.cols());
  }
  uint64_t storage_bytes() const { return codes_.size(); }
  Scratch MakeScratch() const { return Scratch(data_, header_.record_bytes); }
  QueryContext Prepare(const float *query, uint32_t dim) const {
    if (dim != original_dim())
      throw std::invalid_argument("SAQ query dimension mismatch");
    return QueryContext(data_, query, dim, header_.record_bytes);
  }
  PairContext PreparePair(const float *query) const {
    return PairContext{std::vector<float>(query, query + scorer_dim())};
  }
  const float *Transformed(const QueryContext &context) const {
    return context.transformed.data();
  }
  float Score(QueryContext &context, uint32_t id) const {
    Load(id, context.scratch);
    return context.estimator->compAccurateDist(context.scratch.wrapper);
  }
  void ScoreBatch(QueryContext &context, const uint32_t *ids, size_t count,
                  float *scores) const {
    for (size_t i = 0; i < count; ++i) scores[i] = Score(context, ids[i]);
  }
  float PairScore(PairContext &context, Scratch &scratch, uint32_t id) const {
    Load(id, scratch);
    return DotDecoded(context.transformed.data(), scratch);
  }
  void PairScoreBatch(PairContext &context, Scratch &scratch,
                      const uint32_t *ids, size_t count, float *scores) const {
    for (size_t i = 0; i < count; ++i)
      scores[i] = PairScore(context, scratch, ids[i]);
  }
  void Reconstruct(uint32_t id, float *output, Scratch &scratch) const {
    Load(id, scratch);
    std::fill(output, output + scorer_dim(), 0.0f);
    Decode(nullptr, output, scratch);
  }

 private:
  struct Segment {
    uint32_t dim = 0;
    uint32_t bits = 0;
    uint32_t offset = 0;
    std::vector<BitLocation> low_locations;
  };
  void Load(uint32_t id, Scratch &scratch) const {
    if (id >= nodes()) throw std::out_of_range("SAQ node id");
    std::memcpy(scratch.memory.get(),
                code_begin_ + static_cast<uint64_t>(id) * header_.record_bytes,
                header_.record_bytes);
  }
  static uint32_t DecodeShort(const uint8_t *packed, uint32_t coordinate) {
    const uint32_t raw_byte = coordinate / 8;
    const uint32_t stored_byte = raw_byte + 7U - 2U * (raw_byte % 8U);
    return (packed[stored_byte] & (0x80U >> (coordinate % 8U))) ? 1U : 0U;
  }
  float Decode(const float *query, float *output, const Scratch &scratch) const {
    float score = 0.0f;
    for (uint32_t index = 0; index < segments_.size(); ++index) {
      const Segment &layout = segments_[index];
      const auto &segment = scratch.wrapper.get_segment(index);
      if (layout.bits == 0) continue;
      const uint32_t low_bits = layout.bits - 1;
      const uint32_t sign_bit = 1U << low_bits;
      const float delta = 2.0f / static_cast<float>(1U << layout.bits);
      const float scale = segment.long_factor().rescale;
      for (uint32_t coordinate = 0; coordinate < layout.dim; ++coordinate) {
        const uint32_t low = low_bits == 0
                                 ? 0U
                                 : DecodeBits(segment.long_code(),
                                              layout.low_locations, coordinate,
                                              low_bits);
        const uint32_t code =
            low + DecodeShort(segment.short_code(), coordinate) * sign_bit;
        const float value =
            scale * ((static_cast<float>(code) + 0.5f) * delta - 1.0f);
        if (output) output[layout.offset + coordinate] = value;
        if (query) score += query[layout.offset + coordinate] * value;
      }
    }
    return score;
  }
  float DotDecoded(const float *query, const Scratch &scratch) const {
    return Decode(query, nullptr, scratch);
  }

  saqlib::SaqData data_;
  Mapping codes_;
  SaqCodeHeader header_{};
  const uint8_t *code_begin_ = nullptr;
  std::vector<Segment> segments_;
};

class GenericPrimary {
 public:
  using QueryContext = ours::GenericPrimaryIndex::QueryContext;
  using PairContext = ours::GenericPrimaryIndex::QueryContext;
  struct Scratch {
    std::vector<float> reconstruction;
    explicit Scratch(uint32_t dim = 0) : reconstruction(dim) {}
  };

  GenericPrimary(const std::string &model_path, const std::string &codes_path)
      : index_(ours::GenericPrimaryIndex::Load(model_path, codes_path)) {}
  uint64_t nodes() const { return index_.nodes(); }
  uint32_t original_dim() const { return index_.original_dim(); }
  uint32_t scorer_dim() const { return index_.scorer_dim(); }
  uint64_t storage_bytes() const { return index_.storage_bytes(); }
  Scratch MakeScratch() const { return Scratch(scorer_dim()); }
  QueryContext PrepareQuery(const float *query, uint32_t dim) const {
    return index_.Prepare(query, dim);
  }
  QueryContext Prepare(const float *query, uint32_t dim) const {
    return index_.Prepare(query, dim);
  }
  PairContext PreparePair(const float *query) const {
    return index_.PrepareTransformed(query, scorer_dim());
  }
  const float *Transformed(const QueryContext &context) const {
    return context.transformed.data();
  }
  float Score(QueryContext &context, uint32_t id) const {
    return index_.Score(context, id);
  }
  void ScoreBatch(QueryContext &context, const uint32_t *ids, size_t count,
                  float *scores) const {
    index_.ScoreBatch(context, ids, count, scores);
  }
  float PairScore(PairContext &context, Scratch &, uint32_t id) const {
    return index_.Score(context, id);
  }
  void PairScoreBatch(PairContext &context, Scratch &, const uint32_t *ids,
                      size_t count, float *scores) const {
    index_.ScoreBatch(context, ids, count, scores);
  }
  void Reconstruct(uint32_t id, float *output, Scratch &) const {
    index_.Reconstruct(id, output, scorer_dim());
  }

 private:
  ours::GenericPrimaryIndex index_;
};

struct Arguments {
  std::string primary_kind;
  std::string primary_model;
  std::string primary_codes;
  std::string rabitq_config = "exact";
  std::string ji_model;
  std::string ji_side_codes;
  std::string queries;
  std::string knn;
  std::string graph;
  std::string summary;
  uint64_t query_start = 0;
  uint64_t query_limit = 0;
  uint32_t topk = 100;
  uint32_t m_pjbp = 35;
  uint32_t l_pjpq = 500;
  uint32_t threads = 1;
  uint32_t score_batch = 4096;
  uint32_t self_check = 32;
  bool disable_ji = false;
  bool force_knn = false;
  bool validate_only = false;
};

std::string Value(int argc, char **argv, int &index) {
  if (++index >= argc) throw std::invalid_argument("missing option value");
  return argv[index];
}

Arguments Parse(int argc, char **argv) {
  Arguments a;
  a.threads = std::max(1, omp_get_num_procs());
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--primary-kind") a.primary_kind = Value(argc, argv, i);
    else if (key == "--primary-model") a.primary_model = Value(argc, argv, i);
    else if (key == "--primary-codes") a.primary_codes = Value(argc, argv, i);
    else if (key == "--rabitq-config") a.rabitq_config = Value(argc, argv, i);
    else if (key == "--ji-model") a.ji_model = Value(argc, argv, i);
    else if (key == "--ji-side-codes") a.ji_side_codes = Value(argc, argv, i);
    else if (key == "--queries") a.queries = Value(argc, argv, i);
    else if (key == "--knn") a.knn = Value(argc, argv, i);
    else if (key == "--graph") a.graph = Value(argc, argv, i);
    else if (key == "--summary") a.summary = Value(argc, argv, i);
    else if (key == "--query-start") a.query_start = std::stoull(Value(argc, argv, i));
    else if (key == "--query-limit") a.query_limit = std::stoull(Value(argc, argv, i));
    else if (key == "--topk") a.topk = std::stoul(Value(argc, argv, i));
    else if (key == "--M_pjbp") a.m_pjbp = std::stoul(Value(argc, argv, i));
    else if (key == "--L_pjpq") a.l_pjpq = std::stoul(Value(argc, argv, i));
    else if (key == "--threads") a.threads = std::stoul(Value(argc, argv, i));
    else if (key == "--score-batch") a.score_batch = std::stoul(Value(argc, argv, i));
    else if (key == "--self-check") a.self_check = std::stoul(Value(argc, argv, i));
    else if (key == "--disable-ji") a.disable_ji = true;
    else if (key == "--force-knn") a.force_knn = true;
    else if (key == "--validate-only") a.validate_only = true;
    else if (key == "--help" || key == "-h") {
      std::cout << "Code-native primary + JI-Pure RoarGraph builder\n"
                   "Required: --primary-kind --primary-model --primary-codes "
                   "--queries --knn --graph --summary\n"
                   "Optional: --validate-only checks code/decode parity and exits\n";
      std::exit(0);
    } else throw std::invalid_argument("unknown option: " + key);
  }
  if (a.primary_kind.empty() || a.primary_model.empty() ||
      a.primary_codes.empty() || a.queries.empty() || a.knn.empty() ||
      a.graph.empty() || a.summary.empty() || a.query_limit == 0 ||
      a.topk == 0 || a.threads == 0 || a.score_batch == 0)
    throw std::invalid_argument("missing or invalid required arguments");
  if (!a.disable_ji && (a.ji_model.empty() || a.ji_side_codes.empty()))
    throw std::invalid_argument("JI files required unless --disable-ji is set");
  return a;
}

struct Scored {
  float score;
  uint32_t id;
};
bool Better(const Scored &a, const Scored &b) {
  return a.score > b.score || (a.score == b.score && a.id < b.id);
}
struct BetterComparator {
  bool operator()(const Scored &a, const Scored &b) const { return Better(a, b); }
};

void WriteKnn(const std::string &path, uint32_t rows, uint32_t cols,
              const std::vector<uint32_t> &ids) {
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("cannot create KNN: " + path);
  output.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  output.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
  output.write(reinterpret_cast<const char *>(ids.data()),
               static_cast<std::streamsize>(ids.size() * sizeof(uint32_t)));
  if (!output) throw std::runtime_error("failed writing KNN");
}

template <class Primary>
double BuildKnn(const Primary &primary, const ours::JiOnlineIndex *ji,
                const ours::MappedFbin &queries, uint64_t query_start,
                uint32_t query_count, uint32_t topk, uint32_t batch,
                uint32_t threads, const std::string &path) {
  std::vector<uint32_t> result(static_cast<size_t>(query_count) * topk);
  omp_set_num_threads(static_cast<int>(threads));
  const auto started = Clock::now();
#pragma omp parallel for schedule(dynamic, 1)
  for (uint32_t qi = 0; qi < query_count; ++qi) {
    auto primary_query = primary.Prepare(queries.Row(query_start + qi),
                                         queries.dim());
    const std::optional<ours::JiQueryContext> ji_query =
        ji ? std::optional<ours::JiQueryContext>(ji->PrepareQuery(
                 primary.Transformed(primary_query), primary.scorer_dim()))
           : std::nullopt;
    std::priority_queue<Scored, std::vector<Scored>, BetterComparator> heap;
    std::vector<uint32_t> ids(batch);
    std::vector<float> scores(batch), corrections(batch);
    for (uint64_t begin = 0; begin < primary.nodes(); begin += batch) {
      const uint32_t count = static_cast<uint32_t>(
          std::min<uint64_t>(batch, primary.nodes() - begin));
      for (uint32_t i = 0; i < count; ++i) ids[i] = begin + i;
      primary.ScoreBatch(primary_query, ids.data(), count, scores.data());
      if (ji) {
        ji->ComputeCorrectionsBatch(*ji_query, ids.data(), count,
                                    corrections.data());
        for (uint32_t i = 0; i < count; ++i) scores[i] += corrections[i];
      }
      for (uint32_t i = 0; i < count; ++i) {
        const Scored value{scores[i], ids[i]};
        if (heap.size() < topk) heap.push(value);
        else if (Better(value, heap.top())) {
          heap.pop();
          heap.push(value);
        }
      }
    }
    std::vector<Scored> row;
    row.reserve(topk);
    while (!heap.empty()) {
      row.push_back(heap.top());
      heap.pop();
    }
    std::sort(row.begin(), row.end(), Better);
    for (uint32_t k = 0; k < topk; ++k)
      result[static_cast<size_t>(qi) * topk + k] = row[k].id;
  }
  const double seconds =
      std::chrono::duration<double>(Clock::now() - started).count();
  WriteKnn(path, query_count, topk, result);
  return seconds;
}

struct ParityCheck {
  uint32_t samples = 0;
  double query_max_abs_error = 0.0;
  double pair_max_abs_error = 0.0;
};

template <class Primary>
ParityCheck CheckCodeSemantics(const Primary &primary,
                               const ours::JiOnlineIndex *ji,
                               const ours::MappedFbin &queries,
                               uint64_t query_start, uint32_t samples) {
  ParityCheck check;
  check.samples = samples;
  if (samples == 0) return check;
  std::mt19937_64 generator(20260915);
  std::uniform_int_distribution<uint32_t> nodes(
      0, static_cast<uint32_t>(primary.nodes() - 1));
  auto scratch = primary.MakeScratch();
  std::vector<float> decoded(primary.scorer_dim());
  std::vector<float> residual(primary.scorer_dim());
  std::vector<float> anchor(primary.scorer_dim());
  for (uint32_t sample = 0; sample < samples; ++sample) {
    const uint32_t node = nodes(generator);
    auto query = primary.Prepare(queries.Row(query_start + sample %
                                             (queries.rows() - query_start)),
                                 queries.dim());
    float code_score = primary.Score(query, node);
    primary.Reconstruct(node, decoded.data(), scratch);
    if (ji) {
      auto ji_query = ji->PrepareQuery(primary.Transformed(query),
                                       primary.scorer_dim());
      code_score += ji->Correction(ji_query, node);
      ji->ReconstructNode(node, residual.data(), primary.scorer_dim());
      for (uint32_t d = 0; d < primary.scorer_dim(); ++d)
        decoded[d] += residual[d];
    }
    double reconstructed_score = 0.0;
    for (uint32_t d = 0; d < primary.scorer_dim(); ++d)
      reconstructed_score += static_cast<double>(primary.Transformed(query)[d]) *
                             decoded[d];
    check.query_max_abs_error = std::max(
        check.query_max_abs_error,
        std::abs(static_cast<double>(code_score) - reconstructed_score));

    const uint32_t lhs = nodes(generator), rhs = nodes(generator);
    primary.Reconstruct(rhs, anchor.data(), scratch);
    if (ji) {
      ji->ReconstructNode(rhs, residual.data(), primary.scorer_dim());
      for (uint32_t d = 0; d < primary.scorer_dim(); ++d)
        anchor[d] += residual[d];
    }
    auto pair = primary.PreparePair(anchor.data());
    float pair_score = primary.PairScore(pair, scratch, lhs);
    if (ji) {
      auto ji_pair = ji->PrepareQuery(anchor.data(), primary.scorer_dim());
      pair_score += ji->Correction(ji_pair, lhs);
    }
    primary.Reconstruct(lhs, decoded.data(), scratch);
    if (ji) {
      ji->ReconstructNode(lhs, residual.data(), primary.scorer_dim());
      for (uint32_t d = 0; d < primary.scorer_dim(); ++d)
        decoded[d] += residual[d];
    }
    double pair_reconstructed = 0.0;
    for (uint32_t d = 0; d < primary.scorer_dim(); ++d)
      pair_reconstructed += static_cast<double>(anchor[d]) * decoded[d];
    check.pair_max_abs_error = std::max(
        check.pair_max_abs_error,
        std::abs(static_cast<double>(pair_score) - pair_reconstructed));
  }
  return check;
}

template <class Primary>
class BuildOracle final : public efanna2e::BaseBuildDistanceOracle {
 public:
  BuildOracle(const Primary &primary, const ours::JiOnlineIndex *ji,
              uint32_t threads)
      : primary_(primary), ji_(ji) {
    workspaces_.reserve(threads);
    for (uint32_t i = 0; i < threads; ++i)
      workspaces_.push_back(std::make_unique<Workspace>(primary_));
  }
  uint64_t NodeCount() const override { return primary_.nodes(); }
  uint32_t Dimension() const override { return primary_.scorer_dim(); }
  float Compare(uint32_t lhs, uint32_t rhs) const override {
    PrepareAnchor(rhs);
    ++Current().direct;
    return ComparePrepared(lhs, rhs);
  }
  void PrepareAnchor(uint32_t anchor) const override {
    Workspace &w = Current();
    ReconstructCombined(anchor, w.anchor.data(), w);
    w.pair.emplace(primary_.PreparePair(w.anchor.data()));
    if (ji_)
      w.ji_query.emplace(ji_->PrepareQuery(w.anchor.data(), Dimension()));
    else
      w.ji_query.reset();
    w.anchor_id = anchor;
    ++w.anchors;
  }
  float ComparePrepared(uint32_t candidate, uint32_t anchor) const override {
    Workspace &w = Current();
    CheckAnchor(w, anchor);
    ++w.prepared;
    float score = primary_.PairScore(*w.pair, w.primary_scratch, candidate);
    if (ji_) score += ji_->Correction(*w.ji_query, candidate);
    return -score;
  }
  void ComparePreparedBatch(const uint32_t *candidates, size_t count,
                            uint32_t anchor, float *distances) const override {
    Workspace &w = Current();
    CheckAnchor(w, anchor);
    primary_.PairScoreBatch(*w.pair, w.primary_scratch, candidates, count,
                            distances);
    if (ji_) {
      w.corrections.resize(count);
      ji_->ComputeCorrectionsBatch(*w.ji_query, candidates, count,
                                   w.corrections.data());
    }
    for (size_t i = 0; i < count; ++i)
      distances[i] = -(distances[i] + (ji_ ? w.corrections[i] : 0.0f));
    w.prepared += count;
  }
  uint32_t FindEntryPoint() const override {
    std::vector<double> mean(Dimension(), 0.0);
#pragma omp parallel
    {
      Workspace &w = Current();
      std::vector<double> local(Dimension(), 0.0);
#pragma omp for schedule(static)
      for (uint64_t node = 0; node < NodeCount(); ++node) {
        ReconstructCombined(static_cast<uint32_t>(node), w.anchor.data(), w);
        for (uint32_t d = 0; d < Dimension(); ++d) local[d] += w.anchor[d];
      }
#pragma omp critical
      for (uint32_t d = 0; d < Dimension(); ++d) mean[d] += local[d];
    }
    for (double &value : mean) value /= NodeCount();
    uint32_t best = 0;
    double best_distance = std::numeric_limits<double>::infinity();
#pragma omp parallel
    {
      Workspace &w = Current();
      uint32_t local_best = 0;
      double local_distance = std::numeric_limits<double>::infinity();
#pragma omp for schedule(static)
      for (uint64_t node = 0; node < NodeCount(); ++node) {
        ReconstructCombined(static_cast<uint32_t>(node), w.anchor.data(), w);
        double distance = 0.0;
        for (uint32_t d = 0; d < Dimension(); ++d) {
          const double delta = mean[d] - w.anchor[d];
          distance += delta * delta;
        }
        if (distance < local_distance ||
            (distance == local_distance && node < local_best)) {
          local_distance = distance;
          local_best = static_cast<uint32_t>(node);
        }
      }
#pragma omp critical
      if (local_distance < best_distance ||
          (local_distance == best_distance && local_best < best)) {
        best_distance = local_distance;
        best = local_best;
      }
    }
    entry_distance_ = best_distance;
    return best;
  }
  uint64_t Anchors() const { return Sum(&Workspace::anchors); }
  uint64_t Prepared() const { return Sum(&Workspace::prepared); }
  uint64_t Direct() const { return Sum(&Workspace::direct); }
  double entry_distance() const { return entry_distance_; }

 private:
  struct Workspace {
    explicit Workspace(const Primary &primary)
        : primary_scratch(primary.MakeScratch()),
          anchor(primary.scorer_dim()), residual(primary.scorer_dim()) {}
    typename Primary::Scratch primary_scratch;
    std::vector<float> anchor;
    std::vector<float> residual;
    std::vector<float> corrections;
    std::optional<typename Primary::PairContext> pair;
    std::optional<ours::JiQueryContext> ji_query;
    uint32_t anchor_id = std::numeric_limits<uint32_t>::max();
    uint64_t anchors = 0, prepared = 0, direct = 0;
  };
  Workspace &Current() const {
    const uint32_t thread = omp_in_parallel() ? omp_get_thread_num() : 0;
    if (thread >= workspaces_.size()) throw std::runtime_error("workspace overflow");
    return *workspaces_[thread];
  }
  void CheckAnchor(const Workspace &w, uint32_t anchor) const {
    if (!w.pair || w.anchor_id != anchor)
      throw std::runtime_error("prepared anchor mismatch");
  }
  void ReconstructCombined(uint32_t node, float *output, Workspace &w) const {
    primary_.Reconstruct(node, output, w.primary_scratch);
    if (ji_) {
      ji_->ReconstructNode(node, w.residual.data(), Dimension());
      for (uint32_t d = 0; d < Dimension(); ++d) output[d] += w.residual[d];
    }
  }
  uint64_t Sum(uint64_t Workspace::*member) const {
    uint64_t total = 0;
    for (const auto &w : workspaces_) total += (*w).*member;
    return total;
  }
  const Primary &primary_;
  const ours::JiOnlineIndex *ji_;
  mutable std::vector<std::unique_ptr<Workspace>> workspaces_;
  mutable double entry_distance_ = 0.0;
};

uint64_t PeakRss() {
  struct rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
  return static_cast<uint64_t>(usage.ru_maxrss) * 1024ULL;
}

std::string Escape(const std::string &s) {
  std::string out;
  for (char c : s) {
    if (c == '\\' || c == '"') out += '\\';
    out += c;
  }
  return out;
}

template <class Primary>
int Run(const Arguments &a, Primary &primary) {
  ours::MappedFbin queries(a.queries);
  if (queries.dim() != primary.original_dim() ||
      a.query_start + a.query_limit > queries.rows())
    throw std::invalid_argument("query range/dimension mismatch");
  std::optional<ours::JiOnlineIndex> ji;
  if (!a.disable_ji) {
    ji.emplace(ours::JiOnlineIndex::Load(a.ji_model, a.ji_side_codes));
    if (ji->node_count() != primary.nodes() || ji->dim() != primary.scorer_dim())
      throw std::invalid_argument("JI and primary metadata mismatch");
  }

  const ParityCheck parity = CheckCodeSemantics(
      primary, ji ? &*ji : nullptr, queries, a.query_start, a.self_check);
  // The code scorers use float reductions while the audit uses a double dot
  // product.  This tolerance catches layout/decoder errors without rejecting
  // harmless reduction-order differences.
  if (parity.query_max_abs_error > 1e-4 || parity.pair_max_abs_error > 1e-4)
    throw std::runtime_error(
        "code-native scorer/reconstruction parity failed: query=" +
        std::to_string(parity.query_max_abs_error) + " pair=" +
        std::to_string(parity.pair_max_abs_error));
  if (a.validate_only) {
    std::cout << std::setprecision(17)
              << "{\"validated\":true,\"samples\":" << parity.samples
              << ",\"query_max_absolute_error\":"
              << parity.query_max_abs_error
              << ",\"base_pair_max_absolute_error\":"
              << parity.pair_max_abs_error << "}" << std::endl;
    return 0;
  }

  double knn_seconds = 0.0;
  if (a.force_knn || !std::filesystem::exists(a.knn) ||
      std::filesystem::file_size(a.knn) == 0) {
    knn_seconds = BuildKnn(primary, ji ? &*ji : nullptr, queries, a.query_start,
                           static_cast<uint32_t>(a.query_limit), a.topk,
                           a.score_batch, a.threads, a.knn);
  }

  BuildOracle<Primary> oracle(primary, ji ? &*ji : nullptr, a.threads);
  omp_set_num_threads(static_cast<int>(a.threads));
  efanna2e::Parameters parameters;
  parameters.Set<uint32_t>("M_sq", a.topk);
  parameters.Set<uint32_t>("M_pjbp", a.m_pjbp);
  parameters.Set<uint32_t>("L_pjpq", a.l_pjpq);
  parameters.Set<uint32_t>("num_threads", a.threads);
  efanna2e::IndexBipartite graph(primary.scorer_dim(),
      primary.nodes() + a.query_limit, efanna2e::INNER_PRODUCT, nullptr);
  graph.SetBaseBuildDistanceOracle(&oracle);
  graph.LoadLearnBaseKNN(a.knn.c_str());
  const auto graph_started = Clock::now();
  graph.BuildRoarGraph(a.query_limit, nullptr, primary.nodes(), nullptr,
                       parameters);
  const double graph_seconds =
      std::chrono::duration<double>(Clock::now() - graph_started).count();
  std::filesystem::create_directories(std::filesystem::path(a.graph).parent_path());
  graph.SaveProjectionGraph(a.graph.c_str());

  std::filesystem::create_directories(std::filesystem::path(a.summary).parent_path());
  std::ofstream out(a.summary);
  out << std::setprecision(17)
      << "{\n  \"version\": 1,\n"
      << "  \"method\": \"code_native_primary_ji_pure_roargraph_build\",\n"
      << "  \"primary_kind\": \"" << Escape(a.primary_kind) << "\",\n"
      << "  \"representation\": \"" << (ji ? "primary+ji-pure" : "primary") << "\",\n"
      << "  \"query_base_relation\": \"direct_primary_and_ji_code_scoring\",\n"
      << "  \"base_base_comparison\": \"on_demand_anchor_decode_then_direct_candidate_code_scoring\",\n"
      << "  \"base_vectors_loaded\": false,\n"
      << "  \"materialized_base_vectors\": false,\n"
      << "  \"code_native_construction\": true,\n"
      << "  \"nodes\": " << primary.nodes() << ",\n"
      << "  \"original_dim\": " << primary.original_dim() << ",\n"
      << "  \"scorer_dim\": " << primary.scorer_dim() << ",\n"
      << "  \"build_queries\": " << a.query_limit << ",\n"
      << "  \"query_start\": " << a.query_start << ",\n"
      << "  \"query_base_topk\": " << a.topk << ",\n"
      << "  \"threads\": " << a.threads << ",\n"
      << "  \"peak_rss_bytes\": " << PeakRss() << ",\n"
      << "  \"primary_storage_bytes\": " << primary.storage_bytes() << ",\n"
      << "  \"ji_storage_bytes\": " << (ji ? ji->storage_bytes() : 0) << ",\n"
      << "  \"graph_bytes\": " << std::filesystem::file_size(a.graph) << ",\n"
      << "  \"entry_point_squared_distance\": " << oracle.entry_distance() << ",\n"
      << "  \"parity\": {\"samples\": " << parity.samples
      << ", \"query_max_absolute_error\": " << parity.query_max_abs_error
      << ", \"base_pair_max_absolute_error\": "
      << parity.pair_max_abs_error << "},\n"
      << "  \"comparisons\": {\"prepared_anchors\": " << oracle.Anchors()
      << ", \"prepared\": " << oracle.Prepared() << ", \"direct\": "
      << oracle.Direct() << "},\n"
      << "  \"seconds\": {\"query_base_knn\": " << knn_seconds
      << ", \"roargraph_build\": " << graph_seconds << "},\n"
      << "  \"files\": {\"primary_model\": \"" << Escape(a.primary_model)
      << "\", \"primary_codes\": \"" << Escape(a.primary_codes)
      << "\", \"ji_model\": \"" << Escape(a.ji_model)
      << "\", \"ji_side_codes\": \"" << Escape(a.ji_side_codes)
      << "\", \"query_base_knn\": \"" << Escape(a.knn)
      << "\", \"graph\": \"" << Escape(a.graph) << "\"}\n}\n";
  std::cout << "code-native " << (ji ? "primary+JI-Pure" : "primary")
            << " graph built: " << a.graph << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Arguments args = Parse(argc, argv);
    if (args.primary_kind == "extended-rabitq") {
      RabitqPrimary primary(args.primary_model, args.rabitq_config);
      return Run(args, primary);
    }
    if (args.primary_kind == "saq") {
      SaqPrimary primary(args.primary_model, args.primary_codes);
      return Run(args, primary);
    }
    if (args.primary_kind == "opq-pq" || args.primary_kind == "qaq" ||
        args.primary_kind == "leanvec-ood") {
      GenericPrimary primary(args.primary_model, args.primary_codes);
      return Run(args, primary);
    }
    throw std::invalid_argument(
        "unsupported code-native primary kind");
  } catch (const std::exception &error) {
    std::cerr << "code-native build failed: " << error.what() << std::endl;
    return 1;
  }
}
