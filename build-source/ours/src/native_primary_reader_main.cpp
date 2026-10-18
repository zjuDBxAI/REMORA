// Offline decoder bridge. Reuse the exact construction adapters, not a new
// quantizer. No graph is built and no full decoded database is materialized.
#define main unused_primary_builder_main
#include "primary_ji_build_main.cpp"
#undef main

namespace {
void ReadExact(void* data, size_t bytes) {
  std::cin.read(static_cast<char*>(data), bytes);
  if (!std::cin) throw std::runtime_error("truncated decoder request");
}
void WriteExact(const void* data, size_t bytes) {
  std::cout.write(static_cast<const char*>(data), bytes);
  if (!std::cout) throw std::runtime_error("decoder response pipe closed");
}

template<class Primary>
int Serve(Primary& primary) {
  const uint64_t nodes = primary.nodes();
  const uint32_t dim = primary.original_dim(), transformed_dim = primary.scorer_dim();
  std::vector<float> projection(static_cast<size_t>(dim) * transformed_dim);
  std::vector<float> basis(dim, 0.0f);
  for (uint32_t j = 0; j < dim; ++j) {
    basis[j] = 1.0f;
    auto context = primary.Prepare(basis.data(), dim);
    std::copy(primary.Transformed(context), primary.Transformed(context) + transformed_dim,
              projection.data() + static_cast<size_t>(j) * transformed_dim);
    basis[j] = 0.0f;
  }
  const char magic[8] = {'R','M','P','R','D','0','1','\0'};
  WriteExact(magic, sizeof(magic));
  WriteExact(&nodes, sizeof(nodes));
  WriteExact(&dim, sizeof(dim));
  WriteExact(&transformed_dim, sizeof(transformed_dim));
  WriteExact(projection.data(), projection.size() * sizeof(float));
  std::cout.flush();
  while (true) {
    uint32_t operation = 0, count = 0;
    std::cin.read(reinterpret_cast<char*>(&operation), sizeof(operation));
    if (std::cin.eof()) break;
    if (!std::cin) throw std::runtime_error("invalid decoder operation");
    if (operation == 0) break;
    if (operation != 1 && operation != 2) throw std::runtime_error("unknown decoder operation");
    ReadExact(&count, sizeof(count));
    if (count == 0 || count > 1048576) throw std::runtime_error("invalid decoder batch size");
    std::vector<float> query;
    if (operation == 2) {
      query.resize(dim);
      ReadExact(query.data(), query.size() * sizeof(float));
    }
    std::vector<uint32_t> ids(count);
    ReadExact(ids.data(), ids.size() * sizeof(uint32_t));
    for (uint32_t id : ids) if (id >= nodes) throw std::runtime_error("decoder node out of range");
    if (operation == 1) {
      std::vector<float> values(static_cast<size_t>(count) * transformed_dim);
      #pragma omp parallel
      {
        auto scratch = primary.MakeScratch();
        #pragma omp for schedule(static)
        for (uint32_t j = 0; j < count; ++j)
          primary.Reconstruct(ids[j], values.data() + static_cast<size_t>(j) * transformed_dim, scratch);
      }
      WriteExact(values.data(), values.size() * sizeof(float));
    } else {
      auto context = primary.Prepare(query.data(), dim);
      std::vector<float> values(count);
      primary.ScoreBatch(context, ids.data(), ids.size(), values.data());
      WriteExact(values.data(), values.size() * sizeof(float));
    }
    std::cout.flush();
  }
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 5) throw std::runtime_error("usage: reader KIND MODEL CODES THREADS");
    const std::string kind(argv[1]);
    const int threads = std::stoi(argv[4]);
    if (threads <= 0) throw std::runtime_error("threads must be positive");
    omp_set_num_threads(threads);
    // Library startup messages must not contaminate the binary protocol.
    auto* original = std::cout.rdbuf(std::cerr.rdbuf());
    if (kind == "extended-rabitq") {
      RabitqPrimary primary(argv[3], "exact");
      std::cout.rdbuf(original);
      return Serve(primary);
    }
    if (kind == "saq") {
      SaqPrimary primary(argv[2], argv[3]);
      std::cout.rdbuf(original);
      return Serve(primary);
    }
    if (kind != "generic") throw std::runtime_error("unsupported reader kind");
    GenericPrimary primary(argv[2], argv[3]);
    std::cout.rdbuf(original);
    return Serve(primary);
  } catch (const std::exception& error) {
    std::cerr << "native-primary-reader: " << error.what() << '\n';
    return 1;
  }
}
