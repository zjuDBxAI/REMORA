#include <boost/program_options.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace po = boost::program_options;

namespace {

template <typename T>
T Read(std::ifstream& input, const char* name) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) throw std::runtime_error(std::string("truncated ") + name);
    return value;
}

void ReadNeighbors(std::ifstream& input, std::vector<uint32_t>* values) {
    const uint32_t degree = Read<uint32_t>(input, "graph degree");
    values->resize(degree);
    if (degree != 0) {
        input.read(reinterpret_cast<char*>(values->data()),
                   static_cast<std::streamsize>(degree * sizeof(uint32_t)));
        if (!input) throw std::runtime_error("truncated graph neighbors");
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string reference_path;
    std::string candidate_path;
    std::string output_path;
    po::options_description options("Arguments");
    options.add_options()
        ("help,h", "Print command-line help")
        ("reference", po::value<std::string>(&reference_path)->required(),
         "Reference RoarGraph index")
        ("candidate", po::value<std::string>(&candidate_path)->required(),
         "Candidate RoarGraph index")
        ("out_json", po::value<std::string>(&output_path)->required(),
         "Comparison JSON output");
    try {
        po::variables_map variables;
        po::store(po::parse_command_line(argc, argv, options), variables);
        if (variables.count("help")) {
            std::cout << options;
            return 0;
        }
        po::notify(variables);

        std::ifstream reference(reference_path, std::ios::binary);
        std::ifstream candidate(candidate_path, std::ios::binary);
        if (!reference || !candidate) {
            throw std::runtime_error("cannot open graph input");
        }
        std::vector<char> reference_buffer(4U << 20U);
        std::vector<char> candidate_buffer(4U << 20U);
        reference.rdbuf()->pubsetbuf(reference_buffer.data(),
                                     reference_buffer.size());
        candidate.rdbuf()->pubsetbuf(candidate_buffer.data(),
                                     candidate_buffer.size());

        const uint32_t reference_entry = Read<uint32_t>(reference, "entry point");
        const uint32_t reference_nodes = Read<uint32_t>(reference, "node count");
        const uint32_t candidate_entry = Read<uint32_t>(candidate, "entry point");
        const uint32_t candidate_nodes = Read<uint32_t>(candidate, "node count");
        if (reference_nodes != candidate_nodes) {
            throw std::runtime_error("graph node counts differ");
        }

        uint64_t exact_rows = 0;
        uint64_t exact_position_edges = 0;
        uint64_t reference_edges = 0;
        uint64_t candidate_edges = 0;
        uint64_t intersection_edges = 0;
        std::vector<uint32_t> reference_neighbors;
        std::vector<uint32_t> candidate_neighbors;
        std::vector<uint32_t> sorted_reference;
        std::vector<uint32_t> sorted_candidate;
        std::vector<uint32_t> intersection;
        for (uint32_t node = 0; node < reference_nodes; ++node) {
            ReadNeighbors(reference, &reference_neighbors);
            ReadNeighbors(candidate, &candidate_neighbors);
            reference_edges += reference_neighbors.size();
            candidate_edges += candidate_neighbors.size();
            const size_t common_positions =
                std::min(reference_neighbors.size(), candidate_neighbors.size());
            for (size_t i = 0; i < common_positions; ++i) {
                exact_position_edges +=
                    reference_neighbors[i] == candidate_neighbors[i];
            }
            if (reference_neighbors == candidate_neighbors) {
                ++exact_rows;
                intersection_edges += reference_neighbors.size();
                continue;
            }
            sorted_reference = reference_neighbors;
            sorted_candidate = candidate_neighbors;
            std::sort(sorted_reference.begin(), sorted_reference.end());
            std::sort(sorted_candidate.begin(), sorted_candidate.end());
            intersection.clear();
            std::set_intersection(
                sorted_reference.begin(), sorted_reference.end(),
                sorted_candidate.begin(), sorted_candidate.end(),
                std::back_inserter(intersection));
            intersection_edges += intersection.size();
        }

        const uint64_t union_edges =
            reference_edges + candidate_edges - intersection_edges;
        const std::filesystem::path output(output_path);
        if (!output.parent_path().empty()) {
            std::filesystem::create_directories(output.parent_path());
        }
        std::ofstream stream(output_path);
        if (!stream) throw std::runtime_error("cannot write graph comparison");
        stream << std::setprecision(17)
               << "{\n  \"version\": 1,\n"
               << "  \"method\": \"roargraph_ordered_and_set_edge_comparison\",\n"
               << "  \"nodes\": " << reference_nodes << ",\n"
               << "  \"entry_point_match\": "
               << (reference_entry == candidate_entry ? "true" : "false")
               << ",\n"
               << "  \"reference_entry_point\": " << reference_entry << ",\n"
               << "  \"candidate_entry_point\": " << candidate_entry << ",\n"
               << "  \"exact_ordered_rows\": " << exact_rows << ",\n"
               << "  \"exact_ordered_row_match_rate\": "
               << static_cast<double>(exact_rows) / reference_nodes << ",\n"
               << "  \"reference_edges\": " << reference_edges << ",\n"
               << "  \"candidate_edges\": " << candidate_edges << ",\n"
               << "  \"intersection_edges\": " << intersection_edges << ",\n"
               << "  \"reference_edge_retention\": "
               << static_cast<double>(intersection_edges) /
                      std::max<uint64_t>(reference_edges, 1)
               << ",\n"
               << "  \"edge_jaccard\": "
               << static_cast<double>(intersection_edges) /
                      std::max<uint64_t>(union_edges, 1)
               << ",\n"
               << "  \"ordered_edge_position_match_rate\": "
               << static_cast<double>(exact_position_edges) /
                      std::max<uint64_t>(reference_edges, 1)
               << ",\n"
               << "  \"files\": {\n"
               << "    \"reference\": \""
               << std::filesystem::absolute(reference_path).string() << "\",\n"
               << "    \"candidate\": \""
               << std::filesystem::absolute(candidate_path).string() << "\"\n"
               << "  }\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "graph comparison failed: " << error.what() << '\n';
        return 1;
    }
}
