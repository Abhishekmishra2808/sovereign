#include "sovereign/engine.hpp"
#include "sovereign/json_io.hpp"
#include "sovereign/verifier.hpp"

#include <fstream>
#include <iostream>
#include <string>

namespace {

void print_usage() {
  std::cerr << "Usage:\n"
            << "  sovereign solve <model.json> [--verify] [--out <result.json>]\n"
            << "  sovereign version\n";
}

int cmd_version() {
  std::cout << "sovereign 1.0.0 (LP/MILP/QP + presolve/cuts/heuristics)\n";
  return 0;
}

int cmd_solve(int argc, char** argv) {
  if (argc < 3) {
    print_usage();
    return 2;
  }

  std::string model_path = argv[2];
  bool do_verify = false;
  std::string out_path;

  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--verify") {
      do_verify = true;
    } else if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      print_usage();
      return 2;
    }
  }

  try {
    const auto model = sovereign::load_model_from_json_file(model_path);
    sovereign::OptimizationEngine engine;
    const auto result = engine.solve(model);
    const std::string result_json = sovereign::result_to_json_string(result);
    std::cout << result_json << "\n";

    if (!out_path.empty()) {
      std::ofstream out(out_path);
      if (!out) {
        std::cerr << "Failed to write " << out_path << "\n";
        return 1;
      }
      out << result_json << "\n";
    }

    if (do_verify) {
      sovereign::SolutionVerifier verifier;
      const auto verification = verifier.verify(model, result);
      std::cout << "\nverification:\n"
                << sovereign::verification_to_json_string(verification) << "\n";
    }

    return result.status == sovereign::SolverStatus::Error ? 1 : 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    return 2;
  }

  const std::string cmd = argv[1];
  if (cmd == "version") {
    return cmd_version();
  }
  if (cmd == "solve") {
    return cmd_solve(argc, argv);
  }

  print_usage();
  return 2;
}
