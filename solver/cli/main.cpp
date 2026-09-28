#include "sovereign/engine.hpp"
#include "sovereign/json_io.hpp"
#include "sovereign/mps_io.hpp"
#include "sovereign/verifier.hpp"
#include "sovereign/gpu_spmv.hpp"
#include "sovereign/cuda_driver.hpp"
#include <nlohmann/json.hpp>
#include <cstdlib>

#include <fstream>
#include <iostream>
#include <string>

namespace {

void print_usage() {
  std::cerr << "Usage:\n"
            << "  sovereign solve <model.json|model.mps> [--verify] [--out <result.json>]\n"
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
    const std::string device = sovereign::requested_device();
    if (device == "cuda" && !sovereign::gpu_available()) {
      throw std::runtime_error("CUDA requested but unavailable: " + sovereign::cuda::device_info().reason);
    }
    sovereign::reset_gpu_operations();
    // Dispatch on the file extension so .mps and .json both work. Reading an
    // MPS file through the JSON reader produced a raw nlohmann parse exception,
    // which told the user nothing about what was actually wrong.
    const auto model = sovereign::load_model_from_file(model_path);
    sovereign::OptimizationEngine engine;
    sovereign::EngineOptions options;
    const char* presolve = std::getenv("SOVEREIGN_PRESOLVE");
    options.presolve = !presolve || std::string(presolve) != "0";
    const auto result = engine.solve(model, options);
    auto payload = nlohmann::json::parse(sovereign::result_to_json_string(result));
    payload["requested_device"] = device;
    payload["gpu_operations"] = sovereign::gpu_operations();
    payload["gpu_factorizations"] = sovereign::gpu_factorizations();
    payload["gpu_used"] = sovereign::gpu_operations() > 0;
    const std::string result_json = payload.dump(2);
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
  if (cmd == "capabilities") {
    const auto& gpu = sovereign::cuda::device_info();
    nlohmann::json caps = {{"version", "1.2.0"},
                           {"cuda_available", gpu.available},
                           {"cuda_backend", "driver-api"},
                           {"cuda_reason", gpu.reason},
                           {"gpu_acceleration", "dense-lu,sparse-matrix-vector"}};
    if (!gpu.name.empty()) {
      caps["gpu_name"] = gpu.name;
      caps["gpu_compute_capability"] =
          std::to_string(gpu.compute_major) + "." + std::to_string(gpu.compute_minor);
      caps["gpu_memory_mb"] = static_cast<unsigned long long>(gpu.memory_bytes >> 20);
    }
    if (gpu.driver_version > 0) {
      caps["cuda_driver_version"] =
          std::to_string(gpu.driver_version / 1000) + "." + std::to_string(gpu.driver_version % 1000 / 10);
    }
    std::cout << caps.dump() << "\n";
    return 0;
  }
  if (cmd == "version") {
    return cmd_version();
  }
  if (cmd == "convert" && argc == 3) {
    try {
      std::cout << sovereign::model_to_json_string(sovereign::load_model_from_file(argv[2])) << "\n";
      return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
  }
  if (cmd == "solve") {
    return cmd_solve(argc, argv);
  }

  print_usage();
  return 2;
}
