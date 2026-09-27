// sovereign-server: the HTTP process.
//
// One job, one process: serve the bundled UI and the REST/SSE API, and hand
// every solve to JobManager. No optimization logic here.
//
// Security modes:
//   - Local mode (default): binds 127.0.0.1 only, validates Host/Origin headers
//   - Network mode (--allow-network): can bind to 0.0.0.0, supports API key auth
//     and CORS configuration for safe network access.

#include "sovereign/server/http_server.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

namespace fs = std::filesystem;

namespace {

std::atomic<bool> g_stop{false};

void on_signal(int) { g_stop.store(true); }

void print_usage() {
  std::cerr
      << "sovereign-server — API and UI host\n\n"
         "Usage: sovereign-server [options]\n"
         "  --port N          port to bind (default 0 = OS picks a free port)\n"
         "  --bind ADDR       bind address (default 127.0.0.1)\n"
         "  --allow-network   allow binding to non-loopback addresses\n"
         "  --api-key KEY     require API key for authentication\n"
         "  --cors-origin ORG allow CORS from specified origin\n"
         "  --static DIR      directory of built UI assets\n"
         "  --model-root DIR  directory the model browser may read from\n"
         "  --threads N       solve workers (default: one per hardware thread)\n"
         "  --print-port      print the bound port on stdout, then serve\n"
         "  --help\n\n"
         "The chosen port is printed to stdout as: SOVEREIGN_PORT=<n>\n"
         "The launcher reads that to know where to open the browser.\n\n"
         "Network access:\n"
         "  By default, the server binds to 127.0.0.1 only for security.\n"
         "  To expose to the network, use --bind 0.0.0.0 --allow-network.\n"
         "  When exposed, use --api-key for authentication.\n";
}

/** Walk up from the executable looking for the bundled asset directory. */
fs::path find_static_root(const fs::path& exe_dir, const std::string& override_dir) {
  if (!override_dir.empty()) return fs::absolute(override_dir);

  const fs::path candidates[] = {
      exe_dir / "static",
      exe_dir / ".." / "static",
      exe_dir / ".." / ".." / "api" / "static",
      exe_dir / "..\\static",
      fs::current_path() / "api" / "static",
      fs::current_path() / "static",
  };
  for (const auto& c : candidates) {
    std::error_code ec;
    if (fs::is_directory(c, ec)) return fs::weakly_canonical(c, ec);
  }
  return fs::path();
}

}  // namespace

int main(int argc, char** argv) {
  using namespace sovereign;

  server::ServerConfig config;
  std::string static_override;
  std::string model_root_override;
  bool print_port = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "error: " << what << " requires a value\n";
        std::exit(2);
      }
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    } else if (arg == "--port") {
      config.port = std::atoi(next("--port").c_str());
    } else if (arg == "--bind") {
      config.bind_address = next("--bind");
    } else if (arg == "--allow-network") {
      config.allow_network = true;
    } else if (arg == "--api-key") {
      config.api_key = next("--api-key");
    } else if (arg == "--cors-origin") {
      config.cors_origin = next("--cors-origin");
    } else if (arg == "--static") {
      static_override = next("--static");
    } else if (arg == "--model-root") {
      model_root_override = next("--model-root");
    } else if (arg == "--threads") {
      config.worker_threads = static_cast<unsigned>(std::atoi(next("--threads").c_str()));
    } else if (arg == "--print-port") {
      print_port = true;
    } else {
      std::cerr << "error: unknown argument '" << arg << "'\n";
      print_usage();
      return 2;
    }
  }

  // Security check: refuse non-loopback bind unless explicitly allowed
  if (!config.allow_network && config.bind_address != "127.0.0.1" &&
      config.bind_address != "localhost" && config.bind_address != "::1") {
    std::cerr << "error: refusing to bind '" << config.bind_address
              << "'. This server is loopback-only by design for security.\n"
                 "To expose to the network, use --bind 0.0.0.0 --allow-network.\n";
    return 2;
  }

  // Warn if exposing without authentication
  if (config.allow_network && config.api_key.empty()) {
    std::cerr << "WARNING: Server will be accessible from the network without API key authentication.\n"
                 "         This is a security risk. Use --api-key to protect the server.\n";
  }

  std::error_code ec;
  const fs::path exe_dir = fs::weakly_canonical(fs::path(argv[0]).parent_path(), ec);
  config.static_root = find_static_root(exe_dir, static_override).string();

  if (!model_root_override.empty()) {
    config.model_root = fs::absolute(model_root_override, ec).string();
  } else {
    // Default: the repo's example models, so a source checkout works out of the
    // box. A packaged release ships its own examples/ next to the binary.
    const fs::path candidates[] = {
        exe_dir / "examples" / "models",
        exe_dir / "..\\examples\\models",
        fs::current_path() / "examples" / "models",
    };
    for (const auto& c : candidates) {
      if (fs::is_directory(c, ec)) {
        config.model_root = fs::weakly_canonical(c, ec).string();
        break;
      }
    }
  }

  if (config.static_root.empty()) {
    std::cerr << "warning: no UI asset directory found. The API will work but the "
                 "browser will show 404. Build the UI with: cd web && npm run build\n";
  }

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  server::LocalServer srv(config);
  try {
    srv.start();
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  // This line is the launcher's contract. Keep the format stable.
  std::cout << "SOVEREIGN_PORT=" << srv.port() << std::endl;
  if (print_port) {
    std::cout.flush();
  }
  std::cerr << "Sovereign server listening on " << srv.base_url() << "\n";
  std::cerr << "  UI assets : " << (config.static_root.empty() ? "(none found)" : config.static_root) << "\n";
  std::cerr << "  model root: " << config.model_root << "\n";
  std::cerr << "  workers   : " << srv.jobs().worker_count() << "\n";
  if (config.allow_network) {
    std::cerr << "  network   : ENABLED (accessible from LAN)\n";
    if (!config.api_key.empty()) {
      std::cerr << "  auth      : API key required\n";
    } else {
      std::cerr << "  auth      : WARNING - no API key set (open access)\n";
    }
  } else {
    std::cerr << "  network   : loopback only (127.0.0.1)\n";
  }
  std::cerr << "Press Ctrl+C to stop.\n";

  while (!g_stop.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
  }

  std::cerr << "\nshutting down...\n";
  srv.stop();
  return 0;
}
