#include "sovereign/server/http_server.hpp"

#include "sovereign/engine.hpp"
#include "sovereign/json_io.hpp"
#include "sovereign/mps_io.hpp"
#include "sovereign/model_validator.hpp"
#include <nlohmann/json.hpp>
#include <iomanip>

#include <httplib.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

namespace sovereign {
namespace server {
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

namespace {

std::string json_escape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 16);
  for (char c : in) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

std::string progress_to_json(const JobProgress& p) {
  std::ostringstream oss;
      oss << std::setprecision(17);
  oss << "{"
      << "\"state\":\"" << to_string(p.state) << "\","
      << "\"runtime_seconds\":" << p.runtime_seconds << ","
      << "\"lp_iterations\":" << p.lp_iterations << ","
      << "\"nodes\":" << p.nodes << ",";
  if (p.has_incumbent) {
    oss << "\"incumbent\":" << p.incumbent << ",";
  } else {
    oss << "\"incumbent\":null,";
  }
  if (p.has_bound) {
    oss << "\"best_bound\":" << p.best_bound << ",";
  } else {
    oss << "\"best_bound\":null,";
  }
  oss << "\"mip_gap\":" << p.mip_gap << ","
      << "\"device\":\"" << json_escape(p.device) << "\","
      << "\"threads\":" << p.threads << ","
      << "\"peak_memory_bytes\":" << p.peak_memory_bytes << ",";
  if (p.has_presolve) {
    oss << "\"presolve\":{"
        << "\"rows_removed\":" << p.rows_removed << ","
        << "\"columns_removed\":" << p.cols_removed << ","
        << "\"variables_fixed\":" << p.vars_fixed << ","
        << "\"bounds_tightened\":" << p.bounds_tightened << "},";
  } else {
    oss << "\"presolve\":null,";
  }
  oss << "\"solver_status\":\"" << json_escape(p.solver_status) << "\","
      << "\"has_objective\":" << (p.has_objective ? "true" : "false") << ","
      << "\"objective\":" << (p.has_objective ? p.objective : 0.0) << ","
      << "\"message\":\"" << json_escape(p.message) << "\"";
  if (!p.solution_json.empty()) {
    oss << ",\"solution\":" << p.solution_json;
  }
  if (!p.verification_json.empty()) {
    oss << ",\"verification\":" << p.verification_json;
  }
  oss << "}";
  return oss.str();
}

/**
 * Extract a JSON string field from a flat request body.
 *
 * Handles escapes properly. This matters more than it looks: the value of
 * "modelJson" is itself a JSON document full of quote characters, so a naive
 * scan for the next unescaped `"` returns the empty string between the first
 * pair of quotes and every pasted model fails to parse. That bug shipped and
 * made the solve console reject valid input.
 */
std::string json_string_field(const std::string& body, const std::string& key) {
  auto j=nlohmann::json::parse(body,nullptr,false);
  if (!j.is_object() || !j.contains(key) || !j[key].is_string()) return {};
  return j[key].get<std::string>();
}

double json_number_field(const std::string& body, const std::string& key, double fallback) {
  auto j=nlohmann::json::parse(body,nullptr,false);
  if (!j.is_object() || !j.contains(key)) return fallback;
  if (j[key].is_boolean()) return j[key].get<bool>() ? 1 : 0;
  return j[key].is_number() ? j[key].get<double>() : fallback;
}

}  // namespace

// ---------------------------------------------------------------------------
// LocalServer
// ---------------------------------------------------------------------------

struct LocalServer::Impl {
  httplib::Server svr;
  std::thread listener;
  ServerConfig config;
  std::string base;

  /**
   * Held as a member rather than a local in install_routes(). Lambdas capture
   * `this` and read it from here, which avoids capturing a raw local pointer
   * into handlers that outlive the function frame.
   */
  JobManager* jobs = nullptr;

  /** Middleware that enforces authentication, CORS, Host and Origin. */
  void install_security() {
    svr.set_pre_routing_handler(
        [this](const httplib::Request& req, httplib::Response& res) {
          // Handle OPTIONS preflight for CORS
          if (req.method == "OPTIONS") {
            res.status = 200;
            if (!config.cors_origin.empty()) {
              res.set_header("Access-Control-Allow-Origin", config.cors_origin);
              res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
              res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key");
              res.set_header("Access-Control-Max-Age", "86400");
            }
            return httplib::Server::HandlerResponse::Handled;
          }

          // API key authentication for network mode
          if (!config.api_key.empty()) {
            const std::string auth_header = req.get_header_value("Authorization");
            const std::string api_key_header = req.get_header_value("X-API-Key");
            std::string provided_key;

            if (!auth_header.empty() && auth_header.substr(0, 7) == "Bearer ") {
              provided_key = auth_header.substr(7);
            } else if (!api_key_header.empty()) {
              provided_key = api_key_header;
            }

            if (provided_key != config.api_key) {
              res.status = 401;
              res.set_content("{\"error\":\"unauthorized: invalid or missing API key\"}",
                             "application/json");
              return httplib::Server::HandlerResponse::Handled;
            }
          }

          // Host validation (only for loopback mode)
          if (!config.allow_network) {
            const std::string& host = req.get_header_value("Host");
            if (!is_valid_host_header(host)) {
              res.status = 421;  // Misdirected Request
              res.set_content(
                  "{\"error\":\"invalid Host header. Sovereign accepts only loopback "
                  "authorities (127.0.0.1 or localhost).\"}",
                  "application/json");
              return httplib::Server::HandlerResponse::Handled;
            }
          }

          // Origin validation for state-changing requests (loopback mode only)
          if (!config.allow_network) {
            const bool state_changing = req.method != "GET" && req.method != "HEAD";
            if (state_changing) {
              const std::string origin = req.get_header_value("Origin");
              if (!origin.empty() && !is_valid_origin(origin, base)) {
                res.status = 403;
                res.set_content(
                    "{\"error\":\"origin not permitted for a state-changing request\"}",
                    "application/json");
                return httplib::Server::HandlerResponse::Handled;
              }
            }
          }

          // Add CORS headers for regular responses if configured
          if (!config.cors_origin.empty()) {
            res.set_header("Access-Control-Allow-Origin", config.cors_origin);
            res.set_header("Access-Control-Allow-Credentials", "true");
          }

          return httplib::Server::HandlerResponse::Unhandled;
        });
  }

  void install_limits() {
    svr.set_payload_max_length(config.max_body_bytes);
  }

  void install_routes(LocalServer* owner) {
    jobs = &owner->jobs();

    // ---- system -----------------------------------------------------------
    svr.Get("/api/health", [this](const httplib::Request&, httplib::Response& res) {
      std::ostringstream oss;
      oss << std::setprecision(17);
      oss << "{\"status\":\"ok\",\"service\":\"sovereign-local\","
          << "\"workers\":" << jobs->worker_count()
          << ",\"gpu\":{\"implemented\":false,"
          << "\"note\":\"no CUDA kernels exist; CPU does all work. See ROADMAP M11.\"}}";
      res.set_content(oss.str(), "application/json");
    });

    svr.Get("/api/system", [this](const httplib::Request&, httplib::Response& res) {
      std::ostringstream oss;
      oss << std::setprecision(17);
      const unsigned hw = std::thread::hardware_concurrency();
      oss << "{\"version\":\"1.0.0\","
          << "\"hardware_threads\":" << hw << ","
          << "\"worker_threads\":" << jobs->worker_count() << ","
          << "\"device\":\"cpu\","
          << "\"cuda_available\":false,"
          << "\"capabilities\":{"
          << "\"LP\":true,\"MILP\":true,\"QP\":true,\"MIQP\":false,\"NLP\":false,"
          << "\"MINLP\":false,\"MPS\":true},"
          << "\"model_root\":\"" << json_escape(config.model_root) << "\"}";
      res.set_content(oss.str(), "application/json");
    });

    // ---- models -----------------------------------------------------------
    svr.Get("/api/problems", [this](const httplib::Request& req,
                                    httplib::Response& res) {
      std::error_code ec;
      // The root comes from our own configuration, not from the client, so it
      // needs canonicalising but not a containment check. resolve_within_root()
      // deliberately rejects the root itself, so it is only for the optional
      // client-supplied subdirectory below.
      const fs::path root_path =
          fs::weakly_canonical(fs::absolute(fs::path(config.model_root), ec), ec);
      if (ec || !fs::is_directory(root_path, ec)) {
        res.status = 404;
        res.set_content("{\"error\":\"model root is not a readable directory\"}",
                        "application/json");
        return;
      }

      fs::path root = root_path;
      if (req.has_param("root")) {
        const std::string sub = req.get_param_value("root");
        if (!sub.empty() && sub != ".") {
          std::string resolved;
          std::string err;
          // This is the untrusted input, so it does get the containment check.
          if (!resolve_within_root(root_path.string(), sub, resolved, err)) {
            res.status = 400;
            res.set_content("{\"error\":\"" + json_escape(err) + "\"}", "application/json");
            return;
          }
          root = fs::path(resolved);
        }
      }
      std::ostringstream oss;
      oss << std::setprecision(17);
      oss << "{\"models\":[";
      bool first = true;
      for (const auto& entry : fs::recursive_directory_iterator(root, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        const std::string ext = entry.path().extension().string();
        if (ext != ".json" && ext != ".mps") continue;
        if (!first) oss << ",";
        first = false;
        oss << "{\"name\":\"" << json_escape(fs::relative(entry.path(), root_path).generic_string())
            << "\",\"path\":\"" << json_escape(entry.path().string())
            << "\",\"format\":\"" << (ext == ".mps" ? "mps" : "json") << "\"}";
      }
      oss << "]}";
      res.set_content(oss.str(), "application/json");
    });

    // ---- validate --------------------------------------------------------
    /**
     * Parse a pasted model and report its shape, without solving it.
     *
     * This exists so the user gets immediate feedback on a typo'd constraint or
     * a malformed objective instead of discovering it from a failed solve thirty
     * seconds later. It also powers the "problem summary" panel, so what the
     * solver is about to be handed is visible before it is handed over.
     */
    svr.Post("/api/validate", [](const httplib::Request& req, httplib::Response& res) {
      const std::string model_json = json_string_field(req.body, "modelJson");
      if (model_json.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"modelJson is required\"}", "application/json");
        return;
      }

      OptimizationModel model;
      try {
        const auto format=json_string_field(req.body,"modelFormat");
        if (!format.empty() && format!="json" && format!="mps") throw std::runtime_error("unsupported modelFormat");
        model = format=="mps" ? load_model_from_mps_string(model_json).model : load_model_from_json_string(model_json);
        const auto error=ModelValidator::validate(model);
        if (!error.empty()) throw std::runtime_error(error);
      } catch (const std::exception& e) {
        res.status = 200;  // 200: the request was fine, the model was not
        std::ostringstream oss;
      oss << std::setprecision(17);
        oss << "{\"valid\":false,\"error\":\"" << json_escape(e.what()) << "\"}";
        res.set_content(oss.str(), "application/json");
        return;
      }

      std::int64_t nnz = 0;
      std::int64_t integer_vars = 0;
      std::int64_t binary_vars = 0;
      std::int64_t equalities = 0;
      std::int64_t inequalities = 0;
      for (const auto& c : model.constraints) {
        nnz += static_cast<std::int64_t>(c.linear.size());
        if (c.sense == ConstraintSense::Eq) ++equalities; else ++inequalities;
      }
      for (const auto& v : model.variables) {
        if (v.type == VariableType::Integer) ++integer_vars;
        else if (v.type == VariableType::Binary) ++binary_vars;
      }

      std::ostringstream oss;
      oss << std::setprecision(17);
      oss << "{\"valid\":true"
          << ",\"problem_type\":\"" << to_string(model.problem_type) << "\""
          << ",\"sense\":\"" << to_string(model.sense) << "\""
          << ",\"rows\":" << model.constraints.size()
          << ",\"columns\":" << model.variables.size()
          << ",\"nonzeros\":" << nnz
          << ",\"equalities\":" << equalities
          << ",\"inequalities\":" << inequalities
          << ",\"integer_vars\":" << integer_vars
          << ",\"binary_vars\":" << binary_vars
          << ",\"objective_terms\":" << model.objective.linear.size()
          << ",\"quadratic_terms\":" << model.objective.quadratic.size()
          << "}";
      res.set_content(oss.str(), "application/json");
    });

    // ---- jobs -------------------------------------------------------------
    svr.Post("/api/jobs", [this](const httplib::Request& req,
                                 httplib::Response& res) {
      const std::string body = req.body;

      SolveOptions opt;

      // Two ways to supply a problem: paste/upload inline JSON, or name a
      // bundled model. Inline is what makes this an actual solve console rather
      // than a gallery of canned examples.
      const std::string inline_model = json_string_field(body, "modelJson");
      const std::string requested = json_string_field(body, "modelPath");

      if (!inline_model.empty()) {
        opt.model_json = inline_model;
        opt.model_format = json_string_field(body,"modelFormat");
      } else if (!requested.empty()) {
        // Path containment is enforced here, not in the solver. The client never
        // gets to name a file outside the model root.
        std::string resolved;
        std::string err;
        if (!resolve_within_root(config.model_root, requested, resolved, err)) {
          res.status = 400;
          res.set_content("{\"error\":\"" + json_escape(err) + "\"}", "application/json");
          return;
        }
        opt.model_path = resolved;
      } else {
        res.status = 400;
        res.set_content(
            "{\"error\":\"supply either modelJson (pasted or uploaded model) or "
            "modelPath (a bundled model)\"}",
            "application/json");
        return;
      }
      opt.algorithm = json_string_field(body, "algorithm");
      if (opt.algorithm != "simplex" && opt.algorithm != "ipm" &&
          opt.algorithm != "auto") {
        opt.algorithm.clear();  // unrecognised -> engine default
      }
      opt.threads = static_cast<int>(json_number_field(body, "threads", 1));
      if (opt.threads < 1) opt.threads = 1;
      opt.time_limit_seconds = json_number_field(body, "timeLimitSeconds", 0.0);
      opt.mip_gap = json_number_field(body, "mipGap", 0.0);
      opt.presolve = json_number_field(body, "presolve", 1.0) != 0.0;

      // `device` is accepted so the UI contract is stable and so a future CUDA
      // backend has somewhere to land. There is no CUDA implementation yet, so
      // this is forced to CPU. Reporting that honestly is better than silently
      // pretending the work was offloaded -- the response tells the caller which
      // device actually ran.
      const std::string requested_device = json_string_field(body, "device");
      const bool wanted_cuda = requested_device == "cuda";
      opt.device = false;

      auto job = jobs->submit(opt);
      res.status = 202;  // Accepted
      std::ostringstream oss;
      oss << std::setprecision(17);
      oss << "{\"jobId\":\"" << job->id() << "\",\"state\":\"QUEUED\"";
      // Echo what the user asked for alongside what will actually happen. A
      // request for CUDA that silently runs on CPU is worse than a refusal.
      if (wanted_cuda) {
        oss << ",\"deviceRequested\":\"cuda\",\"deviceActual\":\"cpu\""
            << ",\"deviceWarning\":\"CUDA is not implemented; this run used the CPU\"";
      }
      oss << "}";
      res.set_content(oss.str(), "application/json");
    });

    svr.Get(R"(/api/jobs/([^/]+))",
            [this](const httplib::Request& req, httplib::Response& res) {
              auto job = jobs->find(req.matches[1].str());
              if (!job) {
                res.status = 404;
                res.set_content("{\"error\":\"unknown job\"}", "application/json");
                return;
              }
              res.set_content(progress_to_json(job->snapshot()), "application/json");
            });

    svr.Post(R"(/api/jobs/([^/]+)/cancel)",
             [this](const httplib::Request& req, httplib::Response& res) {
               auto job = jobs->find(req.matches[1].str());
               if (!job) {
                 res.status = 404;
                 res.set_content("{\"error\":\"unknown job\"}", "application/json");
                 return;
               }
               job->request_cancel();
               res.set_content("{\"cancelled\":true}", "application/json");
             });

    // SSE progress. text/event-stream, one `data:` line per event.
    //
    // Polling would work, but SSE avoids making the UI invent a poll interval
    // and keeps latency low without extra chatter. The provider is called
    // repeatedly; returning false ends the stream.
    svr.Get(R"(/api/jobs/([^/]+)/events)",
            [this](const httplib::Request& req, httplib::Response& res) {
              auto job = jobs->find(req.matches[1].str());
              if (!job) {
                res.status = 404;
                res.set_content("{\"error\":\"unknown job\"}", "application/json");
                return;
              }
              res.set_header("Cache-Control", "no-cache");
              res.set_header("X-Accel-Buffering", "no");  // defeat proxy buffering
              res.set_chunked_content_provider(
                  "text/event-stream",
                  [job](std::size_t, httplib::DataSink& sink) {
                    const JobProgress p = job->snapshot();
                    const std::string event = "data: " + progress_to_json(p) + "\n\n";
                    if (!sink.write(event.data(), event.size())) return false;
                    if (p.state==JobState::Completed || p.state==JobState::Failed || p.state==JobState::Cancelled) { sink.done(); return true; }
                    std::this_thread::sleep_for(std::chrono::milliseconds(150));
                    return true;
                  },
                  nullptr);
            });

    svr.Get("/api/jobs", [this](const httplib::Request&, httplib::Response& res) {
      std::ostringstream oss;
      oss << std::setprecision(17);
      oss << "{\"jobs\":[";
      const auto all = jobs->list();
      for (std::size_t i = 0; i < all.size(); ++i) {
        if (i) oss << ",";
        oss << progress_to_json(all[i]);
      }
      oss << "]}";
      res.set_content(oss.str(), "application/json");
    });

    // Solution export. Separate from the job snapshot so "download solution"
    // fetches only the answer, not the whole progress envelope.
    svr.Get(R"(/api/jobs/([^/]+)/solution)",
            [this](const httplib::Request& req, httplib::Response& res) {
              auto job = jobs->find(req.matches[1].str());
              if (!job) {
                res.status = 404;
                res.set_content("{\"error\":\"unknown job\"}", "application/json");
                return;
              }
              const JobProgress p = job->snapshot();
              if (p.solution_json.empty()) {
                res.status = 409;  // not ready / no solution produced
                std::ostringstream oss;
      oss << std::setprecision(17);
                oss << "{\"error\":\"no solution yet\",\"state\":\"" << to_string(p.state)
                    << "\",\"message\":\"" << json_escape(p.message) << "\"}";
                res.set_content(oss.str(), "application/json");
                return;
              }
              std::ostringstream oss;
      oss << std::setprecision(17);
              oss << "{\"jobId\":\"" << job->id() << "\""
                  << ",\"solver_status\":\"" << json_escape(p.solver_status) << "\""
                  << ",\"objective\":" << (p.has_objective ? p.objective : 0.0)
                  << ",\"runtime_seconds\":" << p.runtime_seconds
                  << ",\"mip_gap\":" << p.mip_gap
                  << ",\"solution\":" << p.solution_json;
              if (!p.verification_json.empty()) {
                oss << ",\"verification\":" << p.verification_json;
              }
              oss << "}";
              res.set_content(oss.str(), "application/json");
            });

    // ---- static -----------------------------------------------------------
    if (!config.static_root.empty()) {
      /**
       * Routing note.
       *
       * The release ships two front ends in one directory:
       *   /            -> index.html, the static single-viewport landing page
       *   /dashboard/  -> the React engine dashboard
       *
       * httplib's set_mount_point resolves the directory itself, so a bare
       * mount at "/" would send /dashboard to a nonexistent path instead of
       * descending into the subdirectory.
       *
       * Three aliases are handled here because all three have been typed or
       * linked at some point: "/dashboard", "/dashboard.html" and "/". The bare
       * root is NOT redirected blindly -- if a request already resolved to a
       * real file we must not bounce it away, so only the exact "/" path is
       * redirected to index.html.
       */
      svr.Get("/dashboard", [](const httplib::Request&, httplib::Response& res) {
        res.status = 302;
        res.set_header("Location", "/dashboard/index.html");
      });
      // An earlier build of the landing page linked to "dashboard.html", which
      // never existed. Keep the redirect so any cached or hand-typed copy of
      // that page still lands somewhere useful.
      svr.Get("/dashboard.html", [](const httplib::Request&, httplib::Response& res) {
        res.status = 302;
        res.set_header("Location", "/dashboard/index.html");
      });
      svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.status = 302;
        res.set_header("Location", "/index.html");
      });

      svr.set_mount_point("/", config.static_root);
    }
  }
};

LocalServer::LocalServer(ServerConfig config)
    : config_(std::move(config)),
      impl_(new Impl()),
      jobs_(new JobManager(config_.worker_threads)) {
  impl_->config = config_;
}

LocalServer::~LocalServer() {
  stop();
}

void LocalServer::start() {
  impl_->svr.set_read_timeout(30, 0);
  impl_->svr.set_write_timeout(30, 0);
  impl_->svr.set_payload_max_length(config_.max_body_bytes);
  impl_->svr.set_keep_alive_max_count(64);

  impl_->install_limits();
  // The base URL is needed by the Origin check, but the port is not known until
  // bind succeeds. Install routing first (it does not need the base), then set
  // the base and install security.
  impl_->install_routes(this);

  // httplib's bind_to_port() returns *bool*, not the port, and Server exposes no
  // port() accessor. An earlier version of this code assigned the bool to an
  // int, so a successful bind yielded `true == 1` and the server announced
  // port 1 — found by actually running it.
  //
  // So: for an ephemeral port use bind_to_any_port(), which does return the
  // port. For a caller-specified port, bind_to_port() gives us the success flag
  // and we already know which port we asked for.
  if (config_.port == 0) {
    port_ = impl_->svr.bind_to_any_port(config_.bind_address);
    if (port_ <= 0) {
      throw std::runtime_error("could not bind an ephemeral port on " +
                               config_.bind_address);
    }
  } else {
    if (!impl_->svr.bind_to_port(config_.bind_address, config_.port)) {
      throw std::runtime_error("could not bind " + config_.bind_address + ":" +
                               std::to_string(config_.port));
    }
    port_ = config_.port;
  }

  impl_->base = "http://" + config_.bind_address + ":" + std::to_string(port_);
  impl_->install_security();

  running_.store(true);
  impl_->listener = std::thread([this] { impl_->svr.listen_after_bind(); });
}

std::string LocalServer::base_url() const {
  return "http://" + config_.bind_address + ":" + std::to_string(port_);
}

void LocalServer::wait() {
  while (running_.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void LocalServer::stop() {
  if (!running_.exchange(false)) return;
  impl_->svr.stop();
  if (impl_->listener.joinable()) impl_->listener.join();
  jobs_->shutdown();
}

}  // namespace server
}  // namespace sovereign
