#pragma once

#include "sovereign/server/job_manager.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace sovereign {
namespace server {

struct ServerConfig {
  /**
   * Bind address. Default is 127.0.0.1 for local-only use.
   * Set to "0.0.0.0" to allow network access. Requires --allow-network flag.
   */
  std::string bind_address = "127.0.0.1";

  /** Allow binding to non-loopback addresses. Default false for security. */
  bool allow_network = false;

  /** API key for authentication when exposed to network. Empty = no auth. */
  std::string api_key;

  /** CORS origin to allow. Empty = same-origin only. */
  std::string cors_origin;

  /** 0 = let the OS pick an ephemeral port. The launcher reads back the choice. */
  int port = 0;

  /** Directory of pre-built frontend assets, served from disk. */
  std::string static_root;

  /** Directory the model browser is allowed to read from. */
  std::string model_root = "examples/models";

  unsigned worker_threads = 0;

  /** Max request body accepted, in bytes. Models are text; 32 MB is generous. */
  std::size_t max_body_bytes = 32u * 1024u * 1024u;

  std::size_t max_header_bytes = 32u * 1024u;
  std::size_t max_url_bytes = 8u * 1024u;
};

/**
 * HTTP server for Sovereign.
 *
 * Security posture:
 *
 *   - By default, binds 127.0.0.1 only for local use.
 *   - When --allow-network is set, can bind to 0.0.0.0 for network access.
 *   - Validates the Host header in loopback mode. Without this, a DNS-rebinding
 *     attack can point an attacker-controlled domain at 127.0.0.1 and then talk
 *     to this API from their own origin.
 *   - Validates Origin on state-changing methods in loopback mode.
 *   - Supports optional API key authentication for network mode.
 *   - Supports configurable CORS origin for network mode.
 *   - Caps body, header and URL size so a request cannot exhaust memory.
 *   - Never executes a frontend-supplied command. Options are typed fields
 *     mapped to an enum; there is no endpoint that takes a shell string.
 *
 * Transport only. No optimization logic lives here.
 */
class LocalServer {
 public:
  explicit LocalServer(ServerConfig config);
  ~LocalServer();

  LocalServer(const LocalServer&) = delete;
  LocalServer& operator=(const LocalServer&) = delete;

  /** Bind and start serving on a background thread. Throws on failure. */
  void start();

  /** The port actually bound. Valid after start(). */
  int port() const { return port_; }

  /** "http://127.0.0.1:<port>", for the launcher to open. */
  std::string base_url() const;

  /** Block until stop() is called. */
  void wait();

  /** Idempotent. */
  void stop();

  JobManager& jobs() { return *jobs_; }

 private:
  ServerConfig config_;
  int port_ = 0;
  std::atomic<bool> running_{false};
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::unique_ptr<JobManager> jobs_;
};

// ---------------------------------------------------------------------------
// Individually testable security helpers.
//
// These are pure functions on purpose: they are the parts an attacker actually
// targets, so they must be unit-testable without binding a socket.
// ---------------------------------------------------------------------------

/** True if `host` is a loopback authority (127.0.0.1 / localhost, optional port). */
bool is_valid_host_header(const std::string& host);

/**
 * True if `origin` is acceptable for a state-changing request.
 * An absent Origin means a non-browser client (curl, the CLI), which is allowed;
 * a browser-sent Origin must match our own loopback origin.
 */
bool is_valid_origin(const std::string& origin, const std::string& expected_base);

/**
 * Resolve `candidate` against `root` and confirm the result stays inside root.
 *
 * Uses canonicalisation so `..` segments, and symlinks pointing outside, are
 * both rejected. This is the only way models are ever located.
 */
bool resolve_within_root(const std::string& root, const std::string& candidate,
                         std::string& resolved_out, std::string& error_out);

}  // namespace server
}  // namespace sovereign
