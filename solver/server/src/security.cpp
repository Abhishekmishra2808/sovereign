#include "sovereign/server/http_server.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <sstream>

namespace sovereign {
namespace server {
namespace fs = std::filesystem;

namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string trim(const std::string& s) {
  std::size_t a = 0;
  std::size_t b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

}  // namespace

bool is_valid_host_header(const std::string& host) {
  const std::string h = lower(trim(host));
  if (h.empty()) return false;

  // Split off an optional port. Bracketed IPv6 literals are accepted only for
  // ::1, which is the other loopback address.
  std::string name = h;
  if (h.front() == '[') {
    const auto close = h.find(']');
    if (close == std::string::npos) return false;
    name = h.substr(0, close + 1);
    const std::string tail = h.substr(close + 1);
    if (!tail.empty() && tail.front() != ':') return false;
    if (!tail.empty()) {
      for (std::size_t i = 1; i < tail.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(tail[i]))) return false;
      }
    }
    return name == "[::1]";
  }

  const auto colon = h.rfind(':');
  if (colon != std::string::npos) {
    // Anything after the last colon must be a plain port number. A Host like
    // "evil.com:80.evil" or "127.0.0.1:notaport" must not pass.
    const std::string port = h.substr(colon + 1);
    if (port.empty()) return false;
    for (char c : port) {
      if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    if (port.size() > 5) return false;  // no port exceeds 65535
    name = h.substr(0, colon);
  }

  return name == "127.0.0.1" || name == "localhost";
}

bool is_valid_origin(const std::string& origin, const std::string& expected_base) {
  const std::string o = trim(origin);
  if (o.empty()) {
    // No Origin at all means the request did not come from a browser page
    // (curl, the CLI, our own tooling). Those are allowed; the CSRF token and
    // the loopback bind are the controls that matter there.
    return true;
  }
  // A browser-sent Origin must be exactly our own loopback origin. "null"
  // (sandboxed iframe, some file:// pages) is rejected.
  if (o == "null") return false;
  return o == expected_base || o == expected_base + "/";
}

bool resolve_within_root(const std::string& root, const std::string& candidate,
                         std::string& resolved_out, std::string& error_out) {
  namespace fs = std::filesystem;
  std::error_code ec;

  if (candidate.empty()) {
    error_out = "empty path";
    return false;
  }

  // Canonicalise the root once. weakCanonical resolves what exists without
  // requiring the leaf to exist; canonical requires it. The root exists, the
  // candidate may not yet.
  const fs::path root_path = fs::weakly_canonical(fs::absolute(fs::path(root), ec), ec);
  if (ec) {
    error_out = "model root is not resolvable";
    return false;
  }

  fs::path candidate_path(candidate);
  if (candidate_path.is_absolute()) {
    // Absolute paths are allowed only if they land inside the root. Building the
    // joined path is what makes the containment check below meaningful.
    candidate_path = candidate_path.lexically_normal();
  } else {
    candidate_path = (root_path / candidate_path).lexically_normal();
  }

  // The single most important check: does the result start with the root?
  // Compare component-wise, not by string prefix. A string-prefix test would
  // accept "C:\models-evil" for root "C:\models".
  const fs::path& canonical_root = root_path;
  auto root_it = canonical_root.begin();
  auto cand_it = candidate_path.begin();
  for (; root_it != canonical_root.end(); ++root_it, ++cand_it) {
    if (cand_it == candidate_path.end()) {
      // Candidate is a strict prefix of the root, e.g. root itself minus a
      // component. Not acceptable as a model path.
      error_out = "path does not resolve inside the model root";
      return false;
    }
    // Compare component-wise via .string(). A component can be any path type, so
    // std::string(*it) is not valid; .string() is.
    if (root_it->string() != cand_it->string()) {
      error_out = "path escapes the model root";
      return false;
    }
  }
  if (cand_it == candidate_path.end()) {
    error_out = "path is the model root itself, not a file";
    return false;
  }

  resolved_out = candidate_path.string();
  return true;
}

}  // namespace server
}  // namespace sovereign
