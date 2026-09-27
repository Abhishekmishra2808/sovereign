#include "sovereign/ed25519_verify.hpp"

// Monocypher, vendored under third_party/monocypher (CC0 / public domain).
// C translation units; see third_party/monocypher/README.md for why this
// library rather than libsodium, OpenSSL, or a hand-rolled implementation.
//
// Ed25519 lives in Monocypher's `optional` module and SHA-512 (required
// internally by Ed25519) comes with it, so both headers are included.
extern "C" {
#include "monocypher.h"
#include "monocypher-ed25519.h"
}

#include <stdexcept>

namespace sovereign {
namespace {

int hex_value(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

/**
 * Reject degenerate public keys before verifying.
 *
 * Ed25519 verification is malleable for *small-order* public keys. The identity
 * point (all zeroes) and the other low-order points admit trivially forged
 * signatures, so "the signature verified" carries no information at all when
 * the key is one of them.
 *
 * This is not hypothetical: a test in tests/unit/test_ed25519_verify.cpp
 * discovered that Monocypher's crypto_ed25519_check() *accepts* an all-zero
 * signature over an empty message when the public key is all zeroes. Monocypher
 * documents no low-order rejection for this function, unlike libsodium's
 * crypto_sign_verify_detached.
 *
 * The client compiles in exactly one trusted public key, so the practical
 * exposure is limited -- an attacker cannot choose it -- but "limited" is not
 * "none", and the check is two lines. We reject the all-zero key outright.
 *
 * Residual risk, stated plainly: the other seven low-order curve points are NOT
 * filtered here. Properly rejecting all of them requires decoding the point and
 * checking its order, which Monocypher does not expose. If this ever needs to
 * be airtight, either pin libsodium (which does the check) or add point
 * decompression. For gating a locally-stored offline credential against a
 * server-chosen key, the all-zero guard is proportionate.
 */
bool is_degenerate_key(const std::vector<uint8_t>& public_key) {
  for (unsigned char b : public_key) {
    if (b != 0) return false;
  }
  return true;
}

}  // namespace

bool ed25519_verify(const std::vector<uint8_t>& public_key,
                    const std::vector<uint8_t>& message,
                    const std::vector<uint8_t>& signature) {
  // crypto_ed25519_check requires exactly a 32-byte key and a 64-byte signature;
  // anything else is undefined behaviour, so this is checked rather than
  // assumed. An empty message is a valid Ed25519 input and is allowed through.
  if (public_key.size() != 32 || signature.size() != 64) return false;
  if (is_degenerate_key(public_key)) return false;

  // Monocypher's argument order is (signature, public_key, message, size).
  // Returns 0 on success.
  return crypto_ed25519_check(
      signature.data(),
      public_key.data(),
      message.empty() ? nullptr : message.data(),
      static_cast<std::size_t>(message.size())) == 0;
}

bool ed25519_verify_hex(const std::string& public_key_hex,
                        const std::string& message,
                        const std::string& signature_hex) {
  try {
    const auto pub = hex_to_bytes(public_key_hex);
    const auto sig = hex_to_bytes(signature_hex);
    const std::vector<uint8_t> msg(message.begin(), message.end());
    return ed25519_verify(pub, msg, sig);
  } catch (const std::invalid_argument&) {
    return false;
  }
}

std::vector<uint8_t> blake2b256(const std::vector<uint8_t>& data) {
  // 32 bytes is the BLAKE2b-256 variant. Passing the digest size explicitly
  // matters: Monocypher defaults to 64 and a mismatch would silently produce a
  // different (longer) digest than callers expect.
  std::vector<uint8_t> out(32);
  crypto_blake2b(out.data(), 32, data.empty() ? nullptr : data.data(), data.size());
  return out;
}

std::string bytes_to_hex(const std::vector<uint8_t>& bytes) {
  static const char* kDigits = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (unsigned char b : bytes) {
    out.push_back(kDigits[b >> 4]);
    out.push_back(kDigits[b & 0x0f]);
  }
  return out;
}

std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
  if (hex.size() % 2 != 0) {
    throw std::invalid_argument("hex string has an odd number of characters");
  }
  std::vector<uint8_t> out;
  out.reserve(hex.size() / 2);
  for (std::size_t i = 0; i < hex.size(); i += 2) {
    const int hi = hex_value(hex[i]);
    const int lo = hex_value(hex[i + 1]);
    if (hi < 0 || lo < 0) {
      throw std::invalid_argument("hex string contains a non-hex character");
    }
    out.push_back(static_cast<uint8_t>((hi << 4) | lo));
  }
  return out;
}

}  // namespace sovereign
