#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sovereign {

/**
 * Ed25519 signature verification.
 *
 * Sovereign's client NEVER signs. Signing happens on the auth service, which
 * holds the private key. This module only answers one question: "was this
 * payload signed by a key we trust, and has it been altered since?"
 *
 * That asymmetry is the whole point of the offline-authorization design. If a
 * client could sign, any modified client could mint its own entitlement, and
 * server-side revocation would be meaningless.
 *
 * Implementation: vendored Monocypher (CC0), wrapping a well-audited
 * ref10-derived implementation. We do not implement Ed25519 ourselves.
 *
 * All inputs are attacker-influenced, so malformed input returns false or
 * throws std::invalid_argument rather than aborting.
 */

/** Verify a detached Ed25519 signature. False on any malformed input. */
bool ed25519_verify(const std::vector<uint8_t>& public_key,
                    const std::vector<uint8_t>& message,
                    const std::vector<uint8_t>& signature);

/** Verify using a hex-encoded key and signature. Returns false on bad hex. */
bool ed25519_verify_hex(const std::string& public_key_hex,
                        const std::string& message,
                        const std::string& signature_hex);

/**
 * BLAKE2b digest, used for local session identifiers and credential binding.
 *
 * BLAKE2b rather than SHA-256 on purpose. These digests never leave the machine
 * except as opaque local identifiers -- there is no interoperability
 * requirement, no attestation, no third-party verification. BLAKE2b is the
 * primitive Monocypher is built around, so using it avoids vendoring a second
 * crypto library purely to hash a random session token. It is also faster than
 * SHA-256 on 64-bit hardware. If a spec ever requires SHA-256 for these, that is
 * a deliberate change, not an accident.
 */
std::vector<uint8_t> blake2b256(const std::vector<uint8_t>& data);

/** Lowercase hex encoding. */
std::string bytes_to_hex(const std::vector<uint8_t>& bytes);

/**
 * Parse hex into bytes.
 * @throws std::invalid_argument on odd length or a non-hex character.
 */
std::vector<uint8_t> hex_to_bytes(const std::string& hex);

}  // namespace sovereign
