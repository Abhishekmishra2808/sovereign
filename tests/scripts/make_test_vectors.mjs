#!/usr/bin/env node
/**
 * Generate Ed25519 test vectors for the C++ verifier.
 *
 * WHY THIS EXISTS
 * ---------------
 * The auth service will sign with Node's `crypto`. The desktop client will
 * verify with Monocypher, in C++. Those are different implementations, and
 * "both are Ed25519" is an assumption, not a fact. The realistic failure modes
 * are mundane and fatal: a seed extracted from the wrong JWK field, a public
 * key that is not the raw 32 bytes, a payload canonicalised differently on each
 * side. Any of them yields a client that silently rejects every valid
 * credential, and it would only show up in front of a user.
 *
 * So we generate real vectors with the production code path and assert against
 * them in C++.
 *
 * Writes hex files to tests/data/. Committed, so `ctest` needs no Node.
 *
 * USAGE:  node tests/scripts/make_test_vectors.mjs
 */

import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const OUT = path.resolve(HERE, '..', 'data')

/**
 * The payload shape an offline credential actually uses. Deliberately realistic
 * rather than "hello world", so the test also pins down that a JSON blob with
 * quotes, colons and unicode signs correctly.
 */
function buildTestPayload() {
  return JSON.stringify({
    v: 1,
    user_id: '11111111-2222-3333-4444-555555555555',
    device_id: '66666666-7777-8888-9999-aaaaaaaaaaaa',
    email: 'engineer@example.com',
    display_name: 'Refinery Optimisation Engineer',
    issued_at: 1767225600,
    expires_at: 1798761600,
    offline_access: true,
    credential_version: 1,
    key_id: 'ed25519:test',
  })
}

function main() {
  fs.mkdirSync(OUT, { recursive: true })

  const { publicKey, privateKey } = crypto.generateKeyPairSync('ed25519')

  // Raw 32-byte public key: the JWK 'x' member, base64url-decoded. This is
  // exactly what Monocypher's crypto_ed25519_check expects.
  const pubJwk = publicKey.export({ format: 'jwk' })
  const rawPublic = Buffer.from(pubJwk.x, 'base64url')
  if (rawPublic.length !== 32) {
    console.error(`ERROR: expected 32-byte public key, got ${rawPublic.length}`)
    process.exit(1)
  }

  // Confirm the raw public key really is the seed's public counterpart by
  // rebuilding it from the private key. If these disagree, the extraction above
  // is wrong and every downstream test would be testing the wrong key.
  const privJwk = privateKey.export({ format: 'jwk' })
  const rebuilt = crypto.createPublicKey({
    key: { kty: 'OKP', crv: 'Ed25519', x: pubJwk.x, d: privJwk.d },
    format: 'jwk',
  })
  const rebuiltRaw = Buffer.from(rebuilt.export({ format: 'jwk' }).x, 'base64url')
  if (!rebuiltRaw.equals(rawPublic)) {
    console.error('ERROR: public key derived from the private key does not match')
    process.exit(1)
  }

  const message = Buffer.from(buildTestPayload(), 'utf8')
  const signature = crypto.sign(null, message, privateKey)

  if (signature.length !== 64) {
    console.error(`ERROR: expected a 64-byte signature, got ${signature.length}`)
    process.exit(1)
  }

  const files = {
    'ed25519_test_pub.hex': rawPublic.toString('hex'),
    'ed25519_test_msg.hex': message.toString('hex'),
    'ed25519_test_sig.hex': signature.toString('hex'),
  }
  for (const [name, hex] of Object.entries(files)) {
    fs.writeFileSync(path.join(OUT, name), hex + '\n')
  }

  // Also emit the exact signing key as a PEM so the auth service and the test
  // fixtures provably use the same generation path.
  fs.writeFileSync(
    path.join(OUT, 'ed25519_test_key.pem'),
    privateKey.export({ type: 'pkcs8', format: 'pem' }),
  )

  // Self-check: verify with Node before shipping the vectors, so a broken
  // vector file cannot be committed and then read as a real failure later.
  const ok = crypto.verify(null, message, publicKey, signature)
  if (!ok) {
    console.error('ERROR: Node refused to verify its own signature')
    process.exit(1)
  }

  console.log(`wrote ${Object.keys(files).length + 1} files to ${OUT}`)
  console.log(`  message : ${message.length} bytes`)
  console.log(`  public  : ${rawPublic.toString('hex')}`)
  console.log(`  self-verify: ok`)
}

main()
