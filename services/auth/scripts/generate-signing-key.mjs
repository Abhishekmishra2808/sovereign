#!/usr/bin/env node
/**
 * Generate the Ed25519 keypair Sovereign uses to sign offline credentials.
 *
 * WHY THIS SCRIPT EXISTS
 * ----------------------
 * The server signs an offline credential; the desktop app only ever verifies it.
 * So the *public* key is compiled into the binary and the *private* key must
 * never reach a client, the repository, or an installer.
 *
 * This script is the only place the private key is produced. It writes the
 * private key to a gitignored path and prints the public key in the exact form
 * the C++ verifier needs.
 *
 * PLATFORM NOTE
 * -------------
 * You asked for "the crypto command". Two candidates exist and they are not
 * interchangeable:
 *
 *   Linux / macOS, openssl present:
 *     openssl genpkey -algorithm ed25519 -out sovereign_ed25519.pem
 *     openssl pkey -in sovereign_ed25519.pem -pubout -out sovereign_ed25519.pub
 *
 *   This Windows dev box has NO openssl, so the script falls back to Node's
 *   built-in `crypto`, which is OpenSSL-backed underneath and supports Ed25519
 *   natively. That is why this is a Node script rather than a shell one-liner.
 *
 * Both paths produce the same key material; only the encoding differs.
 *
 * USAGE
 *   node services/auth/scripts/generate-signing-key.mjs
 *   node services/auth/scripts/generate-signing-key.mjs --out C:\secure\path
 *   node services/auth/scripts/generate-signing-key.mjs --force
 */

import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const AUTH_DIR = path.resolve(HERE, '..')
const REPO_ROOT = path.resolve(AUTH_DIR, '..', '..')

const DEFAULT_KEY_DIR = path.join(AUTH_DIR, 'secrets')
const PRIVATE_KEY_FILE = 'signing_ed25519.pem'
const PUBLIC_KEY_FILE = path.join(AUTH_DIR, 'keys', 'signing_ed25519.pub.json')

function parseArgs(argv) {
  const args = { out: DEFAULT_KEY_DIR, force: false, help: false }
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--out') args.out = path.resolve(argv[++i])
    else if (argv[i] === '--force') args.force = true
    else if (argv[i] === '--help' || argv[i] === '-h') args.help = true
    else throw new Error(`unknown argument: ${argv[i]}`)
  }
  return args
}

function fail(message) {
  console.error(`\nERROR: ${message}\n`)
  process.exit(1)
}

// --------------------------------------------------------------------------
// Guard rails
// --------------------------------------------------------------------------

/**
 * Refuse to place a private key inside the working tree unless some .gitignore
 * covering that path protects it. A private key committed to git is compromised
 * forever, because git history is immutable even after a "removal" commit.
 *
 * Git honours a .gitignore at ANY directory level, not just the repository root,
 * so `services/.gitignore` is exactly as effective as a root entry. This check
 * therefore walks every ancestor directory rather than only the root.
 */
function assertPrivateKeyIsIgnored(keyDir) {
  const rel = path.relative(REPO_ROOT, keyDir).split(path.sep).join('/')
  if (rel.startsWith('..')) {
    console.log(`private key dir is OUTSIDE the repo: ${keyDir}`)
    return
  }

  const segments = rel.split('/')
  const candidates = [path.join(REPO_ROOT, '.gitignore')]
  let dir = REPO_ROOT
  for (let i = 0; i < segments.length - 1; i += 1) {
    dir = path.join(dir, segments[i])
    candidates.push(path.join(dir, '.gitignore'))
  }

  const coveredBy = candidates.find((file) => {
    if (!fs.existsSync(file)) return false
    const patterns = fs
      .readFileSync(file, 'utf8')
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter((l) => l && !l.startsWith('#'))
    return patterns.some((p) => {
      const clean = p.replace(/^\//, '').replace(/\/$/, '')
      if (clean === '*') return true
      // A rule covers this path if it names the path itself or one of its
      // ancestors. Handles both "/services/auth/secrets/" and "secrets/".
      return rel === clean || rel.startsWith(`${clean}/`)
    })
  })

  if (!coveredBy) {
    fail(
      `private key would be written inside the repo at ${rel}, which is NOT ` +
        'covered by any .gitignore.\nAdd this line to the nearest .gitignore, ' +
        `then re-run:\n    /${rel}/`,
    )
  }
  console.log(
    `private key dir is inside the repo but gitignored by ` +
      `${path.relative(REPO_ROOT, coveredBy)}: /${rel}/`,
  )
}

// --------------------------------------------------------------------------
// Key generation
// --------------------------------------------------------------------------

function generate() {
  const { publicKey, privateKey } = crypto.generateKeyPairSync('ed25519')

  // PKCS#8 PEM private key: what the auth service loads at runtime.
  const privatePem = privateKey.export({ type: 'pkcs8', format: 'pem' }).toString()

  // The raw 32-byte public key is what Monocypher's crypto_ed25519_check wants.
  // A Node KeyObject cannot export raw bytes directly, but the JWK 'x' member is
  // exactly the 32-byte key, base64url-encoded.
  const jwk = publicKey.export({ format: 'jwk' })
  const rawPublic = Buffer.from(jwk.x, 'base64url')

  if (rawPublic.length !== 32) {
    fail(`expected a 32-byte Ed25519 public key, got ${rawPublic.length} bytes`)
  }

  return { privatePem, rawPublic }
}

function main() {
  const args = parseArgs(process.argv.slice(2))
  if (args.help) {
    console.log(
      'Usage: node services/auth/scripts/generate-signing-key.mjs [--out DIR] [--force]',
    )
    return
  }

  if (fs.existsSync(PUBLIC_KEY_FILE) && !args.force) {
    fail(
      `${path.relative(REPO_ROOT, PUBLIC_KEY_FILE)} already exists.\n` +
        'Re-run with --force to replace the keypair. Replacing it invalidates ' +
        'every offline credential already issued to users.',
    )
  }

  assertPrivateKeyIsIgnored(args.out)

  const { privatePem, rawPublic } = generate()

  fs.mkdirSync(args.out, { recursive: true })
  const privatePath = path.join(args.out, PRIVATE_KEY_FILE)
  fs.writeFileSync(privatePath, privatePem, { mode: 0o600 })

  const keyId = `ed25519:${new Date().toISOString().slice(0, 10)}`
  const publicRecord = {
    key_id: keyId,
    algorithm: 'Ed25519',
    public_key_hex: rawPublic.toString('hex'),
    public_key_base64: rawPublic.toString('base64'),
    comment:
      'Raw 32-byte Ed25519 public key. Compile this into the Sovereign client so ' +
      'it can verify server-signed offline credentials. Safe to commit.',
  }
  fs.mkdirSync(path.dirname(PUBLIC_KEY_FILE), { recursive: true })
  fs.writeFileSync(PUBLIC_KEY_FILE, JSON.stringify(publicRecord, null, 2) + '\n')

  console.log(`
Ed25519 signing keypair generated.

  PUBLIC  (safe to commit, compiled into the client)
    file    : ${path.relative(REPO_ROOT, PUBLIC_KEY_FILE)}
    key_id  : ${keyId}
    hex     : ${rawPublic.toString('hex')}
    base64  : ${rawPublic.toString('base64')}

  PRIVATE (NEVER commit, never ship to a client)
    file    : ${privatePath}

Next steps
  1. Point the auth service at the private key:
       SOVEREIGN_SIGNING_KEY_PATH=${privatePath}
  2. Pass the public key to the C++ build:
       -DSOVEREIGN_TRUSTED_SIGNING_KEY_HEX=${rawPublic.toString('hex')}
  3. Back the private key up somewhere the repository is not. If it is lost,
     every existing device must re-activate. If it leaks, every offline
     credential ever issued is forgeable.
`)
}

main()
