# Monocypher

This directory vendors Monocypher (`src/monocypher.c`, `src/monocypher.h`) from
<https://github.com/LoupVaillant/Monocypher>.

## Why

Sovereign needs **Ed25519 signature verification** in the local application so it
can validate a server-issued offline credential with no network access, and
**SHA-256** for credential/session digests.

Requirements that drove this choice:

- **No runtime dependency.** The offline release must not require the user to
  install OpenSSL or any other shared library.
- **No hand-rolled cryptography.** The specification is explicit that crypto
  must use an established implementation, not something written in this repo.
- **Auditable and small.** ~10 KB of C compiles in seconds and can be reviewed
  line by line, which matters for a component that gates offline authorization.

Alternatives considered and rejected:

| Option | Why not |
|---|---|
| libsodium | Excellent, but a large build dependency with its own CMake/pkg-config dance. Overkill for "verify one signature". |
| OpenSSL | Runtime .dll dependency, which breaks the "no external native libraries" packaging goal. |
| Windows CNG / BCrypt | Ties verification to Windows and cannot be unit-tested on other platforms. |
| TweetNaCl | Also good, but Monocypher is a single translation unit with a cleaner API and a maintained modern default. |
| Hand-rolled Ed25519 | Absolutely not. Non-starter. |

## What Sovereign uses from it

- `crypto_ed25519_check()` — verify the server's detached signature over the
  offline credential payload. **Verification only.** Sovereign never signs; the
  server holds the private key.
- `crypto_sha256()` — digesting for session identifiers and credential binding.

Nothing else. If we later need X25519 key agreement or BLAKE2b, they are in the
same file.

## License

Monocypher is released under **CC0 / Unlicense** (public domain dedication).
Intended to be copied into projects. See `LICENSE`.

## Note on the signing side

Monocypher can sign, and Sovereign's CMake does link the module, but **no code
path in the shipped application ever calls `crypto_ed25519_sign`**. The server
private key must never exist on a client machine. The test fixtures generate a
throwaway keypair at test time; that keypair is discarded with the test run.
