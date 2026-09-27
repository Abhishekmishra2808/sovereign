# Sovereign — Authentication and Offline Authorization

Status: **design complete, implementation not started.** This document is the
specification the implementation will follow. Nothing described here exists in
code yet, and the roadmap reflects that.

---

## 1. The two-layer model

The single most important idea in this design: **online account authentication
and offline authorization are different problems solved by different
mechanisms.** Conflating them is the usual way offline-first products end up
either insecure or unusable offline.

```
LAYER A — ONLINE ACCOUNT AUTHENTICATION          "who is this user?"
  email + password ──HTTPS──> Auth API ──> User DB
  Proves identity. Requires network. Rate-limited. Revocable centrally.

LAYER B — LOCAL OFFLINE AUTHORIZATION            "may this install run offline?"
  DPAPI-protected blob ──> Ed25519 verify ──> claims check
  Proves a previously-authenticated device. No network. Locally decidable.
```

Layer A answers "who is this user?". Layer B answers "is this installation
entitled to run without asking again?". Neither substitutes for the other, and
Layer B must never attempt to reach the network to re-validate itself.

**The user's password is never stored locally, in any form, ever.** Not hashed,
not encrypted, not in the OS credential store, not in a config file. It exists
only in the TLS request body to the auth API and in the server's Argon2id hash.

---

## 2. State machine

```
                        START
                          │
                 Secure credential?
                    ╱          ╲
                  NO            YES
                  │               │
              Login UI      Verify locally (Ed25519)
                  │            ╱        ╲
                  │         Valid      Invalid / expired
                  │          │              │
                  │      Dashboard        Login UI
                  │   (+ refresh online
                  │      if reachable)
                  │
            Internet?
             ╱     ╲
           NO       YES
           │         │
    "internet    Authenticate over HTTPS
     required"     │
           │    Activate device
    Show login      │
    (offline        │  server signs credential (Ed25519 private key)
    login is        │  client verifies + stores via DPAPI
    impossible)     │
                 Dashboard

During use:  Dashboard ──internet lost──> continue offline (never sign out)

Logout:      Dashboard ──> delete DPAPI blob ──> clear session ──> Login UI
             (works fully offline; no server round trip required)
```

### Network-unavailable is not authentication-failure

These must be distinguishable in the UI and in the API, because they demand
opposite responses:

| Condition | API | UI | Solver |
|---|---|---|---|
| Valid offline credential, no network | `200 {mode: "offline"}` | Dashboard, offline badge | runs normally |
| No credential, no network | `401 {mode: "offline", reason: "activation_required"}` | Login screen, "internet required to sign in" | still runs via CLI |
| Invalid/tampered credential | `401 {mode: "local", reason: "credential_invalid"}` | Login screen, "sign in again" | still runs via CLI |
| Wrong password, network up | `401 {mode: "online", reason: "bad_credentials"}` | Login screen, inline error | — |

The solver is never gated on authentication. `sovereign solve model.mps` works
with no credential, no server, and no network. Authentication gates the
*dashboard*, not the engine.

---

## 3. Cryptography

**Algorithm: Ed25519** via vendored Monocypher (CC0). Chosen for a fixed 32-byte
public key, small signatures, no parameter-selection footguns, and a
single-translation-unit build with no runtime dependency.

Explicitly rejected:

| Option | Why not |
|---|---|
| Hand-rolled crypto | Non-starter, and the spec forbids it. |
| HMAC shared secret | Client would hold signing capability. Any client could mint credentials. Asymmetric is required. |
| RSA / ECDSA P-256 | Perfectly viable; Ed25519 chosen for smaller keys/signatures and simpler, less error-prone verification. |
| JWT | A JWT is fine as a *transport format*, but it is not a substitute for offline device binding. We'll use a compact custom payload plus signature rather than a general JWT, so the claim set is explicit and auditable. |

### Server side
- Private key generated on the auth service, stored in a KMS/secret manager
  (Azure Key Vault / GCP KMS / AWS Secrets Manager). **Never in this repo.**
- Rotatable via a `key_id` claim; the client carries a small set of trusted
  public keys and accepts any that validates.

### Client side
- Only the public key(s) are compiled in. A tampered binary could swap the public
  key, which is a genuine limitation of client-side verification and is stated
  plainly in the threat model (§9).

### Offline credential payload

Signed envelope, JSON inside a binary wrapper:

```json
{
  "v": 1,
  "user_id": "uuid",
  "device_id": "uuid",
  "email": "user@example.com",
  "display_name": "…",
  "issued_at": 1735689600,
  "expires_at": 1767225600,
  "offline_access": true,
  "credential_version": 1,
  "key_id": "ed25519:2026-01"
}
```

Signed bytes = the canonical JSON of the payload (sorted keys, no whitespace).
Signature appended. **Contains no password, no refresh token, no session
secret.** The refresh token stays server-side; it is never given to the client in
a form that can be replayed offline.

---

## 4. Secure local storage

`SecureCredentialStore` interface, deliberately narrow so platform
implementations are drop-in:

```cpp
struct SecureCredentialStore {
  virtual ~SecureCredentialStore() = default;
  virtual bool save(const std::string& key, const std::vector<uint8_t>& blob) = 0;
  virtual bool load(const std::string& key, std::vector<uint8_t>& out) = 0;
  virtual bool exists(const std::string& key) const = 0;
  virtual bool remove(const std::string& key) = 0;
  virtual std::string name() const = 0;
};
```

### `WindowsSecureCredentialStore`

- **`CryptProtectData` / `CryptUnprotectData`** from `wincrypt.h` (`advapi32`).
- Flags: **`CRYPTPROTECT_UI_FORBIDDEN`**, and deliberately **not**
  `CRYPTPROTECT_LOCAL_MACHINE` — the default per-user scope ties decryption to
  the Windows account, so another user on the same machine cannot read it.
- Optional entropy: a per-installation random 32-byte value, stored in
  `%LOCALAPPDATA%\Sovereign\install.dat` (not itself secret; it raises the cost
  of a blob copied to another machine).
- Persisted at
  `%LOCALAPPDATA%\Sovereign\credentials\<key>.bin` (binary DPAPI envelope).

Alternatives considered:

| Option | Why not |
|---|---|
| `WindowsCredentialManager` (CredWrite) | Genuinely good, but it surfaces entries in the OS credential UI, which is confusing for a non-credential secret, and has a per-credential size limit. DPAPI + file is simpler and sufficient. |
| DPAPI with **roaming** profiles | Avoid. The blob can be exported to the domain and decrypted elsewhere, which would defeat device binding. |
| Plain file + "hidden" attribute | Not security. Explicitly rejected. |
| Hardcoded AES key in the binary | Worse than nothing; the key is public. Explicitly rejected. |

### Other platforms (later)
- Linux: Secret Service API (`libsecret`) → `gnome-keyring` / `kwallet`
- macOS: Keychain via `SecItemAdd`/`SecItemCopyMatching`

### Non-sensitive data
Stays in plain TOML under `%LOCALAPPDATA%\Sovereign\config.toml`: UI
preferences, last-opened directory, window size, algorithm defaults. No
credentials, no tokens.

---

## 5. Data model

PostgreSQL. The local machine **never** receives any of these tables.

```sql
-- accounts
CREATE TABLE users (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email           CITEXT NOT NULL UNIQUE,
  password_hash   TEXT NOT NULL,              -- Argon2id
  display_name    TEXT NOT NULL DEFAULT '',
  email_verified  BOOLEAN NOT NULL DEFAULT FALSE,
  disabled_at     TIMESTAMPTZ,                -- NULL = active
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- one row per installation
CREATE TABLE devices (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  label           TEXT NOT NULL DEFAULT '',
  -- Client-generated install ID, random UUID. NOT a hardware fingerprint:
  -- hardware IDs are spoofable and create privacy problems, so they are
  -- deliberately not used as a security mechanism.
  install_id      UUID NOT NULL UNIQUE,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_seen_at    TIMESTAMPTZ,
  revoked_at      TIMESTAMPTZ
);
CREATE INDEX devices_user_idx ON devices(user_id);

-- server-issued offline entitlements
CREATE TABLE offline_grants (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id           UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  device_id         UUID NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
  key_id            TEXT NOT NULL,            -- which signing key signed it
  credential_version INT NOT NULL DEFAULT 1,
  issued_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at        TIMESTAMPTZ NOT NULL,
  revoked_at        TIMESTAMPTZ
);
CREATE INDEX grants_device_idx ON offline_grants(device_id) WHERE revoked_at IS NULL;

-- online session rotation (hash stored, never the token)
CREATE TABLE refresh_tokens (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  device_id    UUID REFERENCES devices(id) ON DELETE SET NULL,
  token_hash   TEXT NOT NULL,                 -- SHA-256 of the token
  issued_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at   TIMESTAMPTZ NOT NULL,
  revoked_at   TIMESTAMPTZ,
  rotated_from UUID REFERENCES refresh_tokens(id)
);

-- audit trail for a security-relevant product
CREATE TABLE auth_audit (
  id          BIGSERIAL PRIMARY KEY,
  user_id     UUID REFERENCES users(id) ON DELETE SET NULL,
  device_id   UUID,
  event       TEXT NOT NULL,   -- login_failed, device_activated, device_revoked, …
  ip_hash     TEXT,            -- hashed, not raw
  at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX auth_audit_user_idx ON auth_audit(user_id, at DESC);
```

### Local storage, for contrast

`%LOCALAPPDATA%\Sovereign\`
```
config.toml            plain, non-sensitive preferences
install.dat            random install UUID + DPAPI entropy (not secret)
credentials\<key>.bin  DPAPI envelope, per-user scope
logs\                  rotating local logs
```

That is the complete local footprint. **No user table, no password, no database
file, no token that works online.**

---

## 6. Endpoints

### Hosted auth service (`services/auth`, separate deploy)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/auth/register` | create account, send verification mail |
| `POST` | `/v1/auth/login` | verify credentials, register/refresh device, issue offline credential |
| `POST` | `/v1/auth/refresh` | rotate refresh token, silently re-issue offline credential |
| `POST` | `/v1/auth/logout` | revoke refresh token, optionally revoke device |
| `GET`  | `/v1/me` | current account (used by the website) |
| `GET`  | `/v1/devices` | list devices, so a user can revoke one |
| `DELETE` | `/v1/devices/{id}` | revoke a device |
| `GET`  | `/v1/health` | liveness |

### Local server (`sovereign-server`, loopback only)

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET`  | `/api/auth/status` | no | `online` / `offline` / `activated` / `anonymous` — never conflates the two failure kinds |
| `POST` | `/api/auth/login` | no + CSRF | online sign-in, stores credential via DPAPI |
| `POST` | `/api/auth/logout` | session | delete credential, clear session. **Works offline.** |
| `GET`  | `/api/auth/me` | session | `{user_id, display_name, email, device_id, offline, expires_at}` |
| `GET`  | `/api/system` | no | version, CPU, threads, capabilities |
| `GET`  | `/api/system/gpu` | no | GPU detection, CUDA availability, honest about absence |
| `GET`  | `/api/problems` | session | list models found on disk |
| `POST` | `/api/problems` | session | open a local model, path-validated |
| `POST` | `/api/jobs` | session | enqueue a solve, returns `job_id` immediately |
| `GET`  | `/api/jobs/{id}` | session | status snapshot |
| `GET`  | `/api/jobs/{id}/events` | session | **SSE** progress stream |
| `POST` | `/api/jobs/{id}/cancel` | session | cooperative cancellation |
| `GET`  | `/api/jobs/{id}/solution` | session | verified solution + verification report |
| `GET`  | `/api/benchmarks` | session | results from the offline harness |

**The frontend never receives the offline credential.** It gets an opaque local
session cookie and asks `GET /api/auth/me`. The long-lived signed blob stays in
DPAPI, touched only by the native `AuthManager`.

---

## 7. Local HTTP security

Binding to `127.0.0.1` is necessary but **not sufficient**. Any website the user
visits can issue requests to `http://127.0.0.1:<port>` — the browser will not
stop it, and there is no meaningful CORS preflight for simple GETs. Defences:

1. **Bind loopback only.** Never `0.0.0.0` without an explicit flag.
2. **Host header validation.** Reject any `Host` that is not
   `127.0.0.1[:port]` / `localhost[:port]`. Blocks DNS-rebinding attacks, where
   an attacker-controlled domain resolves to `127.0.0.1` and then talks to the
   local API from the attacker's origin.
3. **Origin validation.** For any state-changing method, require `Origin` to be
   absent (non-browser client) or to equal our own loopback origin.
4. **CSRF token.** Random 256-bit token issued in the session, required in a
   custom header (`X-Sovereign-CSRF`). Not readable cross-origin.
5. **Session cookie:** `HttpOnly`, `SameSite=Strict`, `Path=/`, no `Domain`
   attribute, short idle expiry, regenerated on login.
6. **No `Access-Control-Allow-Origin: *`** anywhere. No CORS headers at all is
   the correct default for a single-origin local app.
7. **Path traversal.** Model paths are resolved and then checked to be inside
   an allowlisted root. Reject `..`, absolute paths outside the root, and
   symlink escapes (`std::filesystem::weakly_canonical` + prefix compare).
8. **Request limits.** Cap body size (models are JSON/MPS text; 32 MB is
   generous), header count, and URL length. Refuse to buffer more.
9. **Never execute frontend-supplied commands.** The API accepts structured
   options (algorithm, time limit, threads) and maps them to a typed enum. There
   is no endpoint that takes a shell command or an arbitrary solver argument.
10. **No admin privileges required.** DPAPI per-user, a high ephemeral port, and
    a user-writable data directory all work unprivileged.

---

## 8. Tests required

Unit and integration:

1. First online login succeeds
2. Invalid credentials fail
3. **Password is never persisted locally** (assert no file under the data dir
   contains it)
4. Valid offline credential verifies
5. **Tampered** credential (any byte flipped) fails
6. App starts with no network after activation
7. Logout removes local authorization
8. App cannot authenticate offline after logout
9. Corrupted credential fails *safely* (no crash, no partial state)
10. Wrong-device credential fails when device binding is enforced
11. Valid server signature verifies
12. Invalid signature fails
13. Localhost session cookie is issued and honoured
14. Unauthenticated API requests are rejected
15. Server does **not** bind publicly by default

Plus an end-to-end scenario:

```
ONLINE   install → launch → login → dashboard → solve
OFFLINE  close → disconnect network → launch
         → NO login prompt → dashboard → solve same problem
LOGOUT   logout while offline → close → launch → login required
```

Test approach that does not require physically disconnecting a network:
a `SovereignNetworkPolicy` seam in `AuthManager` that, when set to
`ForceOffline`, makes the online client fail closed **without** touching the
DPAPI path. The offline code path is therefore genuinely exercised rather than
merely asserted. A manual test on a genuinely disconnected machine is still
required before release, and the manual script is in
`docs/testing-offline.md`.

---

## 9. Threat model — what this does and does not achieve

**Protects against**
- Casual reading of the credential blob by another user on the machine
  (per-user DPAPI scope)
- Extraction of the secret from a config file, since there is no config file
- Forging a credential without the server private key (Ed25519)
- Using a credential issued for device A on device B (device binding)
- Password exposure through local storage (no password is ever stored)
- Cross-origin drive-by requests from a malicious website (Origin + Host +
  CSRF + SameSite)

**Does NOT protect against** — and we do not claim otherwise:
- An administrator, kernel-level malware, or a debugger attached to the process.
  DPAPI offers no defence once code is executing as the user with debug rights.
- A user who patches the shipped binary to replace the embedded public key.
  Client-side verification necessarily trusts the client binary.
- A determined attacker with the user's own password, at a time when the device
  is online.
- Rollback of the system clock to resurrect an expired credential, absent
  trusted-time mechanisms. Mitigation: monotonic "last seen" timestamp in the
  DPAPI-protected blob; not a complete defence.
- **A revoked device continues to work offline until its credential expires.**
  This is inherent to offline-first authorization, not an oversight. It is why
  credential expiry policy matters.

The user owns the machine. The goal is no plaintext credentials, OS-protected
storage, unforgeable server signatures, device-scoped authorization, and minimal
local data. Not invulnerability.

---

## 10. Website → desktop handoff

Deferred. Implementing website login and application login independently first
is explicitly sanctioned by the specification, and the handoff adds real
complexity.

The intended design, for the record: the website, already authenticated, issues
a **single-use, ~60-second authorization code**; the desktop app exchanges it
once over HTTPS for a device credential. Passwords, access tokens and offline
credentials never appear in a custom-protocol URL or a query parameter — query
strings end up in browser history, OS process listings and proxy logs.

---

# Provisioning checklist — what we need from you

## Blocking: cannot implement the auth service without these

| # | Item | Why | Notes |
|---|---|---|---|
| 1 | **Cloud provider + region** | where the auth service runs | Azure / AWS / GCP / an existing VPS |
| 2 | **Domain name** | TLS for the auth API | you own it; DNS access needed |
| 3 | **PostgreSQL connection string** | `users`, `devices`, `offline_grants`, `refresh_tokens`, `auth_audit` | managed (Neon, Azure Database for PostgreSQL, Supabase, RDS) or self-hosted Docker Compose. I can write the Compose file if you prefer self-hosting |
| 4 | **Ed25519 signing key decision** | who holds the private key | **(a) dev keypair** — I generate it now, public key compiled in, private key in your `.env`; lets us build and test the entire flow today with zero cloud spend. **(b) production KMS** — Azure Key Vault / GCP KMS / AWS KMS; correct for release. I recommend (a) immediately, (b) before release. |
| 5 | **TLS termination** | HTTPS for the auth API | managed certificate (Key Vault / ACM) or Caddy/Traefik with Let's Encrypt |

## Needed for a real release, not for the core build

| # | Item | Why |
|---|---|---|
| 6 | **Transactional email provider** API key + verified sender domain | account verification and device-alert emails. SendGrid / Postmark / AWS SES / Azure Communication Services. Can be skipped if registration is invite-only and unverified |
| 7 | **Binary hosting** | where users download the installer. GitHub Releases (free, repo already exists) or S3 / Azure Blob + CDN. If blob storage: bucket name + public read URL |
| 8 | **Windows code-signing certificate** (optional, recommended) | unsigned `.exe` triggers Microsoft SmartScreen on every user's machine. OV certificate ≈ $200–400/year. Not needed for a demo or hackathon submission |
| 9 | **Secret manager** for runtime env vars | `.env` on disk is fine for development only |

## Decisions, not credentials

| Question | Recommendation |
|---|---|
| Dev signing key now? | **Yes.** Unblocks the whole auth implementation today |
| Registration policy? | **Invite-only.** Open registration on a public site attracts spam and abuse cost |
| Offline credential expiry? | **90 days, refreshed silently when online.** Unlimited is friendliest UX but means a stolen laptop works forever; 30 days is stricter than most users will tolerate |
| Auth service in this repo? | **`services/auth/`, separate deployable.** `sovereign_core` must never link it |
| Is Python acceptable for the hosted auth service only? | **Yes, strongly recommended.** It never ships to users. FastAPI + SQLAlchemy + Argon2 would move this forward far faster than C++ |
| OAuth providers (Google/GitHub)? | Later. Adds an external dependency to the login path |

## Not needed at all

- No GPU/cloud credentials for the solver. The engine is local and offline.
- No external optimization solver. HiGHS/others remain benchmark-only, offline.
- No analytics, telemetry or crash-reporting account. The product must not
  depend on any of it.
