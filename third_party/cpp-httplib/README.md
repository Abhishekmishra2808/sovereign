# cpp-httplib

This directory vendors a single header, `httplib.h`, from
<https://github.com/yhirose/cpp-httplib> (tag `v0.18.3`).

## Why

The local application server must ship **inside the release binary** with no
runtime dependency, and must support Server-Sent Events for live solve progress.
cpp-httplib is header-only, MIT licensed, and already provides both.

## Why not the Python server

The previous local server was FastAPI + uvicorn. That works for development but
violates the packaging requirement that a user must not need Python, pip, or a
virtualenv after installing a release. The alternative — freezing Python with
PyInstaller — produces a ~40 MB bundle that antivirus software flags, and it
still leaves a Python process resident for the lifetime of the app.

Moving the transport layer to C++ keeps the process count at two (launcher +
server), leaves `sovereign_core` completely untouched, and means the shipped
artifact is a handful of exes plus static assets.

## Scope of use

Sovereign uses only transport concerns:

- Listening on `127.0.0.1` **only** (never `0.0.0.0`)
- Routing, JSON bodies, static file serving
- Server-Sent Events for job progress
- Per-request `Origin` / `Host` validation, implemented by Sovereign
- Graceful shutdown

It is **not** a framework Sovereign's design depends on. All routing, policy,
session handling and solver invocation live in `solver/server/src/` and would
survive a swap to another transport.

## Security note

cpp-httplib does not do authentication, CSRF, or authorization. Sovereign
implements all of that itself in `auth_manager.cpp` and `routes.cpp`. Do not
assume the header provides any protection — it does not, and it is not
maintained by this project.

## License

MIT. See `LICENSE`.
