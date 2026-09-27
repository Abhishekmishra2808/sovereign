"""Offline asset audit.

Fails the build if the shipped bundle references ANY remote origin. The
product's core promise is that it runs with the network disconnected, and a
single stray CDN <link>, font @import, analytics snippet or remote image
silently breaks that promise in a way that is easy to miss locally.

Checks the built output in api/static, not just the source, because the source
can be clean while a transitive dependency (or a Vite plugin) injects something.

Usage:  python benchmarks/tools/check_offline_assets.py [--dir api/static]
"""
from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]

TEXT_SUFFIXES = {".html", ".css", ".js", ".mjs", ".json", ".svg", ".map"}
SCAN_SUFFIXES = TEXT_SUFFIXES | {".woff2", ".woff", ".ttf"}

# Any http(s) URL is a finding. The comment below is the only tolerated class:
# a licence header or documentation link inside a vendored third-party file is
# inert and does not cause a network request.
ALLOWLIST_HOSTS = {
    # Present in vendored licence headers and inline source comments. Never
    # fetched at runtime.
    "opensource.org",
    "www.apache.org",
    "github.com",
    "www.w3.org",
    "creativecommons.org",
    "127.0.0.1",
    "localhost",
    # React embeds these as the text of thrown Error messages ("See
    # https://react.dev/errors/..."). They are never dereferenced; the
    # minifier cannot know that, so they survive into the bundle. Not a
    # network dependency.
    "react.dev",
    # Licence attribution. BubbledotICG-FinePos is CC BY 4.0 and its licence
    # requires crediting the author, so the landing page carries a link to
    # OnlineWebFonts. That is a user-initiated navigation in an <a href>, not a
    # subresource fetch: nothing is requested unless a human clicks it. Removing
    # the link to satisfy this audit would be a licence violation, so it is
    # allowlisted instead, with the reason recorded.
    "www.onlinewebfonts.com",
    "onlinewebfonts.com",
}

# No remote runtime fallback is permitted in the shipped application.
OPTIONAL_FALLBACK_HOSTS: dict[str, str] = {}

URL_RE = re.compile(r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+")


class ResourceParser(HTMLParser):
    """Active HTML resources must never inherit the documentation allowlist."""

    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"script", "img", "source", "video", "audio", "iframe", "embed"}:
            self.urls.extend(attrs.get(key, "") for key in ("src", "poster"))
            self.urls.extend(part.strip().split(" ")[0] for part in attrs.get("srcset", "").split(","))
        if tag == "link" and set(attrs.get("rel", "").split()) & {"stylesheet", "preload", "modulepreload", "prefetch", "icon"}:
            self.urls.append(attrs.get("href", ""))
        if tag == "object":
            self.urls.append(attrs.get("data", ""))


def host_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def scan(target: Path) -> tuple[list[tuple[Path, str, str]], list[tuple[Path, str, str]]]:
    """Returns (blocking_findings, optional_fallback_findings)."""
    findings: list[tuple[Path, str, str]] = []
    optional: list[tuple[Path, str, str]] = []
    for path in sorted(target.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SCAN_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            findings.append((path.relative_to(target), "unreadable", str(exc)))
            continue
        active = []
        if path.suffix.lower() == ".html":
            parser = ResourceParser()
            parser.feed(text)
            active.extend(parser.urls)
        if path.suffix.lower() in {".css", ".html"}:
            active.extend(re.findall(r"url\(\s*['\"]?([^)'\"\s]+)", text, re.I))
            active.extend(re.findall(r"@import\s+['\"]([^'\"]+)", text, re.I))
        # Literal network calls in JS are active even on allowlisted domains.
        if path.suffix.lower() in {".js", ".mjs"}:
            active.extend(re.findall(r"(?:fetch|import|WebSocket|EventSource)\s*\(\s*['\"]([^'\"]+)", text))
        for url in active:
            if url.startswith(("http://", "https://", "//", "ws://", "wss://")):
                findings.append((path.relative_to(target), host_of("https:" + url if url.startswith("//") else url), url))
        # Strip line comments so we do not flag prose in third-party headers.
        text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
        text = re.sub(r"(?m)^\s*(//|\*).*$", " ", text)
        for match in URL_RE.finditer(text):
            url = match.group(0).rstrip(".,;)'\"")
            host = host_of(url)
            if not host or host in ALLOWLIST_HOSTS:
                continue
            record = (path.relative_to(target), host, url)
            if host in OPTIONAL_FALLBACK_HOSTS:
                optional.append(record)
            else:
                findings.append(record)
    return findings, optional


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default=str(ROOT / "api" / "static"),
        help="built asset directory to audit",
    )
    args = parser.parse_args()

    target = Path(args.dir)
    if not target.is_dir():
        print(f"ERROR: {target} does not exist. Run the web build first.", file=sys.stderr)
        return 2

    print(f"auditing {target} for remote references...")
    if not any(p.is_file() and p.suffix.lower() in TEXT_SUFFIXES for p in target.rglob("*")):
        print("ERROR: no web assets found", file=sys.stderr)
        return 2
    findings, optional = scan(target)

    if optional:
        print("\n  optional progressive fallbacks (page works without them):")
        by_host: dict[str, int] = {}
        for _, host, _ in optional:
            by_host[host] = by_host.get(host, 0) + 1
        for host, count in sorted(by_host.items()):
            print(f"    {count:3d}x {host}  -- {OPTIONAL_FALLBACK_HOSTS.get(host, '')}")
        print()

    if not findings:
        count = sum(1 for p in target.rglob("*") if p.is_file())
        print(f"PASS: {count} files scanned, zero blocking remote references.")
        print("      Static reference audit passed; this is not a runtime network trace.")
        return 0

    print(f"FAIL: {len(findings)} remote reference(s) found.\n", file=sys.stderr)
    by_host: dict[str, int] = {}
    for _, host, _ in findings:
        by_host[host] = by_host.get(host, 0) + 1
    for host, count in sorted(by_host.items(), key=lambda kv: -kv[1]):
        print(f"  {count:4d}x  {host}", file=sys.stderr)
    print("", file=sys.stderr)
    for path, host, url in findings[:25]:
        print(f"  {path}: {url[:120]}", file=sys.stderr)
    if len(findings) > 25:
        print(f"  ... and {len(findings) - 25} more", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
