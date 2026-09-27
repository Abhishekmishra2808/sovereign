"""Download the display and UI fonts Sovereign's landing page needs, and rewrite
the CSS to point at local copies.

WHY A SCRIPT
------------
The design brief specifies Inter, BubbledotICG-FinePos and Font Awesome loaded
from three different CDNs. That is a direct conflict with the offline-first
requirement: a page that pulls fonts from fonts.googleapis.com cannot render
correctly with the network disconnected, and the hero falls back to a
different typeface.

So the *visuals* are honoured exactly, but the bytes are served locally. This
script fetches them once and emits self-hosted @font-face rules. The result is
byte-identical typography with no network dependency.

Font Awesome is the awkward one: it is an icon font, not a typeface we can
restyle. We keep it loaded from cdnjs (it degrades to empty icon slots offline
rather than breaking layout) and inline the three brand glyphs we actually use
as SVG, so the trust row looks right either way.

Run:  python benchmarks/tools/fetch_site_fonts.py
"""
from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "site" / "fonts"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

SOURCES = {
    # Display face: retro dot-matrix, the brief's primary headline font.
    "BubbledotICG-FinePos": (
        "https://db.onlinewebfonts.com/c/8cb707a9b8a73f8a7403336b861c3074"
        "?family=BubbledotICG-FinePos"
    ),
    # UI face.
    "Inter": (
        "https://fonts.googleapis.com/css2"
        "?family=Inter:wght@400;500;600;700&display=swap"
    ),
}

KEEP_SUBSETS = {"latin", "latin-ext"}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    faces: list[str] = []

    for family, url in SOURCES.items():
        try:
            css = fetch(url).decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR fetching {family}: {exc}", file=sys.stderr)
            return 1

        # Some providers (OnlineWebFonts in particular) serve minified CSS with
        # everything on a handful of lines, so a block regex anchored on a
        # trailing newline silently matches nothing. Split on @font-face
        # boundaries instead, which works for both pretty-printed and minified
        # input.
        chunks = re.split(r"@font-face", css)[1:]
        if not chunks:
            print(f"ERROR: no @font-face parsed for {family}", file=sys.stderr)
            return 1

        kept = 0
        for chunk in chunks:
            # Keep only the declaration body, and drop any unicode-range that
            # names a subset we are not shipping.
            body_m = re.search(r"\{(.*?)\}", chunk, re.S)
            if body_m is None:
                continue
            block = "@font-face{" + body_m.group(1) + "}"

            range_m = re.search(r"unicode-range:\s*([^;}]+)", block)
            if range_m:
                subset = "latin-ext" if "U+0100" in range_m.group(1) else "latin"
                if subset not in KEEP_SUBSETS:
                    continue
            else:
                # No unicode-range at all: a single-face stylesheet, ship it.
                subset = "latin"

            # Providers list several formats in one src declaration, eot first.
            # Taking the first URL yields an EOT saved under a .woff2 name, which
            # the browser rejects. Select the woff2 explicitly.
            urls = re.findall(r"url\(\s*[\"']?(https?://[^)\"']+)", block)
            if not urls:
                continue
            woff2 = next((u for u in urls if u.lower().split("?")[0].endswith(".woff2")), None)
            if woff2 is None:
                print(f"  WARN {family}: no woff2 source in this face")
                continue

            weight_m = re.search(r"font-weight:\s*(\d+|normal|bold|bolder|lighter)", block)
            raw_weight = (weight_m.group(1) if weight_m else "400").lower()
            weight = {"normal": "400", "bold": "700", "bolder": "700", "lighter": "300"}.get(
                raw_weight, raw_weight
            )
            name = f"{family.lower()}-{weight}-{subset}.woff2"
            dest = OUT / name
            if not dest.exists():
                try:
                    dest.write_bytes(fetch(woff2))
                except Exception as exc:  # noqa: BLE001
                    print(f"  WARN {name}: {exc}")
                    continue

            # Rebuild the @font-face from parsed fields rather than trying to
            # strip the provider's src list in place. OnlineWebFonts emits both a
            # standalone `src: url(...eot);` line and a comma-separated
            # woff/woff2/ttf/svg list, and a regex that misses either leaves a
            # live CDN reference -- which breaks offline rendering and fails the
            # asset audit. Constructing the block means only the one woff2 URL we
            # downloaded can possibly appear.
            def field(pattern: str, default: str = "") -> str:
                m = re.search(pattern, block)
                return m.group(1).strip() if m else default

            family_name = field(r"font-family:\s*([\"']?)([^\"';]+)\1")
            weight_val = raw_weight
            style_val = field(r"font-style:\s*([^;}]+)", "normal")
            display_val = field(r"font-display:\s*([^;}]+)", "swap")
            range_val = field(r"unicode-range:\s*([^;}]+)")

            rebuilt = (
                "@font-face {\n"
                f"  font-family: '{family_name}';\n"
                f"  font-style: {style_val};\n"
                f"  font-weight: {weight_val};\n"
                f"  font-display: {display_val};\n"
                f"  src: url('fonts/{name}') format('woff2');\n"
            )
            if range_val:
                rebuilt += f"  unicode-range: {range_val};\n"
            rebuilt += "}\n"

            faces.append(rebuilt)
            print(f"  {name:<34} {dest.stat().st_size // 1024:>4} KB")
            kept += 1
        print(f"{family}: kept {kept} face(s) from {len(chunks)} block(s)")

    # Hard assertion. If any remote reference survived, the offline guarantee is
    # already broken, so fail loudly here rather than letting it reach the audit.
    combined = "\n".join(faces)
    leaked = sorted(set(re.findall(r"https?://[^\s)'\"]+", combined)))
    if leaked:
        print("\nERROR: remote references remain in the generated fonts.css:", file=sys.stderr)
        for url in leaked:
            print(f"  {url}", file=sys.stderr)
        return 1

    header = (
        "/* Self-hosted fonts for the Sovereign landing page.\n"
        " *\n"
        " * Generated by benchmarks/tools/fetch_site_fonts.py. Inter and\n"
        " * BubbledotICG-FinePos are served from here rather than from\n"
        " * fonts.googleapis.com / db.onlinewebfonts.com so the page renders\n"
        " * identically with the network disconnected.\n"
        " *\n"
        " * Do not reintroduce a remote @import; it will break the offline\n"
        " * guarantee and fail benchmarks/tools/check_offline_assets.py.\n"
        " */\n"
    )
    (OUT / "fonts.css").write_text(header + "\n".join(faces) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT / 'fonts.css'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
