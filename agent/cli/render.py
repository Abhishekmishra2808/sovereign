"""Terminal rendering — Claude Code style: one live step + spinner."""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional


def _supports_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def _safe_write(text: str) -> None:
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        sys.stdout.write(text.encode(enc, errors="replace").decode(enc, errors="replace"))
        sys.stdout.flush()


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    WHITE = "\033[37m"


class Renderer:
    def __init__(self, color: Optional[bool] = None) -> None:
        self.color = _supports_color() if color is None else color
        self._lock = threading.Lock()
        self._spin_stop = threading.Event()
        self._spin_thread: Optional[threading.Thread] = None
        self._spin_label = ""

    def _c(self, code: str, text: str) -> str:
        if not self.color:
            return text
        return f"{code}{text}{Style.RESET}"

    def banner(self) -> None:
        line = "-" * 56
        _safe_print("")
        _safe_print(self._c(Style.CYAN + Style.BOLD, line))
        _safe_print(self._c(Style.CYAN + Style.BOLD, "  sovereign"))
        _safe_print(self._c(Style.DIM, "  LLM formulates  |  tools solve  |  verifier checks"))
        _safe_print(self._c(Style.CYAN + Style.BOLD, line))
        _safe_print("")

    def tip(self, text: str) -> None:
        self._stop_spinner()
        _safe_print(self._c(Style.DIM, f"  {text}"))

    def info(self, text: str) -> None:
        self._stop_spinner()
        _safe_print(self._c(Style.BLUE, f"i  {text}"))

    def ok(self, text: str) -> None:
        self._stop_spinner()
        _safe_print(self._c(Style.GREEN, f"+  {text}"))

    def warn(self, text: str) -> None:
        self._stop_spinner()
        _safe_print(self._c(Style.YELLOW, f"!  {text}"))

    def err(self, text: str) -> None:
        self._stop_spinner()
        _safe_print(self._c(Style.RED, f"x  {text}"))

    def agent(self, text: str) -> None:
        self._stop_spinner()
        _safe_print(self._c(Style.MAGENTA + Style.BOLD, "agent"))
        for line in text.splitlines() or [""]:
            _safe_print(f"  {line}")

    def tool(self, name: str, detail: str = "") -> None:
        self._stop_spinner()
        msg = f"*  {name}"
        if detail:
            msg += f"  {detail}"
        _safe_print(self._c(Style.CYAN, msg))

    def _kind_tag(self, kind: str) -> str:
        if kind == "llm":
            return self._c(Style.MAGENTA + Style.BOLD, "llm")
        if kind == "tool":
            return self._c(Style.CYAN + Style.BOLD, "tool")
        return self._c(Style.BLUE + Style.BOLD, "sys")

    def _format_step_line(self, kind: str, name: str, detail: str, mark: str, mark_color: str) -> str:
        left = f"  {self._c(mark_color, mark)} {self._kind_tag(kind)}  {self._c(Style.BOLD, name)}"
        if detail:
            left += self._c(Style.DIM, f"  -  {detail}")
        return left

    def _clear_line(self) -> None:
        _safe_write("\r" + (" " * 96) + "\r")

    def _start_spinner(self, kind: str, name: str, detail: str) -> None:
        self._stop_spinner()
        tag = {"llm": "llm", "tool": "tool"}.get(kind, "sys")
        label_core = f"{tag}  {name}"
        if detail:
            label_core += f"  -  {detail}"
        self._spin_label = label_core
        self._spin_stop.clear()

        def run() -> None:
            enc = (getattr(sys.stdout, "encoding", None) or "").lower()
            use_unicode = "utf" in enc
            frames = (
                ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                if use_unicode
                else ["|", "/", "-", "\\"]
            )
            i = 0
            while not self._spin_stop.wait(0.08):
                frame = frames[i % len(frames)]
                with self._lock:
                    plain = f"  {frame} {self._spin_label}"
                    if len(plain) > 94:
                        plain = plain[:91] + "..."
                    text = (
                        self._c(Style.CYAN, f"  {frame}")
                        + " "
                        + self._c(Style.DIM, self._spin_label[:88])
                        if self.color
                        else plain
                    )
                    pad = max(0, 96 - len(plain))
                    _safe_write("\r" + text + (" " * pad))
                i += 1

        self._spin_thread = threading.Thread(target=run, daemon=True)
        self._spin_thread.start()

    def _stop_spinner(self) -> None:
        if self._spin_thread is None:
            return
        self._spin_stop.set()
        self._spin_thread.join(timeout=1.0)
        self._spin_thread = None
        with self._lock:
            self._clear_line()

    def tool_event(self, step: Dict[str, Any]) -> None:
        """Show only the active step (spinner), then replace with a done line."""
        kind = str(step.get("kind", "tool"))
        name = str(step.get("tool", "unknown"))
        detail = str(step.get("detail", "") or "")
        # Never leak provider/deployment names in the UI
        lowered = detail.lower()
        if "azure" in lowered or "deployment" in lowered or "openai" in lowered:
            detail = ""
        status = str(step.get("status", "running"))

        if status == "running":
            self._start_spinner(kind, name, detail)
            return

        # Completed / failed: stop spinner and print one final line only
        self._stop_spinner()
        if status == "ok":
            mark, col = "+", Style.GREEN
        elif status == "error":
            mark, col = "x", Style.RED
        else:
            mark, col = "*", Style.DIM
        _safe_print(self._format_step_line(kind, name, detail, mark, col))

    def tools_summary(self, tools: List[str]) -> None:
        # Intentionally quiet — steps already streamed live.
        return

    def prompt_label(self) -> str:
        self._stop_spinner()
        return self._c(Style.GREEN + Style.BOLD, "> ")

    def status_line(self, llm: bool, solver: str, model_loaded: bool) -> None:
        self._stop_spinner()
        llm_s = self._c(Style.GREEN, "on") if llm else self._c(Style.DIM, "off")
        model_s = self._c(Style.GREEN, "loaded") if model_loaded else self._c(Style.DIM, "none")
        # Short solver label only (no long path noise)
        solver_name = "sovereign"
        _safe_print(
            self._c(Style.DIM, "  llm=")
            + llm_s
            + self._c(Style.DIM, "  solver=")
            + self._c(Style.WHITE, solver_name)
            + self._c(Style.DIM, "  model=")
            + model_s
        )

    def result_card(self, payload: Dict[str, Any]) -> None:
        self._stop_spinner()
        result = payload.get("result") or {}
        verification = payload.get("verification") or {}
        status = result.get("status", "?")
        if hasattr(status, "value"):
            status = status.value
        status = str(status)
        obj = result.get("objective_value")
        valid = verification.get("is_valid")

        color = Style.GREEN if status in ("OPTIMAL", "FEASIBLE") else Style.YELLOW
        if status in ("ERROR", "INFEASIBLE", "UNBOUNDED"):
            color = Style.RED

        _safe_print("")
        _safe_print(self._c(Style.BOLD, "  result"))
        _safe_print(self._c(color, f"  {status}") + (f"   objective {obj}" if obj is not None else ""))
        if result.get("primal"):
            assign = ", ".join(f"{k}={v}" for k, v in sorted(result["primal"].items()))
            _safe_print(self._c(Style.DIM, f"  {assign}"))
        if valid is not None:
            _safe_print(self._c(Style.GREEN if valid else Style.RED, "  verified" if valid else "  verification failed"))
        _safe_print("")

    def help(self) -> None:
        self._stop_spinner()
        cmds = [
            ("/help", "Show this help"),
            ("/status", "Session status"),
            ("/load <file.json>", "Load OptimizationModel JSON"),
            ("/solve [file]", "Solve current or given model"),
            ("/model", "Show current model summary"),
            ("/result", "Show last solver result"),
            ("/verify", "Re-verify last result"),
            ("/math <expr>", "Evaluate arithmetic"),
            ("/clear", "Clear session"),
            ("/examples", "List example models"),
            ("/exit", "Quit"),
            ("<text>", "Natural-language solve (LLM + tools)"),
        ]
        _safe_print("")
        _safe_print(self._c(Style.BOLD, "  Commands"))
        for cmd, desc in cmds:
            _safe_print(f"  {self._c(Style.CYAN, f'{cmd:<22}')}{self._c(Style.DIM, desc)}")
        _safe_print("")
        _safe_print(self._c(Style.DIM, "  Flow: formulate_model -> lp|milp|qp_solver -> verify_solution"))
        _safe_print(self._c(Style.DIM, "  LLM only writes the model JSON; tools compute the optimum."))
        _safe_print("")
