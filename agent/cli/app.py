"""Interactive sovereign agent CLI — Claude Code / Cursor-style REPL."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.cli.commands import handle_command
from agent.cli.render import Renderer
from agent.envfile import load_dotenv
from agent.orchestration.agent import OptimizationAgent
from agent.schemas.models import OptimizationModel
from agent.tools.client import ToolClient


@dataclass
class Session:
    tools: ToolClient
    agent: OptimizationAgent
    model: Optional[OptimizationModel] = None
    model_path: Optional[str] = None
    last: Optional[Dict[str, Any]] = None
    history: List[Dict[str, str]] = field(default_factory=list)


def build_session(sovereign_bin: Optional[str] = None) -> Session:
    tools = ToolClient(sovereign_bin=sovereign_bin)
    agent = OptimizationAgent(tools=tools)
    return Session(tools=tools, agent=agent)


def run_repl(session: Session, ui: Renderer) -> int:
    ui.banner()
    ui.status_line(session.agent.llm_enabled, session.tools.sovereign_bin, False)
    ui.tip("type /help | /examples | /exit")
    print()

    while True:
        try:
            line = input(ui.prompt_label()).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            ui.info("interrupted — bye")
            return 0

        if not line:
            continue

        if line.startswith("/"):
            if not handle_command(session, ui, line):
                return 0
            continue

        # Natural-language turn
        session.history.append({"role": "user", "content": line})
        if not session.agent.llm_enabled:
            ui.warn("LLM not configured — NL chat disabled")
            ui.tip("Add credentials to .env, or use /load <model.json> then /solve")
            continue

        try:
            payload = session.agent.solve_natural_language(line, on_event=ui.tool_event)
        except Exception as ex:  # noqa: BLE001
            msg = str(ex).split("\n")[0]
            for token in ("azure", "openai", "deployment"):
                if token in msg.lower():
                    msg = "LLM request failed — check .env credentials"
                    break
            ui.err(msg)
            continue

        if payload.get("error"):
            err = str(payload["error"])
            if any(t in err.lower() for t in ("azure", "openai", "deployment")):
                err = "LLM not configured"
            ui.err(err)
            if payload.get("hint"):
                hint = str(payload["hint"])
                if any(t in hint.lower() for t in ("azure", "openai", "deployment")):
                    hint = "Add credentials to .env, or use /load + /solve"
                ui.tip(hint)
            continue

        if "model" in payload:
            try:
                session.model = OptimizationModel.model_validate(payload["model"])
                session.model_path = "(from chat)"
            except Exception:
                pass

        if "result" in payload:
            session.last = {
                "model": payload.get("model"),
                "result": payload.get("result"),
                "verification": payload.get("verification"),
                "tools_used": payload.get("tools_used"),
                "trace": payload.get("trace"),
            }
            ui.result_card(session.last)

        if payload.get("explanation"):
            ui.agent(str(payload["explanation"]))
        elif payload.get("answer"):
            ui.agent(str(payload["answer"]))

    return 0


def run_oneshot(session: Session, ui: Renderer, model_path: str, verify: bool) -> int:
    path = Path(model_path)
    if not path.is_file():
        ui.err(f"file not found: {model_path}")
        return 1
    session.model = OptimizationModel.model_validate(
        __import__("json").loads(path.read_text(encoding="utf-8"))
    )
    session.model_path = str(path)
    payload = session.agent.solve_structured(session.model, on_event=ui.tool_event)
    session.last = payload
    ui.result_card(payload)
    if verify and not payload.get("verification", {}).get("is_valid"):
        return 2
    return 0 if payload.get("result", {}).get("status") in ("OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNBOUNDED") else 1


def main(argv: Optional[List[str]] = None) -> int:
    loaded = load_dotenv()
    parser = argparse.ArgumentParser(
        prog="sovereign-agent",
        description="Sovereign optimization agent CLI (interactive)",
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Optional model JSON — solve once and exit (non-interactive)",
    )
    parser.add_argument("--verify", action="store_true", help="Fail if verification fails")
    parser.add_argument("--solver", default=None, help="Path to sovereign binary")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args(argv)

    ui = Renderer(color=False if args.no_color else None)

    if args.version:
        print("sovereign-agent 1.0.0")
        return 0

    session = build_session(sovereign_bin=args.solver)
    if loaded and not args.model:
        ui.tip(f"loaded env from {loaded}")

    # Sanity: solver binary
    solver_path = Path(session.tools.sovereign_bin)
    if session.tools.sovereign_bin == "sovereign" or not solver_path.exists():
        ui.warn(f"solver binary not found at '{session.tools.sovereign_bin}'")
        ui.tip("Build first: cmake --build build -j")

    if args.model:
        return run_oneshot(session, ui, args.model, args.verify)

    return run_repl(session, ui)


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()