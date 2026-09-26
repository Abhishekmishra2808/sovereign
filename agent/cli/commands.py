"""Slash-command handlers for the sovereign agent CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from agent.schemas.models import OptimizationModel

if TYPE_CHECKING:
    from agent.cli.app import Session
    from agent.cli.render import Renderer


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "models"


def handle_command(session: "Session", ui: "Renderer", line: str) -> bool:
    """Return False to exit the REPL."""
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit", "/q"):
        ui.info("bye")
        return False

    if cmd in ("/help", "/h", "/?"):
        ui.help()
        return True

    if cmd == "/status":
        ui.status_line(session.agent.llm_enabled, session.tools.sovereign_bin, session.model is not None)
        if not session.agent.llm_enabled:
            ui.tip("Add LLM credentials to .env for natural-language chat")
        return True

    if cmd == "/clear":
        session.model = None
        session.last = None
        session.history.clear()
        ui.ok("session cleared")
        return True

    if cmd == "/examples":
        if not EXAMPLES.exists():
            ui.err("examples/models not found")
            return True
        for p in sorted(EXAMPLES.glob("*.json")):
            ui.tip(str(p.relative_to(ROOT)))
        ui.tip("load with: /load examples/models/sample_lp.json")
        return True

    if cmd == "/load":
        if not arg:
            ui.err("usage: /load <file.json>")
            return True
        path = Path(arg)
        if not path.is_file():
            alt = ROOT / arg
            if alt.is_file():
                path = alt
            else:
                ui.err(f"file not found: {arg}")
                return True
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            session.model = OptimizationModel.model_validate(data)
            session.model_path = str(path)
            session.last = None
            ui.ok(f"loaded {path}  ({session.model.problem_type.value}, {len(session.model.variables)} vars)")
        except Exception as ex:  # noqa: BLE001
            ui.err(str(ex))
        return True

    if cmd == "/model":
        if session.model is None:
            ui.warn("no model loaded — use /load or chat")
            return True
        m = session.model
        ui.agent(
            f"{m.problem_type.value} | {m.sense.value}\n"
            f"variables   {len(m.variables)}\n"
            f"constraints {len(m.constraints)}\n"
            f"path        {session.model_path or '(in-memory)'}"
        )
        return True

    if cmd == "/solve":
        path: Optional[Path] = None
        if arg:
            path = Path(arg)
            if not path.is_file():
                path = ROOT / arg
            if not path.is_file():
                ui.err(f"file not found: {arg}")
                return True
            session.model = OptimizationModel.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
            session.model_path = str(path)

        if session.model is None:
            ui.err("no model — /load <file.json> or /solve <file.json>")
            return True

        try:
            payload = session.agent.solve_structured(session.model, on_event=ui.tool_event)
            session.last = payload
            ui.result_card(payload)
        except Exception as ex:  # noqa: BLE001
            ui.err(str(ex))
        return True

    if cmd == "/result":
        if not session.last:
            ui.warn("no result yet")
            return True
        ui.result_card(session.last)
        return True

    if cmd == "/verify":
        if session.model is None or not session.last or "result" not in session.last:
            ui.err("need a loaded model and a prior /solve")
            return True
        ui.tool("verify_solution")
        vr = session.tools.verify_solution(session.model, session.last["result"])
        session.last["verification"] = vr.model_dump()
        if vr.is_valid:
            ui.ok(vr.message)
        else:
            ui.err(vr.message)
            for issue in vr.issues:
                ui.warn(issue)
        return True

    if cmd == "/math":
        if not arg:
            ui.err("usage: /math <expression>")
            return True
        try:
            out = session.tools.math_calculator(arg)
            ui.ok(f"{out['expression']} = {out['value']}")
        except Exception as ex:  # noqa: BLE001
            ui.err(str(ex))
        return True

    ui.err(f"unknown command: {cmd}  (try /help)")
    return True


def autocomplete_hint(partial: str) -> List[str]:
    cmds = [
        "/help",
        "/status",
        "/load",
        "/solve",
        "/model",
        "/result",
        "/verify",
        "/math",
        "/clear",
        "/examples",
        "/exit",
    ]
    return [c for c in cmds if c.startswith(partial)]
