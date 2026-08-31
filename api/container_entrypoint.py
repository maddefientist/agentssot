"""Owned container launcher with fail-closed gateway worker admission.

Gateway tickets and abuse-control state are process-local. Until those controls
move to a shared atomic backend, an enabled gateway must never start through a
multi-worker ASGI launcher.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Mapping, Sequence


_TRUE_VALUES = {"1", "true", "yes", "on"}
_LAUNCHER_ATTESTATION = "single-worker-v1"


def _enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUE_VALUES


def _uvicorn_worker_count(argv: Sequence[str], env: Mapping[str, str]) -> int:
    """Resolve every supported worker input, rejecting ambiguity or bad values."""
    values: list[tuple[str, str]] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--workers":
            if index + 1 >= len(argv):
                raise RuntimeError("--workers requires an integer value")
            values.append(("--workers", argv[index + 1]))
            index += 2
            continue
        if argument.startswith("--workers="):
            values.append(("--workers", argument.split("=", 1)[1]))
        index += 1

    for name in ("WEB_CONCURRENCY", "UVICORN_WORKERS"):
        if env.get(name) is not None:
            values.append((name, env[name]))

    counts: list[tuple[str, int]] = []
    for source, raw_value in values:
        try:
            count = int(raw_value)
        except ValueError as exc:
            raise RuntimeError(f"{source} must be an integer when the gateway is enabled") from exc
        if count < 1:
            raise RuntimeError(f"{source} must be at least 1")
        counts.append((source, count))

    unsafe = [(source, count) for source, count in counts if count != 1]
    if unsafe:
        source, count = unsafe[0]
        raise RuntimeError(
            "GATEWAY_ENABLED requires exactly one API worker; "
            f"{source} requested {count}. Disable the gateway or add a shared "
            "ticket/rate/revocation backend."
        )
    return 1


def validate_command(argv: Sequence[str], env: Mapping[str, str]) -> None:
    if not _enabled(env.get("GATEWAY_ENABLED")):
        return
    if not argv:
        raise RuntimeError("container command is required")

    # The production image owns this launcher. An enabled gateway is admitted
    # only through the known Uvicorn command whose worker semantics we validate.
    if Path(argv[0]).name != "uvicorn" or "app.main:app" not in argv[1:]:
        raise RuntimeError(
            "GATEWAY_ENABLED requires the owned single-worker "
            "'uvicorn app.main:app' container command"
        )
    _uvicorn_worker_count(argv[1:], env)


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    command = list(sys.argv[1:] if argv is None else argv)
    environment = os.environ if env is None else env
    try:
        validate_command(command, environment)
    except RuntimeError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 78
    if env is None and _enabled(os.environ.get("GATEWAY_ENABLED")):
        # app.main requires this non-secret safety marker. It prevents an
        # accidental direct `uvicorn --workers N` invocation from bypassing
        # the owned launcher's worker validation.
        os.environ["AGENTSSOT_LAUNCHER_ATTESTATION"] = _LAUNCHER_ATTESTATION
    os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
