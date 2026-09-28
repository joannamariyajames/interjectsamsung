"""FDB-v3 Backspace Adapter.

Bridges LiveKit function tool execution with BackspaceCore without modifying
Member 1's Core or the official Full-Duplex-Bench repository.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable

from app.backspace import (
    BackspaceCore,
    DependencyKind,
    FactUpdate,
    WorkItem,
    WorkStatus,
)
from app.backspace.actions import action_id_for

if TYPE_CHECKING:
    from app.session import Session


class FDBBackspaceAdapter:
    """Application-boundary adapter between FDB-v3 tool dispatch and BACKSPACE Core."""

    def __init__(
        self,
        core: BackspaceCore,
        room_name: str,
        telemetry_path: str = "/tmp/agent_tool_calls.log",
        session: Session | None = None,
    ) -> None:
        self.core = core
        self.room_name = room_name
        self.telemetry_path = telemetry_path
        self.session = session
        self._buffer: list[dict[str, Any]] = []
        self._flushed_work_ids: set[str] = set()

    def observe_fact(
        self,
        key: str,
        value: Any,
        *,
        source: str = "user",
        turn_id: str = "",
    ) -> FactUpdate:
        """Observe or correct a fact, invalidating dependent work items."""
        if self.session is not None:
            return self.session.set_fact(key, value, source=source, turn_id=turn_id)
        update = self.core.assert_fact(key=key, value=value, source=source, turn_id=turn_id)
        if update.changeset is not None:
            self.core.invalidate(update.changeset)
        return update

    def execute_tool(
        self,
        func_name: str,
        args: dict[str, Any],
        call_fn: Callable[..., dict[str, Any]],
        *,
        turn_id: str = "",
        parent_work_id: str | None = None,
        work: WorkItem | None = None,
    ) -> dict[str, Any]:
        """Execute an FDB tool call gated by BackspaceCore.

        1. Creates or uses the WorkItem for this invocation.
        2. Checks immediate staleness if an existing WorkItem is provided.
        3. Maps tool arguments to facts and registers FACT_TO_WORK dependencies.
        4. Registers WORK_TO_WORK if parent_work_id is provided.
        5. Checks pre-execution staleness gate. If stale/invalidated/retracted/superseded,
           suppresses call_fn, avoids telemetry logging, and returns cancelled status.
        6. If valid, executes call_fn(**args) and immediately appends the call to the
           official telemetry log with exact start/end timestamps - executed calls
           are always reported - then sets WorkStatus.VALID and attaches the result.
        """
        _stale_statuses = {
            WorkStatus.STALE,
            WorkStatus.INVALIDATED,
            WorkStatus.RETRACTED,
            "stale",
            "invalidated",
            "retracted",
            "superseded",
        }

        # Resolve any missing or None arguments from session.facts if available
        resolved_args = dict(args)
        if self.session is not None:
            for k, v in list(resolved_args.items()):
                if v is None and self.session.has_fact(k):
                    resolved_args[k] = self.session.get_fact(k)

        # 0. If caller provided an already-stale or superseded WorkItem, block immediately
        if work is not None:
            status_val = getattr(work.status, "value", str(work.status)).lower()
            if work.status in _stale_statuses or status_val in _stale_statuses:
                return {
                    "status": "cancelled",
                    "reason": "superseded",
                }
            if work.work_id not in self.core.graph.all_work_ids():
                self.core.register_work(work)
        else:
            work_id = f"work_{func_name}_{uuid.uuid4().hex[:8]}"
            work = WorkItem(
                kind=func_name,
                work_id=work_id,
                status=WorkStatus.PENDING,
                turn_id=turn_id,
                provenance={"args": dict(resolved_args), "action_id": action_id_for(WorkItem(kind=func_name, work_id=work_id), 0)},
            )
            self.core.register_work(work)

        # 1. If parent work is already invalidated, stale, or retracted, immediately propagate and cancel
        if parent_work_id is not None:
            if parent_work_id not in work.depends_on_work:
                work.depends_on_work.append(parent_work_id)
            self.core.register_dependency(
                DependencyKind.WORK_TO_WORK,
                from_id=parent_work_id,
                to_id=work.work_id,
            )
            parent_work = self.core.graph.get_work(parent_work_id)
            if parent_work:
                parent_status_val = getattr(parent_work.status, "value", str(parent_work.status)).lower()
                if parent_work.status in _stale_statuses or parent_status_val in _stale_statuses:
                    work.status = WorkStatus.INVALIDATED
                    return {
                        "status": "cancelled",
                        "reason": "superseded",
                    }

        # 2. Register tool arguments as facts and establish FACT_TO_WORK edges
        for param_name, param_value in resolved_args.items():
            if param_value is None:
                continue
            fact_update = self.core.assert_fact(
                key=param_name,
                value=param_value,
                source="fdb_tool",
                turn_id=turn_id,
            )
            if fact_update.changeset is not None:
                self.core.invalidate(fact_update.changeset)
            work.depends_on_facts.append(fact_update.fact.fact_id)
            self.core.register_dependency(
                DependencyKind.FACT_TO_WORK,
                from_id=fact_update.fact.fact_id,
                to_id=work.work_id,
            )
            if self.session is not None:
                self.session.facts[param_name] = param_value

        # 3. Pre-execution gate
        current_work = self.core.graph.get_work(work.work_id) or work
        curr_status_val = getattr(current_work.status, "value", str(current_work.status)).lower()
        if current_work.status in _stale_statuses or curr_status_val in _stale_statuses:
            return {
                "status": "cancelled",
                "reason": "superseded",
            }

        # 4. Valid execution
        t_start = time.time()
        result = call_fn(**resolved_args)
        t_end = time.time()

        # 5. Record the call in the official telemetry log now, exactly as the
        # official FDB agent does. Once call_fn has run, the call happened: it
        # is reported whatever becomes of it later, so the benchmark scores what
        # the agent actually did. BACKSPACE earns its keep by blocking stale
        # calls *before* they run (the gates above), never by hiding executed ones.
        record = {
            "room": self.room_name,
            "call": {
                "function": func_name,
                "args": resolved_args,
                "timestamp_start": t_start,
                "timestamp_end": t_end,
            },
        }
        self._buffer.append({"work_id": work.work_id, "record": record})
        self._write_telemetry([(work.work_id, record)])

        # Post-execution currency check: newer information arrived while the
        # call ran. It still executed (and is logged above); tell the model the
        # result is outdated rather than pretending the call never happened.
        current_work = self.core.graph.get_work(work.work_id) or work
        post_status_val = getattr(current_work.status, "value", str(current_work.status)).lower()
        if current_work.status in _stale_statuses or post_status_val in _stale_statuses:
            return {
                "status": "superseded",
                "reason": "the request changed while this call was running; its result is outdated",
                "executed": True,
                "result": result,
            }

        work.output = result
        work.status = WorkStatus.VALID
        return result

    def flush(self) -> int:
        """Write any executed call not yet in the telemetry log; return how many.

        Calls are normally written the moment they execute, so this is a safety
        net at session shutdown and usually writes nothing. It never drops an
        executed call - including one that later went stale - and never writes
        one twice.
        """
        pending = [
            (entry["work_id"], entry["record"])
            for entry in self._buffer
            if entry["work_id"] not in self._flushed_work_ids
        ]
        return self._write_telemetry(pending)

    def _write_telemetry(self, records: list[tuple[str, dict[str, Any]]]) -> int:
        if not records:
            return 0
        Path(self.telemetry_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.telemetry_path, "a", encoding="utf-8") as f:
            for work_id, record in records:
                f.write(json.dumps(record) + "\n")
                self._flushed_work_ids.add(work_id)
        return len(records)
