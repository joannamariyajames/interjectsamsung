"""FDB-v3 Backspace Adapter.

Bridges LiveKit function tool execution with BackspaceCore without modifying
Member 1's Core or the official Full-Duplex-Bench repository.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable

from app.backspace import (
    BackspaceCore,
    DependencyKind,
    WorkItem,
    WorkStatus,
)
from app.backspace.actions import action_id_for


class FDBBackspaceAdapter:
    """Application-boundary adapter between FDB-v3 tool dispatch and BACKSPACE Core."""

    def __init__(
        self,
        core: BackspaceCore,
        room_name: str,
        telemetry_path: str = "/tmp/agent_tool_calls.log",
    ) -> None:
        self.core = core
        self.room_name = room_name
        self.telemetry_path = telemetry_path
        self._buffer: list[dict[str, Any]] = []
        self._flushed_work_ids: set[str] = set()

    def execute_tool(
        self,
        func_name: str,
        args: dict[str, Any],
        call_fn: Callable[..., dict[str, Any]],
        *,
        turn_id: str = "",
        parent_work_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute an FDB tool call gated by BackspaceCore.

        1. Creates a WorkItem for this invocation.
        2. Maps tool arguments to facts and registers FACT_TO_WORK dependencies.
        3. Registers WORK_TO_WORK if parent_work_id is provided.
        4. Checks pre-execution staleness gate. If stale/invalidated/retracted,
           suppresses call_fn, avoids telemetry logging, and returns cancelled status.
        5. If valid, executes call_fn(**args), sets WorkStatus.VALID, attaches result,
           and buffers official telemetry with exact start/end timestamps.
        """
        work_id = f"work_{func_name}_{uuid.uuid4().hex[:8]}"
        work = WorkItem(
            kind=func_name,
            work_id=work_id,
            status=WorkStatus.PENDING,
            turn_id=turn_id,
            provenance={"args": dict(args), "action_id": action_id_for(WorkItem(kind=func_name, work_id=work_id), 0)},
        )
        self.core.register_work(work)

        # 1. If parent work is already invalidated, stale, or retracted, immediately propagate and cancel
        if parent_work_id is not None:
            work.depends_on_work.append(parent_work_id)
            self.core.register_dependency(
                DependencyKind.WORK_TO_WORK,
                from_id=parent_work_id,
                to_id=work.work_id,
            )
            parent_work = self.core.graph.get_work(parent_work_id)
            if parent_work and parent_work.status in {
                WorkStatus.STALE,
                WorkStatus.INVALIDATED,
                WorkStatus.RETRACTED,
            }:
                work.status = WorkStatus.INVALIDATED
                return {
                    "status": "cancelled",
                    "reason": "superseded",
                }

        # 2. Register tool arguments as facts and establish FACT_TO_WORK edges
        for param_name, param_value in args.items():
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

        # 3. Pre-execution gate
        current_work = self.core.graph.get_work(work.work_id) or work
        if current_work.status in {
            WorkStatus.STALE,
            WorkStatus.INVALIDATED,
            WorkStatus.RETRACTED,
        }:
            return {
                "status": "cancelled",
                "reason": "superseded",
            }

        # 4. Valid execution
        t_start = time.time()
        result = call_fn(**args)
        t_end = time.time()

        # Check post-execution currency gate: work must not have gone stale during execution
        current_work = self.core.graph.get_work(work.work_id) or work
        if current_work.status in {
            WorkStatus.STALE,
            WorkStatus.INVALIDATED,
            WorkStatus.RETRACTED,
        }:
            return {
                "status": "cancelled",
                "reason": "superseded",
            }

        work.output = result
        work.status = WorkStatus.VALID

        # 5. Buffer telemetry record in-memory
        self._buffer.append({
            "work_id": work.work_id,
            "record": {
                "room": self.room_name,
                "call": {
                    "function": func_name,
                    "args": args,
                    "timestamp_start": t_start,
                    "timestamp_end": t_end,
                },
            },
        })

        return result

    def flush(self) -> int:
        """Flush buffered tool calls whose final status is VALID or RECOMPUTED.

        Prunes STALE, INVALIDATED, or RETRACTED calls from the official telemetry file.
        Guarantees no duplicate records if called repeatedly.
        Returns count of records written.
        """
        to_write = []
        for entry in self._buffer:
            work_id = entry["work_id"]
            if work_id in self._flushed_work_ids:
                continue
            work = self.core.graph.get_work(work_id)
            if work is not None and work.status in {
                WorkStatus.VALID,
                WorkStatus.RECOMPUTED,
            }:
                to_write.append((work_id, entry["record"]))

        if not to_write:
            return 0

        with open(self.telemetry_path, "a", encoding="utf-8") as f:
            for work_id, record in to_write:
                f.write(json.dumps(record) + "\n")
                self._flushed_work_ids.add(work_id)

        return len(to_write)
