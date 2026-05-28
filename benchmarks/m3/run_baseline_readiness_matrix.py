#!/usr/bin/env python3
"""Generate the M3.11 baseline readiness matrix skeleton.

The first implementation is intentionally dry-run first. It fixes the result
schema, run IDs, artifact layout, and summary handoff before wiring the online
vLLM B0-B5 execution paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx

from benchmarks.m3.summarize_baseline_readiness_matrix import (
    summarize_baseline_readiness_matrix,
)
from benchmarks.m3.run_reuse_smoke_matrix import (
    MatrixConfig as ReuseMatrixConfig,
    run_matrix as run_reuse_smoke_matrix,
)


DEFAULT_MODEL = "/root/models/Qwen2.5-14B-Instruct"
DEFAULT_MODEL_CONTEXT_LIMIT = 32_768
DEFAULT_KV_BYTES_PER_TOKEN = 196_608

BASELINES = ["B0", "B1", "B2", "B3", "B4", "B5"]
WORKLOADS = ["session_append", "shared_prefix", "low_locality"]

PHASE_TO_BASELINES = {
    "all": BASELINES,
    "b0_full_prefill": ["B0"],
    "b1_apc_hot": ["B1"],
    "b2_apc_restart": ["B2"],
    "b3_dram_ready": ["B3"],
    "b4_naive_cold": ["B4"],
    "b5_anti_caching": ["B5"],
}

BASELINE_MATRIX_COLUMNS = [
    "run_id",
    "timestamp_utc",
    "model_id",
    "model_context_limit",
    "kv_bytes_per_token",
    "backend",
    "baseline_id",
    "workload",
    "prefix_tokens",
    "actual_prefix_tokens",
    "suffix_tokens",
    "actual_suffix_tokens",
    "output_tokens",
    "concurrency",
    "repeat_id",
    "status",
    "error_type",
    "error_message",
    "ttft_ms",
    "e2e_ms",
    "itl_p50_ms",
    "itl_p95_ms",
    "tpot_p50_ms",
    "tpot_p95_ms",
    "queue_wait_ms",
    "ready_barrier_wait_ms",
    "online_service_ttft_ms",
    "restore_inclusive_ttft_ms",
    "restore_wait_ms",
    "connector_store_elapsed_ms",
    "connector_load_elapsed_ms",
    "connector_total_elapsed_ms",
    "unattributed_ttft_ms",
    "reuse_tokens",
    "delta_prefill_tokens",
    "prefill_tokens_saved",
    "required_historical_tokens",
    "historical_kv_hit_tokens",
    "historical_kv_hit_rate",
    "historical_bytes_required",
    "historical_bytes_hit",
    "historical_byte_hit_rate",
    "cold_probe_decision",
    "restore_status",
    "restore_actual_bytes",
    "restore_useful_bytes",
    "restore_bytes_per_useful_byte",
    "restore_elapsed_ms",
    "restore_batch_ms",
    "restore_checksum_status",
    "storage_backend",
    "prefetch_deadline_miss",
    "sync_cold_miss_total",
    "admission_decision",
    "admission_reason",
    "prefill_reuse_ready",
    "decode_execution_ready",
    "external_load_observed",
    "connector_load_events",
    "connector_store_events",
    "ttft_vs_full_prefill",
    "ttft_vs_dram_ready_reuse",
    "ttft_vs_apc_hot",
]


@dataclass(frozen=True)
class BaselineReadinessConfig:
    result_dir: Path
    model: str = DEFAULT_MODEL
    base_url: str = "http://127.0.0.1:8000"
    model_context_limit: int = DEFAULT_MODEL_CONTEXT_LIMIT
    kv_bytes_per_token: int = DEFAULT_KV_BYTES_PER_TOKEN
    baselines: list[str] = field(default_factory=lambda: list(BASELINES))
    workloads: list[str] = field(default_factory=lambda: list(WORKLOADS))
    prefix_tokens: list[int] = field(
        default_factory=lambda: [512, 2048, 8192, 16_384, 32_768]
    )
    suffix_tokens: list[int] = field(default_factory=lambda: [128, 512, 2048])
    output_tokens: list[int] = field(default_factory=lambda: [1, 16])
    concurrency: list[int] = field(default_factory=lambda: [1])
    repeats: int = 1
    phase: str = "all"
    dry_run: bool = False
    storage_backend: str = "local_posix"
    strict_b2_restart: bool = False
    serve_host: str = "127.0.0.1"
    serve_port: int = 8000
    serve_max_model_len: int | None = None
    gpu_memory_utilization: float = 0.90
    cpu_offload_gb: float | None = None
    enforce_eager: bool = False
    enable_prefix_caching: bool = False
    startup_timeout_sec: float = 420.0
    vllm_extra_args: list[str] = field(default_factory=list)
    sidecar_url: str = "http://127.0.0.1:8010"
    connector_event_log: Path | None = None
    sidecar_decision_log: Path | None = None
    tensor_store: Path | None = None
    cold_root: Path | None = None
    cold_backend: str = "local_posix"
    cold_tier_advance_ms: float = 12_000.0
    block_size: int = 16


@dataclass(frozen=True)
class BaselineMatrixRow:
    run_id: str
    baseline_id: str
    workload: str
    prefix_tokens: int
    suffix_tokens: int
    output_tokens: int
    concurrency: int
    repeat_id: int


@dataclass(frozen=True)
class CompletionTiming:
    status: str
    ttft_ms: float | str = ""
    e2e_ms: float | str = ""
    itl_p50_ms: float | str = ""
    itl_p95_ms: float | str = ""
    tpot_p50_ms: float | str = ""
    tpot_p95_ms: float | str = ""
    error_type: str = ""
    error_message: str = ""
    raw_path: str = ""


class CompletionClient(Protocol):
    def complete(
        self,
        *,
        prompt: str,
        output_tokens: int,
        run_id: str,
        purpose: str,
    ) -> CompletionTiming:
        ...


class VLLMServiceManager(Protocol):
    def start(self, *, run_id: str, purpose: str) -> None:
        ...

    def stop(self, *, run_id: str, purpose: str) -> None:
        ...


class ReuseExecutor(Protocol):
    def run(
        self,
        *,
        config: BaselineReadinessConfig,
        row: BaselineMatrixRow,
    ) -> dict[str, Any]:
        ...


class OpenAICompletionClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        raw_dir: Path,
        timeout: float | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.raw_dir = raw_dir
        self.timeout = timeout

    def complete(
        self,
        *,
        prompt: str,
        output_tokens: int,
        run_id: str,
        purpose: str,
    ) -> CompletionTiming:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": int(output_tokens),
            "temperature": 0,
            "stream": True,
        }
        url = self.base_url + "/v1/completions"
        started = time.perf_counter()
        first_token_at: float | None = None
        chunks = 0
        raw_events: list[str] = []
        try:
            with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        raw_events.append(line)
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            continue
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        chunks += 1
            ended = time.perf_counter()
        except Exception as exc:  # pragma: no cover - online failure path
            ended = time.perf_counter()
            raw_path = self._write_raw(
                run_id=run_id,
                purpose=purpose,
                payload=payload,
                raw_events=raw_events,
                status="ERROR",
                error=str(exc),
                started=started,
                ended=ended,
            )
            return CompletionTiming(
                status="ERROR",
                error_type=type(exc).__name__,
                error_message=str(exc)[:500],
                e2e_ms=_elapsed_ms_between(started, ended),
                raw_path=str(raw_path),
            )
        raw_path = self._write_raw(
            run_id=run_id,
            purpose=purpose,
            payload=payload,
            raw_events=raw_events,
            status="OK",
            error="",
            started=started,
            ended=ended,
        )
        ttft_ms = (
            _elapsed_ms_between(started, first_token_at)
            if first_token_at is not None
            else _elapsed_ms_between(started, ended)
        )
        e2e_ms = _elapsed_ms_between(started, ended)
        tpot = ""
        if chunks > 1 and first_token_at is not None:
            tpot = round((ended - first_token_at) * 1000.0 / max(1, chunks - 1), 3)
        return CompletionTiming(
            status="OK",
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            itl_p50_ms=tpot,
            itl_p95_ms=tpot,
            tpot_p50_ms=tpot,
            tpot_p95_ms=tpot,
            raw_path=str(raw_path),
        )

    def _write_raw(
        self,
        *,
        run_id: str,
        purpose: str,
        payload: dict[str, Any],
        raw_events: list[str],
        status: str,
        error: str,
        started: float,
        ended: float,
    ) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_dir / f"{_safe_file_name(run_id)}_{_safe_file_name(purpose)}.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "purpose": purpose,
                    "status": status,
                    "error": error,
                    "payload": payload,
                    "events": raw_events,
                    "elapsed_ms": _elapsed_ms_between(started, ended),
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return path


class LocalVLLMServiceManager:
    """Own a vLLM serve process for restart-sensitive benchmark rows."""

    def __init__(self, *, config: BaselineReadinessConfig, log_dir: Path) -> None:
        self.config = config
        self.log_dir = log_dir
        self.process: subprocess.Popen[str] | None = None
        self.log_fh: Any | None = None
        self.log_path: Path | None = None

    def start(self, *, run_id: str, purpose: str) -> None:
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("vLLM service is already running")
        if _port_open(self.config.serve_host, self.config.serve_port):
            raise RuntimeError(
                f"{self.config.serve_host}:{self.config.serve_port} is already open"
            )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / (
            f"{_safe_file_name(run_id)}_{_safe_file_name(purpose)}.log"
        )
        command = _build_vllm_serve_command(self.config)
        env = _vllm_serve_env()
        self.log_fh = self.log_path.open("w", encoding="utf-8")
        self.log_fh.write("+ " + " ".join(command) + "\n")
        self.log_fh.flush()
        self.process = subprocess.Popen(
            command,
            stdout=self.log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        started = _wait_for_health(
            f"http://{self.config.serve_host}:{self.config.serve_port}",
            timeout_sec=self.config.startup_timeout_sec,
            process=self.process,
        )
        if not started:
            tail = _tail_file(self.log_path)
            self.stop(run_id=run_id, purpose=purpose)
            raise RuntimeError(
                f"vLLM failed to become healthy for {purpose}; log_tail={tail}"
            )

    def stop(self, *, run_id: str, purpose: str) -> None:
        if self.process is not None:
            _terminate_process(self.process)
            self.process = None
        if self.log_fh is not None:
            self.log_fh.close()
            self.log_fh = None


class ReuseSmokeMatrixExecutor:
    """Bridge existing sidecar/connector smoke rows into the M3.11 matrix."""

    def run(
        self,
        *,
        config: BaselineReadinessConfig,
        row: BaselineMatrixRow,
    ) -> dict[str, Any]:
        result_dir = (
            config.result_dir
            / "reuse_bridge"
            / row.baseline_id
            / row.workload
            / f"p{row.prefix_tokens}_s{row.suffix_tokens}_o{row.output_tokens}_r{row.repeat_id}"
        )
        result = run_reuse_smoke_matrix(
            ReuseMatrixConfig(
                prefix_tokens=[row.prefix_tokens],
                suffix_tokens=row.suffix_tokens,
                output_tokens=row.output_tokens,
                result_dir=result_dir,
                sidecar_url=config.sidecar_url,
                model=config.model,
                dry_run=False,
                phase="both",
                prompt_unit="cache" if row.workload != "low_locality" else "random",
                connector_event_log=config.connector_event_log,
                sidecar_decision_log=config.sidecar_decision_log,
                tensor_store=config.tensor_store,
                cold_root=config.cold_root,
                cold_backend=config.cold_backend,
                cold_tier_restore=row.baseline_id == "B5",
                advance_ms=config.cold_tier_advance_ms,
                block_size=config.block_size,
            )
        )
        csv_path = Path(result["csv_path"])
        rows = _read_csv_rows(csv_path)
        if not rows:
            return {
                "status": "ERROR",
                "error": f"reuse smoke matrix produced no rows at {csv_path}",
            }
        return rows[0]


class ReuseCsvImportExecutor:
    """Import existing reuse_smoke_matrix.csv rows as B3/B5 measurements."""

    def __init__(self, sources: dict[tuple[str, int, int, int], Path]) -> None:
        self.sources = sources

    def run(
        self,
        *,
        config: BaselineReadinessConfig,
        row: BaselineMatrixRow,
    ) -> dict[str, Any]:
        source = self.sources.get(
            (row.baseline_id, row.prefix_tokens, row.suffix_tokens, row.output_tokens)
        ) or self.sources.get((row.baseline_id, 0, 0, 0))
        if source is None:
            raise FileNotFoundError(
                "no reuse CSV source configured for "
                f"{row.baseline_id}/p{row.prefix_tokens}/s{row.suffix_tokens}/o{row.output_tokens}"
            )
        candidates = _read_csv_rows(source)
        for candidate in candidates:
            if (
                str(candidate.get("prefix_tokens")) == str(row.prefix_tokens)
                and str(candidate.get("suffix_tokens")) == str(row.suffix_tokens)
                and str(candidate.get("output_tokens")) == str(row.output_tokens)
            ):
                return candidate
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(
            f"no matching row in {source} for p{row.prefix_tokens}/s{row.suffix_tokens}/o{row.output_tokens}"
        )


def build_matrix_rows(config: BaselineReadinessConfig) -> list[BaselineMatrixRow]:
    rows: list[BaselineMatrixRow] = []
    for baseline_id in config.baselines:
        _validate_choice("baseline", baseline_id, BASELINES)
        for workload in config.workloads:
            _validate_choice(
                "workload",
                workload,
                ["session_append", "shared_prefix", "low_locality", "mixed_short_long"],
            )
            for prefix_tokens in config.prefix_tokens:
                for suffix_tokens in config.suffix_tokens:
                    for output_tokens in config.output_tokens:
                        for concurrency in config.concurrency:
                            for repeat_id in range(config.repeats):
                                rows.append(
                                    BaselineMatrixRow(
                                        run_id=(
                                            f"m3_11_{baseline_id}_{workload}"
                                            f"_p{int(prefix_tokens)}"
                                            f"_s{int(suffix_tokens)}"
                                            f"_o{int(output_tokens)}"
                                            f"_c{int(concurrency)}"
                                            f"_r{int(repeat_id)}"
                                        ),
                                        baseline_id=baseline_id,
                                        workload=workload,
                                        prefix_tokens=int(prefix_tokens),
                                        suffix_tokens=int(suffix_tokens),
                                        output_tokens=int(output_tokens),
                                        concurrency=int(concurrency),
                                        repeat_id=int(repeat_id),
                                    )
                                )
    return rows


def run_baseline_readiness_matrix(
    config: BaselineReadinessConfig,
    completion_client: CompletionClient | None = None,
    service_manager: VLLMServiceManager | None = None,
    reuse_executor: ReuseExecutor | None = None,
) -> dict[str, Any]:
    result_dir = config.result_dir
    summary_dir = result_dir / "summary"
    raw_dir = result_dir / "raw"
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    timestamp = _utc_now()
    matrix_rows = []
    client = completion_client or OpenAICompletionClient(
        base_url=config.base_url,
        model=config.model,
        raw_dir=raw_dir,
    )
    manager = service_manager
    if manager is None and config.strict_b2_restart:
        manager = LocalVLLMServiceManager(config=config, log_dir=result_dir / "logs")
    reuse = reuse_executor or ReuseSmokeMatrixExecutor()
    for row in build_matrix_rows(config):
        if config.dry_run:
            matrix_rows.append(_dry_run_csv_row(config, row, timestamp))
            continue
        matrix_rows.append(_online_csv_row(config, row, timestamp, client, manager, reuse))
    _write_json(result_dir / "env.json", _build_env(config, timestamp))
    _write_run_config(result_dir / "run_config.yaml", config, timestamp)
    _write_csv(result_dir / "baseline_matrix.csv", matrix_rows)
    _write_csv(result_dir / "request_metrics.csv", matrix_rows)
    _touch_jsonl(result_dir / "sidecar_decisions.jsonl")
    _touch_jsonl(result_dir / "connector_events.jsonl")
    _touch_jsonl(result_dir / "restore_events.jsonl")
    if config.dry_run:
        (raw_dir / "README.md").write_text(
            "Raw online response files will be written here after B0-B5 execution is wired.\n",
            encoding="utf-8",
        )

    summary = summarize_baseline_readiness_matrix(result_dir)
    status = _status_from_summary(summary)
    return {
        "status": "DRY_RUN" if config.dry_run else status,
        "rows": len(matrix_rows),
        "result_dir": str(result_dir),
        "baseline_matrix": str(result_dir / "baseline_matrix.csv"),
        "summary": summary,
    }


def _online_csv_row(
    config: BaselineReadinessConfig,
    row: BaselineMatrixRow,
    timestamp: str,
    client: CompletionClient,
    service_manager: VLLMServiceManager | None,
    reuse_executor: ReuseExecutor,
) -> dict[str, Any]:
    boundary = _context_boundary(config, row)
    if boundary:
        csv_row = _base_csv_row(config, row, timestamp)
        csv_row.update(
            {
                "status": "ERROR_BOUNDARY",
                "error_type": "context_limit",
                "error_message": boundary,
                "backend": (
                    "vllm_sidecar_connector"
                    if row.baseline_id in {"B3", "B5"}
                    else "vllm_http"
                ),
                "admission_decision": "REJECT",
                "admission_reason": "context_boundary",
                "prefill_reuse_ready": "false",
                "decode_execution_ready": "false",
            }
        )
        return csv_row

    if row.baseline_id == "B0":
        return _run_b0_row(config, row, timestamp, client)
    if row.baseline_id == "B1":
        return _run_b1_row(config, row, timestamp, client)
    if row.baseline_id == "B2":
        return _run_b2_row(config, row, timestamp, client, service_manager)
    if row.baseline_id in {"B3", "B5"}:
        return _run_external_reuse_row(config, row, timestamp, reuse_executor)
    csv_row = _dry_run_csv_row(config, row, timestamp)
    csv_row.update(
        {
            "status": "ERROR",
            "error_type": "unsupported_online_baseline",
            "error_message": f"online execution for {row.baseline_id} is not wired yet",
        }
    )
    return csv_row


def _run_b0_row(
    config: BaselineReadinessConfig,
    row: BaselineMatrixRow,
    timestamp: str,
    client: CompletionClient,
) -> dict[str, Any]:
    prompt = _prompt_for_row(row, prefix_seed="full")
    timing = client.complete(
        prompt=prompt,
        output_tokens=row.output_tokens,
        run_id=row.run_id,
        purpose="measure_B0",
    )
    return _native_vllm_csv_row(
        config,
        row,
        timestamp,
        timing,
        reuse_tokens=0,
        historical_hits=0,
        admission_reason="full_prefill_apc_off_or_unique_prefix",
    )


def _run_b1_row(
    config: BaselineReadinessConfig,
    row: BaselineMatrixRow,
    timestamp: str,
    client: CompletionClient,
) -> dict[str, Any]:
    prefix_seed = "cache" if row.workload != "low_locality" else "random"
    warm_prompt = _repeat_tokens(prefix_seed, row.prefix_tokens)
    client.complete(
        prompt=warm_prompt,
        output_tokens=row.output_tokens,
        run_id=row.run_id,
        purpose="warm_B1",
    )
    timing = client.complete(
        prompt=_prompt_for_row(row, prefix_seed=prefix_seed),
        output_tokens=row.output_tokens,
        run_id=row.run_id,
        purpose="measure_B1",
    )
    reuse_tokens = row.prefix_tokens if row.workload != "low_locality" else 0
    return _native_vllm_csv_row(
        config,
        row,
        timestamp,
        timing,
        reuse_tokens=reuse_tokens,
        historical_hits=reuse_tokens,
        admission_reason="apc_hot_cache" if reuse_tokens else "low_locality_no_reuse",
    )


def _run_b2_row(
    config: BaselineReadinessConfig,
    row: BaselineMatrixRow,
    timestamp: str,
    client: CompletionClient,
    service_manager: VLLMServiceManager | None,
) -> dict[str, Any]:
    if config.strict_b2_restart:
        if service_manager is None:
            raise RuntimeError("strict B2 restart requires a service manager")
        prefix_seed = "cache" if row.workload != "low_locality" else "random"
        try:
            service_manager.start(run_id=row.run_id, purpose="warm_B2_before_restart")
            warm_timing = client.complete(
                prompt=_repeat_tokens(prefix_seed, row.prefix_tokens),
                output_tokens=row.output_tokens,
                run_id=row.run_id,
                purpose="warm_B2_before_restart",
            )
        except Exception as exc:
            _stop_service_after_error(service_manager, row.run_id, "warm_B2_before_restart")
            return _native_vllm_csv_row(
                config,
                row,
                timestamp,
                CompletionTiming(
                    status="ERROR",
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:500],
                ),
                reuse_tokens=0,
                historical_hits=0,
                admission_reason="apc_restart_warm_failed",
            )
        service_manager.stop(run_id=row.run_id, purpose="warm_B2_before_restart")
        if warm_timing.status != "OK":
            return _native_vllm_csv_row(
                config,
                row,
                timestamp,
                warm_timing,
                reuse_tokens=0,
                historical_hits=0,
                admission_reason="apc_restart_warm_failed",
            )

        try:
            service_manager.start(run_id=row.run_id, purpose="measure_B2_after_restart")
            timing = client.complete(
                prompt=_prompt_for_row(row, prefix_seed=prefix_seed),
                output_tokens=row.output_tokens,
                run_id=row.run_id,
                purpose="measure_B2_after_restart",
            )
        except Exception as exc:
            timing = CompletionTiming(
                status="ERROR",
                error_type=type(exc).__name__,
                error_message=str(exc)[:500],
            )
        finally:
            service_manager.stop(run_id=row.run_id, purpose="measure_B2_after_restart")
        return _native_vllm_csv_row(
            config,
            row,
            timestamp,
            timing,
            reuse_tokens=0,
            historical_hits=0,
            admission_reason="apc_restart_cold_cache",
        )

    timing = client.complete(
        prompt=_prompt_for_row(row, prefix_seed="cold"),
        output_tokens=row.output_tokens,
        run_id=row.run_id,
        purpose="measure_B2",
    )
    return _native_vllm_csv_row(
        config,
        row,
        timestamp,
        timing,
        reuse_tokens=0,
        historical_hits=0,
        admission_reason="apc_cold_cache_or_restart",
    )


def _native_vllm_csv_row(
    config: BaselineReadinessConfig,
    row: BaselineMatrixRow,
    timestamp: str,
    timing: CompletionTiming,
    *,
    reuse_tokens: int,
    historical_hits: int,
    admission_reason: str,
) -> dict[str, Any]:
    csv_row = _base_csv_row(config, row, timestamp)
    required_historical_tokens = row.prefix_tokens if row.workload != "low_locality" else 0
    historical_bytes_required = required_historical_tokens * config.kv_bytes_per_token
    historical_bytes_hit = historical_hits * config.kv_bytes_per_token
    csv_row.update(
        {
            "backend": "vllm_http",
            "status": timing.status,
            "error_type": timing.error_type,
            "error_message": timing.error_message,
            "ttft_ms": timing.ttft_ms,
            "e2e_ms": timing.e2e_ms,
            "itl_p50_ms": timing.itl_p50_ms,
            "itl_p95_ms": timing.itl_p95_ms,
            "tpot_p50_ms": timing.tpot_p50_ms,
            "tpot_p95_ms": timing.tpot_p95_ms,
            "online_service_ttft_ms": timing.ttft_ms,
            "restore_inclusive_ttft_ms": timing.ttft_ms,
            "restore_wait_ms": "0.0",
            "connector_store_elapsed_ms": "0.0",
            "connector_load_elapsed_ms": "0.0",
            "connector_total_elapsed_ms": "0.0",
            "unattributed_ttft_ms": timing.ttft_ms,
            "reuse_tokens": reuse_tokens,
            "delta_prefill_tokens": row.prefix_tokens + row.suffix_tokens - reuse_tokens,
            "prefill_tokens_saved": reuse_tokens,
            "required_historical_tokens": required_historical_tokens,
            "historical_kv_hit_tokens": historical_hits,
            "historical_kv_hit_rate": _safe_rate(historical_hits, required_historical_tokens),
            "historical_bytes_required": historical_bytes_required,
            "historical_bytes_hit": historical_bytes_hit,
            "historical_byte_hit_rate": _safe_rate(
                historical_bytes_hit,
                historical_bytes_required,
            ),
            "cold_probe_decision": "NA",
            "restore_status": "NA",
            "restore_actual_bytes": 0,
            "restore_useful_bytes": 0,
            "restore_bytes_per_useful_byte": "",
            "restore_checksum_status": "NA",
            "storage_backend": "none",
            "prefetch_deadline_miss": "false",
            "sync_cold_miss_total": 0,
            "admission_decision": "ADMIT" if timing.status == "OK" else "REJECT",
            "admission_reason": admission_reason,
            "prefill_reuse_ready": "true",
            "decode_execution_ready": "true",
            "external_load_observed": "false",
            "connector_load_events": 0,
            "connector_store_events": 0,
        }
    )
    return csv_row


def _run_external_reuse_row(
    config: BaselineReadinessConfig,
    row: BaselineMatrixRow,
    timestamp: str,
    reuse_executor: ReuseExecutor,
) -> dict[str, Any]:
    try:
        reuse_row = reuse_executor.run(config=config, row=row)
    except Exception as exc:  # pragma: no cover - online bridge failure path
        csv_row = _base_csv_row(config, row, timestamp)
        csv_row.update(
            {
                "backend": "vllm_sidecar_connector",
                "status": "ERROR",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
                "storage_backend": _storage_backend_for(row.baseline_id, config.storage_backend),
                "admission_decision": "REJECT",
                "admission_reason": "reuse_executor_failed",
                "prefill_reuse_ready": "false",
                "decode_execution_ready": "false",
            }
        )
        return csv_row
    return _external_reuse_csv_row(config, row, timestamp, reuse_row)


def _external_reuse_csv_row(
    config: BaselineReadinessConfig,
    row: BaselineMatrixRow,
    timestamp: str,
    reuse_row: dict[str, Any],
) -> dict[str, Any]:
    csv_row = _base_csv_row(config, row, timestamp)
    status = str(reuse_row.get("status", "ERROR") or "ERROR")
    reuse_tokens = _int_value(reuse_row.get("reuse_tokens"), default=0)
    if row.workload == "low_locality":
        reuse_tokens = 0
    required_historical_tokens = row.prefix_tokens if row.workload != "low_locality" else 0
    historical_hits = reuse_tokens if status == "OK" else 0
    historical_bytes_required = required_historical_tokens * config.kv_bytes_per_token
    historical_bytes_hit = historical_hits * config.kv_bytes_per_token
    storage_backend = (
        str(reuse_row.get("cold_backend") or config.cold_backend)
        if row.baseline_id == "B5"
        else "dram"
    )
    restore_actual_bytes = (
        _int_value(reuse_row.get("cold_restore_actual_bytes"), default=0)
        if row.baseline_id == "B5"
        else 0
    )
    restore_useful_bytes = historical_bytes_required if row.baseline_id == "B5" else 0
    restore_elapsed_ms = (
        _string_value(reuse_row.get("cold_restore_executor_elapsed_ms"))
        if row.baseline_id == "B5"
        else ""
    )
    restore_status = (
        _string_value(reuse_row.get("cold_restore_status")) or "NOT_RESTORED"
        if row.baseline_id == "B5"
        else "NA"
    )
    restore_checksum_status = (
        _string_value(reuse_row.get("cold_restore_checksum_status")) or "NA"
        if row.baseline_id == "B5"
        else "NA"
    )
    online_service_ttft_ms = _string_value(
        reuse_row.get("online_service_ttft_ms") or reuse_row.get("second_ttft_ms")
    )
    restore_wait_ms = _string_value(
        reuse_row.get("restore_wait_ms")
        or (restore_elapsed_ms if row.baseline_id == "B5" else "0.0")
    )
    restore_inclusive_ttft_ms = _string_value(
        reuse_row.get("restore_inclusive_ttft_ms")
    ) or _format_float_string(
        _float_value(online_service_ttft_ms) + _float_value(restore_wait_ms)
    )
    connector_store_elapsed_ms = _string_value(
        reuse_row.get("connector_store_elapsed_ms") or reuse_row.get("store_elapsed_ms")
    )
    connector_load_elapsed_ms = _string_value(
        reuse_row.get("connector_load_elapsed_ms") or reuse_row.get("load_elapsed_ms")
    )
    connector_total_elapsed_ms = _format_float_string(
        _float_value(connector_store_elapsed_ms) + _float_value(connector_load_elapsed_ms)
    )
    unattributed_ttft_ms = _string_value(reuse_row.get("unattributed_ttft_ms")) or (
        _format_float_string(
            _float_value(online_service_ttft_ms)
            - _float_value(connector_load_elapsed_ms)
        )
        if online_service_ttft_ms
        else ""
    )
    csv_row.update(
        {
            "backend": "vllm_sidecar_connector",
            "status": status,
            "error_type": "" if status == "OK" else "reuse_smoke_error",
            "error_message": _string_value(reuse_row.get("error")),
            "ttft_ms": _string_value(reuse_row.get("second_ttft_ms")),
            "e2e_ms": _string_value(reuse_row.get("second_ttft_ms")),
            "queue_wait_ms": "",
            "ready_barrier_wait_ms": restore_elapsed_ms if row.baseline_id == "B5" else "",
            "online_service_ttft_ms": online_service_ttft_ms,
            "restore_inclusive_ttft_ms": restore_inclusive_ttft_ms,
            "restore_wait_ms": restore_wait_ms,
            "connector_store_elapsed_ms": connector_store_elapsed_ms,
            "connector_load_elapsed_ms": connector_load_elapsed_ms,
            "connector_total_elapsed_ms": connector_total_elapsed_ms,
            "unattributed_ttft_ms": unattributed_ttft_ms,
            "reuse_tokens": reuse_tokens,
            "delta_prefill_tokens": row.prefix_tokens + row.suffix_tokens - reuse_tokens,
            "prefill_tokens_saved": reuse_tokens,
            "required_historical_tokens": required_historical_tokens,
            "historical_kv_hit_tokens": historical_hits,
            "historical_kv_hit_rate": _safe_rate(historical_hits, required_historical_tokens),
            "historical_bytes_required": historical_bytes_required,
            "historical_bytes_hit": historical_bytes_hit,
            "historical_byte_hit_rate": _safe_rate(
                historical_bytes_hit,
                historical_bytes_required,
            ),
            "cold_probe_decision": (
                _string_value(reuse_row.get("cold_probe_decision"))
                if row.baseline_id == "B5"
                else "NA"
            ),
            "restore_status": restore_status,
            "restore_actual_bytes": restore_actual_bytes,
            "restore_useful_bytes": restore_useful_bytes,
            "restore_bytes_per_useful_byte": _safe_rate(
                restore_actual_bytes,
                restore_useful_bytes,
            )
            if row.baseline_id == "B5"
            else "",
            "restore_elapsed_ms": restore_elapsed_ms,
            "restore_batch_ms": restore_elapsed_ms,
            "restore_checksum_status": restore_checksum_status,
            "storage_backend": storage_backend,
            "prefetch_deadline_miss": _bool_string(
                _positive_number(reuse_row.get("prefetch_deadline_miss_delta"))
            ),
            "sync_cold_miss_total": _int_value(
                reuse_row.get("sync_ssd_miss_total"),
                default=0,
            ),
            "admission_decision": _string_value(reuse_row.get("decision")) or (
                "ADMIT" if status == "OK" else "REJECT"
            ),
            "admission_reason": _string_value(reuse_row.get("reason")) or (
                "external_dram_ready_reuse"
                if row.baseline_id == "B3"
                else "anti_caching_restore_then_admit"
            ),
            "prefill_reuse_ready": _ready_bool_string(reuse_row),
            "decode_execution_ready": _ready_bool_string(reuse_row),
            "external_load_observed": _bool_string(
                _truthy(reuse_row.get("external_load_observed"))
            ),
            "connector_load_events": _int_value(reuse_row.get("load_events"), default=0),
            "connector_store_events": _int_value(reuse_row.get("store_events"), default=0),
        }
    )
    return csv_row


def _base_csv_row(
    config: BaselineReadinessConfig,
    row: BaselineMatrixRow,
    timestamp: str,
) -> dict[str, Any]:
    csv_row = {column: "" for column in BASELINE_MATRIX_COLUMNS}
    csv_row.update(
        {
            "run_id": row.run_id,
            "timestamp_utc": timestamp,
            "model_id": config.model,
            "model_context_limit": config.model_context_limit,
            "kv_bytes_per_token": config.kv_bytes_per_token,
            "backend": "",
            "baseline_id": row.baseline_id,
            "workload": row.workload,
            "prefix_tokens": row.prefix_tokens,
            "actual_prefix_tokens": row.prefix_tokens,
            "suffix_tokens": row.suffix_tokens,
            "actual_suffix_tokens": row.suffix_tokens,
            "output_tokens": row.output_tokens,
            "concurrency": row.concurrency,
            "repeat_id": row.repeat_id,
        }
    )
    return csv_row


def _dry_run_csv_row(
    config: BaselineReadinessConfig,
    row: BaselineMatrixRow,
    timestamp: str,
) -> dict[str, Any]:
    total_context = row.prefix_tokens + row.suffix_tokens + row.output_tokens
    boundary = total_context > config.model_context_limit
    reusable = row.workload in {"session_append", "shared_prefix"}
    external_reuse = row.baseline_id in {"B3", "B4", "B5"} and reusable
    apc_reuse = row.baseline_id in {"B1", "B2"} and reusable
    required_historical_tokens = row.prefix_tokens if (external_reuse or apc_reuse) else 0
    reuse_tokens = required_historical_tokens if row.baseline_id in {"B1", "B3"} else 0
    if row.baseline_id == "B5" and reusable:
        reuse_tokens = row.prefix_tokens
    if row.baseline_id == "B4" and reusable:
        reuse_tokens = row.prefix_tokens

    status = "ERROR_BOUNDARY" if boundary else "DRY_RUN"
    error_type = "context_limit" if boundary else ""
    error_message = (
        f"prefix+suffix+output={total_context} exceeds model_context_limit="
        f"{config.model_context_limit}"
        if boundary
        else ""
    )
    historical_bytes_required = required_historical_tokens * config.kv_bytes_per_token
    storage_backend = _storage_backend_for(row.baseline_id, config.storage_backend)
    cold_probe_decision = "DELAY_RESTORE" if row.baseline_id == "B5" and reusable else "NA"
    restore_status = "PLANNED" if row.baseline_id in {"B4", "B5"} and reusable else "NA"
    admission_decision = _admission_decision_for(row.baseline_id, row.workload)
    prefill_ready = "false" if row.baseline_id == "B5" and reusable else "true"
    decode_ready = "false" if row.baseline_id in {"B4", "B5"} and reusable else "true"

    csv_row = _base_csv_row(config, row, timestamp)
    csv_row.update(
        {
            "backend": "simulated",
            "status": status,
            "error_type": error_type,
            "error_message": error_message,
            "queue_wait_ms": "",
            "ready_barrier_wait_ms": "",
            "reuse_tokens": reuse_tokens,
            "delta_prefill_tokens": (
                row.suffix_tokens if reuse_tokens else row.prefix_tokens + row.suffix_tokens
            ),
            "prefill_tokens_saved": reuse_tokens,
            "required_historical_tokens": required_historical_tokens,
            "historical_kv_hit_tokens": "",
            "historical_kv_hit_rate": "",
            "historical_bytes_required": historical_bytes_required,
            "historical_bytes_hit": "",
            "historical_byte_hit_rate": "",
            "cold_probe_decision": cold_probe_decision,
            "restore_status": restore_status,
            "restore_actual_bytes": "",
            "restore_useful_bytes": historical_bytes_required if external_reuse else 0,
            "restore_bytes_per_useful_byte": "",
            "restore_elapsed_ms": "",
            "restore_batch_ms": "",
            "restore_checksum_status": "NA",
            "storage_backend": storage_backend,
            "prefetch_deadline_miss": "false",
            "sync_cold_miss_total": 0,
            "admission_decision": admission_decision,
            "admission_reason": "dry_run_plan" if not boundary else "context_boundary",
            "prefill_reuse_ready": prefill_ready,
            "decode_execution_ready": decode_ready,
            "external_load_observed": "false",
            "connector_load_events": 0,
            "connector_store_events": 0,
            "online_service_ttft_ms": "",
            "restore_inclusive_ttft_ms": "",
            "restore_wait_ms": "",
            "connector_store_elapsed_ms": "",
            "connector_load_elapsed_ms": "",
            "connector_total_elapsed_ms": "",
            "unattributed_ttft_ms": "",
        }
    )
    return csv_row


def _admission_decision_for(baseline_id: str, workload: str) -> str:
    if workload == "low_locality":
        return "FULL_PREFILL_FALLBACK"
    if baseline_id == "B5":
        return "DELAY_RESTORE"
    if baseline_id == "B4":
        return "ADMIT_SYNC_COLD_RESTORE"
    return "ADMIT"


def _storage_backend_for(baseline_id: str, storage_backend: str) -> str:
    if baseline_id in {"B0", "B1", "B2"}:
        return "none"
    if baseline_id == "B3":
        return "dram"
    return storage_backend


def _stop_service_after_error(
    service_manager: VLLMServiceManager,
    run_id: str,
    purpose: str,
) -> None:
    try:
        service_manager.stop(run_id=run_id, purpose=purpose)
    except Exception:
        pass


def _context_boundary(config: BaselineReadinessConfig, row: BaselineMatrixRow) -> str:
    total_context = row.prefix_tokens + row.suffix_tokens + row.output_tokens
    if total_context <= config.model_context_limit:
        return ""
    return (
        f"prefix+suffix+output={total_context} exceeds model_context_limit="
        f"{config.model_context_limit}"
    )


def _prompt_for_row(row: BaselineMatrixRow, *, prefix_seed: str) -> str:
    prefix = _repeat_tokens(prefix_seed, row.prefix_tokens)
    suffix = _repeat_tokens("tail", row.suffix_tokens)
    return f"{prefix} {suffix}".strip()


def _repeat_tokens(seed: str, count: int) -> str:
    return " ".join([seed] * max(1, int(count)))


def _safe_rate(numerator: int | float, denominator: int | float) -> str:
    denominator = float(denominator)
    if denominator <= 0:
        return "0.0"
    value = round(float(numerator) / denominator, 6)
    text = str(value).rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def _int_value(value: Any, *, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _positive_number(value: Any) -> bool:
    try:
        return float(value or 0) > 0
    except (TypeError, ValueError):
        return False


def _float_value(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_float_string(value: float) -> str:
    rounded = round(float(value), 6)
    if abs(rounded) == 0:
        rounded = 0.0
    text = f"{rounded:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def _string_value(value: Any) -> str:
    return "" if value is None else str(value)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _bool_string(value: bool) -> str:
    return "true" if value else "false"


def _ready_bool_string(reuse_row: dict[str, Any]) -> str:
    if "ready_barrier_all_ready" in reuse_row:
        return _bool_string(_truthy(reuse_row.get("ready_barrier_all_ready")))
    return _bool_string(str(reuse_row.get("status", "")) == "OK")


def _elapsed_ms_between(start: float, end: float | None) -> float:
    if end is None:
        end = time.perf_counter()
    return round((end - start) * 1000.0, 3)


def _safe_file_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _reuse_import_sources_from_args(args: argparse.Namespace) -> dict[tuple[str, int, int, int], Path]:
    sources: dict[tuple[str, int, int, int], Path] = {}
    for value in args.import_reuse_csv or []:
        parts = value.split(":", 4)
        if len(parts) == 2:
            baseline, path = parts
            key = (baseline, 0, 0, 0)
        elif len(parts) == 5:
            baseline, prefix, suffix, output, path = parts
            key = (baseline, int(prefix), int(suffix), int(output))
        else:
            raise ValueError(
                "--import-reuse-csv must be BASELINE:PATH or "
                "BASELINE:PREFIX:SUFFIX:OUTPUT:PATH"
            )
        _validate_choice("baseline", key[0], ["B3", "B5"])
        sources[key] = Path(path)
    return sources


def _build_vllm_serve_command(config: BaselineReadinessConfig) -> list[str]:
    max_model_len = config.serve_max_model_len or config.model_context_limit
    command = [
        "vllm",
        "serve",
        config.model,
        "--host",
        config.serve_host,
        "--port",
        str(config.serve_port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--disable-log-stats",
        "--disable-uvicorn-access-log",
    ]
    if config.cpu_offload_gb is not None:
        command.extend(["--cpu-offload-gb", str(config.cpu_offload_gb)])
    if config.enforce_eager:
        command.append("--enforce-eager")
    if config.enable_prefix_caching:
        command.append("--enable-prefix-caching")
    command.extend(config.vllm_extra_args)
    return command


def _vllm_serve_env() -> dict[str, str]:
    env = os.environ.copy()
    env["VLLM_PLUGINS"] = ""
    env["NO_PROXY"] = "127.0.0.1,localhost"
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        env.pop(key, None)
    return env


def _wait_for_health(
    base_url: str,
    *,
    timeout_sec: float,
    process: subprocess.Popen[str] | None = None,
) -> bool:
    deadline = time.monotonic() + timeout_sec
    url = base_url.rstrip("/") + "/health"
    with httpx.Client(timeout=3, trust_env=False) as client:
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                return False
            try:
                response = client.get(url)
                if 200 <= response.status_code < 500:
                    return True
            except (httpx.HTTPError, OSError):
                pass
            time.sleep(2)
    return False


def _port_open(host: str, port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(1)
    try:
        sock.connect((host, int(port)))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def _tail_file(path: Path, *, max_chars: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:].strip()


def _status_from_summary(summary: dict[str, Any]) -> str:
    if summary.get("error_count", 0) and not summary.get("ok_count", 0):
        return "PARTIAL"
    if summary.get("error_count", 0):
        return "PARTIAL"
    if summary.get("ok_count", 0):
        return "OK"
    return str(summary.get("status", "EMPTY"))


def _build_env(config: BaselineReadinessConfig, timestamp: str) -> dict[str, Any]:
    return {
        "timestamp_utc": timestamp,
        "cwd": os.getcwd(),
        "model_id": config.model,
        "model_context_limit": config.model_context_limit,
        "kv_bytes_per_token": config.kv_bytes_per_token,
        "dry_run": config.dry_run,
        "git_status_short": _run_text(["git", "status", "--short"]),
        "git_diff_stat": _run_text(["git", "diff", "--stat"]),
        "notes": {
            "purpose": "M3.11 baseline readiness schema and dry-run artifact layout",
            "online_execution": "native_vllm_http_b0_b1_b2_with_optional_strict_b2_restart",
        },
    }


def _write_run_config(
    path: Path,
    config: BaselineReadinessConfig,
    timestamp: str,
) -> None:
    payload = {
        **asdict(config),
        "result_dir": str(config.result_dir),
        "timestamp_utc": timestamp,
    }
    lines = ["# Generated by run_baseline_readiness_matrix.py"]
    for key, value in payload.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=BASELINE_MATRIX_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in BASELINE_MATRIX_COLUMNS})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _touch_jsonl(path: Path) -> None:
    path.write_text("", encoding="utf-8")


def _run_text(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=True,
        cwd=Path.cwd(),
    )
    if completed.returncode != 0:
        return completed.stderr.strip()
    return completed.stdout.strip()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_choice(name: str, value: str, choices: list[str]) -> None:
    if value not in choices:
        raise ValueError(f"unsupported {name}: {value}; choices={choices}")


def _baselines_from_args(args: argparse.Namespace) -> list[str]:
    if args.baselines:
        return [str(item) for item in args.baselines]
    return list(PHASE_TO_BASELINES[args.phase])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=sorted(PHASE_TO_BASELINES),
        default="all",
        help="Convenience selector for one B0-B5 execution phase.",
    )
    parser.add_argument("--baselines", nargs="+", choices=BASELINES)
    parser.add_argument("--workload", "--workloads", nargs="+", default=list(WORKLOADS))
    parser.add_argument("--prefix-tokens", nargs="+", type=int, default=[512, 2048, 8192, 16_384, 32_768])
    parser.add_argument("--suffix-tokens", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--output-tokens", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--result-dir", default="results/m3_11_baseline_readiness")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--sidecar-url", default="http://127.0.0.1:8010")
    parser.add_argument("--model-context-limit", type=int, default=DEFAULT_MODEL_CONTEXT_LIMIT)
    parser.add_argument("--kv-bytes-per-token", type=int, default=DEFAULT_KV_BYTES_PER_TOKEN)
    parser.add_argument("--storage-backend", default="local_posix")
    parser.add_argument("--connector-event-log")
    parser.add_argument("--sidecar-decision-log")
    parser.add_argument("--tensor-store")
    parser.add_argument("--cold-root")
    parser.add_argument(
        "--cold-backend",
        default="local_posix",
        choices=["local_posix", "3fs_posix", "packed_v1"],
    )
    parser.add_argument("--cold-tier-advance-ms", type=float, default=12_000.0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--import-reuse-csv",
        action="append",
        default=[],
        help=(
            "Import existing reuse_smoke_matrix.csv for B3/B5. Format: "
            "BASELINE:PATH or BASELINE:PREFIX:SUFFIX:OUTPUT:PATH."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--strict-b2-restart",
        action="store_true",
        help="For B2, warm APC, stop vLLM, restart vLLM, then measure the same prefix.",
    )
    parser.add_argument("--serve-host", default="127.0.0.1")
    parser.add_argument("--serve-port", type=int, default=8000)
    parser.add_argument(
        "--serve-max-model-len",
        type=int,
        help="Max model length used when strict B2 starts vLLM; defaults to model context limit.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--cpu-offload-gb", type=float)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--startup-timeout-sec", type=float, default=420.0)
    parser.add_argument(
        "--vllm-extra-arg",
        action="append",
        default=[],
        help="Extra argument passed to `vllm serve`; repeat for multiple arguments.",
    )
    return parser.parse_args(argv)


def exit_code_for_status(status: str) -> int:
    return 0 if status in {"OK", "DRY_RUN", "PARTIAL"} else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        reuse_sources = _reuse_import_sources_from_args(args)
        reuse_executor = ReuseCsvImportExecutor(reuse_sources) if reuse_sources else None
        result = run_baseline_readiness_matrix(
            BaselineReadinessConfig(
                result_dir=Path(args.result_dir),
                model=args.model,
                base_url=args.base_url,
                sidecar_url=args.sidecar_url,
                model_context_limit=args.model_context_limit,
                kv_bytes_per_token=args.kv_bytes_per_token,
                baselines=_baselines_from_args(args),
                workloads=[str(item) for item in args.workload],
                prefix_tokens=[int(item) for item in args.prefix_tokens],
                suffix_tokens=[int(item) for item in args.suffix_tokens],
                output_tokens=[int(item) for item in args.output_tokens],
                concurrency=[int(item) for item in args.concurrency],
                repeats=int(args.repeats),
                phase=args.phase,
                dry_run=bool(args.dry_run),
                storage_backend=str(args.storage_backend),
                connector_event_log=(
                    Path(args.connector_event_log) if args.connector_event_log else None
                ),
                sidecar_decision_log=(
                    Path(args.sidecar_decision_log) if args.sidecar_decision_log else None
                ),
                tensor_store=Path(args.tensor_store) if args.tensor_store else None,
                cold_root=Path(args.cold_root) if args.cold_root else None,
                cold_backend=str(args.cold_backend),
                cold_tier_advance_ms=float(args.cold_tier_advance_ms),
                block_size=int(args.block_size),
                strict_b2_restart=bool(args.strict_b2_restart),
                serve_host=str(args.serve_host),
                serve_port=int(args.serve_port),
                serve_max_model_len=args.serve_max_model_len,
                gpu_memory_utilization=float(args.gpu_memory_utilization),
                cpu_offload_gb=args.cpu_offload_gb,
                enforce_eager=bool(args.enforce_eager),
                enable_prefix_caching=bool(args.enable_prefix_caching),
                startup_timeout_sec=float(args.startup_timeout_sec),
                vllm_extra_args=[str(item) for item in args.vllm_extra_arg],
            ),
            reuse_executor=reuse_executor,
        )
    except NotImplementedError as exc:
        print(json.dumps({"status": "NOT_IMPLEMENTED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code_for_status(str(result["status"]))


if __name__ == "__main__":
    raise SystemExit(main())
