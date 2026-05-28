import csv
import subprocess
import sys
from pathlib import Path

from benchmarks.m3.control_plane import (
    ConnectorState,
    CorrectnessKey,
    Decision,
    PrefixCandidate,
    SidecarRequest,
    TieredKVControlPlane,
)
from benchmarks.m3.replay import run_replay, write_replay_outputs


def key(model: str = "/root/models/Qwen2.5-14B-Instruct") -> CorrectnessKey:
    return CorrectnessKey(
        model_fingerprint=model,
        tokenizer_fingerprint="qwen2.5-tokenizer",
        rope_config="native-32768",
        dtype="bf16",
        kv_layout="vllm-paged",
    )


def request(
    request_id: str,
    prefix_id: str,
    token_count: int,
    candidate_end: int = 0,
    decode_sla_ms: float = 12_000.0,
    admission_window_ms: float = 12_000.0,
    correctness_key: CorrectnessKey | None = None,
) -> SidecarRequest:
    candidates = []
    if candidate_end > 0:
        candidates.append(
            PrefixCandidate(
                prefix_id=prefix_id,
                token_start=0,
                token_end=candidate_end,
                committed=True,
            )
        )
    return SidecarRequest(
        request_id=request_id,
        model_id=key().model_fingerprint,
        token_count=token_count,
        decode_sla_ms=decode_sla_ms,
        admission_window_ms=admission_window_ms,
        correctness_key=correctness_key or key(),
        prefix_candidates=candidates,
    )


def test_second_session_turn_reuses_committed_prefix_and_delta_prefills():
    cp = TieredKVControlPlane(
        hbm_capacity_tokens=65_000,
        dram_capacity_tokens=1_000_000,
        ssd_capacity_tokens=8_000_000,
        kv_bytes_per_token=196_608,
        h2d_gbps=25.0,
        storage_gbps=8.8,
    )

    first = cp.admit(request("r1", "session-a", token_count=1_048_576))
    cp.commit_request(first, prefix_id="session-a", token_end=1_048_576)
    second = cp.admit(
        request("r2", "session-a", token_count=1_048_576, candidate_end=786_432)
    )

    assert first.decision == Decision.FULL_PREFILL_FALLBACK
    assert second.decision == Decision.ADMIT
    assert second.reuse_tokens == 786_432
    assert second.delta_prefill_tokens == 262_144
    assert second.sync_ssd_miss_allowed is False
    assert second.required_ranges


def test_ready_barrier_blocks_missing_kv_without_sync_ssd_miss():
    cp = TieredKVControlPlane(
        hbm_capacity_tokens=65_000,
        dram_capacity_tokens=100_000,
        ssd_capacity_tokens=8_000_000,
        kv_bytes_per_token=196_608,
        h2d_gbps=25.0,
        storage_gbps=2.5,
    )
    cp.manifest.commit(
        prefix_id="session-a",
        token_start=0,
        token_end=786_432,
        correctness_key=key(),
        tier=ConnectorState.SSD,
        ready=False,
    )

    response = cp.admit(
        request("r1", "session-a", token_count=1_048_576, candidate_end=786_432)
    )
    barrier = cp.ready_barrier(response)

    assert barrier.all_required_blocks_ready is False
    assert barrier.sync_ssd_miss_total == 0
    assert barrier.missing_blocks
    assert response.decision == Decision.DELAY


def test_mark_prefetched_updates_manifest_and_next_admit_succeeds():
    cp = TieredKVControlPlane(
        hbm_capacity_tokens=65_000,
        dram_capacity_tokens=100_000,
        ssd_capacity_tokens=8_000_000,
        kv_bytes_per_token=196_608,
        h2d_gbps=25.0,
        storage_gbps=8.8,
    )
    cp.manifest.commit(
        prefix_id="session-a",
        token_start=0,
        token_end=64,
        correctness_key=key(),
        tier=ConnectorState.SSD,
        ready=False,
    )
    delayed = cp.admit(request("r1", "session-a", token_count=80, candidate_end=64))

    assert delayed.decision == Decision.DELAY

    cp.mark_prefetched(delayed.required_ranges)
    admitted = cp.admit(request("r2", "session-a", token_count=80, candidate_end=64))

    assert admitted.decision == Decision.ADMIT
    assert cp.ready_barrier(admitted).all_required_blocks_ready is True


def test_newer_unready_capacity_commit_overrides_older_ready_commit():
    cp = TieredKVControlPlane(
        hbm_capacity_tokens=65_000,
        dram_capacity_tokens=100_000,
        ssd_capacity_tokens=8_000_000,
        kv_bytes_per_token=196_608,
        h2d_gbps=25.0,
        storage_gbps=8.8,
    )
    stored = cp.admit(request("store", "session-a", token_count=64))
    cp.commit_request(
        stored,
        prefix_id="session-a",
        token_end=64,
        tier=ConnectorState.DRAM,
        ready=True,
    )
    cp.commit_request(
        stored,
        prefix_id="session-a",
        token_end=64,
        tier=ConnectorState.SSD,
        ready=False,
    )

    delayed = cp.admit(request("reuse", "session-a", token_count=80, candidate_end=64))

    assert delayed.decision == Decision.DELAY
    assert delayed.reason == "required_kv_not_ready_before_decode"
    assert cp.ready_barrier(delayed).all_required_blocks_ready is False


def test_low_locality_1m_request_uses_full_prefill_fallback_not_admit():
    cp = TieredKVControlPlane(
        hbm_capacity_tokens=65_000,
        dram_capacity_tokens=1_000_000,
        ssd_capacity_tokens=8_000_000,
        kv_bytes_per_token=196_608,
        h2d_gbps=25.0,
        storage_gbps=8.8,
    )

    response = cp.admit(request("random-1", "random", token_count=1_048_576))

    assert response.decision == Decision.FULL_PREFILL_FALLBACK
    assert response.reuse_tokens == 0
    assert response.delta_prefill_tokens == 1_048_576
    assert response.sync_ssd_miss_allowed is False


def test_correctness_key_mismatch_rejects_reuse():
    cp = TieredKVControlPlane(
        hbm_capacity_tokens=65_000,
        dram_capacity_tokens=1_000_000,
        ssd_capacity_tokens=8_000_000,
        kv_bytes_per_token=196_608,
        h2d_gbps=25.0,
        storage_gbps=8.8,
    )
    cp.manifest.commit(
        prefix_id="session-a",
        token_start=0,
        token_end=786_432,
        correctness_key=key(model="old-model"),
        tier=ConnectorState.DRAM,
        ready=True,
    )

    response = cp.admit(
        request("r1", "session-a", token_count=1_048_576, candidate_end=786_432)
    )

    assert response.decision == Decision.REJECT
    assert response.reason == "correctness_key_mismatch"


def test_replay_writes_decisions_and_ready_barriers(tmp_path):
    cp = TieredKVControlPlane(
        hbm_capacity_tokens=65_000,
        dram_capacity_tokens=1_000_000,
        ssd_capacity_tokens=8_000_000,
        kv_bytes_per_token=196_608,
        h2d_gbps=25.0,
        storage_gbps=8.8,
    )
    rows = run_replay(
        cp,
        [
            request("r1", "session-a", token_count=1_048_576),
            request("r2", "session-a", token_count=1_048_576, candidate_end=786_432),
        ],
        commit_prefix_id="session-a",
    )

    write_replay_outputs(tmp_path, rows)

    with (tmp_path / "m3_replay_decisions.csv").open(newline="", encoding="utf-8") as fh:
        decisions = list(csv.DictReader(fh))
    report = (tmp_path / "report.md").read_text(encoding="utf-8")

    assert [row["decision"] for row in decisions] == ["FULL_PREFILL_FALLBACK", "ADMIT"]
    assert decisions[1]["reuse_tokens"] == "786432"
    assert "sync_ssd_miss_total=0" in report


def test_replay_cli_writes_outputs(tmp_path):
    output_dir = tmp_path / "out"
    repo_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/m3/replay_cli.py",
            "--simulator-params",
            "results/qwen25_14b_calibration/simulator_params.yaml",
            "--capacity-csv",
            "results/qwen25_14b_calibration/qwen25_capacity_probe.csv",
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "m3_replay_decisions.csv").exists()
    assert (output_dir / "report.md").exists()
