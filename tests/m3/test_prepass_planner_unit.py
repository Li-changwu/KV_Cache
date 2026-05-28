from benchmarks.m3.prepass_planner import build_prepass_plan


def test_prepass_plan_does_not_double_count_completed_restore_when_ready_count_is_final():
    plan = build_prepass_plan(
        required_count=1,
        ready_count=1,
        restore_results=[{"status": "COMPLETED", "deadline_miss": False}],
        errors=[],
    )

    assert plan.status == "READY"
    assert plan.ready_before_request is True
    assert plan.ready_count == 1
