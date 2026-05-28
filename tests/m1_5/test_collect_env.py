from pathlib import Path

from benchmarks.m1_5.collect_env import collect


def test_collect_env_marks_sliding_window_models_limited(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        '{"model_type": "test", "sliding_window": 128}',
        encoding="utf-8",
    )

    env = collect(model)

    assert (
        env["notes"]["dense_attention_representativeness"]
        == "limited_by_sliding_window_attention"
    )


def test_collect_env_marks_non_sliding_window_models_full_path(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        '{"model_type": "qwen3", "use_sliding_window": false, "sliding_window": null}',
        encoding="utf-8",
    )

    env = collect(model)

    assert (
        env["notes"]["dense_attention_representativeness"]
        == "full_attention_path_for_configured_context"
    )


def test_collect_env_respects_disabled_sliding_window_field(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        (
            '{"model_type": "qwen2", "use_sliding_window": false, '
            '"sliding_window": 131072}'
        ),
        encoding="utf-8",
    )

    env = collect(model)

    assert (
        env["notes"]["dense_attention_representativeness"]
        == "full_attention_path_for_configured_context"
    )
