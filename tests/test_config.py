"""Serving config validation: kill a bad run before it costs GPU-minutes."""

from simulator.config import ServingConfig


def test_validate_refuses_an_illegal_expert_parallel_size():
    bad = ServingConfig(model="Qwen/Qwen3.8-27B-FP8", ep_size=3)
    assert bad.validate()


def test_default_config_is_valid():
    assert ServingConfig(model="Qwen/Qwen3.8-27B-FP8").validate() == []


def test_batch_is_left_unbounded_by_default():
    """Production schedulers admit what fits; capping would measure a system we
    would not deploy."""
    assert ServingConfig().max_running_requests >= 256
