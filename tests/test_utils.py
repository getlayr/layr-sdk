from layr.utils import env, estimate_cost_usd


def test_utils_cost_and_env(monkeypatch):
    monkeypatch.setenv("LAYR_TEST", "x")
    assert env("LAYR_TEST") == "x"
    assert estimate_cost_usd("gpt-4o", 1000, 500) > 0.0
