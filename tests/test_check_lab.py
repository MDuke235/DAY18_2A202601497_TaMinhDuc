import io
from contextlib import redirect_stdout

import check_lab


def test_run_tests_allows_slow_model_suite(monkeypatch):
    observed = {}

    class Completed:
        stdout = "======================= 39 passed in 243.75s =======================\n"

    def fake_run(*args, **kwargs):
        observed["command"] = args[0]
        observed.update(kwargs)
        return Completed()

    monkeypatch.setattr(check_lab.subprocess, "run", fake_run)

    assert check_lab.run_tests() == (39, 39)
    assert observed["timeout"] >= 300
    assert "--basetemp" in observed["command"]


def test_checker_supports_windows_console():
    output = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")

    with redirect_stdout(output):
        getattr(check_lab, "_configure_console", lambda: None)()
        assert check_lab.check_file("ASSIGNMENT.md")
