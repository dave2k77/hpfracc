"""The diagnostic parent must preserve all successes and native failures."""

import json
from types import SimpleNamespace

from benchmarks.numerical import jax_convolution_probe as probe


def test_native_failure_does_not_discard_other_processes(monkeypatch, capsys):
    calls = []

    def run(command, **kwargs):
        assert kwargs["env"]["JAX_ENABLE_X64"] == "1"
        assert kwargs["env"]["JAX_PLATFORMS"] == "cpu"
        calls.append(command)
        failed = len(calls) == 2
        return SimpleNamespace(returncode=3221225477 if failed else 0,
                               stdout="before trial", stderr="native fault" if failed
                               else "")

    monkeypatch.setattr(probe.subprocess, "run", run)
    assert probe.main(["--processes", "3"]) == 1
    record = json.loads(capsys.readouterr().out)
    assert len(calls) == len(record["runs"]) == 3
    assert [r["returncode_hex"] for r in record["runs"]] == [
        "0x0", "0xc0000005", "0x0",
    ]
    assert record["runs"][1]["stderr"] == "native fault"
    assert all(r["stdout"] == "before trial" for r in record["runs"])
