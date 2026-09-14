"""Check the measurement gate and a small real compiled comparison."""

from dataclasses import replace

from benchmarks.numerical import windowed_history as report


def test_compiled_measurements_and_failed_gate():
    rows = report.generate_rows(lengths=(32,), dimensions=(2,), windows=(4,),
                                repeats=2)
    assert {row.method for row in rows} == set(report.METHODS)
    assert report.rows_passed(rows, repeats=2)
    for change in ({"agrees_with_dense": False}, {"repeatable": False},
                   {"sample_seconds": (float("nan"), .1)}, {"temporary_bytes": None}):
        assert not report.rows_passed([replace(rows[0], **change)], repeats=2)
    assert not report.rows_passed([], repeats=2)
    assert not report.rows_passed(rows, repeats=3)
