"""Runner exit-code probe (passing). Not collected by normal test discovery
(python_files = test_*.py); used explicitly by tests/test_phase24_infra.py."""


def test_probe_passes():
    assert True
