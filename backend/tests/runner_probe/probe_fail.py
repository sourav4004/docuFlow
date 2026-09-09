"""Runner exit-code probe (failing). Not collected by normal test discovery;
used explicitly by tests/test_phase24_infra.py to verify that the validation
runner propagates pytest failures as a nonzero exit code."""


def test_probe_fails():
    assert False, "intentional probe failure - verifies exit-code propagation"
