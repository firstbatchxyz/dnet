"""Tests: utils.profile_subproc process wrapper without launching real processes."""

import pytest

from dnet.utils.profile_subproc import profile_device_via_subprocess

pytestmark = [pytest.mark.core]


def test_profile_device_via_subprocess_error_path(monkeypatch):
    from tests.fakes import FakeMPContext

    # fake multiprocessing context that avoids real processes and pipes
    monkeypatch.setattr(
        "dnet.utils.profile_subproc.mp.get_context",
        lambda _: FakeMPContext(),
        raising=True,
    )

    with pytest.raises(RuntimeError):
        profile_device_via_subprocess("hf://m", max_batch_exp=1, debug=0, timeout_s=0.1)
