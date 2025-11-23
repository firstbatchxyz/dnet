"""Topology solver and related fakes."""


class FakeSolver:
    async def solve(self, **kwargs):
        return {"ok": True, "profiles": list(kwargs.get("profiles", {}).keys())}


class FakeBadSolver:
    async def solve(self, **kwargs):
        raise RuntimeError("boom")
