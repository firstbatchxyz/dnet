Fakes Package Quick Guide

Purpose
- Centralize small test doubles (no real I/O/network).
- Keep imports stable: `from tests.fakes import FakeX`.

Import
- Prefer re‑exports: `from tests.fakes import FakeShard, FakeDiscovery, FakeRingStub`.
- Module import is allowed (e.g., `from tests.fakes.grpc import FakeRingStub`) but not required.

Layout
- discovery.py: FakeProps, FakeDiscovery, FakeTBConn
- api.py: FakeClient, FakeResponse, FakeTokenizer(+WithTemplate), FakeTokenResult, FakeStrategyAdapter, FakeModelManager, FakeInferenceManager, FakeModelProfile, FakeClusterManager
- grpc.py: FakeGrpcServer, FakeChannel, FakeRingStub, FakeApiStub
- models.py: FakeModelMetadata, FakeRingModel, FakeWeightSize
- weight_cache.py: FakeLayerManagerForCache, FakeWeightCache, patch_layer_manager_for_cache
- runtime.py: FakeRuntimeMinimal, FakeRuntimeForAdapter, FakeRuntimeForPolicy
- shard.py: FakeAdapter, FakeShard
- policies.py: FakePolicyPlan, FakePolicy, FakeComputeModel, FakeSampler
- solver.py: FakeSolver, FakeBadSolver

Patterns
- Prefer real request models over ad‑hoc dicts where practical.
- Keep fakes composable: accept constructor args where it helps (e.g., runtime, flags).
- For error paths, add tiny configurable hooks/flags instead of complex logic.

Adding a new fake
1) Pick the closest module (see Layout).
2) Implement the smallest surface required; add a 1‑line docstring.
3) Lazy‑import heavy modules if needed.
4) Re‑export in `__init__.py`.
5) Use it via `from tests.fakes import YourFake` in tests.

Anti‑patterns
- Don’t mirror full production classes; test intent over completeness.
- Don’t do real network/disk I/O from fakes.
- Don’t hide behavior behind global state; prefer explicit constructor params.

