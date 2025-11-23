"""Shared test fakes package.

This package centralizes lightweight test doubles used across the suite.
Keep fakes small, documented, and focused on test needs only.

Import convenience:
    from tests.fakes import FakeShard, FakeDiscovery, FakeRingStub, ...
"""

# Discovery / device props
from .discovery import FakeProps, FakeDiscovery, FakeTBConn

# API-facing fakes
from .api import (
    FakeClient,
    FakeResponse,
    FakeLatencyRequest,
    FakeProfileRequest,
    FakeLatencyResponse,
    FakeProfileResponse,
    FakeTokenizer,
    FakeTokenizerWithTemplate,
    FakeDetokenizer,
    FakeTokenResult,
    FakeStrategyAdapter,
    FakeModelManager,
    FakeInferenceManager,
    FakeModelProfile,
    FakeClusterManager,
)

# gRPC channel/stubs
from .grpc import FakeGrpcServer, FakeChannel, FakeRingStub, FakeApiStub

# Core models / metadata
from .models import FakeWeightSize, FakeModelMetadata, FakeRingModel

# Weight cache and layer manager
from .weight_cache import (
    FakeLayerManagerForCache,
    patch_layer_manager_for_cache,
    FakeWeightCache,
)

# Shard/runtime/policies
from .runtime import FakeRuntimeMinimal, FakeRuntimeForAdapter, FakeRuntimeForPolicy
from .shard import FakeAdapter, FakeShard
from .policies import FakePolicyPlan, FakePolicy, FakeSampler, FakeComputeModel
from .streams import FakeStreamAck, FakeStreamCall

# Solvers
from .solver import FakeSolver, FakeBadSolver
from .mp import FakeMPConn, FakeMPProc, FakeMPContext

__all__ = [
    name
    for name in globals().keys()
    if name.startswith("Fake") or name.endswith("layer_manager_for_cache")
]
