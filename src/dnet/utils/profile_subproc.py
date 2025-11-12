import json
import multiprocessing as mp
from distilp.common import DeviceProfile

def _child_profile_device(repo_id: str, max_batch_exp: int, debug: int, conn) -> None:
    try:
        # import inside child to avoid pre-initializing Metal/MLX in parent
        from distilp.profiler import profile_device as _profile_device

        device_profile = _profile_device(repo_id, max_batch_exp=max_batch_exp, debug=debug)
        conn.send(device_profile.model_dump_json())
    except ChildProcessError as e:
        try:
            conn.send(json.dumps({"_error": str(e)}))
        except ImportError:
            pass
    finally:
        try:
            conn.close()
        except ConnectionError:
            pass


def profile_device_via_subprocess(
    repo_id: str, *, max_batch_exp: int = 6, debug: int = 0, timeout_s: float = 300.0
) -> DeviceProfile:
    """Run distilp.profiler.profile_device in a fresh subprocess and return a dict.

    This isolates Metal/IOAccelerator allocations to the child so that they are
    reclaimed on exit, avoiding persistent memory bloat.
    """
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(
        target=_child_profile_device,
        args=(repo_id, max_batch_exp, debug, child),
        daemon=True,
    )
    p.start()
    try:
        child.close()
        data = parent.recv()  # blocks until child sends JSON
    finally:
        try:
            parent.close()
        except ChildProcessError:
            pass
    p.join(timeout_s)
    if p.exitcode is None:
        try:
            p.terminate()
        except ChildProcessError:
            pass
        raise RuntimeError("device profiler subprocess timed out")

    obj = json.loads(data)
    if isinstance(obj, dict) and obj.get("_error"):
        raise RuntimeError(f"device profiler failed: {obj['_error']}")
    else:
        return DeviceProfile.model_validate_json(data)
