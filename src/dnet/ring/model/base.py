"""Base class for ring topology models."""

from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Tuple, Dict, Set, List

import mlx.core as mx
import mlx.nn as nn


class BaseRingModel(nn.Module, metaclass=ABCMeta):
    """Base class for models used in ring topology.

    Subclasses must implement embedding, normalization, LM projection,
    and layer-by-layer application for distributed inference.
    """

    model_type: Optional[str] = None

    @abstractmethod
    def embed(self, x: mx.array) -> mx.array:
        """Embed input tokens.

        Args:
            x: Input token IDs

        Returns:
            Embedded representations
        """

    @abstractmethod
    def normalize(self, x: mx.array) -> mx.array:
        """Apply final normalization.

        Args:
            x: Hidden states

        Returns:
            Normalized hidden states
        """

    @abstractmethod
    def lm_project(self, x: mx.array) -> mx.array:
        """Project to vocabulary logits.

        Args:
            x: Normalized hidden states

        Returns:
            Logits over vocabulary
        """

    # Note: forward is intentionally not abstract in ring models.
    # The ring execution path uses apply_single_layer; subclasses
    # may implement forward for convenience/testing, but it is not required.

    @abstractmethod
    def apply_single_layer(
        self, layer_idx: int, x: mx.array, cache: Optional[Any] = None
    ) -> mx.array:
        """Apply a single decoding layer by absolute index.

        Implementations should map `layer_idx` to their local layers if they
        only host a subset, and use the correct per-layer cache entry if
        provided.

        Args:
            layer_idx: Absolute layer index
            x: Layer input
            cache: Optional per-layer cache

        Returns:
            Layer output
        """

    @property
    @abstractmethod
    def decoding_layers(self) -> Any:
        """Get the decoding layers.

        Returns:
            Layer container (e.g., ModuleList)
        """

    @property
    @abstractmethod
    def head_dim(self) -> Tuple[int, int]:
        """Get head dimensions.

        Returns:
            Tuple of (num_heads, dim_per_head)
        """

    @property
    @abstractmethod
    def n_kv_heads(self) -> int:
        """Get number of key/value heads.

        Returns:
            Number of KV heads
        """

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Get total number of layers in the model.

        Returns:
            Number of layers
        """

    def load_weights(self, file_or_weights, strict: bool = False):
        """Bind weights for this shard"""

        wdict: Dict[str, Any]
        if isinstance(file_or_weights, dict):
            wdict = dict(file_or_weights)
        elif isinstance(file_or_weights, (list, tuple)):
            try:
                wdict = {
                    (
                        k.decode("utf-8")
                        if isinstance(k, (bytes, bytearray))
                        else str(k)
                        if not isinstance(k, str)
                        else k
                    ): v
                    for (k, v) in file_or_weights  # type: ignore[misc]
                }
            except UnicodeError:
                wdict = dict(file_or_weights)  # type: ignore[arg-type]
        elif isinstance(file_or_weights, str):
            try:
                wdict = mx.load(file_or_weights)  # type: ignore[assignment]
            except ValueError:
                wdict = {}
        else:
            try:
                wdict = dict(file_or_weights)  # type: ignore[arg-type]
            except ValueError:
                wdict = {}

        # Model-specific canonicalization: prefer tensor-level when available
        try:
            if hasattr(self, "sanitize_weights"):
                # type: ignore[attr-defined]
                wdict = getattr(self, "sanitize_weights")(wdict)
            elif hasattr(self, "sanitize"):
                # type: ignore[attr-defined]
                wdict = getattr(self, "sanitize")(wdict)
        except Exception:
            # Be permissive: proceed even if sanitize fails
            pass

        shard_w: Dict[str, mx.array] = {}

        def _has_module(name: str) -> bool:
            return hasattr(self, name)

        tie = bool(getattr(self.config, "tie_word_embeddings", False))

        for key, value in wdict.items():
            # Per-layer tensors: accept both model.layers.N.* and layers.N.*
            if key.startswith("model.layers.") or key.startswith("layers."):
                parts = key.split(".")
                idx_pos = 2 if parts[0] == "model" else 1
                try:
                    abs_idx = int(parts[idx_pos])
                except Exception:
                    continue
                # Map absolute -> local for hosted layers only
                abs2loc = getattr(self, "abs_to_local", {}) or {}
                if abs_idx not in abs2loc:
                    continue
                local_idx = abs2loc[abs_idx]
                parts[idx_pos] = str(local_idx)
                # Drop leading 'model.' if present
                if parts[0] == "model":
                    parts = parts[1:]
                new_key = ".".join(parts)
                shard_w[new_key] = value
                continue

            # API-layer tensors: embed_tokens.*, norm.*, lm_head.* (strip optional model.)
            if key.startswith("model."):
                bare = key[6:]
            else:
                bare = key
            if bare.startswith("embed_tokens.") and _has_module("embed_tokens"):
                shard_w[bare] = value
                continue
            if bare.startswith("norm.") and _has_module("norm"):
                shard_w[bare] = value
                continue
            if bare.startswith("lm_head.") and _has_module("lm_head") and (not tie):
                shard_w[bare] = value
                continue

        return super().load_weights(list(shard_w.items()), strict=strict)

    def _abskey_to_local_path(self, key: str) -> Optional[str]:
        """Generic mapping of absolute weight/config keys to local module paths.

        - model.layers.ABS.suffix -> layers.LOCAL.suffix (uses abs_to_local)
        - model.embed_tokens*     -> embed_tokens
        - model.norm*             -> norm
        - (model.)lm_head*        -> lm_head
        Returns None if the ABS layer is not hosted locally.
        """
        try:
            if key.startswith("model.layers."):
                parts = key.split(".")
                abs_idx = int(parts[2])
                # tolerate subclasses without abs_to_local
                abs2loc = getattr(self, "abs_to_local", {}) or {}
                local = abs2loc.get(abs_idx)
                if local is None:
                    return None
                suffix = ".".join(parts[3:])
                return f"layers.{local}.{suffix}"
            if key.startswith("model.embed_tokens") or key.startswith("embed_tokens"):
                return "embed_tokens"
            if key.startswith("model.norm") or key.startswith("norm"):
                return "norm"
            if key.startswith("model.lm_head") or key.startswith("lm_head"):
                return "lm_head"
        except Exception:
            return None
        return None

    def apply_quantization_from_config(
        self, model_config: Any, model_metadata: Any
    ) -> bool:
        """Quantize using a simple MLX-style predicate with optional per-path overrides.

        - If config["quantization"][path] exists, use that for this path.
        - Else, quantize only if a matching `{path}.scales` exists in the (sanitized) weight names.
        """
        try:
            # Build a set of all tensor names in the checkpoint
            weight_names: Set[str] = set()
            try:
                for lid, tmap in model_metadata.weight_info.items():
                    for suffix in tmap.keys():
                        weight_names.add(f"model.layers.{int(lid)}.{suffix}")
                for k in model_metadata.embed_tokens.keys():
                    weight_names.add(f"model.embed_tokens.{k}")
                for k in model_metadata.norm.keys():
                    weight_names.add(f"model.norm.{k}")
                for k in model_metadata.lm_head.keys():
                    weight_names.add(f"lm_head.{k}")
            except Exception:
                weight_names = set()

            try:
                if hasattr(self, "sanitize") and weight_names:
                    wdict = {k: True for k in weight_names}
                    sanitized = self.sanitize(dict(wdict))  # type: ignore[attr-defined]
                    if isinstance(sanitized, dict) and sanitized:
                        weight_names = set(sanitized.keys())
            except Exception:
                pass

            # Extract global params and per-path overrides
            qspec = model_config.get("quantization") or {}
            overrides_map: Dict[str, Dict[str, Any]] = {}
            local_overrides_map: Dict[str, Dict[str, Any]] = {}
            g_bits, g_group = 0, 0
            g_mode = "affine"
            has_qspec = bool(qspec)
            if isinstance(qspec, dict) and qspec:
                # Always read global defaults from root when present
                g_bits = int(qspec.get("bits", 0) or 0)
                g_group = int(qspec.get("group_size", 0) or 0)
                g_mode = str(qspec.get("mode", "affine")).strip().lower() or "affine"

                ds = qspec.get("disable_sinks")
                if isinstance(ds, bool):
                    self._disable_sinks_on_quant = ds

                # Collect per-path overrides from remaining dict-valued entries
                for k, v in qspec.items():
                    if k in {"bits", "group_size", "mode"}:
                        continue
                    if isinstance(v, dict):
                        # Use override values exactly as provided; do not fill from globals
                        o: Dict[str, Any] = {}
                        if "bits" in v:
                            try:
                                o["bits"] = int(v.get("bits"))
                            except Exception:
                                pass
                        if "group_size" in v:
                            try:
                                o["group_size"] = int(v.get("group_size"))
                            except Exception:
                                pass
                        if "mode" in v:
                            try:
                                o["mode"] = str(v.get("mode")).strip().lower()
                            except Exception:
                                pass
                        overrides_map[k] = o
                        # Map absolute key to local path used by nn.quantize predicate
                        lp = self._abskey_to_local_path(k)
                        if lp:
                            local_overrides_map[lp] = o
                        # Also support API-layer keys without model. prefix
                        if k.startswith("model.embed_tokens"):
                            local_overrides_map.setdefault("embed_tokens", o)
                        if k.startswith("model.norm"):
                            local_overrides_map.setdefault("norm", o)
                        if k.startswith("lm_head") or k.startswith("model.lm_head"):
                            local_overrides_map.setdefault("lm_head", o)

            qcfg = model_config.get("quantization_config") or {}
            if not g_bits:
                try:
                    g_bits = int(qcfg.get("bits", 0) or 0)
                except Exception:
                    g_bits = 0
            if not g_group:
                try:
                    g_group = int(qcfg.get("group_size", 0) or 0)
                except Exception:
                    g_group = 0
            # Only use legacy quant_method to fill mode if root didn't specify it
            if not g_mode:
                method = str(qcfg.get("quant_method", "")).strip().lower()
                g_mode = method or "affine"
            # Do not hardcode defaults; require explicit config for global params
            # If not provided, leave as zero/empty and rely solely on per-path overrides

            def _local_path_to_abs_prefix(path: str) -> Optional[str]:
                try:
                    # If predicate already supplies absolute-style keys, pass through
                    if path.startswith("model.layers."):
                        return path
                    if path.startswith("layers."):
                        parts = path.split(".")
                        local = int(parts[1])
                        abs2loc = getattr(self, "abs_to_local", {}) or {}
                        loc2abs = {v: k for k, v in abs2loc.items()}
                        abs_idx = loc2abs.get(local)
                        if abs_idx is None:
                            return None
                        suffix = ".".join(parts[2:])
                        return f"model.layers.{abs_idx}.{suffix}"
                    if path.startswith("embed_tokens") or path.startswith(
                        "model.embed_tokens"
                    ):
                        return "model.embed_tokens"
                    if path.startswith("norm") or path.startswith("model.norm"):
                        return "model.norm"
                    if path.startswith("lm_head") or path.startswith("model.lm_head"):
                        return "lm_head"
                except Exception:
                    return None
                return None

            def _predicate(path: str, module: nn.Module):
                # Per-path override if present in config (match on LOCAL path)
                if local_overrides_map:
                    o = local_overrides_map.get(path)
                    if o and hasattr(module, "to_quantized"):
                        return o
                # Also consider absolute form for completeness
                if overrides_map and hasattr(module, "to_quantized"):
                    abs_pref = _local_path_to_abs_prefix(path) or path
                    o = overrides_map.get(abs_pref)
                    if o:
                        return o

                # Only quantize when a quantization section exists in config
                if not has_qspec:
                    return False
                if not hasattr(module, "to_quantized"):
                    return False
                abs_pref = _local_path_to_abs_prefix(path)
                if abs_pref is None:
                    return False
                return f"{abs_pref}.scales" in weight_names

            try:
                # If global params are not provided, call quantize without root bits/group and
                # rely entirely on per-path overrides returned by class_predicate.
                if g_bits and g_group and g_mode:
                    nn.quantize(
                        self,
                        group_size=int(g_group),
                        bits=int(g_bits),
                        mode=str(g_mode),  # type: ignore[arg-type]
                        class_predicate=_predicate,
                    )  # type: ignore[call-arg]
                else:
                    nn.quantize(
                        self,
                        class_predicate=_predicate,
                    )
            except TypeError:
                if g_bits and g_group:
                    nn.quantize(
                        self,
                        group_size=int(g_group),
                        bits=int(g_bits),
                        class_predicate=_predicate,
                    )
                else:
                    nn.quantize(
                        self,
                        class_predicate=_predicate,
                    )
            except Exception:
                self._converted_to_quantized = False
                return False
            self._converted_to_quantized = True
            return True
        except Exception:
            try:
                self._converted_to_quantized = False
            except Exception:
                pass
            return False

    @staticmethod
    def _shrink_linear_like(mod) -> None:
        """Replace common parameter arrays with tiny zero tensors.

        Safe for quantized and unquantized modules. Used to free memory for
        layers that won’t run until re-bound.
        """
        try:
            for name in ("weight", "bias", "scales", "biases"):
                if hasattr(mod, name):
                    arr = getattr(mod, name)
                    try:
                        dt = arr.dtype
                    except Exception:
                        continue
                    if name == "weight":
                        new_arr = mx.zeros((1, 1), dtype=dt)
                    else:
                        new_arr = mx.zeros((1,), dtype=dt)
                    setattr(mod, name, new_arr)
        except Exception:
            pass

    def _shrink_block(self, block) -> None:
        """Shrink attention/MLP submodules when present.

        Handles typical projections (q/k/v/o, gate/up/down) and optional
        structures like routers/experts if exposed by the model.
        """
        try:
            attn = getattr(block, "self_attn", None)
            if attn is not None:
                for pn in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    if hasattr(attn, pn):
                        self._shrink_linear_like(getattr(attn, pn))

            mlp = getattr(block, "mlp", None)
            if mlp is not None:
                for pn in ("gate_proj", "up_proj", "down_proj"):
                    if hasattr(mlp, pn):
                        self._shrink_linear_like(getattr(mlp, pn))
                # Some models expose a router submodule
                if hasattr(mlp, "router"):
                    self._shrink_linear_like(getattr(mlp, "router"))
                # Optional experts container with projections
                experts = getattr(mlp, "experts", None)
                if experts is not None:
                    for pn in ("gate_proj", "up_proj", "down_proj"):
                        if hasattr(experts, pn):
                            self._shrink_linear_like(getattr(experts, pn))
        except Exception:
            pass

    def unload_layers(self, abs_layers: List[int]) -> None:
        """Shrink params for the given absolute layer indices hosted locally."""
        for abs_idx in abs_layers:
            try:
                abs2loc = getattr(self, "abs_to_local", {}) or {}
                local = abs2loc.get(abs_idx)
                if local is None:
                    continue
                layers = getattr(self, "layers", []) or []
                if 0 <= local < len(layers):
                    self._shrink_block(layers[local])
            except Exception:
                continue
