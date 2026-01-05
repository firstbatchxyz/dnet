import mlx.core as mx
import numpy as np
from typing import Optional, Any, Dict
from mlx_lm.sample_utils import make_sampler
from dnet.core.types.messages import TokenResult
from dnet.core.decoding.config import DecodingConfig
from dnet.utils.logger import logger


class GrammarState:
    def __init__(
        self,
        guide,
        index,
        bitmask_allocator,
        vocab_size: int,
        eos_token_id: Optional[int] = None,
    ):
        self.guide = guide
        self.index = index
        self.bitmask_allocator = bitmask_allocator
        self.vocab_size = vocab_size
        self._eos_token_id = eos_token_id
        self._bitmask = None
        self._terminated = False

    def get_bitmask(self):
        if self._bitmask is None:
            self._bitmask = self.bitmask_allocator(self.vocab_size)
        return self._bitmask

    def fill_next_token_bitmask(self):
        if self._terminated:
            return None

        from outlines_core.kernels.mlx import fill_next_token_bitmask

        bitmask = self.get_bitmask()
        fill_next_token_bitmask(self.guide, bitmask)
        return bitmask

    def accept_token(self, token_id: int) -> None:
        if self._terminated:
            return

        if not self.guide.is_finished():
            self.guide.advance(token_id=token_id, return_tokens=False)
        else:
            self._terminated = True

    def is_terminated(self) -> bool:
        if self._terminated:
            return True

        if not self.guide.is_finished():
            return False

        try:
            current_state = self.guide.get_state()
            is_final = self.index.is_final_state(current_state)

            if is_final:
                self._terminated = True
                return True

            self._terminated = True
            logger.debug(
                f"Guide finished but not in final state - marking as terminated anyway. "
                f"state={current_state}, is_final={is_final}"
            )
            return True
        except Exception as e:
            self._terminated = True
            logger.debug(
                f"Could not verify final state: {e}, using is_finished()={self.guide.is_finished()}, "
                f"marking as terminated"
            )
            return True


class Sampler:
    """
    Handles the transformation of logits into tokens based on a DecodingConfig.
    Wraps mlx_lm's make_sampler for consistent sampling behavior.
    Supports structured output via grammar-constrained generation using Outlines.
    """

    # Cache for compiled vocabulary to avoid recomputing per request
    _vocabulary_cache: Dict[int, Any] = {}

    def __init__(self):
        pass

    @staticmethod
    def _get_or_create_vocabulary(tokenizer, vocab_size: int):
        cache_key = id(tokenizer)
        if cache_key in Sampler._vocabulary_cache:
            return Sampler._vocabulary_cache[cache_key]

        try:
            from outlines_core import Vocabulary

            vocab = tokenizer.get_vocab()
            actual_vocab_size = len(vocab)
            if vocab_size != actual_vocab_size:
                logger.warning(
                    f"Vocab size mismatch: expected {vocab_size} (from model/logits) "
                    f"but tokenizer has {actual_vocab_size} tokens. "
                    f"Using model vocab_size {vocab_size} for bitmask allocation."
                )

            eos_token_id = tokenizer.eos_token_id
            eos_token = tokenizer.eos_token or tokenizer.decode([eos_token_id])

            # Build formatted vocabulary for Outlines
            formatted_vocab = {}
            for token, token_id in vocab.items():
                try:
                    token_as_str = tokenizer.convert_tokens_to_string([token])
                    if token_as_str not in formatted_vocab:
                        formatted_vocab[token_as_str] = [token_id]
                    else:
                        formatted_vocab[token_as_str].append(token_id)
                except Exception:
                    if token not in formatted_vocab:
                        formatted_vocab[token] = [token_id]
                    else:
                        formatted_vocab[token].append(token_id)

            # Remove EOS token from vocab (Outlines handles it separately)
            formatted_vocab.pop(eos_token, None)

            vocabulary = Vocabulary(eos_token_id, formatted_vocab)
            Sampler._vocabulary_cache[cache_key] = vocabulary

            logger.debug(
                f"Created vocabulary: {len(formatted_vocab)} entries, vocab_size={vocab_size}, actual={actual_vocab_size}"
            )
            return vocabulary

        except Exception as e:
            logger.warning(f"Failed to create Outlines vocabulary: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return None

    @staticmethod
    def create_grammar_state(
        json_schema: str, tokenizer, model_vocab_size: Optional[int] = None
    ) -> Optional[GrammarState]:
        if not json_schema:
            return None

        try:
            from outlines_core import Index, Guide
            from outlines_core.outlines_core import json_schema as oc_json_schema
            from outlines_core.kernels.mlx import allocate_token_bitmask

            # Get vocab_size: prefer model_vocab_size (from logits shape) over tokenizer.vocab_size
            vocab_size = model_vocab_size or getattr(tokenizer, "vocab_size", None)
            if vocab_size is None:
                logger.warning("Could not determine vocab size for grammar state")
                return None

            if model_vocab_size:
                logger.debug(f"Using model_vocab_size={vocab_size} (from logits shape)")
            else:
                logger.debug(f"Using tokenizer.vocab_size={vocab_size} (fallback)")

            regex_pattern = oc_json_schema.build_regex_from_schema(json_schema)
            logger.debug(f"Built regex from JSON schema (length: {len(regex_pattern)})")

            vocabulary = Sampler._get_or_create_vocabulary(tokenizer, vocab_size)
            if vocabulary is None:
                logger.warning("Failed to create vocabulary for grammar state")
                return None

            index = Index(regex_pattern, vocabulary)
            guide = Guide(index)
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            eos_token_id = getattr(tokenizer, "eos_token_id", None)

            logger.debug("Created grammar state")
            return GrammarState(
                guide=guide,
                index=index,
                bitmask_allocator=allocate_token_bitmask,
                vocab_size=vocab_size,
                eos_token_id=eos_token_id,
            )

        except ImportError as e:
            logger.warning(f"Outlines not installed or import error: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to create grammar state: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return None

    @staticmethod
    def sample(
        logits: mx.array,
        config: DecodingConfig,
        req_logprobs: bool = False,
        req_top_logprobs: int = 0,
        grammar_state: Optional[GrammarState] = None,
    ) -> TokenResult:
        sampler_fn = make_sampler(
            temp=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            min_p=config.min_p if hasattr(config, "min_p") else 0.0,
            min_tokens_to_keep=config.min_tokens_to_keep
            if hasattr(config, "min_tokens_to_keep")
            else 1,
        )
        ndim = getattr(logits, "ndim", None)
        if ndim == 3:
            v = logits[:, -1, :]
            v = v[0]
        elif ndim == 2:
            v = logits[-1]
        else:
            v = logits

        if grammar_state is not None:
            try:
                from outlines_core.kernels.mlx import apply_token_bitmask

                bitmask = grammar_state.fill_next_token_bitmask()

                if bitmask is not None:
                    v_2d = v[None, :] if v.ndim == 1 else v
                    v_masked = apply_token_bitmask(v_2d, bitmask)
                    v = v_masked[0] if v_masked.ndim == 2 else v_masked
                else:
                    v = mx.full_like(v, float("-inf"))

            except Exception as e:
                logger.warning(f"Failed to apply grammar mask: {e}")
                import traceback

                logger.debug(traceback.format_exc())

        token_tensor = sampler_fn(v)
        token_id = int(token_tensor.item())

        if grammar_state is not None:
            try:
                grammar_state.accept_token(token_id)
            except Exception as e:
                logger.warning(f"Failed to accept token in grammar: {e}")
                import traceback

                logger.debug(traceback.format_exc())

        logprob = 0.0
        top_logprobs = {}

        if req_logprobs or req_top_logprobs > 0:
            log_sum_exp = mx.logsumexp(v, axis=-1)
            log_probs = v - log_sum_exp

            if req_logprobs:
                logprob = float(log_probs[token_id].item())

            if req_top_logprobs > 0:
                ti = mx.argsort(v)
                ti_np = np.array(ti.tolist())[::-1][:req_top_logprobs]
                for idx in ti_np:
                    ii = int(idx)
                    top_logprobs[ii] = float(log_probs[ii].item())

        return TokenResult(
            token_id=token_id,
            logprob=logprob,
            top_logprobs=top_logprobs,
        )
