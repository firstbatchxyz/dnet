import mlx.core as mx
import numpy as np
from typing import Optional, Any, Tuple
from mlx_lm.sample_utils import make_sampler
from dnet.core.types.messages import TokenResult
from dnet.core.decoding.config import DecodingConfig
from dnet.utils.logger import logger


class GrammarState:
    """Holds xgrammar state for a single generation session."""
    
    def __init__(self, compiled_grammar, tokenizer_info):
        import xgrammar as xgr
        self.compiled_grammar = compiled_grammar
        self.tokenizer_info = tokenizer_info
        self.matcher = xgr.GrammarMatcher(compiled_grammar)
        self.bitmask = None
        self.vocab_size = tokenizer_info.vocab_size
    
    def get_bitmask(self):
        """Get or create the token bitmask."""
        if self.bitmask is None:
            import xgrammar as xgr
            self.bitmask = xgr.allocate_token_bitmask(1, self.vocab_size)
        return self.bitmask


class Sampler:
    """
    Handles the transformation of logits into tokens based on a DecodingConfig.
    Wraps mlx_lm's make_sampler for consistent sampling behavior.
    Supports structured output via grammar-constrained generation using xgrammar.
    """

    def __init__(self):
        """Initialize sampler."""
        pass

    @staticmethod
    def create_grammar_state(json_schema: str, tokenizer, model_vocab_size: Optional[int] = None) -> Optional[GrammarState]:
        """Create a grammar state for JSON schema constrained generation."""
        if not json_schema:
            return None
            
        try:
            import xgrammar as xgr
            
            vocab_size = model_vocab_size or getattr(tokenizer, 'vocab_size', None)
            
            if vocab_size:
                tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
            else:
                tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
            
            grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
            compiled_grammar = grammar_compiler.compile_json_schema(json_schema)
            
            return GrammarState(compiled_grammar, tokenizer_info)
            
        except ImportError as e:
            logger.warning(f"xgrammar not installed: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to create grammar state: {e}")
            return None

    @staticmethod
    def sample(
        logits: mx.array,
        config: DecodingConfig,
        req_logprobs: bool = False,
        req_top_logprobs: int = 0,
        grammar_state: Optional[GrammarState] = None,
    ) -> TokenResult:
        """
        Sample a token from logits using the provided configuration.
        If grammar_state is provided, applies grammar constraints before sampling.
        """
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

        # Apply grammar-constrained logits processing if available
        if grammar_state is not None:
            try:
                import xgrammar as xgr
                import torch
                
                bitmask = grammar_state.get_bitmask()
                grammar_state.matcher.fill_next_token_bitmask(bitmask)
                
                # Convert to float32 first (handles bfloat16 which NumPy doesn't support)
                # Use buffer protocol instead of .tolist() for better performance
                v_np = np.array(v.astype(mx.float32))
                v_torch = torch.from_numpy(v_np).unsqueeze(0)
                xgr.apply_token_bitmask_inplace(v_torch, bitmask.to(v_torch.device))
                v = mx.array(v_torch.squeeze(0).numpy())
                
            except Exception as e:
                logger.warning(f"Failed to apply grammar mask: {e}")

        token_tensor = sampler_fn(v)
        token_id = int(token_tensor.item())
        
        # Update grammar state with accepted token
        if grammar_state is not None:
            try:
                grammar_state.matcher.accept_token(token_id)
            except Exception as e:
                logger.warning(f"Failed to accept token in grammar: {e}")

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
