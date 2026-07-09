"""JAX policy wrapper for inference."""

from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import v1 as ocp

from poliwrap.policy import FilePolicyWrapper


class JaxPolicyWrapper(FilePolicyWrapper[dict[str, np.ndarray], np.ndarray]):
    """Wrapper for JAX policies."""

    def __init__(self, model_path: Path, apply_fn: Callable[..., jax.Array]) -> None:
        """Initialize the JAX policy wrapper."""
        super().__init__(model_path)
        self.apply_fn = apply_fn
        self._load_model()

    def _load_model(self) -> None:
        self.model = ocp.load(str(self.model_path.resolve()))
        self._jitted_apply = jax.jit(self.apply_fn)

    def __call__(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        # Preprocess observations
        obs = self.preprocess_observations(obs)
        # Run inference
        actions = self._jitted_apply(self.model, **{k: jnp.asarray(v) for k, v in obs.items()})
        # Postprocess actions
        return self.postprocess_actions(np.asarray(actions))
