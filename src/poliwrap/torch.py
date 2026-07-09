"""Pytorch policy wrapper for inference."""

from pathlib import Path

import torch

from poliwrap.policy import FilePolicyWrapper


class TorchPolicyWrapper(FilePolicyWrapper[dict[str, torch.Tensor], torch.Tensor]):
    """Wrapper for Pytorch policies."""

    def __init__(self, model_path: Path) -> None:
        """Initialize the Pytorch policy wrapper."""
        super().__init__(model_path)
        self._load_model()

    def _load_model(self) -> None:
        self.device = torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")
        self.model = torch.load(self.model_path, map_location=self.device)
        self.model.eval()
        self.model = torch.compile(self.model)

    def __call__(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # Preprocess observations
        obs = self.preprocess_observations(obs)
        obs = {k: v.to(self.device) for k, v in obs.items()}
        # Run inference
        with torch.no_grad():
            actions = self.model(**obs)
        # Postprocess actions
        return self.postprocess_actions(actions)
