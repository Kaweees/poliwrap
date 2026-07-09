"""Policy wrapper for inference."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class PolicyWrapper(ABC, Generic[ObservationT, ActionT]):
    """Base wrapper for policy inference."""

    def preprocess_observations(self, observations: ObservationT) -> ObservationT:
        """
        Preprocess observations before inference.

        Args:
            observations: Raw observations from the environment

        Returns:
            Preprocessed observations ready for model input.
        """
        return observations

    @abstractmethod
    def __call__(self, observations: ObservationT) -> ActionT:
        """
        Perform inference on the given observations.

        Args:
            observations: Observations accepted by the policy implementation.
                         Keys depend on the specific policy implementation.
                         Example: {"actor_obs": np.ndarray, "estimator_obs": np.ndarray}

        Returns:
            Policy actions in the implementation's native representation.
        """

    def postprocess_actions(self, actions: ActionT) -> ActionT:
        """
        Postprocess actions after inference (e.g., clipping, scaling).

        Args:
            actions: Raw actions from the model

        Returns:
            Postprocessed actions ready for execution.
        """
        return actions

    def reset(self) -> None:
        """
        Reset the policy state (e.g., hidden states for RNN policies).

        Override this method if your policy maintains internal state.
        """


class FilePolicyWrapper(PolicyWrapper[ObservationT, ActionT]):
    """Base wrapper for policies loaded from a local model file."""

    def __init__(self, model_path: Path) -> None:
        """
        Initialize the file policy wrapper.

        Args:
            model_path: Path to the trained model file (e.g., ONNX, PyTorch, etc.)
        """
        self.model_path: Path = model_path

    @abstractmethod
    def _load_model(self) -> None:
        """
        Load the model from ``model_path``.

        This method should initialize self.model with the loaded model.
        Implementation depends on the model format (ONNX, PyTorch, TensorFlow, etc.)
        """
