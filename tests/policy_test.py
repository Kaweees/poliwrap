from pathlib import Path
from unittest import TestCase

import numpy as np

from poliwrap import FilePolicyWrapper, PolicyWrapper


class RemotePolicy(PolicyWrapper[dict[str, np.ndarray], np.ndarray]):
    def __init__(self) -> None:
        self._load_model()

    def _load_model(self) -> None:
        self.connected = True

    def __call__(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        observations = self.preprocess_observations(observations)
        return self.postprocess_actions(observations["actor_obs"])


class FileArrayPolicy(FilePolicyWrapper[dict[str, np.ndarray], np.ndarray]):
    def __init__(self, model_path: str | Path) -> None:
        super().__init__(model_path)
        self._load_model()

    def _load_model(self) -> None:
        self.loaded_from = self.model_path

    def __call__(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        return observations["actor_obs"]


class PolicyWrapperTest(TestCase):
    def test_supports_non_file_backed_policy(self) -> None:
        policy = RemotePolicy()
        observations = {"actor_obs": np.array([[1.0, 2.0]])}

        self.assertTrue(policy.connected)
        np.testing.assert_array_equal(policy(observations), observations["actor_obs"])

    def test_file_wrapper_normalizes_model_path(self) -> None:
        policy = FileArrayPolicy("models/policy.bin")

        self.assertEqual(policy.model_path, Path("models/policy.bin"))
        self.assertEqual(policy.loaded_from, policy.model_path)

    def test_default_hooks_preserve_native_types(self) -> None:
        policy = RemotePolicy()
        observations = {"actor_obs": np.array([1.0])}
        actions = np.array([2.0])

        self.assertIs(policy.preprocess_observations(observations), observations)
        self.assertIs(policy.postprocess_actions(actions), actions)
