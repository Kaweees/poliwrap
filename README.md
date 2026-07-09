# PoliWrap

A lightweight, framework-agnostic wrapper for reinforcement learning policy inference.

## Installation

```bash
pip install poliwrap            # core only (numpy)
pip install poliwrap[onnx]      # + ONNX Runtime backend (CPU)
pip install poliwrap[onnx-gpu]  # + ONNX Runtime backend (CUDA)
pip install poliwrap[torch]     # + PyTorch backend
pip install poliwrap[all]       # all backends (CPU ONNX)
```

Each wrapper is importable only when its backend is installed.

## Usage

### ONNX

```python
from poliwrap import ONNXPolicyWrapper

policy = ONNXPolicyWrapper("policy.onnx")
actions = policy({"actor_obs": obs})  # obs: dict[str, np.ndarray]
```

### PyTorch

```python
from poliwrap import TorchPolicyWrapper

policy = TorchPolicyWrapper("policy.pt")
actions = policy({"actor_obs": obs})  # obs: dict[str, torch.Tensor]
```

## Extending

`PolicyWrapper` is generic over observation and action types. File-backed
implementations can inherit from `FilePolicyWrapper`, which provides a
`model_path`:

```python
from pathlib import Path

import numpy as np

from poliwrap import FilePolicyWrapper


class MyPolicyWrapper(FilePolicyWrapper[dict[str, np.ndarray], np.ndarray]):
    def __init__(self, model_path: str | Path) -> None:
        super().__init__(model_path)
        self._load_model()

    def _load_model(self) -> None:
        self.model = ...  # load from self.model_path

    def __call__(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        return self.model(observations)
```

`PolicyWrapper` has no path requirement, so policies backed by other sources can
subclass it directly. Both base classes provide `preprocess_observations`,
`postprocess_actions`, and `reset` hooks.

## License

MIT
