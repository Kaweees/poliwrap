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
from poliwrap import PytorchPolicyWrapper

policy = PytorchPolicyWrapper("policy.pt")
actions = policy({"actor_obs": obs})  # obs: dict[str, torch.Tensor]
```

## Extending

All wrappers subclass `PolicyWrapper`. To support a new backend, implement
`_load_model` and `__call__`:

```python
from poliwrap import PolicyWrapper


class MyPolicyWrapper(PolicyWrapper):
    def _load_model(self) -> None:
        self.model = ...  # load from self.model_path

    def __call__(self, observations):
        return self.model(observations)
```

`PolicyWrapper` also has overridable `preprocess_observations`, `postprocess_actions`,
and `reset` hooks for input/output transforms and stateful (e.g. RNN) policies.

## License

MIT
