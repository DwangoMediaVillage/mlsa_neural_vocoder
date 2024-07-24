from typing import Literal

from torch import nn, Tensor, LongTensor, device as TorchDevice

ActivationType = Literal["hardtanh", "tanh", "relu", "selu", "swish"]


def get_activation(act: ActivationType) -> nn.Module:
    """Return activation function."""
    # Lazy load to avoid unused import
    from models.conformer.swish import Swish

    activation_funcs = {
        "hardtanh": nn.Hardtanh,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "swish": Swish,
    }

    return activation_funcs[act]()


def to_device(data: dict, device: TorchDevice):
    """Move data to device."""
    for k, v in data.items():
        if isinstance(v, Tensor) or isinstance(v, LongTensor):
            data[k] = v.to(device)

    return data
