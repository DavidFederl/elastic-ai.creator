"""Checks that integer layers reject calls in the wrong mode or order.

The checks raise RuntimeError instead of using assert, which is removed when
Python runs with -O.
"""

from collections.abc import Callable
from typing import Any

import pytest
import torch

from elasticai.creator.nn.integer.batchnorm1d.batchnorm1d import BatchNorm1d
from elasticai.creator.nn.integer.conv1d.conv1d import Conv1d
from elasticai.creator.nn.integer.conv1dbn.conv1dbn import Conv1dBN
from elasticai.creator.nn.integer.hardtanh.hardtanh import HardTanh
from elasticai.creator.nn.integer.layernorm1d.layernorm1d import LayerNorm1d
from elasticai.creator.nn.integer.linearrelu.linearrelu import LinearReLU
from elasticai.creator.nn.integer.relu.relu import ReLU

# The layers have no common base class with their QParams and methods.
Factory = Callable[[], Any]

LAYERS: dict[str, tuple[Factory, tuple[int, ...]]] = {
    "Conv1d": (
        lambda: Conv1d(
            name="l",
            in_channels=2,
            out_channels=3,
            kernel_size=3,
            padding=0,
            seq_len=8,
            quant_bits=8,
        ),
        (4, 2, 8),
    ),
    "Conv1dBN": (
        lambda: Conv1dBN(
            name="l",
            in_channels=2,
            out_channels=3,
            kernel_size=3,
            seq_len=8,
            padding=0,
            quant_bits=8,
        ),
        (4, 2, 8),
    ),
    "BatchNorm1d": (
        lambda: BatchNorm1d(
            name="l",
            norm_dim=4,
            in_features=4,
            out_features=4,
            num_dimensions=5,
            quant_bits=8,
        ),
        (3, 5, 4),
    ),
    "LayerNorm1d": (
        lambda: LayerNorm1d(
            name="l",
            norm_dim=8,
            num_dimensions=4,
            in_features=8,
            out_features=8,
            quant_bits=8,
        ),
        (4, 8),
    ),
    "LinearReLU": (
        lambda: LinearReLU(
            name="l", in_features=3, out_features=2, bias=True, quant_bits=8
        ),
        (4, 3),
    ),
    "HardTanh": (lambda: HardTanh(name="l", quant_bits=8), (4, 3)),
    "ReLU": (lambda: ReLU(name="l", quant_bits=8), (4, 3)),
}

LAYERS_WITH_PRECOMPUTE_CHECK = {
    name: LAYERS[name]
    for name in ["Conv1d", "Conv1dBN", "BatchNorm1d", "LayerNorm1d", "LinearReLU"]
}


def trained(factory: Factory, shape: tuple[int, ...]) -> tuple[Any, torch.Tensor]:
    torch.manual_seed(0)
    layer = factory()
    inputs = torch.randn(shape)
    layer.train()
    layer(inputs, None)
    layer.eval()
    return layer, inputs


@pytest.mark.parametrize(
    "factory, shape",
    LAYERS_WITH_PRECOMPUTE_CHECK.values(),
    ids=LAYERS_WITH_PRECOMPUTE_CHECK,
)
def test_precompute_in_training_mode_raises(factory, shape) -> None:
    layer, _ = trained(factory, shape)
    layer.train()

    with pytest.raises(RuntimeError, match="eval mode"):
        layer.precompute()


@pytest.mark.parametrize("factory, shape", LAYERS.values(), ids=LAYERS)
def test_int_forward_in_training_mode_raises(factory, shape) -> None:
    layer, inputs = trained(factory, shape)
    if hasattr(layer, "precompute"):
        layer.precompute()
    q_inputs = layer.inputs_QParams.quantize(inputs)
    layer.train()

    with pytest.raises(RuntimeError, match="eval mode"):
        layer.int_forward(q_inputs)


@pytest.mark.parametrize(
    "factory, shape",
    LAYERS_WITH_PRECOMPUTE_CHECK.values(),
    ids=LAYERS_WITH_PRECOMPUTE_CHECK,
)
def test_int_forward_without_precompute_raises(factory, shape) -> None:
    layer, inputs = trained(factory, shape)

    with pytest.raises(RuntimeError, match="precompute"):
        layer.int_forward(layer.inputs_QParams.quantize(inputs))
