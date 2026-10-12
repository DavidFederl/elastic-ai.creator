"""Checks of the QParams handling that all integer layers share."""

from collections.abc import Callable
from typing import Any

import pytest
import torch

from elasticai.creator.nn.integer.batchnorm1d.batchnorm1d import BatchNorm1d
from elasticai.creator.nn.integer.conv1d.conv1d import Conv1d
from elasticai.creator.nn.integer.conv1dbn.conv1dbn import Conv1dBN
from elasticai.creator.nn.integer.depthconv1d.depthconv1d import DepthConv1d
from elasticai.creator.nn.integer.fusedffn.fc1relu.linearrelu import (
    LinearReLU as FusedLinearReLU,
)
from elasticai.creator.nn.integer.fusedffn.fc2.linear import Linear as FusedLinear
from elasticai.creator.nn.integer.hardtanh.hardtanh import HardTanh
from elasticai.creator.nn.integer.linearrelu.linearrelu import LinearReLU
from elasticai.creator.nn.integer.pointconv1dbn.pointconv1dbn import PointConv1dBN
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
)
from elasticai.creator.nn.integer.sepconv1dbn.depthconv1d.depthconv1d import (
    DepthConv1d as SepDepthConv1d,
)
from elasticai.creator.nn.integer.sepconv1dbn.pointconv1dbn.pointconv1dbn import (
    PointConv1dBN as SepPointConv1dBN,
)

# The layers have no common base class with their QParams and methods.
Factory = Callable[[], Any]

CONV_SHAPE = (4, 2, 8)
LINEAR_SHAPE = (4, 3)

LAYERS_WITH_BIAS: dict[str, tuple[Factory, tuple[int, ...]]] = {
    "LinearReLU": (
        lambda: LinearReLU(
            name="l", in_features=3, out_features=2, bias=True, quant_bits=8
        ),
        LINEAR_SHAPE,
    ),
    "fusedffn.LinearReLU": (
        lambda: FusedLinearReLU(
            name="l", in_features=3, out_features=2, bias=True, quant_bits=8
        ),
        LINEAR_SHAPE,
    ),
    "fusedffn.Linear": (
        lambda: FusedLinear(
            name="l", in_features=3, out_features=2, bias=True, quant_bits=8
        ),
        LINEAR_SHAPE,
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
        CONV_SHAPE,
    ),
    "PointConv1dBN": (
        lambda: PointConv1dBN(
            name="l", in_channels=2, out_channels=3, seq_len=8, quant_bits=8
        ),
        CONV_SHAPE,
    ),
    "sepconv1dbn.PointConv1dBN": (
        lambda: SepPointConv1dBN(
            name="l", in_channels=2, out_channels=3, seq_len=8, quant_bits=8
        ),
        CONV_SHAPE,
    ),
    "sepconv1dbn.DepthConv1d": (
        lambda: SepDepthConv1d(
            name="l",
            in_channels=2,
            out_channels=2,
            kernel_size=3,
            padding=0,
            seq_len=8,
            quant_bits=8,
        ),
        CONV_SHAPE,
    ),
}

LAYERS_WITH_INPUTS_QPARAMS: dict[str, tuple[Factory, tuple[int, ...]]] = {
    **LAYERS_WITH_BIAS,
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
        CONV_SHAPE,
    ),
    "DepthConv1d": (
        lambda: DepthConv1d(
            name="l",
            in_channels=2,
            out_channels=2,
            kernel_size=3,
            padding=0,
            seq_len=8,
            quant_bits=8,
        ),
        CONV_SHAPE,
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
    "HardTanh": (lambda: HardTanh(name="l", quant_bits=8), LINEAR_SHAPE),
}

BIAS_QPARAMS_BUFFERS = ["scale_factor", "zero_point", "min_quant", "max_quant"]


def quantized_bias(layer: Any) -> torch.Tensor:
    # Layers that fold a BatchNorm1d into their weights store the folded bias.
    if hasattr(layer, "q_fused_bias"):
        return layer.q_fused_bias
    return layer.q_bias


def trained(factory: Factory, shape: tuple[int, ...]) -> tuple[Any, torch.Tensor]:
    torch.manual_seed(0)
    layer = factory()
    inputs = torch.randn(shape)
    layer.train()
    layer(inputs)
    layer.eval()
    return layer, inputs


@pytest.mark.parametrize(
    "factory, shape", LAYERS_WITH_BIAS.values(), ids=LAYERS_WITH_BIAS
)
def test_precompute_keeps_bias_QParams(factory, shape) -> None:
    layer, _ = trained(factory, shape)
    before = {
        name: getattr(layer.bias_QParams, name).clone() for name in BIAS_QPARAMS_BUFFERS
    }

    layer.precompute()

    for name in BIAS_QPARAMS_BUFFERS:
        assert torch.equal(getattr(layer.bias_QParams, name), before[name]), name


@pytest.mark.parametrize(
    "factory, shape", LAYERS_WITH_BIAS.values(), ids=LAYERS_WITH_BIAS
)
def test_precompute_does_not_change_forward(factory, shape) -> None:
    layer, inputs = trained(factory, shape)
    with torch.no_grad():
        before = layer(inputs)
        layer.precompute()
        after = layer(inputs)

    assert torch.equal(before, after)


@pytest.mark.parametrize(
    "factory, shape", LAYERS_WITH_BIAS.values(), ids=LAYERS_WITH_BIAS
)
def test_precompute_twice_gives_same_quantized_bias(factory, shape) -> None:
    layer, _ = trained(factory, shape)
    layer.precompute()
    q_bias = quantized_bias(layer).clone()

    layer.precompute()

    assert torch.equal(quantized_bias(layer), q_bias)


@pytest.mark.parametrize(
    "factory, shape", LAYERS_WITH_BIAS.values(), ids=LAYERS_WITH_BIAS
)
def test_precompute_does_not_change_state_dict_keys(factory, shape) -> None:
    layer, _ = trained(factory, shape)
    keys = set(layer.state_dict())

    layer.precompute()

    # Otherwise a state_dict saved after precompute does not load into a new
    # layer.
    assert set(layer.state_dict()) == keys


@pytest.mark.parametrize(
    "factory, shape",
    LAYERS_WITH_INPUTS_QPARAMS.values(),
    ids=LAYERS_WITH_INPUTS_QPARAMS,
)
def test_forward_stores_a_copy_of_given_inputs_QParams(factory, shape) -> None:
    torch.manual_seed(0)
    layer = factory()
    inputs = torch.randn(shape)
    given_inputs_QParams = AsymmetricSignedQParams(
        quant_bits=8, observer=GlobalMinMaxObserver()
    )
    given_inputs_QParams.update_quant_params(inputs)
    layer.train()

    layer(inputs, given_inputs_QParams)

    # A shared module would be registered in both layers and saved twice.
    assert layer.inputs_QParams is not given_inputs_QParams
    assert torch.equal(
        layer.inputs_QParams.scale_factor, given_inputs_QParams.scale_factor
    )
    assert torch.equal(layer.inputs_QParams.zero_point, given_inputs_QParams.zero_point)
