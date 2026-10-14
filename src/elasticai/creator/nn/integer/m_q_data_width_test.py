"""Checks that the designs of integer layers can represent M_Q as signed.

The templates convert M_Q with `to_signed(M_Q, M_Q_DATA_WIDTH)`.
"""

from collections.abc import Callable
from typing import Any

import pytest
import torch

from elasticai.creator.nn.integer.batchnorm1d.batchnorm1d import BatchNorm1d
from elasticai.creator.nn.integer.conv1d.conv1d import Conv1d
from elasticai.creator.nn.integer.depthconv1d.depthconv1d import DepthConv1d
from elasticai.creator.nn.integer.linearrelu.linearrelu import LinearReLU
from elasticai.creator.nn.integer.pointconv1dbn.pointconv1dbn import PointConv1dBN

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
    "LinearReLU": (
        lambda: LinearReLU(
            name="l", in_features=3, out_features=2, bias=True, quant_bits=8
        ),
        (4, 3),
    ),
    "PointConv1dBN": (
        lambda: PointConv1dBN(
            name="l", in_channels=2, out_channels=3, seq_len=8, quant_bits=8
        ),
        (4, 2, 8),
    ),
}


@pytest.mark.parametrize("m_q", [1, 2, 3, 4, 255, 256])
@pytest.mark.parametrize("factory, shape", LAYERS.values(), ids=LAYERS)
def test_m_q_fits_m_q_data_width_as_signed(factory, shape, m_q) -> None:
    torch.manual_seed(0)
    layer = factory()
    layer.train()
    layer(torch.randn(shape), None)
    layer.eval()
    layer.precompute()
    layer.scale_factor_m_q = torch.tensor(m_q)

    width = layer.create_design("l")._m_q_data_width

    assert m_q < 2 ** (width - 1)
