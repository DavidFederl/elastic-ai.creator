import pytest
import torch

from elasticai.creator.nn.integer.conv1dbn.conv1dbn import Conv1dBN


@pytest.fixture
def layer() -> Conv1dBN:
    torch.manual_seed(0)
    return Conv1dBN(
        name="conv1dbn",
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        seq_len=8,
        padding=0,
        quant_bits=8,
    )


@pytest.fixture
def inputs() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(4, 2, 8)


def test_running_statistics_are_not_attached_to_the_graph(layer, inputs) -> None:
    layer.train()
    for _ in range(3):
        layer(inputs.clone().requires_grad_()).sum().backward()

    assert layer.bn1d.running_mean.grad_fn is None
    assert layer.bn1d.running_var.grad_fn is None


def test_running_statistics_stay_buffers(layer, inputs) -> None:
    running_mean = layer.bn1d.running_mean

    layer.train()
    layer(inputs)

    assert layer.bn1d.running_mean is running_mean
