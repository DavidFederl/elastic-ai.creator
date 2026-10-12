import pytest
import torch

from elasticai.creator.nn.integer.MPQ.batchnorm1d.batchnorm1d import BatchNorm1d


def create_batchnorm(affine: bool = True) -> BatchNorm1d:
    return BatchNorm1d(
        name="bn_0",
        norm_dim=3,
        num_dimensions=4,
        in_features=3,
        out_features=3,
        eps=1e-5,
        momentum=0.1,
        track_running_stats=True,
        affine=affine,
        device="cpu",
    )


def test_affine_false_is_not_implemented():
    with pytest.raises(NotImplementedError):
        create_batchnorm(affine=False)


def test_is_affine_by_default():
    assert create_batchnorm().affine


def test_bias_quant_bits_are_derived_from_inputs_and_weights():
    layer = create_batchnorm()

    layer.set_quant_bits_from_config(
        {"bn_0.inputs": 6, "bn_0.weights": 4, "bn_0.bias": 3, "bn_0.outputs": 8}
    )

    assert layer.bias_QParams.quant_bits == (6 + 1) + (4 + 1)


def test_missing_quant_bits_in_config_raises():
    layer = create_batchnorm()

    with pytest.raises(ValueError, match="bn_0.weights"):
        layer.set_quant_bits_from_config({"bn_0.inputs": 8, "bn_0.outputs": 8})


def test_running_statistics_are_not_attached_to_the_graph():
    layer = create_batchnorm()
    layer.set_quant_bits_from_config(
        {"bn_0.inputs": 8, "bn_0.weights": 8, "bn_0.bias": 8, "bn_0.outputs": 8}
    )
    running_mean = layer.running_mean
    inputs = torch.randn(2, 4, 3)

    layer.train()
    for _ in range(3):
        layer(inputs.clone().requires_grad_(), None).sum().backward()

    assert layer.running_mean is running_mean
    assert layer.running_mean.grad_fn is None
    assert layer.running_var.grad_fn is None


def test_precompute_keeps_bias_QParams():
    layer = create_batchnorm()
    layer.set_quant_bits_from_config(
        {"bn_0.inputs": 8, "bn_0.weights": 8, "bn_0.bias": 8, "bn_0.outputs": 8}
    )
    torch.manual_seed(0)
    inputs = torch.randn(2, 4, 3)
    layer.train()
    layer(inputs, None)
    layer.eval()
    quant_bits = layer.bias_QParams.quant_bits
    scale_factor = layer.bias_QParams.scale_factor.clone()

    layer.precompute()
    q_bias = layer.q_bias.clone()
    layer.precompute()

    assert layer.bias_QParams.quant_bits == quant_bits
    assert torch.equal(layer.bias_QParams.scale_factor, scale_factor)
    assert torch.equal(layer.q_bias, q_bias)
    assert layer.create_design("bn_0")._b_data_width == (8 + 1) + (8 + 1)
