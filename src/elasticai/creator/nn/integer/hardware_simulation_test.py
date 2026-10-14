"""GHDL simulations that compare the hardware of integer layers to int_forward.

Each layer is trained on scaled inputs and then run on unscaled inputs. With
a scale below 1, the trained output range is too small for some outputs, so
the hardware must saturate them like int_forward does. The layers run on two
samples, so that their state must be reset between inferences.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from elasticai.creator.file_generation.on_disk_path import OnDiskPath
from elasticai.creator.nn.integer.addition.addition import Addition
from elasticai.creator.nn.integer.avgpooling1d.avgpooling1d import AVGPooling1d
from elasticai.creator.nn.integer.avgpooling1dflatten.avgpooling1dflatten import (
    AVGPooling1dFlatten,
)
from elasticai.creator.nn.integer.batchnorm1d.batchnorm1d import BatchNorm1d
from elasticai.creator.nn.integer.concatenate.concatenate import Concatenate
from elasticai.creator.nn.integer.conv1dbn.conv1dbn import Conv1dBN
from elasticai.creator.nn.integer.fusedffn.fc1relu.linearrelu import (
    LinearReLU as FusedLinearReLU,
)
from elasticai.creator.nn.integer.fusedffn.fc2.linear import Linear as FusedLinear
from elasticai.creator.nn.integer.hadamardproduct.hadamardproduct import (
    HadamardProduct,
)
from elasticai.creator.nn.integer.hardsigmoid.hardsigmoid import HardSigmoid
from elasticai.creator.nn.integer.linearrelu.linearrelu import LinearReLU
from elasticai.creator.nn.integer.matrixmulti.matrixmulti import MatrixMulti
from elasticai.creator.nn.integer.MPQ.addition.addition import Addition as MPQAddition
from elasticai.creator.nn.integer.MPQ.avgpooling1dflatten.avgpooling1dflatten import (
    AVGPooling1dFlatten as MPQAVGPooling1dFlatten,
)
from elasticai.creator.nn.integer.MPQ.batchnorm1d.batchnorm1d import (
    BatchNorm1d as MPQBatchNorm1d,
)
from elasticai.creator.nn.integer.MPQ.linear.linear import Linear as MPQLinear
from elasticai.creator.nn.integer.MPQ.matrixmulti.matrixmulti import (
    MatrixMulti as MPQMatrixMulti,
)
from elasticai.creator.nn.integer.pointconv1dbn.pointconv1dbn import PointConv1dBN
from elasticai.creator.nn.integer.sepconv1dbn.sepconv1dbn import SepConv1dBN
from elasticai.creator.testing.ghdl_simulation import GHDLSimulator, SimulationError

pytestmark = pytest.mark.simulation

NAME = "layer_0"

# Takes the directory for the quantized data and returns the layer.
Factory = Callable[[str], Any]


def with_8_bits(layer: Any, elements: list[str]) -> Any:
    layer.set_quant_bits_from_config({f"{NAME}.{e}": 8 for e in elements})
    return layer


def conv1dbn(padding: int | str) -> Factory:
    return lambda data_dir: Conv1dBN(
        name=NAME,
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        seq_len=8,
        padding=padding,
        quant_bits=8,
        quant_data_dir=data_dir,
    )


def sepconv1dbn(kernel_size: int, padding: int | str) -> Factory:
    return lambda data_dir: SepConv1dBN(
        name=NAME,
        in_channels=2,
        out_channels=3,
        kernel_size=kernel_size,
        padding=padding,
        seq_len=8,
        quant_bits=8,
        quant_data_dir=data_dir,
    )


# Dimensions of the attention scores `Q @ K^T` (score) and of
# `softmax_out @ V` (att) with a window of 3 and 2 heads.
SCORE = dict(
    x_1_dim_a=3,
    x_1_dim_b=2,
    x_1_dim_c=4,
    x_2_dim_a=3,
    x_2_dim_b=2,
    x_2_dim_c=4,
    y_dim_a=2,
    y_dim_b=3,
    y_dim_c=3,
    operation_mode="score",
)
ATT = dict(
    x_1_dim_a=2,
    x_1_dim_b=3,
    x_1_dim_c=3,
    x_2_dim_a=3,
    x_2_dim_b=2,
    x_2_dim_c=4,
    y_dim_a=3,
    y_dim_b=2,
    y_dim_c=4,
    operation_mode="att",
)

ONE_INPUT_LAYERS: dict[str, tuple[Factory, tuple[int, ...]]] = {
    "Conv1dBN-padding_0": (conv1dbn(0), (2, 2, 8)),
    "Conv1dBN-padding_1": (conv1dbn(1), (2, 2, 8)),
    "Conv1dBN-padding_same": (conv1dbn("same"), (2, 2, 8)),
    "SepConv1dBN-padding_0": (sepconv1dbn(3, 0), (2, 2, 8)),
    "SepConv1dBN-padding_1": (sepconv1dbn(3, 1), (2, 2, 8)),
    "SepConv1dBN-padding_same": (sepconv1dbn(3, "same"), (2, 2, 8)),
    "SepConv1dBN-kernel_5-padding_1": (sepconv1dbn(5, 1), (2, 2, 8)),
    "PointConv1dBN": (
        lambda data_dir: PointConv1dBN(
            name=NAME,
            in_channels=2,
            out_channels=3,
            seq_len=8,
            quant_bits=8,
            quant_data_dir=data_dir,
        ),
        (2, 2, 8),
    ),
    "PointConv1dBN-in_channels_1": (
        lambda data_dir: PointConv1dBN(
            name=NAME,
            in_channels=1,
            out_channels=3,
            seq_len=8,
            quant_bits=8,
            quant_data_dir=data_dir,
        ),
        (2, 1, 8),
    ),
    "PointConv1dBN-out_channels_1": (
        lambda data_dir: PointConv1dBN(
            name=NAME,
            in_channels=2,
            out_channels=1,
            seq_len=8,
            quant_bits=8,
            quant_data_dir=data_dir,
        ),
        (2, 2, 8),
    ),
    "LinearReLU-1d": (
        lambda data_dir: LinearReLU(
            name=NAME,
            in_features=3,
            out_features=2,
            bias=True,
            quant_bits=8,
            quant_data_dir=data_dir,
        ),
        (2, 3),
    ),
    "LinearReLU-2d": (
        lambda data_dir: LinearReLU(
            name=NAME,
            in_features=3,
            out_features=2,
            bias=True,
            num_dimensions=4,
            quant_bits=8,
            quant_data_dir=data_dir,
        ),
        (2, 4, 3),
    ),
    "LinearReLU-2d-in_features_1": (
        lambda data_dir: LinearReLU(
            name=NAME,
            in_features=1,
            out_features=2,
            bias=True,
            num_dimensions=4,
            quant_bits=8,
            quant_data_dir=data_dir,
        ),
        (2, 4, 1),
    ),
    "fusedffn.LinearReLU": (
        lambda data_dir: FusedLinearReLU(
            name=NAME,
            in_features=3,
            out_features=2,
            bias=True,
            num_dimensions=4,
            quant_bits=8,
            quant_data_dir=data_dir,
        ),
        (2, 4, 3),
    ),
    "fusedffn.Linear": (
        lambda data_dir: FusedLinear(
            name=NAME,
            in_features=3,
            out_features=2,
            bias=True,
            num_dimensions=4,
            quant_bits=8,
            quant_data_dir=data_dir,
        ),
        (2, 4, 3),
    ),
    "BatchNorm1d": (
        lambda data_dir: BatchNorm1d(
            name=NAME,
            norm_dim=4,
            in_features=4,
            out_features=4,
            num_dimensions=5,
            quant_bits=8,
            quant_data_dir=data_dir,
        ),
        (2, 5, 4),
    ),
    "AVGPooling1d": (
        lambda data_dir: AVGPooling1d(
            name=NAME,
            in_features=6,
            out_features=1,
            in_num_dimensions=4,
            out_num_dimensions=4,
            quant_bits=8,
            quant_data_dir=data_dir,
        ),
        (2, 6, 4),
    ),
    "AVGPooling1dFlatten": (
        lambda data_dir: AVGPooling1dFlatten(
            name=NAME,
            in_features=6,
            out_features=1,
            in_num_dimensions=4,
            out_num_dimensions=4,
            quant_bits=8,
            quant_data_dir=data_dir,
        ),
        (2, 6, 4),
    ),
    "MPQ.Linear": (
        lambda data_dir: with_8_bits(
            MPQLinear(
                name=NAME,
                in_features=3,
                out_features=2,
                num_dimensions=4,
                quant_data_dir=data_dir,
            ),
            ["inputs", "weights", "outputs"],
        ),
        (2, 4, 3),
    ),
    "MPQ.Linear-in_features_1": (
        lambda data_dir: with_8_bits(
            MPQLinear(
                name=NAME,
                in_features=1,
                out_features=2,
                num_dimensions=4,
                quant_data_dir=data_dir,
            ),
            ["inputs", "weights", "outputs"],
        ),
        (2, 4, 1),
    ),
    "MPQ.BatchNorm1d": (
        lambda data_dir: with_8_bits(
            MPQBatchNorm1d(
                name=NAME,
                norm_dim=4,
                in_features=4,
                out_features=4,
                num_dimensions=5,
                eps=1e-5,
                momentum=0.1,
                track_running_stats=True,
                quant_data_dir=data_dir,
            ),
            ["inputs", "weights", "bias", "outputs"],
        ),
        (2, 5, 4),
    ),
    "MPQ.AVGPooling1dFlatten": (
        lambda data_dir: with_8_bits(
            MPQAVGPooling1dFlatten(
                name=NAME,
                in_features=6,
                out_features=1,
                in_num_dimensions=4,
                out_num_dimensions=4,
                quant_data_dir=data_dir,
            ),
            ["inputs", "outputs"],
        ),
        (2, 6, 4),
    ),
}

TWO_INPUT_LAYERS: dict[str, tuple[Factory, tuple[int, ...], tuple[int, ...]]] = {
    "Addition": (
        lambda data_dir: Addition(
            name=NAME,
            num_features=4,
            num_dimensions=3,
            quant_bits=8,
            quant_data_dir=data_dir,
        ),
        (2, 3, 4),
        (2, 3, 4),
    ),
    "Addition-pipeline": (
        lambda data_dir: Addition(
            name=NAME,
            num_features=4,
            num_dimensions=3,
            quant_bits=8,
            quant_data_dir=data_dir,
            use_pipeline_template=True,
        ),
        (2, 3, 4),
        (2, 3, 4),
    ),
    "HadamardProduct": (
        lambda data_dir: HadamardProduct(
            name=NAME,
            num_features=4,
            num_dimensions=3,
            quant_bits=8,
            quant_data_dir=data_dir,
        ),
        (2, 3, 4),
        (2, 3, 4),
    ),
    "HadamardProduct-pipeline": (
        lambda data_dir: HadamardProduct(
            name=NAME,
            num_features=4,
            num_dimensions=3,
            quant_bits=8,
            quant_data_dir=data_dir,
            use_pipeline_template=True,
        ),
        (2, 3, 4),
        (2, 3, 4),
    ),
    "Concatenate": (
        lambda data_dir: Concatenate(
            name=NAME,
            inputs_size=3,
            hidden_size=2,
            num_dimensions=1,
            quant_bits=8,
            quant_data_dir=data_dir,
        ),
        (2, 3),
        (2, 2),
    ),
    "MatrixMulti-score": (
        lambda data_dir: MatrixMulti(
            name=NAME, quant_bits=8, quant_data_dir=data_dir, **SCORE
        ),
        (2, 3, 2, 4),
        (2, 3, 2, 4),
    ),
    "MatrixMulti-att": (
        lambda data_dir: MatrixMulti(
            name=NAME, quant_bits=8, quant_data_dir=data_dir, **ATT
        ),
        (2, 2, 3, 3),
        (2, 3, 2, 4),
    ),
    "MPQ.MatrixMulti-score": (
        lambda data_dir: with_8_bits(
            MPQMatrixMulti(name=NAME, quant_data_dir=data_dir, **SCORE),
            ["inputs1", "inputs2", "outputs"],
        ),
        (2, 3, 2, 4),
        (2, 3, 2, 4),
    ),
    "MPQ.MatrixMulti-att": (
        lambda data_dir: with_8_bits(
            MPQMatrixMulti(name=NAME, quant_data_dir=data_dir, **ATT),
            ["inputs1", "inputs2", "outputs"],
        ),
        (2, 2, 3, 3),
        (2, 3, 2, 4),
    ),
    "MPQ.Addition": (
        lambda data_dir: with_8_bits(
            MPQAddition(
                name=NAME, num_features=4, num_dimensions=3, quant_data_dir=data_dir
            ),
            ["inputs1", "inputs2", "outputs"],
        ),
        (2, 3, 4),
        (2, 3, 4),
    ),
}

TRAIN_SCALES = [
    pytest.param(1.0, id="in_range"),
    pytest.param(0.3, id="saturating"),
]


def simulate_and_compare(layer: Any, build_dir: Path, data_dir: Path) -> None:
    layer.create_design(NAME).save_to(OnDiskPath("build", parent=str(build_dir.parent)))
    simulator = GHDLSimulator(workdir=str(build_dir), top_design_name=f"{NAME}_tb")
    simulator.initialize()
    # The testbench stops the simulation with an assertion failure once all
    # inputs are processed, so GHDL always exits with an error.
    with pytest.raises(SimulationError, match="Simulation done"):
        simulator.run()

    expected = (data_dir / f"{NAME}_q_y.txt").read_text().split()
    actual = (data_dir / f"{NAME}_q_out.txt").read_text().split()
    assert actual == expected


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "build" / "data"
    data_dir.mkdir(parents=True)
    return data_dir


@pytest.mark.parametrize("train_scale", TRAIN_SCALES)
@pytest.mark.parametrize(
    "factory, shape", ONE_INPUT_LAYERS.values(), ids=ONE_INPUT_LAYERS
)
def test_hardware_of_one_input_layer_matches_int_forward(
    data_dir: Path, factory: Factory, shape: tuple[int, ...], train_scale: float
) -> None:
    torch.manual_seed(0)
    layer = factory(str(data_dir))
    inputs = torch.randn(shape)
    layer.train()
    for _ in range(3):
        layer(inputs * train_scale, None)
    layer.eval()
    layer.precompute()
    # Writes the inputs and expected outputs that the testbench reads.
    layer.int_forward(layer.inputs_QParams.quantize(inputs))

    simulate_and_compare(layer, data_dir.parent, data_dir)


@pytest.mark.parametrize("train_scale", TRAIN_SCALES)
@pytest.mark.parametrize(
    "factory, shape1, shape2", TWO_INPUT_LAYERS.values(), ids=TWO_INPUT_LAYERS
)
def test_hardware_of_two_input_layer_matches_int_forward(
    data_dir: Path,
    factory: Factory,
    shape1: tuple[int, ...],
    shape2: tuple[int, ...],
    train_scale: float,
) -> None:
    torch.manual_seed(0)
    layer = factory(str(data_dir))
    inputs1, inputs2 = torch.randn(shape1), torch.randn(shape2)
    layer.train()
    for _ in range(3):
        layer(inputs1 * train_scale, inputs2 * train_scale, None, None)
    layer.eval()
    layer.precompute()
    layer.int_forward(
        layer.inputs1_QParams.quantize(inputs1),
        layer.inputs2_QParams.quantize(inputs2),
    )

    simulate_and_compare(layer, data_dir.parent, data_dir)


def test_hardware_of_hardsigmoid_matches_int_forward(data_dir: Path) -> None:
    layer = HardSigmoid(name=NAME, quant_bits=8, quant_data_dir=str(data_dir))
    # Inputs from -4 to 4 hit every quantized value between -3 and 3, also the
    # ones next to the thresholds.
    inputs = torch.linspace(-4, 4, 512).reshape(2, 256)
    layer.train()
    layer(inputs, None)
    layer.eval()
    layer.precompute()
    layer.int_forward(layer.inputs_QParams.quantize(inputs))

    simulate_and_compare(layer, data_dir.parent, data_dir)
