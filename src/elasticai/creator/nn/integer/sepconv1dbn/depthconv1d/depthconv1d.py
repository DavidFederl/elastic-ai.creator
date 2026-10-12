import copy

import torch
from torch import nn
from torch.nn import functional as F

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.math_operations import (
    get_padded_q_inputs,
    get_padding_len,
)
from elasticai.creator.nn.integer.math_operations.math_operations import MathOperations
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
    SimQuant,
    SymmetricSignedQParams,
    scaling_M,
    simulate_bitshifting,
)
from elasticai.creator.nn.integer.quant_utils.QParams import QParams
from elasticai.creator.nn.integer.sepconv1dbn.depthconv1d.design import (
    DepthConv1d as DepthConv1dDesign,
)
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class DepthConv1d(DesignCreatorModule, nn.Conv1d):
    def __init__(
        self,
        *,
        name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int | str,
        seq_len: int,
        quant_bits: int,
        quant_data_dir: str | None = None,
        device: torch.device | str = torch.device("cpu"),
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=True,
        )
        self.seq_len = seq_len
        self.padding_len = get_padding_len(self.padding, self.kernel_size)

        self.name = name
        self.quant_bits = quant_bits

        self.quant_data_dir = quant_data_dir

        self.weight_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.bias_QParams = SymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.inputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        self.outputs_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)

        self.math_ops = MathOperations()
        self.precomputed = False

    def create_design(self, name: str) -> DepthConv1dDesign:
        return DepthConv1dDesign(
            name=name,
            data_width=self.quant_bits,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            seq_len=self.seq_len,
            kernel_size=self.kernel_size[0],
            padding=self.padding,
            padding_len=self.padding_len,
            weights=self.q_weights.tolist(),
            bias=self.q_bias.tolist(),
            m_q=self.scale_factor_m_q.item(),
            m_q_shift=self.scale_factor_m_q_shift.item(),
            z_x=self.inputs_QParams.zero_point.item(),
            z_w=self.weight_QParams.zero_point.item(),
            z_b=self.q_bias_zero_point,
            z_y=self.outputs_QParams.zero_point.item(),
            work_library_name="work",
            resource_option="auto",
        )

    def _get_quantized_weights(self) -> torch.Tensor:
        q_weights = self.weight_QParams.quantize(self.weight).to("cpu")

        if not self.weight_QParams.is_symmetric:
            q_weights = self.math_ops.intsub(
                q_weights,
                self.weight_QParams.zero_point,
                self.weight_QParams.quant_bits + 1,
            )
        return q_weights

    def _get_quantized_bias(self) -> tuple[torch.Tensor, QParams]:
        q_bias_QParams = copy.deepcopy(self.bias_QParams)
        new_bias_scale_factor = (
            self.inputs_QParams.scale_factor * self.weight_QParams.scale_factor
        )

        q_bias_QParams.set_scale_factor(new_bias_scale_factor)
        q_bias_QParams.set_zero_point(torch.zeros((1), dtype=torch.int32))
        q_bias_QParams.set_quant_range(self.tmp_quant_bits)
        q_bias = q_bias_QParams.quantize(self.bias).to("cpu")
        if not q_bias_QParams.is_symmetric:
            q_bias = self.math_ops.intsub(
                q_bias, q_bias_QParams.zero_point, self.tmp_quant_bits + 1
            )
        return q_bias, q_bias_QParams

    def precompute(self) -> None:
        if self.training:
            raise RuntimeError("precompute must be called in eval mode")
        self.q_weights = self._get_quantized_weights()
        self.tmp_quant_bits = (
            self.inputs_QParams.quant_bits + 1 + self.weight_QParams.quant_bits + 1
        )
        if self.bias is not None:
            self.q_bias, q_bias_QParams = self._get_quantized_bias()
            # Only the zero point is kept. Storing the QParams module would add
            # its buffers to the state_dict of the layer.
            self.q_bias_zero_point = int(q_bias_QParams.zero_point.item())
            self.tmp_quant_bits += 1

        self.scale_factor_M = (
            self.inputs_QParams.scale_factor * self.weight_QParams.scale_factor
        ) / self.outputs_QParams.scale_factor

        self.scale_factor_m_q_shift, self.scale_factor_m_q = scaling_M(
            self.scale_factor_M
        )
        self.precomputed = True

    def int_forward(
        self,
        q_inputs: torch.Tensor,
    ) -> torch.Tensor:
        if self.training:
            raise RuntimeError("int_forward must be called in eval mode")
        if not self.precomputed:
            raise RuntimeError("precompute must be called before int_forward")

        save_quant_data(q_inputs, self.quant_data_dir, f"{self.name}_q_x")

        # Pad with the zero point before subtracting it, so that the padded
        # values represent 0.
        q_inputs = get_padded_q_inputs(
            padding_len=self.padding_len,
            q_inputs=q_inputs,
            inputs_QParams=self.inputs_QParams,
        )
        q_inputs = self.math_ops.intsub(
            q_inputs, self.inputs_QParams.zero_point, self.inputs_QParams.quant_bits + 1
        )

        if self.q_bias is not None:
            tmp = F.conv1d(
                q_inputs,
                self.q_weights,
                self.q_bias,
                padding=0,
                groups=self.groups,
            )
            tmp = self.math_ops.clamp_result(tmp, self.tmp_quant_bits + 1)
        else:
            tmp = F.conv1d(
                q_inputs,
                self.q_weights,
                padding=0,
                groups=self.groups,
            )
            tmp = self.math_ops.clamp_result(tmp, self.tmp_quant_bits)

        tmp = simulate_bitshifting(
            tmp, self.scale_factor_m_q_shift, self.scale_factor_m_q
        )

        q_outputs = self.math_ops.intadd(
            tmp, self.outputs_QParams.zero_point, self.outputs_QParams.quant_bits
        )
        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")
        return q_outputs

    def forward(
        self,
        inputs: torch.Tensor,
        given_inputs_QParams: QParams | None = None,
        enable_simquant: bool = True,
    ) -> torch.Tensor:
        if enable_simquant:
            if self.training:
                if given_inputs_QParams is None:
                    self.inputs_QParams.update_quant_params(inputs)
                else:
                    self.inputs_QParams = copy.deepcopy(given_inputs_QParams)

                self.weight_QParams.update_quant_params(self.weight)
                if self.bias is not None:
                    self.bias_QParams.update_quant_params(self.bias)

            inputs = SimQuant.apply(inputs, self.inputs_QParams)
            weight = SimQuant.apply(self.weight, self.weight_QParams)
            if self.bias is not None:
                bias = SimQuant.apply(self.bias, self.bias_QParams)
        else:
            weight = self.weight
            bias = self.bias

        outputs = (
            F.conv1d(inputs, weight, bias, padding=self.padding, groups=self.groups)
            if self.bias is not None
            else F.conv1d(inputs, weight, padding=self.padding, groups=self.groups)
        )

        if enable_simquant:
            if self.training:
                self.outputs_QParams.update_quant_params(outputs)

            outputs = SimQuant.apply(outputs, self.outputs_QParams)
        return outputs
