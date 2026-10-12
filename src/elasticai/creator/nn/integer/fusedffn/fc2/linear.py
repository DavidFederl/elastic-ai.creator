import copy

import torch
from torch import nn
from torch.nn import functional as F

from elasticai.creator.nn.integer.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.integer.fusedffn.fc2.design import Linear as LinearDesign
from elasticai.creator.nn.integer.math_operations import MathOperations
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
    SimQuant,
    SymmetricSignedQParams,
    scaling_M,
    simulate_bitshifting,
)
from elasticai.creator.nn.integer.quant_utils.QParams import QParams
from elasticai.creator.nn.integer.vhdl_test_automation.file_save_utils import (
    save_quant_data,
)


class Linear(DesignCreatorModule, nn.Linear):
    def __init__(
        self,
        *,
        name: str,
        in_features: int,
        out_features: int,
        bias: bool,
        quant_bits: int,
        num_dimensions: int | None = None,
        quant_data_dir: str | None = None,
        device: torch.device | str = torch.device("cpu"),
        enable_error_analysis: bool = False,
    ):
        super().__init__(in_features, out_features, bias)
        self.num_dimensions = num_dimensions

        self.name = name
        self.quant_bits = quant_bits
        self.quant_data_dir = quant_data_dir
        self.enable_error_analysis = enable_error_analysis

        # TODO: quantization scheme for each quantiztaion objects should be chosen by the user
        self.weight_QParams = AsymmetricSignedQParams(
            quant_bits=self.quant_bits, observer=GlobalMinMaxObserver()
        ).to(device)
        if bias:
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

    def create_design(self, name: str) -> LinearDesign:
        return LinearDesign(
            name=name,
            data_width=self.quant_bits,
            in_features=self.in_features,
            num_dimensions=self.num_dimensions,
            out_features=self.out_features,
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
        new_bias_quant_bits = (self.inputs_QParams.quant_bits + 1) + (
            self.weight_QParams.quant_bits + 1
        )
        q_bias_QParams.set_scale_factor(new_bias_scale_factor)
        q_bias_QParams.set_zero_point(torch.zeros((1), dtype=torch.int32))
        q_bias_QParams.set_quant_range(new_bias_quant_bits)
        q_bias = q_bias_QParams.quantize(self.bias).to("cpu")
        if not q_bias_QParams.is_symmetric:
            q_bias = self.math_ops.intsub(
                q_bias, q_bias_QParams.zero_point, new_bias_quant_bits + 1
            )
        return q_bias, q_bias_QParams

    def precompute(self) -> None:
        if self.training:
            raise RuntimeError("precompute must be called in eval mode")
        self.q_weights = self._get_quantized_weights()
        if self.bias is not None:
            self.q_bias, q_bias_QParams = self._get_quantized_bias()
            # Only the zero point is kept. Storing the QParams module would add
            # its buffers to the state_dict of the layer.
            self.q_bias_zero_point = int(q_bias_QParams.zero_point.item())
        else:
            self.q_bias = None
            self.q_bias_zero_point = 0

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

        q_inputs = self.math_ops.intsub(
            q_inputs, self.inputs_QParams.zero_point, self.inputs_QParams.quant_bits + 1
        )

        tmp = self.math_ops.int_mac(
            x=q_inputs,
            w=self.q_weights,
            b=self.q_bias,
            x_quant_bits=self.inputs_QParams.quant_bits,
            w_quant_bits=self.weight_QParams.quant_bits,
        )

        tmp = simulate_bitshifting(
            tmp, self.scale_factor_m_q_shift, self.scale_factor_m_q
        )

        q_outputs = self.math_ops.intadd(
            tmp, self.outputs_QParams.zero_point, self.outputs_QParams.quant_bits
        )

        save_quant_data(q_outputs, self.quant_data_dir, f"{self.name}_q_y")
        if self.enable_error_analysis:
            save_quant_data(
                self.outputs_QParams.dequantize(q_outputs),
                self.quant_data_dir,
                f"{self.name}_dq_y",
            )
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
            F.linear(inputs, weight, bias)
            if self.bias is not None
            else F.linear(inputs, weight)
        )

        if enable_simquant:
            if self.training:
                self.outputs_QParams.update_quant_params(outputs)

            outputs = SimQuant.apply(outputs, self.outputs_QParams)
            if self.enable_error_analysis:
                save_quant_data(
                    outputs,
                    self.quant_data_dir,
                    f"{self.name}_y",
                )
        return outputs
