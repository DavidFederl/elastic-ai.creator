from typing import Any, cast

import torch

from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
from elasticai.creator.base_modules.linear import Linear as LinearBase
from elasticai.creator.nn.delta_compression import DeltaOperations
from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.fixed_point import MathOperations

type LinearDesign = (
    Any  # Placeholder for the actual design class that will be implemented later
)


class Linear(DesignCreatorModule, LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        total_bits: int,
        frac_bits: int,
        delta_bits: int,
        bias: bool = True,
        device: Any = None,
    ) -> None:
        self._params = FxpParams(
            total_bits=total_bits, frac_bits=frac_bits, signed=True
        )
        self._config = FxpArithmetic(self._params)
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            operations=MathOperations(config=self._config),
            bias=bias,
            device=device,
        )
        self._deltaops = DeltaOperations(delta_bits, self._config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._deltaops.simulate(self._operations.quantize(self.weight))
        if self.bias is None:
            return self._operations.matmul(x, weight.T)
        else:
            return self._operations.add(
                self._operations.matmul(x, weight.T),
                self._deltaops.simulate(self._operations.quantize(self.bias)),
            )

    def get_params(self) -> tuple[list[list[float]], list[float]]:
        bias = (
            torch.Tensor([0] * self.out_features).tolist()
            if self.bias is None
            else self.bias.tolist()
        )
        weights = self.weight.tolist()
        return weights, bias

    def get_params_quant(self) -> tuple[list[list[int]], list[int]]:
        weights, bias = self.get_params()
        q_weights = cast(list[list[int]], self._config.cut_as_integer(weights))
        q_bias = cast(list[int], self._config.cut_as_integer(bias))
        return q_weights, q_bias

    def get_params_compressed(self) -> tuple[list[list[int]], list[int]]:
        bias = (
            self.bias
            if self.bias is not None
            else torch.Tensor([0] * self.out_features)
        )
        c_weights = self._deltaops.compress(self.weight).tolist()
        c_bias = self._deltaops.compress(bias).tolist()
        return c_weights, c_bias

    def create_design(self, name: str) -> LinearDesign:
        raise NotImplementedError("Design creation not implemented yet")
