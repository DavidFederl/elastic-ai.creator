import torch

from elasticai.creator.arithmetic.fxp_arithmetic import FxpArithmetic


class DeltaOperations:
    def __init__(self, delta_bits: int, fxp_arithmetic: FxpArithmetic) -> None:
        self._delta_bits = delta_bits
        self._fxp_arithmetic = fxp_arithmetic

    @property
    def delta_bits(self) -> int:
        return self._delta_bits

    def simulate(self, input: torch.Tensor) -> torch.Tensor:
        return self.inflate(self.compress(input))

    def compress(self, input: torch.Tensor) -> torch.Tensor:
        input_shape = input.shape
        input = self._fxp_arithmetic.cut_as_integer(input)

        for index in reversed(range(1, len(input))):
            input[index] = input[index] - input[index - 1]

        min_val = -(2 ** (self._delta_bits - 1))
        max_val = (2 ** (self._delta_bits - 1)) - 1
        input[1:].clamp_(min=min_val, max=max_val)

        return input.reshape(input_shape)

    def inflate(self, input: torch.Tensor) -> torch.Tensor:
        input_shape = input.shape

        for index in range(1, len(input)):
            input[index] = input[index - 1] + input[index]

        min_val = -(2 ** (self._fxp_arithmetic.total_bits - 1))
        max_val = (2 ** (self._fxp_arithmetic.total_bits - 1)) - 1
        input[1:].clamp_(min=min_val, max=max_val)

        input = self._fxp_arithmetic.as_rational(input)

        return input.reshape(input_shape)
