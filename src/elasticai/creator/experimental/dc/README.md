# Delta compression (`experimental.dc`)

Delta compression utilities for PyTorch tensors. A tensor is stored as a
reference value plus a sequence of deltas, and each delta is quantized to a fixed
bit width. The original tensor can be reconstructed from the reference and the
deltas.

## Concepts

Two delta schemes are available:

| Scheme            | Reference value                 | Delta                   |
| ----------------- | ------------------------------- | ----------------------- |
| `fixed_reference` | Computed by a function you pass | `data - reference`      |
| `consecutive`     | First element, `data[0]`        | `data[i + 1] - data[i]` |

Two quantizers limit each delta to `delta_bit_width` bits (signed):

| Quantizer | Behaviour                                                 |
| --------- | --------------------------------------------------------- |
| `clamp`   | Saturates deltas to `[-2^(w-1), 2^(w-1) - 1]`             |
| `cut`     | Wraps deltas into the same range (keeps the low `w` bits) |

A wrapped delta fixes a value only modulo `2^w`. To decode it, pass the bounds of
the original data to `inflate` (see [Decoding wrapped deltas](#decoding-wrapped-deltas)).

Input must be an integer tensor (`int8` to `int32`, `uint8` to `uint32`) with at
least one dimension and one element. `uint16` and `uint32` need torch 2.3 or newer. Floats, `bool`, `int64`, `uint64` and complex tensors raise a
`TypeError`. Inputs are widened to `int64`, so unsigned types produce correct
negative deltas. `compress` and `inflate` return `int64` tensors, and `inflate`
also accepts `int64`. The delta bit width must
be an integer between 1 and 34, the most that 32 bit input can use.

Reconstruction is exact when every delta fits into the chosen bit width.
Otherwise it is lossy. With `consecutive` and `clamp`, the encoder tracks the
decoder's reconstruction, so an error does not accumulate: it is corrected as
soon as the data moves back into the delta range. A jump larger than the delta
range is still followed, but only by the largest delta that fits per step
(`2^(w-1) - 1` upwards, `-2^(w-1)` downwards). `cut` keeps every value
modulo `2^w`, but `inflate` can only pick the right one if it knows the range of
the original data.

## Project structure

```
src/elasticai/creator/experimental/dc/
    base.py                    DeltaCompression, the abstract base class
    fixed_reference_delta.py   FixedReferenceDelta
    consecutive_delta.py       ConsecutiveDelta
    builder.py                 DeltaCompressionBuilder, the public entry point,
                               and the quantizers clamp_delta and cut_delta
```

`DeltaCompression` and `DeltaCompressionBuilder` are exported from the package.

## Usage

```python
import torch

from elasticai.creator.experimental.dc import DeltaCompressionBuilder

data = torch.tensor([5, 7, 4, 4, 9], dtype=torch.int32)

codec = (
    DeltaCompressionBuilder()
    .consecutive()
    .delta_bit_width(8)
    .clamp()
    .build_delta()
)

reference, deltas = codec.compress(data)
# reference: tensor(5)
# deltas:    tensor([ 2, -3,  0,  5])

restored = codec.inflate(reference, deltas)
assert torch.equal(restored, data)
```

For a fixed reference, pass a module-level function that computes the reference
from the data:

```python
def minimum(data: torch.Tensor) -> torch.Tensor:
    return data.min(dim=0).values


codec = (
    DeltaCompressionBuilder()
    .fixed_reference(minimum)
    .delta_bit_width(4)
    .cut()
    .build_delta()
)
```

`build_delta()` raises a `ValueError` if the delta type, the bit width or the
quantizer has not been set. `delta_bit_width()` raises a `TypeError` for a
non-integer and a `ValueError` for a width outside 1 to 34.

### Decoding wrapped deltas

Pass the inclusive `(low, high)` bounds of the original data as `value_range`.
`inflate` then wraps the reconstruction into the window of `2^w` values that
starts at `low`. The result is exact when `high - low < 2^w`. This example uses
the `fixed_reference` codec with `cut` from above:

```python
data = torch.tensor([10, 12, 25, 11, 20, 18], dtype=torch.int32)
reference, deltas = codec.compress(data)

codec.inflate(reference, deltas)                          # wrong where deltas wrapped
codec.inflate(reference, deltas, value_range=(10, 25))    # equals data
```

A range that is empty, not a pair of integers, or spans `2^w` values or more
cannot be decoded unambiguously and raises an error. Do not pass `value_range`
for `clamp`.

### Validation in `inflate`

- The shapes must fit together: for `consecutive` the delta has shape
  `(n - 1, *reference.shape)`, for `fixed_reference` the reference must broadcast
  to the delta. Otherwise a `ValueError` names both shapes.
- Every delta must fit into `delta_bit_width` bits, otherwise a `ValueError` is
  raised. This catches corrupt data and costs one pass over the delta. Pass
  `check_delta_range=False` to skip it, for example for a custom quantizer that
  does not respect the width.

## Development

This package is part of the elasticai.creator repository. Set up the
environment and run checks as described in the repository's `CLAUDE.md` and
`CONTRIBUTING.md`:

```sh
uv sync
uv run ruff check --fix src/elasticai tests
uv run ruff format src/elasticai tests
uv run ty check
uv run tach check
```

Commit messages follow Conventional Commits, enforced by commitlint through
pre-commit. Add or update tests for every behaviour change. Prefer
[Hypothesis](https://hypothesis.readthedocs.io) properties, such as "the round
trip is exact if and only if all deltas fit", over single examples.

When adding a delta scheme:

- Subclass `DeltaCompression`, implement the abstract methods (including
  `_validate_inflate_shapes`) and expose it through `DeltaCompressionBuilder`.
- Use module-level functions, not lambdas, for reference and quantizer functions
  stored on a compressor, so that compressors stay picklable.
