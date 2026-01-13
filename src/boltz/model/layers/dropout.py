from typing import Union

import torch
from jaxtyping import Float32
from torch import Tensor


def get_dropout_mask(
    dropout: float,
    t: Float32[Tensor, "d0 d1 d2 ..."],
    training: bool,
    column_wise: bool = False,
) -> Union[Float32[Tensor, "d0 1 d2 ..."], Float32[Tensor, "d0 d1 1 ..."]]:
    """Generates a dropout mask that can be applied to a tensor `t`, selectively dropping out elements based
    on the specified dropout rate.
    The mask is scaled to maintain the expected sum of activations. The mask's shape is the same as `x`,
    except that `d1` or `d2` are set to 1 (depending on whether column_wise is True or False),
    matching the shape of the input tensor `t` for broadcasting.


    Args:
        dropout: The dropout rate (a float between 0.0 and 1.0).
        t: The input tensor to which dropout would be applied. Its shape is used to
            determine the shape of the generated mask.
        training: A boolean indicating whether the model is in training mode. If False,
            the dropout rate is effectively zero.
        column_wise: A boolean indicating whether to apply dropout across columns (True)
            or rows (False) of the tensor's relevant dimensions.

    Returns:
        A tensor representing the binary dropout mask, scaled by `1.0 / (1.0 - dropout)`.
    """
    dropout = dropout * training
    v = t[:, 0:1, :, 0:1] if column_wise else t[:, :, 0:1, 0:1]
    d = torch.rand_like(v) > dropout
    d = d * 1.0 / (1.0 - dropout)
    return d
