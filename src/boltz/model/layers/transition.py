from typing import Optional

from jaxtyping import Float32
from torch import Tensor, nn

import boltz.model.layers.initialize as init


class Transition(nn.Module):
    """Perform a two-layer MLP."""

    def __init__(
        self,
        dim: int = 128,
        hidden: int = 512,
        out_dim: Optional[int] = None,
    ) -> None:
        """Initializes the TransitionUpdate module.

        Args:
            dim: The dimension of the input.
            hidden: The dimension of the hidden layer.
            out_dim: The dimension of the output. If None, it defaults to the input dimension (`dim`).
        """
        super().__init__()
        if out_dim is None:
            out_dim = dim

        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(dim, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, out_dim, bias=False)
        self.silu = nn.SiLU()
        self.hidden = hidden

        init.bias_init_one_(self.norm.weight)
        init.bias_init_zero_(self.norm.bias)

        init.lecun_normal_init_(self.fc1.weight)
        init.lecun_normal_init_(self.fc2.weight)
        init.final_init_(self.fc3.weight)

    def forward(
        self, x: Float32[Tensor, "b dim ..."], chunk_size: Optional[int] = None
    ) -> Float32[Tensor, "b out_dim ..."]:
        """Perform a forward pass.

        Args:
            x: the input tensor
            chunk_size: if not None, the chunk size to use for computing the output in chunks.

        Returns:
            The output tensor.

        """
        x = self.norm(x)

        if chunk_size is None or self.training:
            x = self.silu(self.fc1(x)) * self.fc2(x)
            x = self.fc3(x)
            return x
        else:
            # Compute in chunks
            for i in range(0, self.hidden, chunk_size):
                fc1_slice = self.fc1.weight[i : i + chunk_size, :]
                fc2_slice = self.fc2.weight[i : i + chunk_size, :]
                fc3_slice = self.fc3.weight[:, i : i + chunk_size]
                x_chunk = self.silu(x @ fc1_slice.T) * (x @ fc2_slice.T)
                if i == 0:
                    x_out = x_chunk @ fc3_slice.T
                else:
                    x_out = x_out + x_chunk @ fc3_slice.T
            return x_out
