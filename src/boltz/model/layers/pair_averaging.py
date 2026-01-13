import torch
from jaxtyping import Float32
from torch import Tensor, nn

import boltz.model.layers.initialize as init


class PairWeightedAveraging(nn.Module):
    """Pair weighted averaging layer.

    This module facilitates communication from the pairwise representation (z) to the MSA
    representation (m). It allows each MSA sequence to gather information from all
    other positions in the sequence, with the weighting of this gathering being
    conditioned by the pairwise features. This is analogous to the MSA-to-Pair communication
    in Alphafold's Evo-former blocks.
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_h: int,
        num_heads: int,
        inf: float = 1e6,
    ) -> None:
        """Initializes the pair weighted averaging layer.

        Args:
            c_m: The embedding dimension of the input MSA tensor.
            c_z: The embedding dimension of the input pairwise tensor.
            c_h: The hidden dimension per attention head.
            num_heads: The number of attention heads to use.
            inf: A large positive value used for masking padded elements in attention calculations.
        """
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_h = c_h
        self.num_heads = num_heads
        self.inf = inf

        self.norm_m = nn.LayerNorm(c_m)
        self.norm_z = nn.LayerNorm(c_z)

        self.proj_m = nn.Linear(c_m, c_h * num_heads, bias=False)
        self.proj_g = nn.Linear(c_m, c_h * num_heads, bias=False)
        self.proj_z = nn.Linear(c_z, num_heads, bias=False)
        self.proj_o = nn.Linear(c_h * num_heads, c_m, bias=False)
        init.final_init_(self.proj_o.weight)

    def forward(
        self,
        m: Float32[Tensor, "batch msa_size len msa_s"],
        z: Float32[Tensor, "batch len len token_z"],
        mask: Float32[Tensor, "batch len len"],
        chunk_heads: False = bool,
    ) -> Float32[Tensor, "batch msa_size len msa_s"]:
        """Performs the forward pass of the pair weighted averaging.

        It updates the MSA representation `m` by incorporating contextual information
        from the pairwise representation `z` through an attention-like mechanism.

        Args:
            m: The input MSA representation tensor.
            z: The input pairwise representation tensor.
            mask: A 0/1 mask tensor for sequence padding.
            chunk_heads: A boolean flag indicating whether to compute attention heads sequentially
                         for memory efficiency during inference.

        Returns:
            The updated MSA representation tensor, with the same shape as `m`.
        """
        # Compute layer norms
        m = self.norm_m(m)
        z = self.norm_z(z)

        if chunk_heads and not self.training:
            # Compute heads sequentially
            o_chunks = []
            for head_idx in range(self.num_heads):
                sliced_weight_proj_m = self.proj_m.weight[head_idx * self.c_h : (head_idx + 1) * self.c_h, :]
                sliced_weight_proj_g = self.proj_g.weight[head_idx * self.c_h : (head_idx + 1) * self.c_h, :]
                sliced_weight_proj_z = self.proj_z.weight[head_idx : (head_idx + 1), :]
                sliced_weight_proj_o = self.proj_o.weight[:, head_idx * self.c_h : (head_idx + 1) * self.c_h]

                # Project input tensors
                v: Tensor = m @ sliced_weight_proj_m.T
                v = v.reshape(*v.shape[:3], 1, self.c_h)
                v = v.permute(0, 3, 1, 2, 4)

                # Compute weights
                b: Tensor = z @ sliced_weight_proj_z.T
                b = b.permute(0, 3, 1, 2)
                b = b + (1 - mask[:, None]) * -self.inf
                w = torch.softmax(b, dim=-1)

                # Compute gating
                g: Tensor = m @ sliced_weight_proj_g.T
                g = g.sigmoid()

                # Compute output
                o = torch.einsum("bhij,bhsjd->bhsid", w, v)
                o = o.permute(0, 2, 3, 1, 4)
                o = o.reshape(*o.shape[:3], 1 * self.c_h)
                o_chunks = g * o
                if head_idx == 0:
                    o_out = o_chunks @ sliced_weight_proj_o.T
                else:
                    o_out += o_chunks @ sliced_weight_proj_o.T
            return o_out
        else:
            # Project input tensors
            v: Tensor = self.proj_m(m)
            v = v.reshape(*v.shape[:3], self.num_heads, self.c_h)
            v = v.permute(0, 3, 1, 2, 4)

            # Compute weights
            b: Tensor = self.proj_z(z)
            b = b.permute(0, 3, 1, 2)
            b = b + (1 - mask[:, None]) * -self.inf
            w = torch.softmax(b, dim=-1)

            # Compute gating
            g: Tensor = self.proj_g(m)
            g = g.sigmoid()

            # Compute output
            o = torch.einsum("bhij,bhsjd->bhsid", w, v)
            o = o.permute(0, 2, 3, 1, 4)
            o = o.reshape(*o.shape[:3], self.num_heads * self.c_h)
            o = self.proj_o(g * o)
            return o
