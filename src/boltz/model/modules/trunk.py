from typing import Optional

import torch
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from jaxtyping import Bool, Float, Float32, Int64
from torch import Tensor, nn

from boltz.data import const
from boltz.model.layers.attention import AttentionPairBias
from boltz.model.layers.dropout import get_dropout_mask
from boltz.model.layers.outer_product_mean import OuterProductMean
from boltz.model.layers.pair_averaging import PairWeightedAveraging
from boltz.model.layers.transition import Transition
from boltz.model.layers.triangular_attention.attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from boltz.model.layers.triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from boltz.model.modules.encoders import AtomAttentionEncoder


class InputEmbedder(nn.Module):
    """Transforms the input features (sequence, msa profile, etc.) into a single embedding."""

    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        atom_feature_dim: int,
        atom_encoder_depth: int,
        atom_encoder_heads: int,
        no_atom_encoder: bool = False,
    ) -> None:
        """Initialize the input embedder.

        Args:
            atom_s: the atom single representation embedding size.
            atom_z: the atom pair representation embedding size.
            token_s: the single token representation embedding size.
            token_z: the pair token representation embedding size.
            atoms_per_window_queries: the number of atoms per window for queries.
            atoms_per_window_keys: the number of atoms per window for keys.
            atom_feature_dim: the atom feature dimension.
            atom_encoder_depth: the atom encoder depth.
            atom_encoder_heads: the atom encoder heads.
            no_atom_encoder: whether to use the atom encoder (if disabled, the corresponding embedding is set to zero).

        """
        super().__init__()
        self.token_s = token_s
        self.no_atom_encoder = no_atom_encoder

        if not no_atom_encoder:
            self.atom_attention_encoder = AtomAttentionEncoder(
                atom_s=atom_s,
                atom_z=atom_z,
                token_s=token_s,
                token_z=token_z,
                atoms_per_window_queries=atoms_per_window_queries,
                atoms_per_window_keys=atoms_per_window_keys,
                atom_feature_dim=atom_feature_dim,
                atom_encoder_depth=atom_encoder_depth,
                atom_encoder_heads=atom_encoder_heads,
                structure_prediction=False,
            )

    def forward(self, feats: dict[str, Tensor]) -> Float32[Tensor, "batch len embed=455"]:
        """Perform the forward pass.

        Args:
            feats: dictionary of input feature name to input feature tensor. The following features are being
                processed:
                 - `res_type`: the residue type, a one-hot encoding of the 33 token types (amino acids, nucleotides, etc.)
                 - `profile`: the amino acid frequency in the MSA
                 - `deletion_mean`: the average number of deletions per position in the MSA (see featurizer.py for details)
                 - `pocket_feature`: the pocket feature, a one-hot encoding of the 4 pocket types (see const.py::pocket_contact_info)
                 - if `self.no_atom_encoder` is False, the atom features are also processed by self.atom_attention_encoder.

        Return: the embedded tokens.
        """
        res_type: Int64[Tensor, " batch len num_tokens=33"] = feats["res_type"]
        profile: Float32[Tensor, " batch len num_tokens=33"] = feats["profile"]
        deletion_mean: Float32[Tensor, " batch len 1"] = feats["deletion_mean"].unsqueeze(-1)
        pocket_feature: Int64[Tensor, " batch len 4"] = feats["pocket_feature"]

        # Compute input embedding
        if self.no_atom_encoder:
            a = torch.zeros(
                (res_type.shape[0], res_type.shape[1], self.token_s),
                device=res_type.device,
            )
        else:
            a, _, _, _, _ = self.atom_attention_encoder(feats)
        # embed size is: num_tokens*2 + 1 + 4=len(const.pocket_contact_info) + token_s
        s: Float32[Tensor, "batch len embed=455"] = torch.cat(
            [a, res_type, profile, deletion_mean, pocket_feature], dim=-1
        )
        return s


class MSAModule(nn.Module):
    """MSA module, which processes multiple sequence alignments (MSA) and pairwise embeddings.
    It consists of `msa_blocks` MSA layers that called in sequence.
    """

    def __init__(
        self,
        msa_s: int,
        token_z: int,
        s_input_dim: int,
        msa_blocks: int,
        msa_dropout: float,
        z_dropout: float,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        activation_checkpointing: bool = False,
        use_paired_feature: bool = False,
        offload_to_cpu: bool = False,
        subsample_msa: bool = False,
        num_subsampled_msa: int = 1024,
        **kwargs,
    ) -> None:
        """Initializes the MSA module.

        Args:
            msa_s: The MSA embedding size (typically 64).
            token_z: The token pairwise embedding size (typically 128).
            s_input_dim: The input sequence dimension (typically 455).
            msa_blocks: The number of MSA blocks (typically 4).
            msa_dropout: The MSA dropout (typically 0.15).
            z_dropout: The pairwise dropout (typically 0.25).
            pairwise_head_width: The pairwise head width. Defaults to 32.
            pairwise_num_heads: The number of pairwise heads. Defaults to 4.
            activation_checkpointing: Whether to use activation checkpointing. Defaults to False.
            use_paired_feature: if true, the MSA module will use `feats["msa_paired"]` to distinguish betwee
                simple homologs, and homologs that are paired (e.g. in a complex).
            offload_to_cpu: Whether to offload to CPU. Defaults to False.
            subsample_msa: whether to subsample the MSA (for efficiency).
            kwargs: extra keyword arguments (ignored).
            num_subsampled_msa: the number of MSA sequences to subsample from the full MSA.
        """
        super().__init__()
        del kwargs
        self.msa_blocks = msa_blocks
        self.msa_dropout = msa_dropout
        self.z_dropout = z_dropout
        self.use_paired_feature = use_paired_feature
        self.subsample_msa = subsample_msa
        self.num_subsampled_msa = num_subsampled_msa

        self.s_proj = nn.Linear(s_input_dim, msa_s, bias=False)
        self.msa_proj = nn.Linear(
            const.num_tokens + 2 + int(use_paired_feature),  # 33 + 2 + 1/0
            msa_s,
            bias=False,
        )

        def no_op_checkpoint_wrapper(module: nn.Module, offload_to_cpu: bool) -> nn.Module:
            del offload_to_cpu
            return module

        maybe_checkpoint = checkpoint_wrapper if activation_checkpointing else no_op_checkpoint_wrapper

        self.layers = nn.ModuleList(
            [
                maybe_checkpoint(
                    MSALayer(
                        msa_s,
                        token_z,
                        msa_dropout,
                        z_dropout,
                        pairwise_head_width,
                        pairwise_num_heads,
                    ),
                    offload_to_cpu=offload_to_cpu,
                )
                for _ in range(msa_blocks)
            ]
        )

    def forward(
        self,
        z: Float[Tensor, "batch len len token_z"],
        s_inputs: Float32[Tensor, "batch len embed=455"],
        feats: dict[str, Tensor],
        use_trifast: bool = False,
    ) -> Float[Tensor, "batch len len token_z"]:
        """Processes MSA features and pairwise embeddings through a series of MSA layers
        to refine the pairwise representation. It integrates information from both the MSA and sequence inputs.

        Args:
            z: the pairwise embeddings
            s_inputs: the input embeddings
            feats: the input features
            use_trifast: whether to use fast triangular attention
        Returns:
            The output pairwise embeddings.
        """
        # Set chunk sizes
        if not self.training:
            if z.shape[1] > const.chunk_size_threshold:
                chunk_heads_pwa = True
                chunk_size_transition_z = 64
                chunk_size_transition_msa = 32
                chunk_size_outer_product = 4
                chunk_size_tri_attn = 128
            else:
                chunk_heads_pwa = False
                chunk_size_transition_z = None
                chunk_size_transition_msa = None
                chunk_size_outer_product = None
                chunk_size_tri_attn = 512
        else:
            chunk_heads_pwa = False
            chunk_size_transition_z = None
            chunk_size_transition_msa = None
            chunk_size_outer_product = None
            chunk_size_tri_attn = None

        # Load relevant features
        msa: Int64[Tensor, "batch msa_size len num_tokens=33"] = feats["msa"]
        has_deletion: Bool[Tensor, "batch msa_size len 1"] = feats["has_deletion"].unsqueeze(-1)
        deletion_value: Float32[Tensor, "batch msa_size len 1"] = feats["deletion_value"].unsqueeze(-1)
        is_paired: Float32[Tensor, "batch msa_size len 1"] = feats["msa_paired"].unsqueeze(-1)
        msa_mask: Int64[Tensor, "batch msa_size len"] = feats["msa_mask"]
        token_mask: Float32[Tensor, "batch len"] = feats["token_pad_mask"].float()
        token_mask: Float32[Tensor, "batch len len"] = token_mask[:, :, None] * token_mask[:, None, :]

        # Compute MSA embeddings
        if self.use_paired_feature:
            m = torch.cat([msa, has_deletion, deletion_value, is_paired], dim=-1)
        else:
            m = torch.cat([msa, has_deletion, deletion_value], dim=-1)

        if self.subsample_msa:
            msa_indices = torch.randperm(m.shape[1])[: self.num_subsampled_msa]
            m = m[:, msa_indices]
            msa_mask = msa_mask[:, msa_indices]

        # Compute input projections
        m: Float32["batch msa_size len msa_s=64"] = self.msa_proj(m)
        m = m + self.s_proj(s_inputs).unsqueeze(1)

        # Perform MSA blocks
        for i in range(self.msa_blocks):
            # z has shape (batch, len, len, token_z)
            z, m = self.layers[i](
                z,
                m,
                token_mask,
                msa_mask,
                chunk_heads_pwa,
                chunk_size_transition_z,
                chunk_size_transition_msa,
                chunk_size_outer_product,
                chunk_size_tri_attn,
                use_trifast=use_trifast,
            )
        return z


class MSALayer(nn.Module):
    """MSA module."""

    def __init__(
        self,
        msa_s: int,
        token_z: int,
        msa_dropout: float,
        z_dropout: float,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
    ) -> None:
        """Initializes the MSA module.

        Args:
            msa_s: The MSA embedding size.
            token_z: The pair representation dimension.
            msa_dropout: The MSA dropout.
            z_dropout: The pair dropout.
            pairwise_head_width: The pairwise head width. Defaults to 32.
            pairwise_num_heads: The number of pairwise heads. Defaults to 4.
        """
        super().__init__()
        self.msa_dropout = msa_dropout
        self.z_dropout = z_dropout
        self.msa_transition = Transition(dim=msa_s, hidden=msa_s * 4)
        self.pair_weighted_averaging = PairWeightedAveraging(
            c_m=msa_s,
            c_z=token_z,
            c_h=32,
            num_heads=8,
        )

        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)
        self.tri_att_start = TriangleAttentionStartingNode(token_z, pairwise_head_width, pairwise_num_heads, inf=1e9)
        self.tri_att_end = TriangleAttentionEndingNode(token_z, pairwise_head_width, pairwise_num_heads, inf=1e9)
        self.z_transition = Transition(
            dim=token_z,
            hidden=token_z * 4,
        )
        self.outer_product_mean = OuterProductMean(
            c_in=msa_s,
            c_hidden=32,
            c_out=token_z,
        )

    def forward(
        self,
        z: Float32[Tensor, "batch len len token_z"],
        m: Float32[Tensor, "batch msa_size len msa_s"],
        token_mask: Float32[Tensor, "batch len len"],
        msa_mask: Int64[Tensor, "batch msa_size len"],
        chunk_heads_pwa: bool = False,
        chunk_size_transition_z: Optional[int] = None,
        chunk_size_transition_msa: Optional[int] = None,
        chunk_size_outer_product: Optional[int] = None,
        chunk_size_tri_attn: Optional[int] = None,
        use_trifast: bool = False,
    ) -> tuple[Float32[Tensor, "batch len len token_z"], Float32[Tensor, "batch msa_size len msa_s"]]:
        """Performs the forward pass of a single MSA layer.

        This layer updates both the MSA representation (`m`) and the pairwise representation (`z`)
        through a series of attention, multiplicative, and transition operations, facilitating communication
        between the two representations.

        Args:
            z: The current pairwise representation tensor
            m: The current MSA representation tensor
            token_mask: A boolean mask for valid tokens (sequence positions), typically used for padding.
            msa_mask: A boolean mask for valid MSA sequences, typically used for padding.
            chunk_heads_pwa: A boolean indicating whether to chunk heads in PairWeightedAveraging for memory
                efficiency
            chunk_size_transition_z: the chunk size for the pairwise transition operation.
                If None, no chunking is applied.
            chunk_size_transition_msa: the chunk size for the MSA transition operation. If None, no chunking is applied.
            chunk_size_outer_product: the chunk size for the outer product mean operation.
                If None, no chunking is applied.
            chunk_size_tri_attn: the chunk size for triangular attention operations.
                If None, no chunking is applied.
            use_trifast: A boolean indicating whether to use an optimized (faster) implementation for
                triangular attention operations. Defaults to False.

        Returns:
            A tuple containing:
            - z: The updated pairwise representation tensor.
            - m: The updated MSA representation tensor.
        """
        # Communication to MSA stack
        msa_dropout = get_dropout_mask(self.msa_dropout, m, self.training)
        m = m + msa_dropout * self.pair_weighted_averaging(m, z, token_mask, chunk_heads_pwa)
        m = m + self.msa_transition(m, chunk_size_transition_msa)

        # Communication to pairwise stack
        z = z + self.outer_product_mean(m, msa_mask, chunk_size_outer_product)

        # Compute pairwise stack
        dropout = get_dropout_mask(self.z_dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(z, mask=token_mask)

        dropout = get_dropout_mask(self.z_dropout, z, self.training)
        z = z + dropout * self.tri_mul_in(z, mask=token_mask)

        dropout = get_dropout_mask(self.z_dropout, z, self.training)
        z = z + dropout * self.tri_att_start(
            z,
            mask=token_mask,
            chunk_size=chunk_size_tri_attn,
            use_trifast=use_trifast,
        )

        dropout = get_dropout_mask(self.z_dropout, z, self.training, column_wise=True)
        z = z + dropout * self.tri_att_end(
            z,
            mask=token_mask,
            chunk_size=chunk_size_tri_attn,
            use_trifast=use_trifast,
        )

        z = z + self.z_transition(z, chunk_size_transition_z)

        return z, m


class PairformerModule(nn.Module):
    """Pairformer module."""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_blocks: int,
        num_heads: int = 16,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        activation_checkpointing: bool = False,
        no_update_s: bool = False,
        no_update_z: bool = False,
        offload_to_cpu: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the Pairformer module.

        Parameters
        ----------
        token_s : int
            The token single embedding size.
        token_z : int
            The token pairwise embedding size.
        num_blocks : int
            The number of blocks.
        num_heads : int, optional
            The number of heads, by default 16
        dropout : float, optional
            The dropout rate, by default 0.25
        pairwise_head_width : int, optional
            The pairwise head width, by default 32
        pairwise_num_heads : int, optional
            The number of pairwise heads, by default 4
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False
        no_update_s : bool, optional
            Whether to update the single embeddings, by default False
        no_update_z : bool, optional
            Whether to update the pairwise embeddings, by default False
        offload_to_cpu : bool, optional
            Whether to offload to CPU, by default False

        """
        super().__init__()
        del kwargs
        self.token_z = token_z
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.num_heads = num_heads

        self.layers = nn.ModuleList()
        for i in range(num_blocks):
            if activation_checkpointing:
                self.layers.append(
                    checkpoint_wrapper(
                        PairformerLayer(
                            token_s,
                            token_z,
                            num_heads,
                            dropout,
                            pairwise_head_width,
                            pairwise_num_heads,
                            no_update_s,
                            False if i < num_blocks - 1 else no_update_z,
                        ),
                        offload_to_cpu=offload_to_cpu,
                    )
                )
            else:
                self.layers.append(
                    PairformerLayer(
                        token_s,
                        token_z,
                        num_heads,
                        dropout,
                        pairwise_head_width,
                        pairwise_num_heads,
                        no_update_s,
                        False if i < num_blocks - 1 else no_update_z,
                    )
                )

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
        chunk_size_tri_attn: Optional[int] = None,
        use_trifast: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        s : Tensor
            The sequence embeddings
        z : Tensor
            The pairwise embeddings
        mask : Tensor
            The token mask
        pair_mask : Tensor
            The pairwise mask

        Returns:
        -------
        Tensor
            The updated sequence embeddings.
        Tensor
            The updated pairwise embeddings.

        """
        if not self.training:
            if z.shape[1] > const.chunk_size_threshold:
                chunk_size_tri_attn = 128
            else:
                chunk_size_tri_attn = 512
        else:
            chunk_size_tri_attn = None

        for layer in self.layers:
            s, z = layer(s, z, mask, pair_mask, chunk_size_tri_attn, use_trifast=use_trifast)
        return s, z


class PairformerLayer(nn.Module):
    """Pairformer module."""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_heads: int = 16,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        no_update_s: bool = False,
        no_update_z: bool = False,
    ) -> None:
        """Initialize the Pairformer module.

        Parameters
        ----------
        token_s : int
            The token single embedding size.
        token_z : int
            The token pairwise embedding size.
        num_heads : int, optional
            The number of heads, by default 16
        dropout : float, optiona
            The dropout rate, by default 0.25
        pairwise_head_width : int, optional
            The pairwise head width, by default 32
        pairwise_num_heads : int, optional
            The number of pairwise heads, by default 4
        no_update_s : bool, optional
            Whether to update the single embeddings, by default False
        no_update_z : bool, optional
            Whether to update the pairwise embeddings, by default False

        """
        super().__init__()
        self.token_z = token_z
        self.dropout = dropout
        self.num_heads = num_heads
        self.no_update_s = no_update_s
        self.no_update_z = no_update_z
        if not self.no_update_s:
            self.attention = AttentionPairBias(token_s, token_z, num_heads)
        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)
        self.tri_att_start = TriangleAttentionStartingNode(token_z, pairwise_head_width, pairwise_num_heads, inf=1e9)
        self.tri_att_end = TriangleAttentionEndingNode(token_z, pairwise_head_width, pairwise_num_heads, inf=1e9)
        if not self.no_update_s:
            self.transition_s = Transition(token_s, token_s * 4)
        self.transition_z = Transition(token_z, token_z * 4)

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
        chunk_size_tri_attn: Optional[int] = None,
        use_trifast: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Perform the forward pass."""
        # Compute pairwise stack
        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(z, mask=pair_mask)

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_in(z, mask=pair_mask)

        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_att_start(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
            use_trifast=use_trifast,
        )

        dropout = get_dropout_mask(self.dropout, z, self.training, column_wise=True)
        z = z + dropout * self.tri_att_end(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
            use_trifast=use_trifast,
        )

        z = z + self.transition_z(z)

        # Compute sequence stack
        if not self.no_update_s:
            s = s + self.attention(s, z, mask)
            s = s + self.transition_s(s)

        return s, z


class DistogramModule(nn.Module):
    """Distogram Module."""

    def __init__(self, token_z: int, num_bins: int) -> None:
        """Initialize the distogram module.

        Parameters
        ----------
        token_z : int
            The token pairwise embedding size.
        num_bins : int
            The number of bins.

        """
        super().__init__()
        self.distogram = nn.Linear(token_z, num_bins)

    def forward(self, z: Tensor) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : Tensor
            The pairwise embeddings

        Returns:
        -------
        Tensor
            The predicted distogram.

        """
        z = z + z.transpose(1, 2)
        return self.distogram(z)
