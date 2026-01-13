# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang
from functools import partial
from math import pi

import torch
from einops import rearrange
from jaxtyping import Bool, Float32, Int64
from torch import Tensor, nn
from torch.nn import Module, ModuleList
from torch.nn.functional import one_hot

import boltz.model.layers.initialize as init
from boltz.data import const
from boltz.model.layers.transition import Transition
from boltz.model.modules.transformers import AtomTransformer
from boltz.model.modules.utils import LinearNoBias


class FourierEmbedding(Module):
    """Fourier embedding layer."""

    def __init__(self, dim):
        """Initialize the Fourier Embeddings.

        Parameters
        ----------
        dim : int
            The dimension of the embeddings.

        """
        super().__init__()
        self.proj = nn.Linear(1, dim)
        torch.nn.init.normal_(self.proj.weight, mean=0, std=1)
        torch.nn.init.normal_(self.proj.bias, mean=0, std=1)
        self.proj.requires_grad_(False)

    def forward(
        self,
        times,
    ):
        times = rearrange(times, "b -> b 1")
        rand_proj = self.proj(times)
        return torch.cos(2 * pi * rand_proj)


class RelativePositionEncoder(Module):
    """Relative position encoder. Computes the relative position of the tokens by one-hot encoding the
    relative position of the tokens within a chain, cyclic structures, identical chains and the relative positions of
    chains and then pushing them through a linear layer.
    """

    def __init__(self, token_z: int, r_max: int = 32, s_max: int = 2):
        """Initialize the relative position encoder.

        Params:
            token_z: the pair representation dimension.
            r_max: the maximum index distance
            s_max: the maximum chain distance
        """
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.linear_layer = LinearNoBias(4 * (r_max + 1) + 2 * (s_max + 1) + 1, token_z)

    def forward(self, feats):
        # True if two tokens are in the same chain, False otherwise
        b_same_chain: Bool[Tensor, "batch len len"] = torch.eq(
            feats["asym_id"][:, :, None],
            feats["asym_id"][:, None, :],
        )
        # True if two tokens are in the same position within their chains, False otherwise
        b_same_residue: Bool[Tensor, "batch len len"] = torch.eq(
            feats["residue_index"][:, :, None],
            feats["residue_index"][:, None, :],
        )
        # True if two tokens are in the same entity, false otherwise
        # For example, in an antibody, the two light chains are in the same entity, same for the 2 heavy chains
        # For a bi-specific antibody with identical light chains, the two light chains will have the same entity id,
        # while the heavy chains will have different entity ids
        b_same_entity: Bool[Tensor, "batch len len"] = torch.eq(
            feats["entity_id"][:, :, None],
            feats["entity_id"][:, None, :],
        )
        # relative position of tokens within their chains
        rel_pos: Int64[Tensor, "batch len len"] = (
            feats["residue_index"][:, :, None] - feats["residue_index"][:, None, :]
        )
        if torch.any(feats["cyclic_period"] != 0):
            # set atoms that are not part of a cyclic structure to 10_000, the rest are untouched
            period: Int64[Tensor, "batch 1 len"] = torch.where(
                feats["cyclic_period"] > 0,
                feats["cyclic_period"],
                torch.zeros_like(feats["cyclic_period"]) + 10000,
            ).unsqueeze(1)

            # correct the relative position for cyclic structures; i.e. the first and last atoms in
            # a cyclic structure are actually close to each other
            rel_pos = (rel_pos - period * torch.round(rel_pos / period)).long()

        # keep only tokens within the range of r_max
        d_residue: Int64[Tensor, "batch len len"] = torch.clip(rel_pos + self.r_max, 0, 2 * self.r_max)

        # set the distance to 2 * r_max + 1 if the two tokens are not in the same chain
        d_residue = torch.where(b_same_chain, d_residue, torch.zeros_like(d_residue) + 2 * self.r_max + 1)
        # one hot encoding of the relative position
        a_rel_pos: Int64[Tensor, "batch len len embed=66"] = one_hot(d_residue, 2 * self.r_max + 2)

        # relative position of tokens within a chain, clipped to r_max
        d_token: Int64[Tensor, "batch len len"] = torch.clip(
            feats["token_index"][:, :, None] - feats["token_index"][:, None, :] + self.r_max,
            0,
            2 * self.r_max,
        )
        # set distance to 2 * r_max + 1 if the two tokens are not in the same chain and don't have the same residue_idx
        # basically, only the diagonal elements will be `rmax` and the residues that all have the same residue_idx (maybe ligands?)
        d_token = torch.where(
            b_same_chain & b_same_residue,
            d_token,
            torch.zeros_like(d_token) + 2 * self.r_max + 1,
        )
        a_rel_token: Int64[Tensor, "batch len len embed=66"] = one_hot(d_token, 2 * self.r_max + 2)

        # sym_id is a unique integer within a set of identical chains. For example, in an A3B2 stoichiometry complex
        # the “A” chains would have IDs [0, 1, 2] and the “B” chains would have IDs [0, 1]
        d_chain: Int64[Tensor, "batch len len"] = torch.clip(
            feats["sym_id"][:, :, None] - feats["sym_id"][:, None, :] + self.s_max,
            0,
            2 * self.s_max,
        )
        # set to 2 * s_max + 1 = 5 if the two tokens are in the same chain, otherwise to the relative chain distance
        d_chain = torch.where(b_same_chain, torch.zeros_like(d_chain) + 2 * self.s_max + 1, d_chain)
        a_rel_chain: Int64[Tensor, "batch len len embed=66"] = one_hot(d_chain, 2 * self.s_max + 2)

        p: Float32[Tensor, " batch, len, len, token_z"] = self.linear_layer(
            torch.cat(
                [
                    a_rel_pos.float(),
                    a_rel_token.float(),
                    b_same_entity.unsqueeze(-1).float(),
                    a_rel_chain.float(),
                ],
                dim=-1,
            )  # (batch, len, len, 4 * (r_max + 1) + 2 * (s_max + 1) + 1))
        )
        return p


class SingleConditioning(Module):
    """Single conditioning layer."""

    def __init__(
        self,
        sigma_data: float,
        token_s=384,
        dim_fourier=256,
        num_transitions=2,
        transition_expansion_factor=2,
        eps=1e-20,
    ):
        """Initialize the single conditioning layer.

        Parameters
        ----------
        sigma_data : float
            The data sigma.
        token_s : int, optional
            The single representation dimension, by default 384.
        dim_fourier : int, optional
            The fourier embeddings dimension, by default 256.
        num_transitions : int, optional
            The number of transitions layers, by default 2.
        transition_expansion_factor : int, optional
            The transition expansion factor, by default 2.
        eps : float, optional
            The epsilon value, by default 1e-20.

        """
        super().__init__()
        self.eps = eps
        self.sigma_data = sigma_data

        input_dim = 2 * token_s + 2 * const.num_tokens + 1 + len(const.pocket_contact_info)
        self.norm_single = nn.LayerNorm(input_dim)
        self.single_embed = nn.Linear(input_dim, 2 * token_s)
        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = nn.LayerNorm(dim_fourier)
        self.fourier_to_single = LinearNoBias(dim_fourier, 2 * token_s)

        transitions = ModuleList([])
        for _ in range(num_transitions):
            transition = Transition(dim=2 * token_s, hidden=transition_expansion_factor * 2 * token_s)
            transitions.append(transition)

        self.transitions = transitions

    def forward(
        self,
        *,
        times,
        s_trunk,
        s_inputs,
    ):
        s = torch.cat((s_trunk, s_inputs), dim=-1)
        s = self.single_embed(self.norm_single(s))
        fourier_embed = self.fourier_embed(times)
        normed_fourier = self.norm_fourier(fourier_embed)
        fourier_to_single = self.fourier_to_single(normed_fourier)

        s = rearrange(fourier_to_single, "b d -> b 1 d") + s

        for transition in self.transitions:
            s = transition(s) + s

        return s, normed_fourier


class PairwiseConditioning(Module):
    """Pairwise conditioning layer."""

    def __init__(
        self,
        token_z,
        dim_token_rel_pos_feats,
        num_transitions=2,
        transition_expansion_factor=2,
    ):
        """Initialize the pairwise conditioning layer.

        Parameters
        ----------
        token_z : int
            The pair representation dimension.
        dim_token_rel_pos_feats : int
            The token relative position features dimension.
        num_transitions : int, optional
            The number of transitions layers, by default 2.
        transition_expansion_factor : int, optional
            The transition expansion factor, by default 2.

        """
        super().__init__()

        self.dim_pairwise_init_proj = nn.Sequential(
            nn.LayerNorm(token_z + dim_token_rel_pos_feats),
            LinearNoBias(token_z + dim_token_rel_pos_feats, token_z),
        )

        transitions = ModuleList([])
        for _ in range(num_transitions):
            transition = Transition(dim=token_z, hidden=transition_expansion_factor * token_z)
            transitions.append(transition)

        self.transitions = transitions

    def forward(
        self,
        z_trunk,
        token_rel_pos_feats,
    ):
        z = torch.cat((z_trunk, token_rel_pos_feats), dim=-1)
        z = self.dim_pairwise_init_proj(z)

        for transition in self.transitions:
            z = transition(z) + z

        return z


def get_indexing_matrix(k: int, w: int, h: int, device: torch.device) -> Float32[Tensor, "2*k num_key_windows*k"]:
    """Get the indexing matrix for the keys in the attention mechanism.

    Args:
        k: the number of windows.
        w: the number of atoms per window for queries.
        h: the number of atoms per window for keys.
        device: the device to create the tensor on.
    """
    assert w % 2 == 0
    assert h % (w // 2) == 0

    num_key_windows = h // (w // 2)
    assert num_key_windows % 2 == 0

    arange: Int64[Tensor, " 2*k"] = torch.arange(2 * k, device=device)
    index: Int64[Tensor, "2*k 2*k"] = ((arange.unsqueeze(0) - arange.unsqueeze(1)) + num_key_windows // 2).clamp(
        min=0, max=num_key_windows + 1
    )
    index: Int64[Tensor, "k 2*k"] = index.view(k, 2, 2 * k)[:, 0, :]
    onehot: Int64[Tensor, "2*k k num_key_windows"] = one_hot(index, num_classes=num_key_windows + 2)[
        ..., 1:-1
    ].transpose(1, 0)
    return onehot.reshape(2 * k, num_key_windows * k).float()


def single_to_keys(
    single: Float32[Tensor, "batch num_atoms atom_s"],
    indexing_matrix: Float32[Tensor, "2*k num_key_cols"],
    w: int,
    h: int,
) -> Float32[Tensor, "batch k h atom_s"]:
    """Convert single representation into a key tensor format for attention.

    Args:
        single: the single representation tensor of shape (batch, num_atoms, atom_s).
        indexing_matrix: the indexing matrix for the keys.
        w: the number of atoms per window for queries.
        h: the number of atoms per window for keys.
    """
    b, n, d = single.shape  # b=batch, n=num_atoms, d=atom_s (embedding size)
    k = n // w
    # break down the initial single representation (of size n) into 2*k blocks of length w/2
    single = single.view(b, 2 * k, w // 2, d)
    return torch.einsum("b j i d, j k -> b k i d", single, indexing_matrix).reshape(b, k, h, d)


class AtomAttentionEncoder(Module):
    """Atom attention encoder."""

    def __init__(
        self,
        atom_s,
        atom_z,
        token_s,
        token_z,
        atoms_per_window_queries,
        atoms_per_window_keys,
        atom_feature_dim,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        structure_prediction=True,
        activation_checkpointing=False,
    ):
        """Initialize the atom attention encoder.

        Args:
            atom_s: the atom single representation embedding size.
            atom_z: the atom pair representation embedding size.
            token_s: the single token representation embedding size.
            token_z: the pair token representation embedding size.
            atoms_per_window_queries: the number of atoms per window for queries.
            atoms_per_window_keys: the number of atoms per window for keys.
            atom_feature_dim: the atom feature dimension (389 in the full model).
            atom_encoder_depth: the atom encoder depth.
            atom_encoder_heads: number of attention heads in the atom encoder.
            structure_prediction: true when used in the DiffusionModule, false when used in the trunk
            activation_checkpointing: whether to use activation checkpointing

        """
        super().__init__()

        self.embed_atom_features = LinearNoBias(atom_feature_dim, atom_s)
        self.embed_atompair_ref_pos = LinearNoBias(3, atom_z)
        self.embed_atompair_ref_dist = LinearNoBias(1, atom_z)
        self.embed_atompair_mask = LinearNoBias(1, atom_z)
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys

        self.structure_prediction = structure_prediction
        if structure_prediction:
            self.s_to_c_trans = nn.Sequential(nn.LayerNorm(token_s), LinearNoBias(token_s, atom_s))
            init.final_init_(self.s_to_c_trans[1].weight)

            self.z_to_p_trans = nn.Sequential(nn.LayerNorm(token_z), LinearNoBias(token_z, atom_z))
            init.final_init_(self.z_to_p_trans[1].weight)

            self.r_to_q_trans = LinearNoBias(10, atom_s)
            init.final_init_(self.r_to_q_trans.weight)

        self.c_to_p_trans_k = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_s, atom_z),
        )
        init.final_init_(self.c_to_p_trans_k[1].weight)

        self.c_to_p_trans_q = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_s, atom_z),
        )
        init.final_init_(self.c_to_p_trans_q[1].weight)

        self.p_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
        )
        init.final_init_(self.p_mlp[5].weight)

        self.atom_encoder = AtomTransformer(
            dim=atom_s,
            dim_single_cond=atom_s,
            dim_pairwise=atom_z,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            depth=atom_encoder_depth,
            heads=atom_encoder_heads,
            activation_checkpointing=activation_checkpointing,
        )

        self.atom_to_token_trans = nn.Sequential(
            LinearNoBias(atom_s, 2 * token_s if structure_prediction else token_s),
            nn.ReLU(),
        )

    def forward(
        self,
        feats,
        s_trunk=None,
        z=None,
        r=None,
        multiplicity=1,
        model_cache=None,
    ):
        """Forward pass of the atom attention encoder.

        Args:
            feats: dictionary of input feature name to input feature tensor. The following features are being
                processed:
                - ref_pos: the reference position of the atoms, shape (batch, num_atoms, 3)
                - atom_pad_mask: the atom padding mask, shape (batch, num_atoms)
                - atom_uid: the atom unique identifier, shape (batch, num_atoms)
            s_trunk: the single trunk representation, shape (batch, num_tokens, token_s)
            z: the pair trunk representation, shape (batch, num_tokens, num_tokens, token_z)
            r: the relative positions of the atoms, shape (batch, num_atoms, 7)
            multiplicity: number of independent diffusion samples to run in parallel
            model_cache: a cache for the model to speed up the computation,
                it is a dictionary that contains pre-computed tensors
        """
        batch, n, _ = feats["ref_pos"].shape
        atom_mask: Bool[Tensor, "batch num_atoms"] = feats["atom_pad_mask"].bool()

        layer_cache = None
        if model_cache is not None:
            cache_prefix = "atomencoder"
            if cache_prefix not in model_cache:
                model_cache[cache_prefix] = {}
            layer_cache = model_cache[cache_prefix]

        if model_cache is None or len(layer_cache) == 0:
            # either model is not using the cache or it is the first time running it

            atom_ref_pos: Float32[Tensor, "batch num_atoms 3"] = feats["ref_pos"]
            atom_uid: Int64[Tensor, "batch num_atoms"] = feats["ref_space_uid"]
            # embed dim size is 389=3+1+1+128+256
            atom_feats: Float32[Tensor, "batch num_atoms embed_dim=389"] = torch.cat(
                [
                    atom_ref_pos,
                    feats["ref_charge"].unsqueeze(-1),
                    feats["atom_pad_mask"].unsqueeze(-1),
                    feats["ref_element"],
                    feats["ref_atom_name_chars"].reshape(batch, n, 4 * 64),
                ],
                dim=-1,
            )

            c: Float32[Tensor, "batch num_atoms atom_s"] = self.embed_atom_features(atom_feats)

            # NOTE: we are already creating the windows to make it more efficient
            w, h = self.atoms_per_window_queries, self.atoms_per_window_keys
            k = n // w
            keys_indexing_matrix: Float32[Tensor, "2*k num_key_cols"] = get_indexing_matrix(k, w, h, c.device)
            to_keys = partial(single_to_keys, indexing_matrix=keys_indexing_matrix, w=w, h=h)

            atom_ref_pos_queries: Float32[Tensor, "batch k w 1 3"] = atom_ref_pos.view(batch, k, w, 1, 3)
            atom_ref_pos_keys: Float32[Tensor, "batch k 1 h 3"] = to_keys(atom_ref_pos).view(batch, k, 1, h, 3)

            d: Float32[Tensor, "batch k w h 3"] = atom_ref_pos_keys - atom_ref_pos_queries
            d_norm: Float32[Tensor, "batch k w h 1"] = torch.sum(d * d, dim=-1, keepdim=True)
            d_norm = 1 / (1 + d_norm)

            atom_mask_queries: Bool[Tensor, "batch k w 1"] = atom_mask.view(batch, k, w, 1)
            atom_mask_keys: Bool[Tensor, "batch k 1 h"] = (
                to_keys(atom_mask.unsqueeze(-1).float()).view(batch, k, 1, h).bool()
            )
            atom_uid_queries: Int64[Tensor, "batch k w 1"] = atom_uid.view(batch, k, w, 1)
            atom_uid_keys: Int64[Tensor, "batch k 1 h"] = (
                to_keys(atom_uid.unsqueeze(-1).float()).view(batch, k, 1, h).long()
            )
            v: Float32[Tensor, "batch k w h 1"] = (
                (atom_mask_queries & atom_mask_keys & (atom_uid_queries == atom_uid_keys)).float().unsqueeze(-1)
            )

            # Next 3 multiplications are: "batch k w h atom_z" * "batch k w h 1" -> "batch k w h atom_z"
            p: Float32[Tensor, "batch k w h atom_z=16"] = self.embed_atompair_ref_pos(d) * v
            p = p + self.embed_atompair_ref_dist(d_norm) * v
            p = p + self.embed_atompair_mask(v) * v

            q: Float32[Tensor, "batch num_atoms atom_s"] = c

            if self.structure_prediction:
                # run only in structure model not in initial encoding
                atom_to_token: Float32[Tensor, "batch num_atoms num_tokens"] = feats["atom_to_token"].float()

                # s_trunk shape:  (batch, num_tokens, token_s")
                s_to_c: Float32[Tensor, "batch num_tokens atom_s"] = self.s_to_c_trans(s_trunk)
                # (batch, num_atoms, num_tokens) @ # (batch, num_tokens, atom_s) -> (batch, num_atoms, atom_s)
                s_to_c: Float32[Tensor, "batch num_atoms atom_s"] = torch.bmm(atom_to_token, s_to_c)
                c = c + s_to_c

                atom_to_token_queries: Float32[Tensor, "batch k w num_tokens"] = atom_to_token.view(
                    batch, k, w, atom_to_token.shape[-1]
                )
                atom_to_token_keys: Float32[Tensor, "batch k h atom_s"] = to_keys(atom_to_token)
                z_to_p: Float32[Tensor, "batch num_tokens num_tokens atom_z"] = self.z_to_p_trans(z)

                # "batch num_tokens num_tokens atom_z", "batch k w num_tokens", "batch k h atom_s" -> "batch k w h atom_z"
                z_to_p: Float32[Tensor, "batch k w h atom_z=16"] = torch.einsum(
                    "bijd,bwki,bwlj->bwkld",
                    z_to_p,
                    atom_to_token_queries,
                    atom_to_token_keys,
                )
                p = p + z_to_p

            p = p + self.c_to_p_trans_q(c.view(batch, k, w, 1, c.shape[-1]))
            p = p + self.c_to_p_trans_k(to_keys(c).view(batch, k, 1, h, c.shape[-1]))
            p = p + self.p_mlp(p)

            if model_cache is not None:
                layer_cache["q"] = q
                layer_cache["c"] = c
                layer_cache["p"] = p
                layer_cache["to_keys"] = to_keys

        else:
            q = layer_cache["q"]
            c = layer_cache["c"]
            p = layer_cache["p"]
            to_keys = layer_cache["to_keys"]

        if self.structure_prediction:
            # only here the multiplicity kicks in because we use the different positions r
            q = q.repeat_interleave(multiplicity, 0)
            r_input = torch.cat(
                [r, torch.zeros((batch * multiplicity, n, 7)).to(r)],
                dim=-1,
            )
            r_to_q = self.r_to_q_trans(r_input)
            q = q + r_to_q

        c = c.repeat_interleave(multiplicity, 0)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        q = self.atom_encoder(
            q=q,
            mask=atom_mask,
            c=c,
            p=p,
            multiplicity=multiplicity,
            to_keys=to_keys,
            model_cache=layer_cache,
        )

        q_to_a = self.atom_to_token_trans(q)
        atom_to_token = feats["atom_to_token"].float()
        atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)
        atom_to_token_mean = atom_to_token / (atom_to_token.sum(dim=1, keepdim=True) + 1e-6)
        a = torch.bmm(atom_to_token_mean.transpose(1, 2), q_to_a)

        return a, q, c, p, to_keys


class AtomAttentionDecoder(Module):
    """Atom attention decoder."""

    def __init__(
        self,
        atom_s,
        atom_z,
        token_s,
        attn_window_queries,
        attn_window_keys,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        activation_checkpointing=False,
    ):
        """Initialize the atom attention decoder.

        Parameters
        ----------
        atom_s : int
            The atom single representation dimension.
        atom_z : int
            The atom pair representation dimension.
        token_s : int
            The single representation dimension.
        attn_window_queries : int
            The number of atoms per window for queries.
        attn_window_keys : int
            The number of atoms per window for keys.
        atom_decoder_depth : int, optional
            The number of transformer layers, by default 3.
        atom_decoder_heads : int, optional
            The number of transformer heads, by default 4.
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False.

        """
        super().__init__()

        self.a_to_q_trans = LinearNoBias(2 * token_s, atom_s)
        init.final_init_(self.a_to_q_trans.weight)

        self.atom_decoder = AtomTransformer(
            dim=atom_s,
            dim_single_cond=atom_s,
            dim_pairwise=atom_z,
            attn_window_queries=attn_window_queries,
            attn_window_keys=attn_window_keys,
            depth=atom_decoder_depth,
            heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
        )

        self.atom_feat_to_atom_pos_update = nn.Sequential(nn.LayerNorm(atom_s), LinearNoBias(atom_s, 3))
        init.final_init_(self.atom_feat_to_atom_pos_update[1].weight)

    def forward(
        self,
        a,
        q,
        c,
        p,
        feats,
        to_keys,
        multiplicity=1,
        model_cache=None,
    ):
        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        atom_to_token = feats["atom_to_token"].float()
        atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)

        a_to_q = self.a_to_q_trans(a)
        a_to_q = torch.bmm(atom_to_token, a_to_q)
        q = q + a_to_q

        layer_cache = None
        if model_cache is not None:
            cache_prefix = "atomdecoder"
            if cache_prefix not in model_cache:
                model_cache[cache_prefix] = {}
            layer_cache = model_cache[cache_prefix]

        q = self.atom_decoder(
            q=q,
            mask=atom_mask,
            c=c,
            p=p,
            multiplicity=multiplicity,
            to_keys=to_keys,
            model_cache=layer_cache,
        )

        r_update = self.atom_feat_to_atom_pos_update(q)
        return r_update
