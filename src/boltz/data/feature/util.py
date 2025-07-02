import numba
import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float, Int64
from numba import types
from torch import Tensor

from boltz.data import const
from boltz.data.types import (
    MSA,
    MSADeletion,
    MSAResidue,
    MSASequence,
)

deletions_dict_type = types.DictType(types.UniTuple(types.int64, 3), types.int64)


def compute_collinear_mask(v1, v2):
    norm1 = np.linalg.norm(v1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(v2, axis=1, keepdims=True)
    v1 = v1 / (norm1 + 1e-6)
    v2 = v2 / (norm2 + 1e-6)
    mask_angle = np.abs(np.sum(v1 * v2, axis=1)) < 0.9063
    mask_overlap1 = norm1.reshape(-1) > 1e-2
    mask_overlap2 = norm2.reshape(-1) > 1e-2
    return mask_angle & mask_overlap1 & mask_overlap2


def dummy_msa(residues: np.ndarray) -> MSA:
    """Create a dummy MSA for a chain that comes with no MSA.

    Args:
        residues: the residues for the chain.

    Returns:
        A dummy MSA, containing the residues as the only sequence and no deletions.
    """
    residues = [res["res_type"] for res in residues]
    deletions = []
    sequences = [(0, -1, 0, len(residues), 0, 0)]
    return MSA(
        residues=np.array(residues, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )


def prepare_msa_arrays(
    tokens,
    pairing: list[dict[int, int]],
    is_paired: list[dict[int, int]],
    deletions: dict[tuple[int, int, int], int],
    msa: dict[int, MSA],
) -> tuple[Int64[Tensor, "n_tokens n_pairs"], Float[Tensor, "n_tokens n_pairs"], Float[Tensor, "n_tokens n_pairs"]]:
    """Reshape the MSA data to play nicely with numba jit.

    Args:
        tokens: list of tokens, each token is a dict with "asym_id" and "res_idx".
        pairing: a list where each element is a dictionary mapping `chain_id` to the seq_idx
            (index within that chain's original MSA) that will be used for that row in the combined MSA
        is_paired: a list with the same length as `pairing`, where each element is a dictionary mapping `chain_id` to:
           - 1, if the sequence for that chain in this row is "paired" or
           - 0 if it's an "unpaired" sequence or a gap
        deletions: dictionary mapping (chain_id, seq_idx, res_idx) to deletion value.
        msa: dictionary mapping chain IDs to MSA objects.
        deletions: map of (chain_id, seq_idx, res_idx) to number of deletions.

    """
    token_asym_ids_arr = np.array([t["asym_id"] for t in tokens], dtype=np.int64)
    token_res_idxs_arr = np.array([t["res_idx"] for t in tokens], dtype=np.int64)

    chain_ids = sorted(msa.keys())

    # chain_ids are not necessarily contiguous (e.g. they might be 0, 24, 25).
    # This allows us to look up a chain_id by it's index in the chain_ids list.
    chain_id_to_idx = {chain_id: i for i, chain_id in enumerate(chain_ids)}
    token_asym_ids_idx_arr = np.array([chain_id_to_idx[asym_id] for asym_id in token_asym_ids_arr], dtype=np.int64)

    pairing_arr = np.zeros((len(pairing), len(chain_ids)), dtype=np.int64)
    is_paired_arr = np.zeros((len(is_paired), len(chain_ids)), dtype=np.int64)

    for i, row_pairing in enumerate(pairing):
        for chain_id in chain_ids:
            pairing_arr[i, chain_id_to_idx[chain_id]] = row_pairing[chain_id]

    for i, row_is_paired in enumerate(is_paired):
        for chain_id in chain_ids:
            is_paired_arr[i, chain_id_to_idx[chain_id]] = row_is_paired[chain_id]

    max_seq_len = max(len(msa[chain_id].sequences) for chain_id in chain_ids)

    # we want res_start from sequences
    msa_sequences = np.full((len(chain_ids), max_seq_len), -1, dtype=np.int64)
    for chain_id in chain_ids:
        for i, seq in enumerate(msa[chain_id].sequences):
            msa_sequences[chain_id_to_idx[chain_id], i] = seq["res_start"]

    max_residues_len = max(len(msa[chain_id].residues) for chain_id in chain_ids)
    msa_residues = np.full((len(chain_ids), max_residues_len), -1, dtype=np.int64)
    for chain_id in chain_ids:
        residues = msa[chain_id].residues.astype(np.int64)
        idxs = np.arange(len(residues))
        chain_idx = chain_id_to_idx[chain_id]
        msa_residues[chain_idx, idxs] = residues

    deletions_dict = numba.typed.Dict.empty(
        key_type=numba.types.Tuple([numba.types.int64, numba.types.int64, numba.types.int64]),
        value_type=numba.types.int64,
    )
    deletions_dict.update(deletions)

    msa_data, del_data, paired_data = _prepare_msa_arrays_inner(
        token_asym_ids_arr,
        token_res_idxs_arr,
        token_asym_ids_idx_arr,
        pairing_arr,
        is_paired_arr,
        deletions_dict,
        msa_sequences,
        msa_residues,
        const.token_ids["-"],
    )
    msa_data_t = torch.tensor(msa_data, dtype=torch.long)
    del_data_t = torch.tensor(del_data, dtype=torch.float)
    paired_data_t = torch.tensor(paired_data, dtype=torch.float)
    return msa_data_t, del_data_t, paired_data_t


@numba.njit(
    [
        types.Tuple(
            (
                types.int64[:, ::1],  # msa_data
                types.int64[:, ::1],  # del_data
                types.int64[:, ::1],  # paired_data
            )
        )(
            types.int64[::1],  # token_asym_ids
            types.int64[::1],  # token_res_idxs
            types.int64[::1],  # token_asym_ids_idx
            types.int64[:, ::1],  # pairing
            types.int64[:, ::1],  # is_paired
            deletions_dict_type,  # deletions
            types.int64[:, ::1],  # msa_sequences
            types.int64[:, ::1],  # msa_residues
            types.int64,  # gap_token
        )
    ],
    cache=True,
)
def _prepare_msa_arrays_inner(
    token_asym_ids: npt.NDArray[np.int64],
    token_res_idxs: npt.NDArray[np.int64],
    token_asym_ids_idx: npt.NDArray[np.int64],
    pairing: npt.NDArray[np.int64],
    is_paired: npt.NDArray[np.int64],
    deletions: dict[tuple[int, int, int], int],
    msa_sequences: npt.NDArray[np.int64],
    msa_residues: npt.NDArray[np.int64],
    gap_token: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    n_tokens = len(token_asym_ids)
    n_pairs = len(pairing)
    msa_data = np.full((n_tokens, n_pairs), gap_token, dtype=np.int64)
    paired_data = np.zeros((n_tokens, n_pairs), dtype=np.int64)
    del_data = np.zeros((n_tokens, n_pairs), dtype=np.int64)

    # Add all the token MSA data
    for token_idx in range(n_tokens):
        chain_id_idx = token_asym_ids_idx[token_idx]
        chain_id = token_asym_ids[token_idx]
        res_idx = token_res_idxs[token_idx]

        for pair_idx in range(n_pairs):
            seq_idx = pairing[pair_idx, chain_id_idx]
            paired_data[token_idx, pair_idx] = is_paired[pair_idx, chain_id_idx]

            # Add residue type
            if seq_idx != -1:
                res_start = msa_sequences[chain_id_idx, seq_idx]
                res_type = msa_residues[chain_id_idx, res_start + res_idx]
                k = (chain_id, seq_idx, res_idx)
                if k in deletions:
                    del_data[token_idx, pair_idx] = deletions[k]
                msa_data[token_idx, pair_idx] = res_type

    return msa_data, del_data, paired_data
