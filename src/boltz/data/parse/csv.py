from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from boltz.data import const
from boltz.data.types import MSA, MSADeletion, MSAResidue, MSASequence


def parse_csv(
    path: Path,
    max_seqs: Optional[int] = None,
) -> MSA:
    """Processes an A3M-like CSV file into an MSA object.

    Args:
        path: The path to the A3M-like CSV file (can be gzipped). In this file, alignments are shown with
             inserts as lower case characters, matches as upper case characters, deletions as ' - ',
             and gaps aligned to inserts as ' . '
        max_seqs: The maximum number of sequences to parse from the file. Defaults to None,
         meaning all sequences are parsed.

    Returns:
        The constructed MSA object.
    """
    data = pd.read_csv(path)

    # Check columns
    if set(data.columns) != {"key", "sequence"}:
        raise ValueError("Invalid CSV format, expected columns: ['sequence', 'key']")

    # Create taxonomy mapping
    visited = set()
    sequences = []
    deletions = []
    residues = []

    seq_idx = 0
    for line, key in zip(data["sequence"], data["key"]):
        line: str = line.strip()
        if not line:
            continue

        # Get taxonomy, if annotated
        taxonomy_id = -1
        if (key is not None) and (key != "") and (str(key) != "nan"):
            taxonomy_id = key

        # Skip if duplicate sequence
        str_seq = line.replace("-", "").upper()
        if str_seq in visited:
            continue

        visited.add(str_seq)

        # Process sequence
        residue = []
        deletion = []
        count = 0
        res_idx = 0
        for c in line:
            if c != "-" and c.islower():
                count += 1
                continue
            token = const.prot_letter_to_token[c]
            token = const.token_ids[token]
            residue.append(token)
            if count > 0:
                deletion.append((res_idx, count))
                count = 0
            res_idx += 1

        res_start = len(residues)
        res_end = res_start + len(residue)

        del_start = len(deletions)
        del_end = del_start + len(deletion)

        sequences.append((seq_idx, taxonomy_id, res_start, res_end, del_start, del_end))
        residues.extend(residue)
        deletions.extend(deletion)

        seq_idx += 1
        if (max_seqs is not None) and (seq_idx >= max_seqs):
            break

    # Create MSA object
    msa = MSA(
        residues=np.array(residues, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )
    return msa
