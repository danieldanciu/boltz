from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from boltz.data import const
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.pad import pad_to_max
from boltz.data.tokenize.boltz import BoltzTokenizer
from boltz.data.types import (
    MSA,
    Connection,
    Input,
    Manifest,
    Record,
    ResidueConstraints,
    Structure,
)


def load_input(
    record: Record,
    target_dir: Path,
    msa_dir: Path,
    constraints_dir: Optional[Path] = None,
) -> Input:
    """Loads the given input data.

    Args:
        record: The record to load.
        target_dir: The path to the data directory.
        msa_dir: The path to the MSA directory.
        constraints_dir: The path to the residue constraints directory.

    Returns:
        The loaded input.
    """
    # Load the structure
    structure = np.load(target_dir / f"{record.id}.npz")
    structure = Structure(
        atoms=structure["atoms"],
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=structure["chains"],
        connections=structure["connections"].astype(Connection),
        interfaces=structure["interfaces"],
        mask=structure["mask"],
    )

    msas = {}
    for chain in record.chains:
        msa_id = chain.msa_id
        # Load the MSA for this chain, if any
        if msa_id != -1:
            msa = np.load(msa_dir / f"{msa_id}.npz")
            msas[chain.chain_id] = MSA(**msa)

    residue_constraints = None
    if constraints_dir is not None:
        residue_constraints = ResidueConstraints.load(constraints_dir / f"{record.id}.npz")

    return Input(structure, msas, record, residue_constraints)


def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The data to collate.

    Returns:
    -------
    Dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in {
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "record",
        }:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


class PredictionDataset(torch.utils.data.Dataset):
    """Dataset for preparing data for Boltz-1 inference.

    Loads molecular records, tokenizes their structures, and computes
    a comprehensive set of features (token, atom, MSA, and optional symmetries/constraints)
    suitable for feeding into the Boltz-1 model. It handles potential loading and
    featurization errors by skipping problematic records.
    """

    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        constraints_dir: Optional[Path] = None,
    ) -> None:
        """Initializes the prediction dataset.

        Args:
            manifest: The manifest object containing a list of records (molecular IDs and associated metadata) to
                be processed.
            target_dir: The path to the directory containing the molecular structure files
            msa_dir: The path to the directory containing the Multiple Sequence Alignment (MSA) files
            constraints_dir: An optional path to the directory containing residue constraint files
        """
        super().__init__()
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir
        self.constraints_dir = constraints_dir
        self.tokenizer = BoltzTokenizer()
        self.featurizer = BoltzFeaturizer()

    def __getitem__(self, idx: int) -> dict:
        """Gets a processed feature dictionary for a single record from the dataset.

        This method loads a record, tokenizes its associated structure, applies
        featurization (including handling inference-specific pocket constraints),
        and returns a dictionary of prepared tensors suitable for model input.
        Skips to the next record if any step fails.

        Args:
            idx: The integer index of the record to retrieve from the manifest.

        Returns:
            A dictionary containing the sampled and computed data features (tensors)
            for the specified record, along with the original record object.
        """
        # Get a sample from the dataset
        record = self.manifest.records[idx]

        # Get the structure
        try:
            input_data = load_input(
                record,
                self.target_dir,
                self.msa_dir,
                self.constraints_dir,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Failed to load input for {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Tokenize structure
        try:
            tokenized = self.tokenizer.tokenize(input_data)
        except Exception as e:  # noqa: BLE001
            print(f"Tokenizer failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Inference specific options
        options = record.inference_options
        if options is None or len(options.pocket_constraints) == 0:
            binder, pocket = None, None
        else:
            binder, pocket = options.pocket_constraints[0][0], options.pocket_constraints[0][1]

        # Compute features
        try:
            features = self.featurizer.process(
                tokenized,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=const.max_msa_seqs,
                pad_to_max_seqs=False,
                symmetries={},
                compute_symmetries=False,
                inference_binder=binder,
                inference_pocket=pocket,
                compute_constraint_features=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        features["record"] = record
        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
        -------
        int
            The length of the dataset.

        """
        return len(self.manifest.records)


class BoltzInferenceDataModule(pl.LightningDataModule):
    """DataModule for Boltz inference."""

    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        num_workers: int,
        constraints_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        config : DataConfig
            The data configuration.

        """
        super().__init__()
        self.num_workers = num_workers
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir
        self.constraints_dir = constraints_dir

    def predict_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns:
        -------
        DataLoader
            The training dataloader.

        """
        dataset = PredictionDataset(
            manifest=self.manifest,
            target_dir=self.target_dir,
            msa_dir=self.msa_dir,
            constraints_dir=self.constraints_dir,
        )
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate,
        )

    @staticmethod
    def transfer_batch_to_device(
        batch: dict,
        device: torch.device,
        dataloader_idx: int,
    ) -> dict:
        """Transfer a batch to the given device.

        Parameters
        ----------
        batch : Dict
            The batch to transfer.
        device : torch.device
            The device to transfer to.
        dataloader_idx : int
            The dataloader index.

        Returns:
        -------
        np.Any
            The transferred batch.

        """
        del dataloader_idx
        for key, value in batch.items():
            if key not in {
                "all_coords",
                "all_resolved_mask",
                "crop_to_all_atom_map",
                "chain_symmetries",
                "amino_acids_symmetries",
                "ligand_symmetries",
                "record",
            }:
                batch[key] = value.to(device)
        return batch
