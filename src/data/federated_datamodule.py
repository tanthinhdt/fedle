from typing import Any, Dict, Optional, Tuple

import torch
import torchvision.transforms.v2 as transforms
from flwr_datasets import FederatedDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader


class FederatedDataModule(LightningDataModule):
    """
    """

    def __init__(
        self,
        dataset: str = "uoft-cs/cifar10",
        num_clients: int = 10,
        train_test_split: Tuple[float, float] = (0.8, 0.2),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.dataset: Optional[FederatedDataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """
        Get the number of classes.

        Returns
        -------
        int
            The number of classes.
        """
        raise NotImplementedError("Method 'num_classes' must be implemented.")

    def transform_batch(self, batch: dict) -> dict:
        """Apply the transforms to the batch.

        Parameters
        ----------
        batch : dict
            The batch.

        Returns
        -------
        dict
            The transformed batch.
        """
        raise NotImplementedError("Method 'transform_batch' must be implemented.")

    def setup(
        self,
        partition_id: int,
        stage: Optional[str] = None,
    ) -> None:
        """
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible "
                    f"by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.dataset:
            fds = FederatedDataset(
                dataset=self.hparams.dataset,
                partitioners={
                    "train": self.hparams.num_clients,
                },
            )
            partition = fds.load_partition(partition_id)
            self.dataset = partition.train_test_split(
                test_size=self.hparams.train_test_split[1],
                train_size=self.hparams.train_test_split[0],
            )
            self.dataset = self.dataset.with_transform(self.transform_batch)

    def train_dataloader(self) -> DataLoader[Any]:
        """
        Create and return the train dataloader.

        Returns
        -------
        DataLoader
            The train dataloader.
        """
        return DataLoader(
            dataset=self.dataset["train"],
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """
        Create and return the test dataloader.

        Returns
        -------
        DataLoader
            The test dataloader.
        """
        return DataLoader(
            dataset=self.dataset["test"],
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """
        Create and return the validation dataloader.

        Returns
        -------
        DataLoader
            The validation dataloader.
        """
        return self.test_dataloader()

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        Parameters
        ----------
        stage : Optional[str], optional
            Stage of the teardown process, by default None
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """
        Called when saving a checkpoint. Implement to generate and save the datamodule state.

        Returns
        -------
        Dict[Any, Any]
            The datamodule state.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        Parameters
        ----------
        state_dict : Dict[str, Any]
            The datamodule state.
        """
        pass


if __name__ == "__main__":
    _ = FederatedDataModule()
