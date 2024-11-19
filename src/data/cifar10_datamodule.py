from typing import Tuple

import torch
from src.data.federated_datamodule import FederatedDataModule


class CIFAR10DataModule(FederatedDataModule):
    """
    """

    def __init__(
        self,
        num_clients: int = 10,
        train_test_split: Tuple[float, float] = (0.8, 0.2),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """
        """
        super().__init__(
            dataset="uoft-cs/cifar10",
            num_clients=num_clients,
            train_test_split=train_test_split,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    @property
    def num_classes(self) -> int:
        """
        Get the number of classes for CIFAR-10.

        Returns
        -------
        int
            The number of classes for CIFAR-10.
        """
        return 10

    def transform_batch(self, batch: dict) -> dict:
        """
        Apply the transforms to the batch for CIFAR-10.

        Parameters
        ----------
        batch : dict
            The batch to transform.

        -------
        dict
            The transformed batch.
        """
        return {
            "pixel_values": torch.stack([self.transforms(img) for img in batch["img"]]),
            "labels": torch.tensor(batch["label"]),
        }


if __name__ == "__main__":
    _ = CIFAR10DataModule()
