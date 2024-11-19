import pytest
from src.data import CIFAR10DataModule


PARAMS = ["batch_size", "num_clients"]
VALUES = [
    (1, 1), (1, 10), (32, 1), (32, 10)
]


@pytest.mark.parametrize(PARAMS, VALUES)
def test_cifar10_datamodule(batch_size: int, num_clients: int) -> None:
    """
    """
    train_test_split = (0.8, 0.2)
    num_workers = 0

    dm = CIFAR10DataModule(
        num_clients=num_clients,
        train_test_split=train_test_split,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    assert dm.num_classes == 10
    assert dm.dataset is None

    dm.setup(partition_id=0)
    assert dm.dataset is not None
    assert dm.train_dataloader() is not None and dm.test_dataloader() is not None

    # num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    # assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    assert len(batch["pixel_values"]) == batch_size
    assert len(batch["labels"]) == batch_size
