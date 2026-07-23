import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class IndexedImageDataset(Dataset):
    """Apply an independent transform to a fixed list of ImageFolder samples."""

    def __init__(self, samples, classes, indices, transform):
        self.samples = samples
        self.classes = classes
        self.indices = list(map(int, indices))
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        path, label = self.samples[self.indices[position]]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


def _create_split(targets, test_fraction, validation_fraction_of_train, seed):
    indices = np.arange(len(targets))
    train_val, test = train_test_split(
        indices, test_size=test_fraction, stratify=targets, random_state=seed
    )
    train, validation = train_test_split(
        train_val,
        test_size=validation_fraction_of_train,
        stratify=np.asarray(targets)[train_val],
        random_state=seed,
    )
    return {
        "seed": seed,
        "train": train.tolist(),
        "validation": validation.tolist(),
        "test": test.tolist(),
    }


def load_or_create_split(dataset, split_file, test_fraction, validation_fraction, seed):
    split_path = Path(split_file)
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        split = _create_split(dataset.targets, test_fraction, validation_fraction, seed)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")

    combined = split["train"] + split["validation"] + split["test"]
    if len(combined) != len(dataset) or len(set(combined)) != len(dataset):
        raise ValueError("Saved split is not a disjoint partition of the dataset")
    if max(combined, default=-1) >= len(dataset):
        raise ValueError("Saved split does not match the current dataset")
    return split


def create_dataloaders(config, model_name, device):
    source = datasets.ImageFolder(config["data_dir"])
    image_size = 128 if model_name == "cnn" else 224
    batch_size = (
        config["cnn_batch_size"] if model_name == "cnn" else config["transformer_batch_size"]
    )
    train_transform, evaluation_transform = build_transforms(image_size)
    split = load_or_create_split(
        source,
        config["split_file"],
        config["test_fraction"],
        config["validation_fraction_of_train"],
        config["seed"],
    )
    datasets_by_split = {
        "train": IndexedImageDataset(source.samples, source.classes, split["train"], train_transform),
        "validation": IndexedImageDataset(
            source.samples, source.classes, split["validation"], evaluation_transform
        ),
        "test": IndexedImageDataset(source.samples, source.classes, split["test"], evaluation_transform),
    }
    generator = torch.Generator().manual_seed(config["seed"])
    loaders = {
        name: DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=config["num_workers"],
            pin_memory=device.type == "cuda",
            generator=generator if name == "train" else None,
        )
        for name, subset in datasets_by_split.items()
    }
    return loaders, source.classes

