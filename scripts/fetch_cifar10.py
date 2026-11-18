"""Download CIFAR-10 and write a small PNG subset in the layout main.py expects."""

import argparse
import pickle
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
LABEL_TO_CLASS = {1: "cars", 3: "cats", 5: "dogs", 9: "trucks"}  # automobile, cat, dog, truck
SPLITS = {"train": "data_batch_1", "test": "test_batch"}
ROOT = Path(__file__).resolve().parent.parent


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def download(cache):
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "cifar-10-python.tar.gz"
    if archive.exists():
        print(f"Using cached {archive}")
    else:
        print(f"Downloading {URL} (about 163 MB)")
        # download to a temp name so an interrupted download is never reused
        part = archive.with_name(archive.name + ".part")
        urllib.request.urlretrieve(URL, part)
        part.replace(archive)
    return archive


def load_batch(archive, name):
    with tarfile.open(archive) as tar:
        member = tar.extractfile(f"cifar-10-batches-py/{name}")
        return pickle.load(member, encoding="bytes")


def write_split(batch, split, per_class, out):
    data, labels = batch[b"data"], np.array(batch[b"labels"])
    counts = {}
    for label, class_name in LABEL_TO_CLASS.items():
        class_dir = out / f"{split}_subset" / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        indices = np.where(labels == label)[0][:per_class]
        for n, index in enumerate(indices):
            rgb = data[index].reshape(3, 32, 32).transpose(1, 2, 0)
            bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(class_dir / f"cifar10_{split}_{n}.png"), bgr)
        counts[class_name] = len(indices)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--train-per-class", type=int, default=25)
    parser.add_argument("--test-per-class", type=int, default=10)
    parser.add_argument("--out", default="image_set")
    parser.add_argument("--cache", default=".cache")
    args = parser.parse_args()

    out = resolve(args.out)
    archive = download(resolve(args.cache))
    per_class = {"train": args.train_per_class, "test": args.test_per_class}
    for split, batch_name in SPLITS.items():
        counts = write_split(load_batch(archive, batch_name), split, per_class[split], out)
        print(f"{split}: {counts} -> {out / (split + '_subset')}")


if __name__ == "__main__":
    main()
