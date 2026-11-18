import cv2
import numpy as np
import pytest

CLASS_NAMES = ("trucks", "dogs", "cats", "cars")


@pytest.fixture
def write_class_images():
    """Return a function that writes random 32 x 32 PNG images into <folder>/<class>/<i>.png for the four classes."""

    def write(folder, per_class=2):
        rng = np.random.default_rng(0)
        for class_name in CLASS_NAMES:
            class_folder = folder / class_name
            class_folder.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                image = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
                cv2.imwrite(str(class_folder / f"{i}.png"), image)

    return write
