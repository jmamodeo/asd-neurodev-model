# image_set

Images are not included in this repository. `main.py` expects:

```
image_set/
  train_subset/<class>/   # training images
  test_subset/<class>/    # testing images
```

Class folder names must match `class_list` in `main.py` (default: `trucks`, `dogs`, `cats`, `cars`).
Run `python scripts/fetch_cifar10.py` from the repository root to fill this layout with a CIFAR-10 subset,
or add your own images. Keep only image files in the class folders.
