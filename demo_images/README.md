# Demo images

The 24 small PNG images used by the live training pane of the demo (`streamlit run app.py`).
They are a 3-train / 3-test subset of CIFAR-10 for each of `trucks`, `dogs`, `cats` and `cars`
(automobile is mapped to `cars`).

Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images.*
https://www.cs.toronto.edu/~kriz/cifar.html

```
demo_images/
  train_subset/<class>/   # 3 PNGs per class
  test_subset/<class>/    # 3 PNGs per class
```
