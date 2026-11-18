# Methods-to-code map

Where each step described in the paper's Methods (`docs/paper/paper.md`) lives in the code. Function and file names are used instead of line numbers so the map stays valid as the code is edited. All modules are in `src/asd_model/` (for example `visual_input.image_to_matrix()` is `src/asd_model/visual_input.py`).

## The model

| Paper (Methods) | Code | Notes |
|---|---|---|
| Visual input: grayscale, resize to 30 x 30, scale to 0-1, flatten to 900 x 1 | `visual_input.image_to_matrix()` | OpenCV `imread`, `cvtColor`, `resize`; side length is `sqrt(image_vector_size)` |
| One-hot encoded labels | `visual_input.encode_image_labels()` | One column vector of 0s and a single 1 per class |
| Layer sizes; all hidden layers start equal to the input layer | `NeuralNetwork.__init__()` (`self.layer_sizes`) | 900 inputs, 4 hidden layers, 4 outputs in the paper |
| Wnt: each hidden layer gains `int(previous_size * wnt_delta)` nodes | `NeuralNetwork.__init__()` (`additional_nodes`) | `wnt_delta` = 0.41859 gives 900, 1276, 1810, 2567, 3641 (drawn in Figure 1) |
| Weights and biases drawn from a standard normal distribution | `NeuralNetwork.__init__()` (`np.random.randn`) | |
| CCL5: stochastic release maps, Poisson(100 (1 + `ccl5_delta`)) / 100 | `network_tuning.get_release_maps()` | `ccl5_delta` = 0.40467 in the paper; drawn in Figure 2 |
| Release maps multiply the weights in the forward pass | `NeuralNetwork.feedforward()` | The sampled maps are kept in `self.last_release_maps` |
| Sigmoid hidden layers, softmax output | `network_tuning.apply_sigmoid()`, `network_tuning.apply_softmax()`, used in `NeuralNetwork.feedforward()` | |
| Cross-entropy loss; the same release maps in the backward pass | `NeuralNetwork.feedback()` | Output delta is `p - label`; gradients flow through the effective weights `W * M` |
| Gradient clipping (global norm 1.0) | `NeuralNetwork.feedback()` (`max_gradient_norm`) | |
| One session = one pass over all images in shuffled, class-mixed order | `NeuralNetwork.run_session()` | `np.random.permutation` |
| Training for a number of sessions; mean loss per training example | `NeuralNetwork.train()` | Default 100 sessions; learning rate 0.03 (`DEFAULT_LEARNING_RATE`). Returns a `TrainingHistory`; writes no files |
| Held-out accuracy (argmax of the output vs the label) and loss | `NeuralNetwork.evaluate()`, `NeuralNetwork.test()` | Chance is 25% with four classes |
| Loss plot and spreadsheet from `main.py` | `loss_outputs.plot_loss_over_time()`, `loss_outputs.export_loss_table()` | Written to `results/<date> results/` by `main.py` after `train()` returns |

## The experiment and analysis

| Paper (Methods and Results) | Code | Notes |
|---|---|---|
| 2 x 2 design: control, CCL5, Wnt, Wnt + CCL5 | `experiment.CONDITIONS` | |
| 20 independently seeded networks per condition and image set, same seeds in all conditions | `experiment.run_experiment()`, `reproduce.collect_runs()` | `np.random.seed(seed)` sets the initial weights, shuffle order and release maps |
| Two engineering choices for speed (float32; release maps drawn from a pre-generated pool) | `experiment.py` module docstring, `experiment.make_pooled_sampler()` | Same distribution as `get_release_maps()`, applied identically to every condition |
| Image sets: the authors' images and the CIFAR-10 subset | `scripts/fetch_cifar10.py` (CIFAR-10), `--original-images DIR` in `reproduce.py` (authors' images) | The authors' images are not redistributed; the shipped runs include them |
| Shipped finished runs | `reproduce_data/runs_lr0.03.json.gz`, `runs_lr0.1.json.gz` | Written by `scripts/export_run_bundle.py` |
| Per-run measures: AUC, final loss, sessions to half of the first loss, sessions to ln 4, held-out accuracy and loss | `paper_analysis.per_run_table()` | `LN4` is the cross-entropy of a uniform guess over four classes |
| Two-way ANOVA with partial eta squared | `paper_analysis._anova()` | `statsmodels`, type II |
| Welch t-tests against the control, Holm correction, Cohen's d with bootstrap CI | `paper_analysis.compute_statistics()`, `_holm_correction()`, `_cohens_d()` | |
| Interaction checked with a permutation test within seeds | `paper_analysis._permutation_interaction()` | |
| Observed vs additive effect with 95% bootstrap CIs | `paper_analysis.observed_vs_additive()` | Resamples seeds |
| Held-out accuracy against chance (25%) | `paper_analysis.compute_statistics()` (the `"accuracy"` entry) | One-sample t-tests |
| Tables 1-3 | `paper_analysis.markdown_tables()`, `paper_analysis.write_tables()` | Written to `tables/` |
| Figures 1-6 and Supplementary Figures S1-S2 | `model_figures.py` (Figures 1 and 2), `paper_analysis.py` (Figures 3-6, S1-S2), driven by `reproduce.py` | One script per figure in `scripts/figures/` |
| Pre-specified analysis plan | `docs/paper/analysis_plan.md` | |
| Numbers in the paper text | `docs/paper/build_paper.py` | Fills placeholders from `tables/` and builds `paper.pdf` |

## Checks

`tests/test_model_learning_rules.py` checks the model's learning rules (softmax output, release maps in the backward pass, shuffling, accuracy from the output layer). `tests/test_reproduce.py` checks that the shipped runs reproduce the paper's numbers and that runs are deterministic. `tests/test_main_outputs.py` checks that a short training run writes the plot and spreadsheet and that `python -m asd_model` starts.
