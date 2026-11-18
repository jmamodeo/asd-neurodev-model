# Pre-specified protocol (written before any corrected-network run; Nov 17 2026, ET)

Design: the paper's 4 conditions form a 2x2: Wnt (dWnt 0 / 0.41859) x CCL5 (dCCL5 0 / 0.40467). Original code formulas kept
(layer widths x(1+dWnt) compounding; release map Poisson(100(1+dCCL5))/100; N(0,1) init; global grad-norm clip 1.0; lr 0.1;
4 hidden layers; image vector 900; 100 sessions (=epochs); per-image SGD).
Datasets: (1) John's original image set (12 train / 12 held-out test images, 3 per class), (2) CIFAR-10 subset (4 classes, 3 per class train/test,
the same "cifar3" set the earlier analysis used). 20 seeds (0..19) per condition per dataset, same seed set in all conditions.
Matched seeds: seed s gives the same shuffle sequence in all 4 conditions; control and CCL5 additionally share the same initial weights
(Wnt changes layer sizes so initial weights cannot match).

Outcomes (per run):
 - Training loss per epoch = mean cross-entropy per example over that epoch's shuffled pass (as in the original train()).
 - Held-out loss and accuracy after each epoch on the test set (release maps on, same distribution as training).
 - PRIMARY learning-speed outcome: AUC = mean training loss over epochs 1..100 (area under the loss curve / 100). Lower = faster/better.
 - Epochs-to-threshold: first epoch with training loss <= 50% of that run's epoch-1 training loss (relative threshold; PRIMARY).
   Runs that never reach it are right-censored at 101 (and the proportion reaching is reported).
   Secondary absolute threshold: first epoch with training loss <= ln(4)=1.386 (chance-level cross-entropy).
 - Final loss = mean training loss over epochs 91-100; final held-out loss/accuracy after epoch 100.
 - Held-out accuracy (argmax of output layer vs label) compared to chance 25%.

Statistics (fixed before running), done separately per dataset on the 20 seeds x 4 conditions = 80 runs:
 1. Primary: two-way fixed-effects ANOVA (statsmodels OLS, type II SS) with Wnt, CCL5, Wnt:CCL5 on AUC and on epochs-to-threshold
    (alpha 0.05, two-sided); partial eta-squared effect sizes. Sensitivity: the same on log(AUC); mixed model with seed random
    intercept; permutation test (10,000 shuffles of condition labels within seed) for the interaction.
 2. Directional replication of the paper's test: Welch one-tailed t-test (control < perturbed) of AUC and of final loss for each
    perturbed condition vs control, Holm-adjusted over the 3 contrasts; Cohen's d with bootstrap 95% CI.
 3. Accuracy: per condition, one-sample t-test of per-seed held-out accuracy against 25%; two-way ANOVA on accuracy as secondary.
 4. Datasets are analysed separately (no pooling); no outcomes beyond those listed are used for conclusions.

## Addendum (written 22:58 ET, after a 4-run pilot at seed 0 showed that with the original lr=0.1 the corrected (shuffled, softmax) network
oscillates at loss ~10-17 and stays at chance accuracy in both control and CCL5, i.e. it does not learn):
 - Primary configuration = the original hyperparameters (lr 0.1), 20 seeds, as pre-specified above.
 - Secondary configuration ("lr-pilot"): same everything but the learning rate, chosen from a CONTROL-ONLY pilot (seeds 100-102, both datasets, lr in {0.1, 0.03, 0.01, 0.003, 0.001}),
   pick = the lr with the lowest mean final training loss (epochs 91-100) of the control network. No perturbed network is used to choose it. Then run all 4 conditions x 20 seeds x 2 datasets at that lr, same statistics.

## Addendum 2 (23:30 ET): lr pilot result (control-only, seeds 100-102, mean final train loss orig/cifar3):
 lr 0.1: 11.97/10.43; 0.03: 0.151/0.031; 0.01: 0.201/0.372; 0.003: 1.12/1.52; 0.001: 5.27/5.69  -> pre-specified rule selects lr = 0.03.
 Secondary configuration "lr0.03" = all four conditions x 20 seeds x 2 datasets at lr 0.03; all else identical; same statistics. Reported as a deviation/sensitivity analysis, clearly labelled.
