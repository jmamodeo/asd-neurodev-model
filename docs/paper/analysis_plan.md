# Analysis plan (specified before the runs)

**Design.** Four conditions forming a 2 x 2: Wnt (ΔWnt 0 or 0.41859) x CCL5 (ΔCCL5 0 or 0.40467). Layer widths are multiplied by (1+ΔWnt) per hidden layer; release maps are Poisson(100(1+ΔCCL5))/100 and multiply the weights in the forward and backward pass. 4 hidden layers, 900 inputs, 100 sessions, per-image SGD, gradient-norm clip 1.0, N(0,1) initial weights, softmax output with cross-entropy loss, images shuffled across classes every session.

**Data.** (1) the authors' image set, (2) a CIFAR-10 subset (automobile, cat, dog, truck; the first 3 training images and first 3 test images of each class). Three training and three held-out test images per class in both sets. 20 seeds (0-19) per condition and image set. Seed s gives the same image order in all four conditions; control and CCL5 also share initial weights.

**Outcomes.** Per session: training loss (mean cross-entropy per image over the session's shuffled pass), held-out loss and accuracy. Per run:
- AUC = mean training loss over sessions 1-100 (primary; lower = faster loss reduction);
- sessions to reach a loss of ln 4 = 1.386 (chance level for four classes) and sessions to reach half of the first-session loss (right-censored at 101);
- final loss (sessions 91-100);
- held-out accuracy after session 100, compared with chance (25%).

**Statistics** (each image set separately):
1. Two-way ANOVA (type II) with Wnt, CCL5 and Wnt x CCL5 on AUC, final loss and sessions-to-threshold, with partial η²; sensitivity analyses: log(AUC), a mixed model with a random intercept for seed, and a within-seed permutation test (10,000 shuffles) of the interaction.
2. Each perturbed condition against the control: one-tailed Welch t-test (control smaller), Holm correction over the three contrasts, Cohen's d with a bootstrap 95% CI.
3. Held-out accuracy: one-sample t-tests against 25% per condition; two-way ANOVA on accuracy.

**Learning rate.** A pilot on control networks only (seeds 100-102, both image sets, learning rates 0.1, 0.03, 0.01, 0.003, 0.001) selected the rate with the lowest final control training loss. Final control training loss (authors' images / CIFAR-10): 0.1: 11.97 / 10.43; 0.03: 0.151 / 0.031; 0.01: 0.201 / 0.372; 0.003: 1.12 / 1.52; 0.001: 5.27 / 5.69. The selected rate, 0.03, is used for all main results; all four conditions were also run at 0.1 and are reported as a supplementary analysis.
