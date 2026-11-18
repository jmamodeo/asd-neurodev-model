"""ASD neurodevelopment model: a NumPy neural network for the effects of elevated Wnt and CCL5 signaling on learning.

Modules: neural_network (the model), network_tuning (release maps, sigmoid, softmax), visual_input (image loading),
loss_outputs (the loss plot and table written by main), experiment / paper_analysis / model_figures / reproduce (the paper's
runs, statistics and figures). Run one network with `python -m asd_model`; re-create the paper's figures and tables with
`python -m asd_model.reproduce` (or `python reproduce.py` in the repository root).
"""

__version__ = "1.0.0"
