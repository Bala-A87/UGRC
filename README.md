# UGRC
Research work done as part of UGRC-I (CS4900) on the topic 'Feature learning in neural networks' under the guidance of prof. Harish Guruprasad

## Abstract

The goal of the project is to analyze the ability of neural networks to learn useful features/representations from the data, beyond what kernel machines, such as SVMs, can. The neural tangent kernel is a tool that is particularly useful, in order to understand the similarities and shortcomings of SVMs wrt NNs.

## Key experiments

- To understand the validity of the neural tangent kernel, a correlation between the predictions of a neural network and its corresponding NTK-SVM was observed. A weak correlation exists before training the NN, which develops into a strong correlation at the end of training. This was done with the help of a small fraction of the MNIST dataset, with only 0 & 1 classes and modelled as a regression problem rather than classification. [Code](./Code/ntk-mnist.ipynb)
- Probably the most significant and spanning experiment was on analyzing the ability of NNs to learn symmetry, with the help of a highly-symmetrical 7-dimensional 2-class dataset, of 2 concentric spheres in each of the 128 orthants. Multiple variants of the experiment were performed to observe symmetry, generalization comparison with SVMs and trying to understand the cause of the difference. [Notebook 1](./Code/orthants-symmetry.ipynb), [Notebook 2](./Code/orthants-single-empty.ipynb), [Script](./Code/orthants-crossval.py)
- To check whether NNs perform better at picking out hidden feature transformations from reasonably high-dimensional data and further transforming it to the labels, as opposed to SVMs. [Code](./Code/hidden-function.ipynb), [Script1](./Code/hidden-function.py), [Script2](./Code/hidden-func-poly.py)

## Repository structure

The proposal, report and presentation (may be added later) are present at the top level, along with the Code folder which contains all relevant details, code and supporting material for all experiments, as follows:
- [scripts](./Code/scripts/) contains the helper scripts with modularized code that is used frequently, grouped into files such as ntk, train, utils, etc.
- [animations](./Code/animations/), [logs](./Code/logs/), [configs](./Code/configs/), [models](./Code/models/) and [plots](./Code/plots/) contain supporting material saved during runtime for ease of making observations and their permanence for later use.
- Notebooks and scripts used for running experiments are present as such in [Code](./Code/).