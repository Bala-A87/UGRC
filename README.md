# UGRC
Research work done as part of UGRC-I (CS4900) on the topic 'Feature learning in neural networks' under the guidance of prof. Harish Guruprasad

## Abstract

The goal of the project is to analyze the ability of neural networks to learn useful features/representations from the data, beyond what kernel machines, such as SVMs, can. The neural tangent kernel is a tool that is particularly useful, in order to understand the similarities and shortcomings of SVMs wrt NNs.

## Key experiments

- Analyzing the ability of NNs to learn symmetry, with the help of a highly-symmetrical 7-dimensional 2-class dataset, of 2 concentric spheres in each of the 128 orthants. Multiple variants of the experiment were performed to observe symmetry, generalization comparison with SVMs and trying to understand the cause of the difference. Code for this experiment is present in:
    - [Notebook 1](./Code/orthants-symmetry.ipynb): Initial experimentation
    - [Notebook 2](./Code/orthants-single-empty.ipynb): Further experimentation, with only one empty orthant and the rest dense
    - [Script 1](./Code/orthants-crossval.py): Script to perform cross-validation and generate plots
    - **[Script 2](./Code/orthants-final.py)**: Final script to generate values and plots over 5 different initializations
- To check whether NNs perform better at picking out hidden feature transformations from reasonably high-dimensional data and further transforming it to the labels, as opposed to SVMs. Code for this experiment is present in:
    - [Notebook 1](./Code/hidden-function.ipynb): Initial experimentation with cosine transformation
    - [Notebook 2](./Code/alignment.ipynb): Comparing learning hidden representations to the silent alignment effect
    - [Script 1](./Code/hidden-function.py): Cross-validation and plot generation with cosine transformed data
    - **[Script 2](./Code/hidden-func-poly.py)**: Cross-validation and plot generation with polynomial data
    - **[Script 3](./Code/direct-func-poly.py)**: Cross-validation and plot generation with 1-d polynomial data
    - **[Script 4](./Code/hidden-function-align.py)**: Plot generation for alignment comparison

## Repository structure

The proposal, report and presentation (may be added later) are present at the top level, along with the Code folder which contains all relevant details, code and supporting material for all experiments, as follows:
- [scripts](./Code/scripts/) contains the helper scripts with modularized code that is used frequently, grouped into files such as ntk, train, utils, etc.
- [animations](./Code/animations/), [logs](./Code/logs/), [configs](./Code/configs/), and [plots](./Code/plots/) contain supporting material saved during runtime for ease of making observations and their permanence for later use.
- Notebooks and scripts used for running experiments are present as such in [Code](./Code/).