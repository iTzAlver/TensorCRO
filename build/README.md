# TensorCRO: A Tensorflow-based implementation of the Coral Reef Optimization algorithm.

<p align="center">
    <img src="https://github.com/iTzAlver/TensorCRO/blob/master/multimedia/logo.png" width="400px">
</p>

<p align="center">
    <a href="https://github.com/iTzAlver/TensorCRO/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/iTzAlver/basenet_api?color=purple&style=plastic" /></a>
    <a href="https://github.com/iTzAlver/TensorCRO/tree/master/test">
        <img src="https://img.shields.io/badge/coverage-100%25-green?color=green&style=plastic" /></a>
    <a href="https://github.com/iTzAlver/TensorCRO/blob/master/build/requirements.txt">
        <img src="https://img.shields.io/badge/requirements-python3.8-red?color=blue&style=plastic" /></a>
    <a href="https://github.com/iTzAlver/TensorCRO/tree/master/multimedia/notebooks">
        <img src="https://img.shields.io/badge/doc-notebook-green?color=orange&style=plastic" /></a>
    <a href="">
        <img src="https://img.shields.io/badge/release-1.2.0-white?color=white&style=plastic" /></a>
</p>

<p align="center">
    <a href="https://www.tensorflow.org/">
        <img src="https://img.shields.io/badge/dependencies-tensorflow-red?color=orange&style=for-the-badge" /></a>
    <a href="https://developer.nvidia.com/cuda-downloads">
        <img src="https://img.shields.io/badge/dependencies-CUDA-red?color=green&style=for-the-badge" /></a>
</p>

# Table of contents

1. [About](#about)
2. [What's new?](#whats-new)
3. [Install](#install)
4. [Usage](#usage)

## About ##
    

```biblitex
Implementation author:     A.Palomo-Alonso (alberto.palomo@uah.es) 
Original Algorithm author: S.Salcedo-Sanz  (sancho.salcedo@uah.es)
Universidad de Alcalá (Madrid - Spain). Escuela Politécnica Superior
Signal Processing and Communications Department (TDSC)
```

This is a Tensorflow-based implementation of the Coral Reef Optimization algorithm. The algorithm is implemented
as a Tensorflow graph, which allows to run it in GPU and TPU. The algorithm is implemented as a set of substrate layers
that can be combined with other algorithms such as Differential Evolution, Harmony Search and Random Search. The
framework also allows to implement crossover operators as blxalpha, gaussian, uniform, masked and multipoint.

The framework also includes a Jupyter Notebook with an example of use of the algorithm.

## What's new?

### 1.0.0
1. First release.
2. CRO-SL: Coral Reef Optimization algorithm with substrate layers.
3. GPU runnable: The algorithm can be run in GPU and TPU as a graph, with +``x2` speed-up over the conventional implementations.
4. Substrate crossovers: The framework allows to implement crossover operators as blxalpha, gaussian, uniform, 
masked and multipoint.
5. Algorithms: The framework allows to implement algorithms as substrate layers such as Differential Evolution,
Harmony Search and Random Search.
6. Watch Replay: The algorithm also allows to watch the replay of the solutions found in the training process, with
an interactive GUI.
7. Jupyter Notebook: The framework includes a Jupyter Notebook with example of use for the Max-Ones-From-Zeros problem.

### 1.2.0
1. Progress bar: The framework now also includes a progress bar to monitor the training process.
2. Minor bug fixing.
3. Jupyter Notebook: The framework includes a Jupyter Notebook with example of use for the Max-Ones-From-Zeros problem.

## Install

To install it you must install the dependencies. Then, you can install the package with the following command
using PIP:

```bash
pip install tensorcro
```

Or you can clone the repository and install it with the following commands
using Git:

```bash
git clone https://github.com.iTzAlver/TensorCRO.git
cd TensorCRO/dist/
pip install ./tensorcro-1.2.0-py3-none-any.whl
```

### Requirements

* Python 3.6 or higher
* Tensorflow 2.0 or higher
* Numpy 1.18.1 or higher
* Matplotlib 3.1.3 or higher
* Pandas 1.0.1 or higher
* CUDA for GPU support (optional but strongly recommended)

# Usage:

We have a JuPyter Notebook with an example of use of the algorithm. You can find it in the folder `/multimedia/notebooks` 
of the repository.

# Cite:

If you use this code, please cite the following paper:

```bibtex
@inproceedings{palomo2022tensorcro,
  title={TensorCRO: A Tensorflow-based implementation of the Coral Reef Optimization algorithm},
  author={Palomo-Alonso, A and Salcedo-Sanz, S},
  journal={arXiv preprint arXiv:X.Y},
  year={2022}
}
```
