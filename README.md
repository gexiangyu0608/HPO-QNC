# Hierarchical Progressive Pauli Noise Modeling and Residual Compensation

This repository provides the official core source code for the research paper: **"Hierarchical Progressive Pauli Noise Modeling and Residual Compensation for Multi-Qubit Circuits"**.

## Overview

Traditional Quantum Process Tomography (QPT) suffers from exponential parameter explosion. Our **Hierarchical Progressive Optimization (HPO)** framework breaks this bottleneck by leveraging hierarchical freezing and progressive masking to efficiently extract high-order spatial crosstalk.

## Repository Structure

This repository contains two core Python scripts that reproduce the key results in our paper:

* `HPO.py`: The core PyTorch-based framework for Hierarchical Progressive Optimization. It includes the logic for the combinatorial projection mask and generates the convergence dynamics comparison (reproducing the optimization loss curves).
* `HHL10bit.py`: The PennyLane-based validation script for the 10-qubit Harrow-Hassidim-Lloyd (HHL) algorithm. It simulates the spatial crosstalk mitigation process and calculates the algorithmic state fidelity recovery (reproducing the QEM fidelity improvement results).

## Requirements

To run the scripts, ensure you have the following dependencies installed in your Python 3.x environment:

* `torch`
* `pennylane`
* `numpy`
* `matplotlib`

## Usage

You can run the scripts directly to observe the convergence optimization and the HHL fidelity outcomes:

    python HPO.py
    python HHL10bit.py

Citation

If you find this methodology or code useful, please cite our paper (link to be updated upon publication).
