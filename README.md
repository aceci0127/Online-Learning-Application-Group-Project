# Online Learning Application Group Project

This repository contains an implementation of various online learning and pricing experiments. It provides both a programmatic framework (using Python scripts) and an interactive Jupyter Notebook interface.

## Contents

- **[agents.py](agents.py)** – Contains agent implementations for different experiments.
- **[data_generators.py](data_generators.py)** – Functions to generate datasets for the experiments.
- **[environments.py](environments.py)** – Environment definitions for simulating pricing and budget scenarios.
- **[runner.py](runner.py)** – Standard experiment runners that execute the experiments with given configurations.
- **[plotter.py](plotter.py)** – Utility functions for plotting experiment results and analysis.
- **[utils.py](utils.py)** – Common utility functions, including computation utilities and helper methods.
- **[notebook.ipynb](notebook.ipynb)** – Interactive notebook demonstrating configuration and execution of experiments.
- **requirements.txt** – List of Python package dependencies.

## Overview

This project implements and compares several online learning strategies for pricing, including:

- UCB1 based pricing
- Constrained UCB pricing
- Combinatorial multi-product pricing
- Primal-Dual methods (with both full-feedback and bandit feedback)
- Sliding window adaptations for non-stationary environments

Each experiment is configured via an experiment configuration object and executed using dedicated experiment runners. Results are visualized with comparative plots and performance analyses.

## How to Run

1. **Install Dependencies**

   Make sure you have Python 3.8+ installed. Then install required packages:

   ```sh
   pip install -r requirements.txt
   ```
