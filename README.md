# Equilibrium-Learning

A simulation framework for studying learning dynamics in game theory, focusing on how different learning algorithms lead to convergence in empirical play.

## Overview

This project implements and compares two classic no-regret learning algorithms (with random initialization):
- **MWU (Multiplicative Weights Update)**: Updates strategies based on cumulative utilities
- **RM (Regret Matching)**: Updates strategies based on cumulative regret

## Features
- Implementation of MWU and Regret Matching learners
- Support for n-player games with customizable payoff matrices
- Visualization of empirical play probabilities over time
- Jinx game generator for testing

## Usage

1. Open `main.ipynb` in Jupyter Notebook
2. Modify hyperparameters in the "Set Hyperparameters" section:
   - `n`: Game size (number of actions per player)
   - `T`: Number of learning rounds
   - `rs`: Tuple of random step values to compare
   - `learner_types`: Choose `MWULearner` or `RMLearner`
   - `N`: Number of simulations to run
3. Run all cells to execute experiments and generate visualizations
4. Results are saved in the `results/` folder

## Results

The generated plots show how average empirical play probabilities converge over time for different random step values.

## Dependencies

- numpy
- matplotlib
- tqdm
