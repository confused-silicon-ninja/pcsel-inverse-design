# PCSEL Inverse Design with Deep Reinforcement Learning

This repository contains Python scripts for reinforcement-learning-based inverse design of a photonic crystal surface-emitting laser (PCSEL) using COMSOL Multiphysics and a Deep Q-Network (DQN).

The workflow couples a DQN agent to a COMSOL eigenfrequency model. The agent iteratively modifies selected device parameters, evaluates the resulting optical response through COMSOL, and receives feedback based on the cavity quality factor and resonance wavelength.

## Repository Contents

### `comsol_env_v1.py`

Defines the COMSOL simulation environment used by the reinforcement learning agent.

Main functions include:

- Connecting Python to COMSOL through the `mph` package
- Updating selected PCSEL design parameters
- Running the COMSOL eigenfrequency study
- Extracting resonance wavelength and Q-factor
- Selecting a suitable optical mode
- Calculating the reward for the reinforcement learning agent
- Defining the state and action spaces

### `optim_PhC_dqn_v1.py`

Implements the Deep Q-Network optimization workflow.

Main features include:

- DQN policy and target networks
- Experience replay
- Epsilon-greedy exploration
- Periodic model checkpointing
- Automatic recovery from saved checkpoints
- Training history logging
- Q-factor and wavelength tracking
- Storage of optimization data for later analysis
- Identification of the best-performing designs

## Optimization Objective

The optimization seeks device configurations that:

1. Produce a resonance wavelength close to the target wavelength of **2200 nm**
2. Achieve a high optical quality factor, **Q**

The COMSOL environment evaluates each modified device configuration and returns the corresponding wavelength and Q-factor to the DQN agent.

## Design Parameters

The optimization environment currently includes the following parameters:

- Photonic crystal lattice constant
- Hole radius / radius-related parameter
- Active-layer thickness
- n-cladding thickness
- p-cladding thickness
- Simulation/domain dimension
- Air-region thickness
- p-contact thickness
- Ge-related layer thickness

The parameter ranges and step sizes are defined in `comsol_env_v1.py`.

## Requirements

The scripts require a Python environment with packages including:

```text
numpy
torch
gym
mph
tensorboard
