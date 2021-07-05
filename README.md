# LinkScheduling

This repository contains code and notes used for the
research conducted in An Architecture for Meeting 
Quality-of-Service Requirements in Multi-User Quantum Networks.

## Running Simulations

To run the simulations used in the paper, the requirements
specified in `requirements.txt` using:

`pip install -r requirements.txt`

Once the requirements have been installed, the simulation for each
network may be run using:

Star Topology:

`python simulations/load_simulations/star_load_simulations.py`

H Topology:

`python simulations/load_simulations/H_load_simulations.py`

Symmetric Topology:

`python simulations/load_simulations/symm_load_simulations.py`

Line Topology:

`python simulations/load_simulations/line_load_simulations.py`
