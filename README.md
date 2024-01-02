# MiniHack Crossing River

## Description
The project is based on the standard "River" environment from the MiniHack Environment Zoo (https://minihack.readthedocs.io/en/latest/envs/navigation/river.html).\
The final goal of the agent is to reach the stairs, which are on the other side of the river. To do this, the agent must push a boulder into the river. After that the agent can cross it; in Nethack, swimming across a river could prove fatal for the agent.

The goal of the project is to find a solution that is able to complete as many games as possible in the most efficient way in terms of the number of steps performed.

### Methodologies
Two solutions to the problem are described and analyzed. The first naive solution is an offline approach to the problem, the agent simply executes the initial computed plan without considering changes in the environment. \
However this is not sufficient, so an online approach was developed to solve the problem. In both solutions, pathfinding is based on the classic A* algorithm in order to optimize the number of steps performed by the agent.

### Assessment
At the end of the notebook, the empirical evaluations carried out for the assessment of the final solution are presented. In particular, three metrics are considered:
* The percentage of success;
* The average number of steps;
* The average execution time.

## Dependencies
The Minihack Crossing River dependencies are:

- [NLE](https://github.com/facebookresearch/nle) -> (version 0.9.0) 
- [MiniHack](https://github.com/facebookresearch/minihack) -> (version 0.1.5)
- [matplotlib](https://github.com/matplotlib/matplotlib) -> (version 3.8.1) 

Other software dependencies are closely related to the requirements for installing these libraries.

## How to install
Clone the repository 
```bash
git clone https://github.com/JacopoRaffi/Minihack_Crossing_River.git
```

Move into the directory 
```bash
cd Minihack_Crossing_River
```

Install the requirements
```bash
pip install --requirement requirements.txt
```

## Usage
To execute the code, simply run the notebook *report.ipynb*, which contains the project's description, the evaluation and the demos.

## Repository Structure
```bash
📂Minihack_Crossing_River
├── 🐍algorithms.py # set of functions specifically for the pathfinding aspects
├── 🐍evaluation.py # tests for the final assesment 
├── 🐍logic.py      # set of functions representing the logic of the agent
├── 📒report.ipynb  # final report describing both problem and solution
├── 🐍utils.py      # set of utility functions
```

## Authors
[Chiara Cecchetti](https://github.com/cecchiara99) \
[Nicola Emmolo](https://github.com/nicolaemmolo) \
[Andrea Piras](https://github.com/aprs3) \
[Jacopo Raffi](https://github.com/JacopoRaffi) 
