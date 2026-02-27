## Getting Started

This repository contains the code to generate the plots from the article: Chartouny, A., Khamassi, M., & Girard, B. (2025). Multi-model reinforcement learning with online retrospective change-point detection.

## Description


* The agent classes are in `agents.py`, `task_change_agents.py`, and `rlcd.py`
* The environment classes are in `envs.py`
* The constants and variables used are in `consts.py`, `variables.py`, and `const_maze.py`
* The play functions are in `play_function.py`
* The plotting functions are in `plots.py`
* The generation process of the mazes and their generation is in `generation_mazes.py`
* To launch the experiment and get all the result figures, launch `main.py`
* To install the libraries, use `requirements.txt` or `requirements.yml`
* The folder **Env** contains the images, transitions, and rewards of the environments generated
* The folder **results** contains the results generated with `main.py`
* You can find a preprint of the article on biorXiv.

## Installation 

To clone this repository, use `git clone https://github.com/AugustinChrtn/switch/`

Then, install the required libraries indicated in the `requirements.txt` or `requirements.yml` file.

After these two steps, you can:
* Launch `main.py` to get the figures and the data from the article. You can indicate in the `main.py` file which experiment you would like to launch.
* The results file contains all the data used to generate the plots from the article. Feel free to generate new plots with this data.