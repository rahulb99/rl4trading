# Stock Trading using Reinforcement Learning

## Project Description
Our project aims to develop and evaluate a suite of deep reinforcement learning agents for
automated stock trading. We will utilize the csce642-deepRL GitHub repository as a foundation,
integrating data pipelines from providers like Yahoo Finance and Alpaca. The project will
leverage the Gymnasium and RLlib libraries to implement, train, and compare several key RL
algorithms. Our agent's state space is a comprehensive vector capturing its financial position
and market conditions, supplemented by technical indicators like the Moving Average
Convergence Divergence and the Relative Strength Index. The action space is discrete, defined
as the set of integers \{−k, ..., −1, 0, 1, ..., k\}, where k represents the maximum number of shares
to be transacted, and the positive sign indicates a buy action while the negative sign represents
a sell action.

## Project Resources
[Project Presentation](https://www.youtube.com/watch?v=DmmIUyizI_8)

[Project Report](./CSCE_642_Final_Report.pdf)

## Setup
SWIG is required for installing Box2D. It can be installed on Linux by running 
```bash
sudo apt-get install swig build-essential python-dev python3-dev
```
and on Mac by running
```bash
brew install swig
```
or on windows by following the instructions [here](https://open-box.readthedocs.io/en/latest/installation/install_swig.html).

For setting up the environment, we recommend using conda + pip or virtual env + pip. The Python environment required is 3.9.16 (version)

 Install the packages given by
```bash
pip install -r requirements.txt
```

### Single Stock Trading Experiment
Run the notebook at https://github.com/rahulb99/rl4trading/blob/master/notebooks/RL4Trading.ipynb

### Multi Stock Trading Experiment
Run the python file run_multi.py to run the launch configurations to test the models. There are 5 launch configurations corresponding to each model

Usage example: 
```bash
python run_multi.py 0 5
```

Use 
```bash
python run_multi.py --help
```

for assistance with running this script
