# CSCE-642: Deep Reinforcement Learning

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