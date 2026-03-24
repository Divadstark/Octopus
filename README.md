# *Quick Start Guide* 

A README for training the **Octopus** agent and plotting results.

---

## Directory map

| Folder      | Contents (core files)         |
|-------------|-------------------------------|
| `octopus/`  | `agent.py`, `run_atari.py`    |
| other dirs  | other baselines      |
| root        | `plot.py`, `requirements.txt` |

---

## Agents Implemented

1. **Double DQN**: From the paper "Deep Reinforcement Learning with Double Q-learning"
2. **Prioritized DQN**: From the paper "Prioritized Experience Replay"
3. **C51**: From the paper "A Distributional Perspective on Reinforcement Learning"
4. **QR—DQN**: From the paper "Distributional Reinforcement Learning with Quantile Regression"
5. **Rainbow**: From the paper "Rainbow: Combining Improvements in Deep Reinforcement Learning“
6. **MDQN**: From the paper "Munchausen Reinforcement Learning"
7. **MIQN_star**: From the paper "Munchausen Reinforcement Learning", MIQN_star is equipped with same enhancements as Octopus. 
8. **Octopus**: The proposed framework

---

##  Setup

```bash
# create & activate a Python ≥3.9 env
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# one-off: import Atari ROMs
python -m atari_py.import_roms /path/to/roms
```

---

## Train the listed  agent
To train an agent on an Atari game, navigate to the corresponding directory and run:

```bash
cd [agent name]
python run_atari.py   --environment_name amidar --seed=1 --results_csv_path /results/[agent name]_amidar_seed1.csv
```
All the algorithms are used as this.

*Key flags*:  
`--environment_name` (Atari game to train on), 

`--seed`(Random seed for reproducibility),

`--results_csv_path` (Output path for training metrics CSV file).

---

## Plot human-normalised returns

To generate human-normalized score plots:
```bash
python plot.py 
```
This creates `normalized_score.pdf` in the working directory.

The script assumes training results are stored in /results/ with each algorithm having its own subdirectory (Rainbow, C51, Double_Q, Prioritized, MDQN, MIQN*, QR-DQN, and Octopus). Within each algorithm's folder, CSV files should follow the naming pattern of algorithm name, game name, and seed number (e.g., `results_octopus_amidar_seed1.csv`).

---

