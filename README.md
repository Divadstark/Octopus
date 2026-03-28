![human_normalized_score_3](https://github.com/user-attachments/assets/fc6cc6f2-7e60-4fe9-8c12-10ec601176ea)![human_normalized_score_3 (1)](https://github.com/user-attachments/assets/fffc3305-8f03-4256-9862-59da8a440d07)
# Octopus
Octopus is a reinforcement learning repository built around the proposed Octopus agent, a regularized distributional DQN method for improved stability and performance.

It is designed to be self-contained, and extensible. The repository includes Octopus together with standard baseline agents, implemented in JAX, Haiku, and RLax, and provides a unified framework for training and evaluation on the Atari 2600 benchmark.

Beyond Atari benchmarks, the Octopus framework is intended to be applicable to real-world sequential decision-making problems, including electricity network optimization. This application is part of ongoing work and is not included in the current repository.

---

## Agents Implemented

| Agent      | Method Summary                                                                 | Reference |
|------------|--------------------------------------------------------------------------------|-----------|
| Double DQN | Reduces overestimation via decoupled action selection and evaluation          | [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) |
| Prioritized DQN | Improves sample efficiency through prioritized experience replay        | [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) |
| C51        | Learns a categorical approximation of the return distribution                 | [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887) |
| QR-DQN     | Models return distribution via quantile regression                            | [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044) |
| Rainbow    | Combines multiple DQN improvements into a unified framework                   | [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298) |
| MDQN       | Incorporates Munchausen reward shaping for improved exploration and stability | [Munchausen Reinforcement Learning](https://arxiv.org/abs/2007.14430) |
| MIQN*      | Munchausen-based distributional variant with enhancements aligned to Octopus  | [Munchausen Reinforcement Learning](https://arxiv.org/abs/2007.14430) |
| Octopus    | Regularized distributional RL framework with dynamic scaling                  | — |


Plot of median human-normalized score over 15 Atari games for each agent:


---

## Directory map

| Folder      | Contents (core files)         |
|-------------|-------------------------------|
| `octopus/`  | `agent.py`, `run_atari.py`    |
| other dirs  | other baselines      |
| root        | `plot.py`, `requirements.txt` |


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

## Acknowledgements

This repository is adapted and extended from the original implementation:

* DeepMind DQN Zoo: https://github.com/google-deepmind/dqn_zoo

We build upon the original codebase and extend it with additional algorithms, including the proposed **Octopus** framework and related variants.
