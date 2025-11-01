# 🚦 FlowSync: A Multi-Agent Deep Reinforcement Learning Framework for Coordinated Traffic Signal Control

### 🧠 Overview

**FlowSync** is a Multi-Agent Deep Reinforcement Learning (MARL) framework for adaptive traffic signal control.  
Each intersection is modeled as an agent that cooperatively learns to optimize signal timing to minimize **global traffic congestion**.

A key innovation is a **neighbor-state attention mechanism**, enabling agents to dynamically weigh neighboring intersections’ states and learn coordinated behaviour.

> Built with **Python**, **TensorFlow/Keras**, and validated using **SUMO (Simulation of Urban MObility)**.

---

## 1. Problem Formulation: Multi-Agent MDP

We formulate Traffic Signal Control (TSC) as a Multi-Agent Markov Decision Process (MDP):

$$
(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)
$$

### Agents
- $\mathcal{I} = \{1, 2, \dots, N\}$ — each agent $i$ is an intersection.

### Global state ($\mathcal{S}$)
- $s = (s_1, s_2, \dots, s_N)$  
- Each agent observes only its local state $s_i$.

### Action space ($\mathcal{A}$)
- $A_i = \{\text{Keep Phase}, \text{Change Phase}\}$

### Transition probability ($\mathcal{P}$)
- Determined by the SUMO simulator: $P(s' \mid s, a)$

### Reward function ($\mathcal{R}$)
- Each agent receives local reward $r_i = R(s, a, s')$

### Discount factor ($\gamma$)
- $\gamma \in [0, 1]$

**Objective:**

$$
\pi_i^{*} = \arg\max_{\pi_i} \mathbb{E}\left[\sum_{k=t}^{\infty} \gamma^{k-t} r_k \right]
$$

---

## 1.1. State Space ($s_i$)

Each agent’s local state combines visual, tabular and neighbor information:

| Component | Shape / Type | Description |
|---|---:|---|
| Local Map | `150 × 150 × 1` | Visual occupancy map of nearby vehicles |
| Queue Length | `1 × 12` | Per-lane queue counts |
| Vehicle Count | `1 × 12` | Per-lane vehicle counts |
| Waiting Time | `1 × 12` | Per-lane accumulated waiting time |
| Current Phase | one-hot | Current signal phase |
| Neighbor Queues | `4 × 12` | Up to 4 neighbors' queue lengths |
| Neighbor Phases | `4 × 1` | Up to 4 neighbors' current phases |
| Neighbor Mask | `1 × 4` | Binary mask indicating active neighbors |

---

## 1.2. Action Space ($a_i$)

| Action | Meaning |
|---:|---|
| `0` — Keep Phase | Continue current traffic phase |
| `1` — Change Phase | End current phase (yellow) and move to next |

---

## 1.3. Reward Function ($r_i$)

Penalty-based reward designed to reduce congestion:

$$
r_i = -\alpha_1 \sum \text{queue} - \alpha_2 \sum \text{wait} - \alpha_3 \sum\left(1 - \frac{v}{v_{\max}}\right) - \alpha_4 \cdot \text{flicker} + \alpha_5 \sum v_{\text{left}}
$$

Weights $\alpha$ are set in `conf/grid_2x2/sumo_agent.conf`.

---

## 2. Methodology

### 2.1. Deep Q-Network (DQN) Formulation

We approximate the optimal action-value function:

![Equation](https://latex.codecogs.com/png.image?\dpi{150}\bg_white\fn_phv\huge\color{White}Q^*(s,a)=\mathbb{E}[\,r+\gamma\max_{a'}Q^*(s',a')\;|\;s,a\,])


Training minimizes the TD loss:

$$
L(\theta) = \mathbb{E}_{(s,a,r,s')}\left[\left(y_i - Q(s,a;\theta)\right)^2\right]
$$

with

$$
y_i = r + \gamma \max_{a'} Q(s', a'; \theta^{-})
$$

---

### 2.2. FlowSync Neural Architecture

**Inputs:** visual map, local vectors, neighbor states.

**Local feature extraction**
- **Visual Encoder:** small CNN (32 filters @ 8×8, 16 filters @ 4×4) → $v_{\text{map}}$
- **Vector Encoder:** dense layers on concatenated per-lane vectors → $v_{\text{local}}$
- **Local embedding:** concatenate $v_{\text{map}}$ and $v_{\text{local}}$ → dense → $e_{\text{local}}$

**Coordination via attention**
- Form neighbor embeddings then apply dot-product attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^{T}}{\sqrt{d_k}}\right)V
$$

Output: $c_{\text{neighbor}}$.

**Q-value head**
- Concatenate $e_{\text{local}}$ and $c_{\text{neighbor}}$, pass through two dense layers, output 2 linear units for `{Keep, Change}`.

---

## 3. Experimental Setup

| Setting | Value |
|---|---|
| Environment | SUMO |
| Scenario | 2×2 grid (J1–J4) |
| Controller | `traffic_light_dqn.py` |

### 3.1. Key Hyperparameters

| Parameter | Value |
|---|---:|
| LEARNING_RATE | 0.001 |
| GAMMA | 0.8 |
| BATCH_SIZE | 20 |
| MAX_MEMORY_LEN | 1000 |
| UPDATE_PERIOD | 300 |
| UPDATE_Q_BAR_FREQ | 5 |
| D_DENSE | 20 |
| EPSILON | 0.00 |

---

## 4. Results & Analysis

Agents were trained on the `grid_2x2` scenario (`data/grid_2x2/grid.sumocfg`).

![Learning Curve](./rewards_over_time.png)

**Observations**
- Learning trend: average reward improves over training (penalties reduced).
- Agent specialization: some intersections handle more traffic; all agents still show learning improvement.

---

## 5. How to Run

### 5.1. Prerequisites

```bash
sudo apt install sumo sumo-tools
pip install tensorflow numpy traci pandas matplotlib
````

### 5.2. Generate 2×2 Map (first-time only)

```bash
cd data/grid_2x2/
netconvert --node-files=grid.nod.xml --edge-files=grid.edg.xml --output-file=grid.net.xml
```

### 5.3. Configure and run

Edit `runexp.py`:

```python
sumoBinary_nogui = "/usr/bin/sumo"
sumoBinary_gui   = "/usr/bin/sumo-gui"
setting_memo     = "grid_2x2"
```

Run:

```bash
python runexp.py
```

For headless runs, ensure `sumo_cmd_str` points to the nogui binary.

### 5.4. Visualize results

```bash
python plot_results.py
```

This generates `rewards_over_time.png` from the latest `memories.txt`.

---

## 6. Citation

This implementation builds on the ideas in the IntelliLight work:

> Hua Wei*, Guanjie Zheng*, Huaxiu Yao, Zhenhui Li,
> *IntelliLight: A Reinforcement Learning Approach for Intelligent Traffic Light Control*, KDD 2018.
> [IntelliLight repository](https://github.com/wingsweihua/IntelliLight)

---

## Repository structure

```
FlowSync/
├── conf/
│   └── grid_2x2/
├── data/
│   └── grid_2x2/
├── models/
│   ├── deeplight_agent.py
│   ├── network_agent.py
│   └── agent.py
├── runexp.py
├── traffic_light_dqn.py
├── plot_results.py
├── rewards_over_time.png
└── README.md
```

---

## Acknowledgements

* Built on **IntelliLight** architecture (Wei et al., 2018)
* Inspired by **MARL attention-based coordination** frameworks
* Simulation powered by **SUMO**

```
