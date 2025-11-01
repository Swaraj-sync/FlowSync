# 🚦 FlowSync: A Multi-Agent Deep Reinforcement Learning Framework for Coordinated Traffic Signal Control

### 🧠 Overview

**FlowSync** is a novel **Multi-Agent Deep Reinforcement Learning (MARL)** framework for **adaptive traffic signal control**.
Each intersection in a network is modeled as an independent agent that cooperatively learns to optimize signal timing to minimize **global traffic congestion**.

A key innovation of FlowSync is the **neighbor-state attention mechanism**, enabling agents to dynamically weigh the state of nearby intersections to achieve **coordinated, emergent behavior**.

> 🧩 Built with **Python**, **TensorFlow/Keras**, and validated in the **SUMO (Simulation of Urban MObility)** environment.

---

## 1. 🧩 Problem Formulation: Multi-Agent MDP

We formulate the Traffic Signal Control (TSC) task as a **Multi-Agent Markov Decision Process (MDP)**:
[
(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)
]

### Agents

* ( \mathcal{I} = {1, 2, ..., N} )
* Each agent ( i \in \mathcal{I} ) corresponds to one intersection.

### Global State (( \mathcal{S} ))

* ( s = (s_1, s_2, ..., s_N) )
* Each agent only observes its **local state** ( s_i ).

### Action Space (( \mathcal{A} ))

* ( A_i = {\text{Keep Phase}, \text{Change Phase}} )

### Transition Probability (( \mathcal{P} ))

* Defined by the SUMO simulator: ( P(s' | s, a) )

### Reward Function (( \mathcal{R} ))

* Each agent receives a **local reward** ( r_i = R(s, a, s') )

### Discount Factor (( \gamma ))

* Discounting future rewards: ( \gamma \in [0, 1] )

Objective:
[
\pi^**i = \arg\max*{\pi_i} \mathbb{E} \left[ \sum_{k=t}^{\infty} \gamma^{k-t} r_k \right]
]

---

## 1.1. 🔍 State Space (( s_i ))

Each agent’s **local state** is hierarchical, combining **visual**, **tabular**, and **neighbor** information:

| Component       | Type             | Description                        |
| --------------- | ---------------- | ---------------------------------- |
| Local Map       | 150×150×1 tensor | Visual vehicle positions           |
| Queue Length    | 1×12 vector      | Per-lane vehicle queues            |
| Vehicle Count   | 1×12 vector      | Per-lane vehicle counts            |
| Waiting Time    | 1×12 vector      | Per-lane accumulated wait          |
| Current Phase   | One-hot          | Current signal phase               |
| Neighbor Queues | 4×12 tensor      | Queue lengths of up to 4 neighbors |
| Neighbor Phases | 4×1 tensor       | Current phase of up to 4 neighbors |
| Neighbor Mask   | 1×4 binary       | Indicates active neighbors         |

---

## 1.2. ⚙️ Action Space (( a_i ))

| Action               | Description                               |
| -------------------- | ----------------------------------------- |
| **0 — Keep Phase**   | Continue the current signal phase         |
| **1 — Change Phase** | Switch to the next phase (yellow → green) |

---

## 1.3. 💰 Reward Function (( r_i ))

A **penalty-based reward** encouraging efficient traffic flow:

[
r_i = -\alpha_1 \sum \text{queue} - \alpha_2 \sum \text{wait} - \alpha_3 \sum (1 - v/v_{\max}) - \alpha_4 \cdot \text{flicker} + \alpha_5 \sum v_{\text{left}}
]

Weights ( \alpha ) are defined in
[`conf/grid_2x2/sumo_agent.conf`](conf/grid_2x2/sumo_agent.conf)

---

## 2. 🧠 Methodology

### 2.1. Deep Q-Network (DQN) Formulation

We approximate the optimal action-value function:

[
Q^*(s, a) = \mathbb{E}[r + \gamma \max_{a'} Q^*(s', a') | s, a]
]

Training minimizes TD error:
[
L(\theta) = \mathbb{E}*{(s,a,r,s')}[(y_i - Q(s,a; \theta))^2]
]
[
y_i = r + \gamma \max*{a'} Q(s', a'; \theta^-)
]

---

### 2.2. 🧩 FlowSync Neural Architecture

**Inputs:**

* Visual, tabular, and neighbor data streams.

**Steps:**

1. **Local Feature Extraction**

   * **Visual Encoder:** CNN → 32@8×8 + 16@4×4 filters
     → Output: ( v_{\text{map}} )
   * **Vector Encoder:** Dense layers over queue, wait, phase vectors
     → Output: ( v_{\text{local}} )
   * **Local Embedding:** Concatenate → Dense → ( e_{\text{local}} )

2. **Coordination via Attention**

   * Compute **neighbor embeddings** and apply **dot-product attention**:
     [
     \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
     ]

3. **Q-Value Head**

   * Concatenate ( e_{\text{local}} ) + ( c_{\text{neighbor}} )
   * Pass through 2 Dense (sigmoid) layers
   * Output: 2 linear units for {Keep, Change}

---

## 3. 🧪 Experimental Setup

| Setting        | Value                  |
| -------------- | ---------------------- |
| Environment    | SUMO                   |
| Scenario       | 2×2 grid (J1–J4)       |
| Control Script | `traffic_light_dqn.py` |

---

### 3.1. ⚙️ Key Hyperparameters

| Parameter         | Value | Description                   |
| ----------------- | ----- | ----------------------------- |
| LEARNING_RATE     | 0.001 | RMSprop LR                    |
| GAMMA             | 0.8   | Discount factor               |
| BATCH_SIZE        | 20    | Batch per update              |
| MAX_MEMORY_LEN    | 1000  | Replay buffer size            |
| UPDATE_PERIOD     | 300   | Simulation update period      |
| UPDATE_Q_BAR_FREQ | 5     | Target network sync frequency |
| D_DENSE           | 20    | Hidden Dense size             |
| EPSILON           | 0.00  | Exploration rate              |

---

## 4. 📊 Results & Analysis

Agents trained on `grid_2x2` scenario:
[`data/grid_2x2/grid.sumocfg`](data/grid_2x2/grid.sumocfg)

![Learning Curve]

### Observations

* **Learning Trend:** Upward trend in rewards (−0.0801 → −0.0794)
  → agents reduce penalties → better traffic flow.
* **Agent Specialization:** J4 (low congestion) vs. J1–J3 (high load).
  Despite differences, all exhibit **positive learning slopes**.

---

## 5. 🧭 How to Run

### 5.1. Prerequisites

```bash
sudo apt install sumo sumo-tools
pip install tensorflow numpy traci pandas matplotlib
```

---

### 5.2. Generate 2×2 Map (First-Time Only)

```bash
cd data/grid_2x2/
netconvert --node-files=grid.nod.xml --edge-files=grid.edg.xml --output-file=grid.net.xml
```

---

### 5.3. Configure and Run

Edit `runexp.py`:

```python
sumoBinary_nogui = "/usr/bin/sumo"
sumoBinary_gui   = "/usr/bin/sumo-gui"
setting_memo     = "grid_2x2"
```

Run training:

```bash
python runexp.py
```

> ⚡ For faster training, set `sumo_cmd_str = sumoCmd_nogui`.

---

### 5.4. Visualize Results

```bash
python plot_results.py
```

Generates `learning_curve.png` from the latest `memories.txt` log.

---

## 6. 📝 Citation

This project extends the concepts from **FlowSync** and builds upon the single-agent **IntelliLight** framework:

> Hua Wei*, Guanjie Zheng*, Huaxiu Yao, Zhenhui Li,
> *IntelliLight: A Reinforcement Learning Approach for Intelligent Traffic Light Control*,
> KDD 2018, London, UK.
> [[Source]](https://github.com/wingsweihua/IntelliLight)

---

## 🧱 Repository Structure

```
FlowSync/
│
├── conf/
│   └── grid_2x2/
│       ├── deeplight_agent.conf
│       └── sumo_agent.conf
│
├── data/
│   └── grid_2x2/
│       ├── grid.nod.xml
│       ├── grid.edg.xml
│       ├── grid.rou.xml
│       └── grid.sumocfg
│
├── models/
│   ├── deeplight_agent.py
│   ├── network_agent.py
│   └── agent.py
│
├── runexp.py
├── traffic_light_dqn.py
├── plot_results.py
└── README.md
```

---

## 🧩 Acknowledgements

* Built on **IntelliLight** architecture (Wei et al., 2018)
* Inspired by **MARL attention-based coordination** frameworks
* Simulation powered by **SUMO**

---
