# FlowSync: A Multi-Agent Deep Reinforcement Learning Framework for Adaptive Traffic Signal Optimization

**FlowSync** is an advanced, multi-agent deep reinforcement learning (MARL) framework designed to optimize urban traffic signal control. This project extends the concepts of a single-intersection agent (like IntelliLight) to a coordinated, multi-intersection network.

It uses an **Attention-based neural network** to allow each traffic signal agent to intelligently weigh the importance of its neighbors, leading to more efficient, coordinated traffic flow across an entire city grid. The system is built to interface with the **SUMO (Simulation of Urban MObility)** traffic simulator.

-----

## Key Features

  * **Multi-Agent Coordination:** Deploys a separate, autonomous Deep Q-Network (DQN) agent for each intersection. A central coordinator synchronizes the simulation steps, allowing for parallel learning in a complex network.
  * **Neighbor-State Attention:** The agent's neural network (defined in `deeplight_agent.py`) uses an attention mechanism to analyze the state of neighboring intersections (their queue lengths and current phases). This allows an agent to make context-aware decisions, for example, by "holding" a green light for an approaching platoon of cars from another intersection.
  * **Dynamic State Representation:** The state for each agent is a hierarchical representation, including:
      * **Local State:** A 2D "map" of vehicle positions in its vicinity.
      * **Local Vectors:** Queue lengths, waiting times, and current phase information.
      * **Neighbor State:** The queue lengths and phases of adjacent intersections.
  * **SUMO/TraCI Integration:** Connects directly to the SUMO simulation via the TraCI API. It dynamically discovers all traffic light junctions in a provided map, automatically creating and assigning an agent to each one.
  * **Explainable AI (XAI):** Includes an `ExplainableAI.py` module (stubbed) to interpret the "black box" decisions of the agent, providing insights into *why* it chose to change or keep a phase.

-----

## How It Works: System Architecture

The framework is managed by a central, synchronous coordinator (`traffic_light_dqn.py`) that orchestrates the learn-act-reward loop for all agents simultaneously.

1.  **Initialization:**

      * The `TrafficLightDQN` coordinator starts SUMO.
      * `map_computor.py` discovers all traffic light nodes (e.g., `['node0', 'node1']`) and their neighbors from the SUMO map.
      * The coordinator creates one `DeeplightAgent` (the "brain") and one `SumoAgent` (the "body") for each discovered node.

2.  **Synchronous Training Loop:** At each simulation step, the coordinator instructs all agents to:

      * **Observe:** Each `SumoAgent` gathers its local state (queues, map) and its neighbor states (neighbor queues/phases) from `map_computor.py`.
      * **Choose Action:** The state is passed to the `DeeplightAgent`'s neural network. The attention mechanism weighs the neighbor data, and the agent outputs a Q-value for "Keep Phase" or "Change Phase".
      * **Act:** All agents commit their chosen action. The `SumoAgent` sets the traffic light in SUMO (e.g., initiating a yellow phase).
      * **Step:** The coordinator advances the SUMO simulation by one global step.
      * **Reward & Remember:** Each `SumoAgent` observes the *new* state and calculates a reward (e.g., negative total queue length). This (`state`, `action`, `reward`, `next_state`) tuple is stored in the `DeeplightAgent`'s memory.
      * **Learn:** Each `DeeplightAgent` samples from its memory to update its neural network via backpropagation.

-----

## Getting Started

### Prerequisites

  * **Python 3.10+**

  * **SUMO (Simulation of Urban MObility):** The SUMO simulation suite must be installed and its `bin` directory added to your system's PATH, or the path must be specified in `runexp.py`.

  * **Python Libraries:** This project requires Keras 3 (TensorFlow).

    ```bash
    pip install tensorflow numpy traci
    ```

### How to Run

1.  **Configure SUMO Path:**
    Open `runexp.py`. Modify the `sumoBinary_gui` and `sumoBinary_nogui` variables to point to the correct path of `sumo-gui.exe` and `sumo.exe` on your system.

    ```python
    # (in runexp.py)
    sumoBinary_nogui = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
    sumoBinary_gui = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"
    ```

2.  **Configure Experiment Settings:**
    Open `conf/one_run/exp.conf`. This file controls the experiment's length.

      * `"RUN_COUNTS"`: The total number of simulation seconds for the main training (e.g., `72000`). For a quick test, set this to `5000`.
      * `"RUN_COUNTS_PRETRAIN"`: The duration for the pre-training phase (e.g., `10000`). For a quick test, set this to `1000`.

3.  **Choose a Mode (GUI vs. No-GUI):**
    Open `runexp.py` and find the `traffic_light_dqn.main(...)` call at the end.

      * **To Watch (Slow):** `sumo_cmd_str=sumoCmd`
      * **To Train (Fast):** `sumo_cmd_str=sumoCmd_nogui`

    <!-- end list -->

    ```python
    # (in runexp.py)
    traffic_light_dqn.main(
        memo=setting_memo,
        f_prefix=prefix,
        sumo_cmd_str=sumoCmd_nogui,  # <-- Set to sumoCmd_nogui for fast training
        sumo_cmd_pretrain_str=sumoCmd_nogui_pretrain
    )
    ```

4.  **Run:**

    ```bash
    python runexp.py
    ```

-----

## Testing & Analysis

### Running on a Multi-Intersection Map

The true power of **FlowSync** is unlocked on a multi-intersection grid.

1.  **Get a SUMO Map:** Find or create a SUMO map with multiple connected traffic light junctions (e.g., a 2x2 or 3x3 grid).

2.  **Update Configs:** Place your new `.net.xml`, `.rou.xml`, and other files in the `data/one_run/` folder.

3.  **Point to Map:** Open `data/one_run/cross.sumocfg` and `data/one_run/cross_pretrain.sumocfg`. Change the `<net-file>` and `<route-files>` values to point to your new map files.

4.  **Run\!** When you run `runexp.py`, the console will now show that it has discovered multiple nodes and their neighbors, and the attention mechanism will be fully utilized.

    ```
    --- FlowSync Network Discovery ---
    Discovered 4 nodes: ['J1', 'J2', 'J3', 'J4']
      Node: J1
        Phases: 8
        Lanes (12): [...]
        Neighbors: ['J2', 'J3']
      Node: J2
        Phases: 8
        Lanes (12): [...]
        Neighbors: ['J1', 'J4']
    ...
    ```

### Analyzing Results

All outputs are saved to folders named after your experiment (e.g., `Deeplight_cross_all_synthetic...`).

  * **Trained Models:** `model/one_run/<experiment_name>/`
      * Contains the `.h5` files for the trained agent models.
  * **Logs & Rewards:** `records/one_run/<experiment_name>/`
      * `memories.txt`: A detailed log of every decision, state, and reward for all agents. This file can be plotted (e.g., `time` vs. `reward`) to visualize learning.
      * `log_rewards_node0.txt`, etc.: A per-agent breakdown of rewards.

-----

## Citation

This project is a functional, multi-agent implementation and is built upon the original single-agent **IntelliLight** framework.

> Hua Wei\*, Guanjie Zheng\*, Huaxiu Yao, Zhenhui Li, **IntelliLight: A Reinforcement Learning Approach for Intelligent Traffic Light Control**, in Proceedings of the 2018 ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD'18), London, UK, August 2018.
>
> (\*Co-First author.)
