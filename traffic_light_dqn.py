# -*- coding: utf-8 -*-

'''
@author: hzw77, gjz_5038

python TrafficLightDQN.py SEED setting_memo

SEED: random number for initializing the experiment
setting_memo: the folder name for this experiment
    The conf, data files will should be placed in conf/setting_memo, data/setting_memo respectively
    The records, model files will be generated in records/setting_memo, model/setting_memo respectively

--- FLOWSYNC MODIFICATION ---
This file is now the central coordinator for a Multi-Agent system.
It initializes N agents (one per intersection) and runs a
synchronous training loop.
'''


import copy
import json
import shutil
import traci

import os
import time
import math
import map_computor as map_computor
from deeplight_agent import DeeplightAgent

from sumo_agent import SumoAgent
import xml.etree.ElementTree as ET


class TrafficLightDQN:

    DIC_AGENTS = {
        "Deeplight": DeeplightAgent, # This will now be our FlowSync agent
    }

    NO_PRETRAIN_AGENTS = []

    class ParaSet:

        def __init__(self, dic_paras):
            for key, value in dic_paras.items():
                setattr(self, key, value)

    class PathSet:

        # ======================================= conf files ========================================
        EXP_CONF = "exp.conf"
        SUMO_AGENT_CONF = "sumo_agent.conf"
        PATH_TO_CFG_TMP = os.path.join("data", "tmp")
        # ======================================= conf files ========================================

        def __init__(self, path_to_conf, path_to_data, path_to_output, path_to_model):

            self.PATH_TO_CONF = path_to_conf
            self.PATH_TO_DATA = path_to_data
            self.PATH_TO_OUTPUT = path_to_output
            self.PATH_TO_MODEL = path_to_model

            if not os.path.exists(self.PATH_TO_OUTPUT):
                os.makedirs(self.PATH_TO_OUTPUT)
            if not os.path.exists(self.PATH_TO_MODEL):
                os.makedirs(self.PATH_TO_MODEL)

            dic_paras = json.load(open(os.path.join(self.PATH_TO_CONF, self.EXP_CONF), "r"))
            self.AGENT_CONF = "{0}_agent.conf".format(dic_paras["MODEL_NAME"].lower())
            self.TRAFFIC_FILE = dic_paras["TRAFFIC_FILE"]
            self.TRAFFIC_FILE_PRETRAIN = dic_paras["TRAFFIC_FILE_PRETRAIN"]

    def __init__(self, memo, f_prefix, sumo_cmd_str):

        self.path_set = self.PathSet(os.path.join("conf", memo),
                                     os.path.join("data", memo),
                                     os.path.join("records", memo, f_prefix),
                                     os.path.join("model", memo, f_prefix))

        self.para_set = self.load_conf(conf_file=os.path.join(self.path_set.PATH_TO_CONF, self.path_set.EXP_CONF))
        shutil.copy(
            os.path.join(self.path_set.PATH_TO_CONF, self.path_set.EXP_CONF),
            os.path.join(self.path_set.PATH_TO_OUTPUT, self.path_set.EXP_CONF))

        # --- FLOWSYNC: Start SUMO here to discover nodes ---
        print("Starting SUMO for node discovery...")
        map_computor.start_sumo(sumo_cmd_str)
        self.all_node_ids = traci.trafficlight.getIDList()
        print(f"Discovered {len(self.all_node_ids)} nodes: {self.all_node_ids}")

        # --- FLOWSYNC: Create N agents and N sumo_agents ---
        self.agents = {}
        self.sumo_agents = {}

        for node_id in self.all_node_ids:
            # Each agent gets its own model (or shared model, depending on implementation)
            self.agents[node_id] = self.DIC_AGENTS[self.para_set.MODEL_NAME](
                num_phases=len(map_computor.NODE_PHASE_DEFINITIONS[node_id]["phases"]),
                num_actions=2, # Keep or Change
                path_set=self.path_set
            )
            # Each sumo_agent manages the state for its node
            self.sumo_agents[node_id] = SumoAgent(
                sumo_cmd_str=sumo_cmd_str,
                path_set=self.path_set,
                node_id=node_id
            )

        self.shared_dic_vehicles = {} # Global vehicle dictionary

        # Initialize all agent states
        if self.all_node_ids:
            self.shared_dic_vehicles = self.sumo_agents[self.all_node_ids[0]].update_vehicles(self.shared_dic_vehicles)
            for node_id in self.all_node_ids:
                self.sumo_agents[node_id].get_observation() # This populates the initial state
        else:
            print("Warning: No traffic light nodes found.")

        print("All agents initialized.")


    def load_conf(self, conf_file):

        dic_paras = json.load(open(conf_file, "r"))
        return self.ParaSet(dic_paras)

    def check_if_need_pretrain(self):

        if self.para_set.MODEL_NAME in self.NO_PRETRAIN_AGENTS:
            return False
        else:
            return True

    def _generate_pre_train_ratios(self, phase_min_time, em_phase):
        phase_traffic_ratios = [phase_min_time]

        # generate how many varients for each phase
        for i, phase_time in enumerate(phase_min_time):
            if i == em_phase:
                for j in range(1, 5, 1):
                    gen_phase_time = copy.deepcopy(phase_min_time)
                    gen_phase_time[i] += j
                    phase_traffic_ratios.append(gen_phase_time)
            else:
                # pass
                for j in range(1, 5, 1):
                    gen_phase_time = copy.deepcopy(phase_min_time)
                    gen_phase_time[i] += j
                    phase_traffic_ratios.append(gen_phase_time)
            for j in range(5, 20, 5):
                gen_phase_time = copy.deepcopy(phase_min_time)
                gen_phase_time[i] += j
                phase_traffic_ratios.append(gen_phase_time)

        return phase_traffic_ratios

    @staticmethod
    def _set_traffic_file(sumo_config_file_tmp_name, sumo_config_file_output_name, list_traffic_file_name):

        # update sumocfg
        sumo_cfg = ET.parse(sumo_config_file_tmp_name)
        config_node = sumo_cfg.getroot()
        input_node = config_node.find("input")
        for route_files in input_node.findall("route-files"):
            input_node.remove(route_files)
        input_node.append(
            ET.Element("route-files", attrib={"value": ",".join(list_traffic_file_name)}))
        sumo_cfg.write(sumo_config_file_output_name)

    def set_traffic_file(self):

        try:
            self._set_traffic_file(
                os.path.join(self.path_set.PATH_TO_DATA, "cross_pretrain.sumocfg"),
                os.path.join(self.path_set.PATH_TO_DATA, "cross_pretrain.sumocfg"),
                self.para_set.TRAFFIC_FILE_PRETRAIN)
        except FileNotFoundError:
            print(f"Warning: could not find cross_pretrain.sumocfg at {self.path_set.PATH_TO_DATA}")
        except Exception as e:
            print(f"Warning: error setting pretrain traffic file: {e}")

        try:
            self._set_traffic_file(
                os.path.join(self.path_set.PATH_TO_DATA, "cross.sumocfg"),
                os.path.join(self.path_set.PATH_TO_DATA, "cross.sumocfg"),
                self.para_set.TRAFFIC_FILE)
        except FileNotFoundError:
            print(f"Warning: could not find cross.sumocfg at {self.path_set.PATH_TO_DATA}")
        except Exception as e:
            print(f"Warning: error setting main traffic file: {e}")

        for file_name in self.path_set.TRAFFIC_FILE_PRETRAIN:
            try:
                shutil.copy(
                    os.path.join(self.path_set.PATH_TO_DATA, file_name),
                    os.path.join(self.path_set.PATH_TO_OUTPUT, file_name))
            except FileNotFoundError:
                print(f"Warning: Could not copy traffic file {file_name}")
        for file_name in self.path_set.TRAFFIC_FILE:
            try:
                shutil.copy(
                    os.path.join(self.path_set.PATH_TO_DATA, file_name),
                    os.path.join(self.path_set.PATH_TO_OUTPUT, file_name))
            except FileNotFoundError:
                print(f"Warning: Could not copy traffic file {file_name}")


    def train(self, if_pretrain, use_average):

        # --- FLOWSYNC: This is the main multi-agent training loop ---

        if not self.all_node_ids:
            print("Error: No traffic light nodes to train. Exiting.")
            return

        if if_pretrain:
            # Pre-training logic would go here
            print("Pre-training is not fully implemented for FlowSync MARL. Skipping.")
            total_run_cnt = 0 # self.para_set.RUN_COUNTS_PRETRAIN
        else:
            total_run_cnt = self.para_set.RUN_COUNTS

        file_name_memory = os.path.join(self.path_set.PATH_TO_OUTPUT, "memories.txt")
        f_memory = open(file_name_memory, "w") # Clear file
        f_memory.close()

        current_time = self.sumo_agents[self.all_node_ids[0]].get_current_time()

        # The main simulation loop
        while current_time < total_run_cnt:

            dic_states = {}
            dic_actions_pred = {}
            dic_q_values = {}
            dic_rewards = {}
            dic_next_states = {}
            dic_actual_actions = {}

            # 1. Get STATE and CHOOSE ACTION for all agents
            for node_id in self.all_node_ids:
                s_agent = self.sumo_agents[node_id]
                agent = self.agents[node_id]

                # Get current state (which was updated in the *previous* step)
                state = s_agent.get_observation()
                state = agent.get_state(state, current_time)
                dic_states[node_id] = state

                # Choose action
                action_pred, q_values = agent.choose(count=current_time, if_pretrain=if_pretrain)
                dic_actions_pred[node_id] = action_pred
                dic_q_values[node_id] = q_values

            # 2. APPLY ACTION for all agents (this sets lights but does not step)
            for node_id in self.all_node_ids:
                s_agent = self.sumo_agents[node_id]
                action_pred = dic_actions_pred[node_id]

                # The action is applied *before* the sim step
                # This updates the agent's internal phase/duration
                # It returns 0 reward (as it's pre-step) and the actual action taken
                _, actual_action = s_agent.take_action(action_pred, self.shared_dic_vehicles)
                dic_actual_actions[node_id] = actual_action

            # 3. STEP SIMULATION GLOBALLY
            # This is a synchronous execution
            # All agents act, then the environment steps once.
            traci.simulationStep()
            current_time = map_computor.get_current_time()

            # Update global vehicle dictionary
            self.shared_dic_vehicles = self.sumo_agents[self.all_node_ids[0]].update_vehicles(self.shared_dic_vehicles)

            # 4. Get NEXT_STATE, REWARD, and REMEMBER for all agents
            for node_id in self.all_node_ids:
                s_agent = self.sumo_agents[node_id]
                agent = self.agents[node_id]

                # Get reward based on the *new* state
                reward = s_agent.cal_reward_post_step(dic_actual_actions[node_id], self.shared_dic_vehicles)
                dic_rewards[node_id] = reward

                # Get next state (which includes neighbor states from the new sim step)
                next_state = s_agent.get_observation()
                next_state = agent.get_next_state(next_state, current_time)
                dic_next_states[node_id] = next_state

                # Remember
                agent.remember(dic_states[node_id], dic_actual_actions[node_id], reward, next_state)

                # Log
                memory_str = 'time = %d\tnode = %s\taction = %d\tphase = %d\treward = %f\t%s' \
                             % (current_time, node_id, dic_actual_actions[node_id],
                                dic_states[node_id].cur_phase[0][0],
                                reward, repr(dic_q_values[node_id]))
                print(memory_str)
                with open(file_name_memory, "a") as f_mem:
                    f_mem.write(memory_str + "\n")

            # 5. UPDATE NETWORKS for all agents
            if not if_pretrain:
                for node_id in self.all_node_ids:
                    agent = self.agents[node_id]
                    agent.update_network(if_pretrain, use_average, current_time)
                    agent.update_network_bar()

        # End of training
        map_computor.end_sumo()
        print("END")


def main(memo, f_prefix, sumo_cmd_str, sumo_cmd_pretrain_str):

    # Pass the *non-pretrain* command string for discovery
    player = TrafficLightDQN(memo, f_prefix, sumo_cmd_str)
    player.set_traffic_file()

    # Pre-training (if implemented)
    # player.train(if_pretrain=True, use_average=True)

    # Main training
    player.train(if_pretrain=False, use_average=False)