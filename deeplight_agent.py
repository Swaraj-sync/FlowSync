# -*- coding: utf-8 -*-

'''
@author: hzw77, gjz5038

Deep reinforcement learning agent

'''

import numpy as np
# --- FLOWSYNC ADDITIONS ---
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, Multiply, Add, concatenate
from keras.layers import GlobalAveragePooling1D, Reshape, Permute, multiply, Lambda, Dot
from keras.models import Model, model_from_json, load_model
# --- END ADDITIONS ---
from keras.optimizers import RMSprop  # MODIFIED: Use .legacy for compatibility
from keras.callbacks import EarlyStopping, TensorBoard
import random
import os
import keras.backend as K

from network_agent import NetworkAgent, conv2d_bn, Selector, State


MEMO = "Deeplight"


class DeeplightAgent(NetworkAgent):

    def __init__(self,
                 num_phases,
                 num_actions,
                 path_set):

        super(DeeplightAgent, self).__init__(
            num_phases=num_phases,
            path_set=path_set)

        self.num_actions = num_actions

        self.q_network = self.build_network_flowsync() # Changed from build_network
        self.save_model("init_model")
        self.update_outdated = 0

        self.q_network_bar = self.build_network_from_copy(self.q_network)
        self.q_bar_outdated = 0
        if not self.para_set.SEPARATE_MEMORY:
            self.memory = self.build_memory()
        else:
            self.memory = self.build_memory_separate()
        self.average_reward = None

    def reset_update_count(self):

        self.update_outdated = 0
        self.q_bar_outdated = 0

    def set_update_outdated(self):

        self.update_outdated = - 2*self.para_set.UPDATE_PERIOD
        self.q_bar_outdated = 2*self.para_set.UPDATE_Q_BAR_FREQ

    def convert_state_to_input(self, state):

        ''' convert a state struct to the format for neural network input'''

        # This will now automatically pick up the new neighbor features
        # based on the modified self.para_set.LIST_STATE_FEATURE in agent.py
        input_list = []
        for feature_name in self.para_set.LIST_STATE_FEATURE:
            # Add a check to ensure the feature exists before getting it
            if hasattr(state, feature_name) and getattr(state, feature_name) is not None:
                input_list.append(getattr(state, feature_name))
            else:
                # If feature is missing, create a correctly-shaped zero array
                print(f"Warning: State object missing '{feature_name}'. Using zeros.")
                feature_shape = getattr(State, "D_"+feature_name.upper())
                zeros_input = np.zeros((1,) + feature_shape) # Add batch dimension
                input_list.append(zeros_input)

        return input_list


    def build_network_flowsync(self):
        '''
        Initialize a Q network for FlowSync
        This architecture uses an Attention Mechanism to weigh neighbor information.
        '''

        # --- 1. DEFINE INPUTS ---
        dic_input_node = {}
        for feature_name in self.para_set.LIST_STATE_FEATURE:
            dic_input_node[feature_name] = Input(shape=getattr(State, "D_"+feature_name.upper()),
                                                 name="input_"+feature_name)

        # --- 2. PROCESS LOCAL STATE ---
        # Process image feature (map_feature)
        cnn_features = self._cnn_network_structure(dic_input_node["map_feature"])

        # Process other local features
        local_features_list = []
        for feature_name in self.para_set.LIST_STATE_FEATURE:
            # Check if feature is NOT one of the special/neighbor ones
            if feature_name not in ["map_feature", "neighbor_queue_lengths", "neighbor_phases", "neighbor_mask", "historical_traffic"]:
                if feature_name in dic_input_node: # Ensure it's an active feature
                    local_features_list.append(Flatten()(dic_input_node[feature_name]))

        if not local_features_list:
            # Handle case where there are no other local features
            local_features_vec = Lambda(lambda x: K.zeros((K.shape(x)[0], 1)))(cnn_features) # Dummy 1-dim zero tensor
        else:
            local_features_vec = concatenate(local_features_list)

        local_features_vec = Dense(self.para_set.D_DENSE, activation="relu", name="local_features_dense")(local_features_vec)

        # Combine local image and vector features
        combined_local_features = concatenate([cnn_features, local_features_vec], name="combined_local_features")
        local_state_embedding = Dense(self.para_set.D_DENSE, activation="relu", name="local_state_embedding")(combined_local_features)


        # --- 3. PROCESS NEIGHBOR STATES (Value) ---
        # We create an "embedding" for each neighbor's state

        # Reshape queues from (Batch, 4, 12) to (Batch, 4, 12) - already correct
        neighbor_q = dic_input_node["neighbor_queue_lengths"]
        # Reshape phases from (Batch, 4, 1) to (Batch, 4, 1) - already correct
        neighbor_p = dic_input_node["neighbor_phases"]

        neighbor_state_features = concatenate([neighbor_q, neighbor_p], name="neighbor_state_features", axis=-1) # (Batch, 4, 13)

        # Create neighbor embeddings (These are the "Values" in Attention)
        # Use a shared Dense layer for all neighbors
        neighbor_embedding = Dense(self.para_set.D_DENSE, activation="relu", name="neighbor_embedding")(neighbor_state_features)


        # --- 4. ATTENTION MECHANISM ---
        # Use the agent's local state as the "Query"
        query = Dense(self.para_set.D_DENSE, activation="relu", name="attention_query")(local_state_embedding)
        query_reshaped = Reshape((1, self.para_set.D_DENSE))(query) # (Batch, 1, 20)

        # Use the neighbor embeddings as the "Key"
        key = Dense(self.para_set.D_DENSE, activation="relu", name="attention_key")(neighbor_embedding) # (Batch, 4, 20)

        # Calculate attention scores (dot-product attention)
        # (Batch, 1, 20) dot (Batch, 20, 4) -> (Batch, 1, 4)
        attention_scores = Dot(axes=[2, 1], name="attention_scores")([query_reshaped, Permute((2, 1))(key)])

        attention_scores = Reshape((4,))(attention_scores) # (Batch, 4)

        # Apply mask to ignore padded/non-existent neighbors
        # We add a large negative number to masked scores before softmax
        mask_adder = Lambda(lambda x: (1.0 - x) * -1e9, name="mask_adder")(dic_input_node["neighbor_mask"])
        attention_scores_masked = Add(name="attention_scores_masked")([attention_scores, mask_adder])

        # Softmax to get attention weights
        attention_weights = Activation("softmax", name="attention_weights")(attention_scores_masked)
        attention_weights_reshaped = Reshape((1, 4))(attention_weights) # (Batch, 1, 4)

        # Apply weights to neighbor embeddings (the "Value")
        # (Batch, 1, 4) dot (Batch, 4, D_DENSE) -> (Batch, 1, D_DENSE)
        context_vector = Dot(axes=[2, 1], name="context_vector")([attention_weights_reshaped, neighbor_embedding])
        context_vector = Reshape((self.para_set.D_DENSE,))(context_vector) # (Batch, D_DENSE)


        # --- 5. FINAL DECISION ---
        # Concatenate local state embedding with the neighbor context vector
        final_combined_state = concatenate([local_state_embedding, context_vector], name="final_state_representation")

        # Shared dense layer
        shared_dense = self._shared_network_structure(final_combined_state, self.para_set.D_DENSE)

        # --- THIS IS THE FIX ---
        # Build phase selector layer (as in original code)
        if "cur_phase" in self.para_set.STATE_FEATURE and self.para_set.STATE_FEATURE["cur_phase"] and self.para_set.PHASE_SELECTOR:
            list_selected_q_values = []
            for phase in range(self.num_phases):
                # Create the layers with regular variables, not locals()
                q_values_for_phase = self._separate_network_structure(
                    shared_dense, self.para_set.D_DENSE, self.num_actions, memo=phase)

                selector_for_phase = Selector(
                    phase, name="selector_{0}".format(phase))(dic_input_node["cur_phase"])

                q_values_selected = Multiply(name="multiply_{0}".format(phase))(
                    [q_values_for_phase,  # <-- Use the variable directly
                     selector_for_phase] # <-- Use the variable directly
                )

                list_selected_q_values.append(q_values_selected)
            q_values = Add()(list_selected_q_values)
        else:
            q_values = self._separate_network_structure(shared_dense, self.para_set.D_DENSE, self.num_actions)

        # Create the list of inputs for the Model
        model_inputs = [dic_input_node[feature_name] for feature_name in self.para_set.LIST_STATE_FEATURE]

        network = Model(inputs=model_inputs,
                        outputs=q_values)
        network.compile(optimizer=RMSprop(learning_rate=self.para_set.LEARNING_RATE), # Keras 3 uses 'learning_rate'
                        loss="mean_squared_error")
        network.summary()

        return network


    def build_network(self):
        # This function is retained for compatibility but now defaults
        # to the new FlowSync architecture.
        return self.build_network_flowsync()

    def build_memory_separate(self):
        memory_list=[]
        for i in range(self.num_phases):
            memory_list.append([[] for j in range(self.num_actions)])
        return memory_list

    def remember(self, state, action, reward, next_state):

        if self.para_set.SEPARATE_MEMORY:
            ''' log the history separately '''
            # Ensure cur_phase is valid
            phase_index = state.cur_phase[0][0]
            if phase_index < self.num_phases:
                self.memory[phase_index][action].append([state, action, reward, next_state])
            else:
                print(f"Warning: Invalid phase index {phase_index} in remember(). Max is {self.num_phases-1}")
        else:
            self.memory.append([state, action, reward, next_state])

    def forget(self, if_pretrain):

        if self.para_set.SEPARATE_MEMORY:
            ''' remove the old history if the memory is too large, in a separate way '''
            for phase_i in range(self.num_phases):
                for action_i in range(self.num_actions):
                    if if_pretrain:
                        random.shuffle(self.memory[phase_i][action_i])
                    if len(self.memory[phase_i][action_i]) > self.para_set.MAX_MEMORY_LEN:
                        print("length of memory (state {0}, action {1}): {2}, before forget".format(
                            phase_i, action_i, len(self.memory[phase_i][action_i])))
                        self.memory[phase_i][action_i] = self.memory[phase_i][action_i][-self.para_set.MAX_MEMORY_LEN:]
                    # print("length of memory (state {0}, action {1}): {2}, after forget".format(
                    #     phase_i, action_i, len(self.memory[phase_i][action_i])))
        else:
            if len(self.memory) > self.para_set.MAX_MEMORY_LEN:
                print("length of memory: {0}, before forget".format(len(self.memory)))
                self.memory = self.memory[-self.para_set.MAX_MEMORY_LEN:]
            # print("length of memory: {0}, after forget".format(len(self.memory)))

    def _cal_average(self, sample_memory):

        list_reward = []
        average_reward = np.zeros((self.num_phases, self.num_actions))
        for phase_i in range(self.num_phases):
            list_reward.append([])
            for action_i in range(self.num_actions):
                list_reward[phase_i].append([])
        for [state, action, reward, _] in sample_memory:
            phase = state.cur_phase[0][0]
            if phase < self.num_phases:
                list_reward[phase][action].append(reward)

        for phase_i in range(self.num_phases):
            for action_i in range(self.num_actions):
                if len(list_reward[phase_i][action_i]) != 0:
                    average_reward[phase_i][action_i] = np.average(list_reward[phase_i][action_i])

        return average_reward

    def _cal_average_separate(self,sample_memory):
        ''' Calculate average rewards for different cases '''

        average_reward = np.zeros((self.num_phases, self.num_actions))
        for phase_i in range(self.num_phases):
            for action_i in range(self.num_actions):
                len_sample_memory = len(sample_memory[phase_i][action_i])
                if len_sample_memory > 0:
                    list_reward = []
                    for i in range(len_sample_memory):
                        state, action, reward, _ = sample_memory[phase_i][action_i][i]
                        list_reward.append(reward)
                    average_reward[phase_i][action_i]=np.average(list_reward)
        return average_reward

    def get_sample(self, memory_slice, dic_state_feature_arrays, Y, gamma, prefix, use_average):

        len_memory_slice = len(memory_slice)
        if len_memory_slice == 0:
            return dic_state_feature_arrays, Y

        f_samples = None
        try:
            f_samples = open(os.path.join(self.path_set.PATH_TO_OUTPUT, "{0}_memory.log".format(prefix)), "a")
        except IOError as e:
            print(f"Warning: Could not open memory log file. {e}")

        for i in range(len_memory_slice):
            state, action, reward, next_state = memory_slice[i]

            # Add state features to the dictionary for training
            for feature_name in self.para_set.LIST_STATE_FEATURE:
                dic_state_feature_arrays[feature_name].append(getattr(state, feature_name)[0]) # [0] to remove batch dim

            if state.if_terminal:
                next_estimated_reward = 0
            else:
                next_estimated_reward = self._get_next_estimated_reward(next_state)

            total_reward = reward + gamma * next_estimated_reward

            if not use_average:
                target = self.q_network.predict(
                    self.convert_state_to_input(state))
            else:
                # Ensure average_reward is initialized
                if self.average_reward is None:
                    self.average_reward = np.zeros((self.num_phases, self.num_actions))
                phase_index = state.cur_phase[0][0]
                if phase_index >= self.num_phases:
                    phase_index = 0 # Safety clamp
                target = np.copy(np.array([self.average_reward[phase_index]]))

            pre_target = np.copy(target)
            target[0][action] = total_reward
            Y.append(target[0])

            if f_samples:
                try:
                    log_line = ""
                    for feature_name in self.para_set.LIST_STATE_FEATURE:
                        if "map" not in feature_name and hasattr(state, feature_name):
                            log_line += "{0}\t".format(str(getattr(state, feature_name)))
                    log_line += "{0}\t{1}\t{2}\t{3}\t{4}\n".format(
                        str(pre_target), str(target),
                        str(action), str(reward), str(next_estimated_reward)
                    )
                    f_samples.write(log_line)
                except Exception as e:
                    print(f"Error writing to log: {e}")

        if f_samples:
            f_samples.close()

        return dic_state_feature_arrays, Y

    def train_network(self, Xs, Y, prefix, if_pretrain):

        if if_pretrain:
            epochs = self.para_set.EPOCHS_PRETRAIN
        else:
            epochs = self.para_set.EPOCHS
        batch_size = min(self.para_set.BATCH_SIZE, len(Y))

        if batch_size == 0:
            print("Warning: No data to train on. Skipping train_network.")
            return

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=self.para_set.PATIENCE, verbose=0, mode='min')

        hist = self.q_network.fit(Xs, Y, batch_size=batch_size, epochs=epochs,
                                  shuffle=False,
                                  verbose=2, validation_split=0.3, callbacks=[early_stopping])
        self.save_model(prefix)

    def update_network(self, if_pretrain, use_average, current_time):

        ''' update Q network '''

        if current_time - self.update_outdated < self.para_set.UPDATE_PERIOD:
            return

        self.update_outdated = current_time

        # prepare the samples
        if if_pretrain:
            gamma = self.para_set.GAMMA_PRETRAIN
        else:
            gamma = self.para_set.GAMMA

        dic_state_feature_arrays = {}
        for feature_name in self.para_set.LIST_STATE_FEATURE:
            dic_state_feature_arrays[feature_name] = []
        Y = []

        # get average state-action reward
        if self.para_set.SEPARATE_MEMORY:
            self.average_reward = self._cal_average_separate(self.memory)
        else:
            self.average_reward = self._cal_average(self.memory)

        # ================ sample memory ====================
        if self.para_set.SEPARATE_MEMORY:
            for phase_i in range(self.num_phases):
                for action_i in range(self.num_actions):
                    sampled_memory = self._sample_memory(
                        gamma=gamma,
                        with_priority=self.para_set.PRIORITY_SAMPLING,
                        memory=self.memory[phase_i][action_i],
                        if_pretrain=if_pretrain)
                    dic_state_feature_arrays, Y = self.get_sample(
                        sampled_memory, dic_state_feature_arrays, Y, gamma, current_time, use_average)
        else:
            sampled_memory = self._sample_memory(
                gamma=gamma,
                with_priority=self.para_set.PRIORITY_SAMPLING,
                memory=self.memory,
                if_pretrain=if_pretrain)
            dic_state_feature_arrays, Y = self.get_sample(
                sampled_memory, dic_state_feature_arrays, Y, gamma, current_time, use_average)
        # ================ sample memory ====================

        if len(Y) == 0:
            print("Not enough memory to train. Skipping update.")
            return

        Xs = [np.array(dic_state_feature_arrays[feature_name]) for feature_name in self.para_set.LIST_STATE_FEATURE]
        Y = np.array(Y)
        sample_weight = np.ones(len(Y))

        # Check for data integrity
        try:
            Xs, Y, _ = self._unison_shuffled_copies(Xs, Y, sample_weight)
        except ValueError as e:
            print(f"Error during shuffling, likely due to inconsistent array lengths: {e}")
            for i, (feature_name, arr) in enumerate(zip(self.para_set.LIST_STATE_FEATURE, Xs)):
                print(f"  Input {i} ({feature_name}): {arr.shape}")
            print(f"  Labels (Y): {Y.shape}")
            return # Skip update

        # ============================  training  =======================================

        self.train_network(Xs, Y, current_time, if_pretrain)
        self.q_bar_outdated += 1
        self.forget(if_pretrain=if_pretrain)

    def _sample_memory(self, gamma, with_priority, memory, if_pretrain):

        len_memory = len(memory)

        if len_memory == 0:
            return []

        if not if_pretrain:
            sample_size = min(self.para_set.SAMPLE_SIZE, len_memory)
        else:
            sample_size = min(self.para_set.SAMPLE_SIZE_PRETRAIN, len_memory)

        if sample_size == 0:
            return []

        if with_priority:
            # sample with priority
            sample_weight = []
            for i in range(len_memory):
                state, action, reward, next_state = memory[i]

                if state.if_terminal:
                    next_estimated_reward = 0
                else:
                    next_estimated_reward = self._get_next_estimated_reward(next_state)

                total_reward = reward + gamma * next_estimated_reward
                target = self.q_network.predict(
                    self.convert_state_to_input(state))
                pre_target = np.copy(target)
                target[0][action] = total_reward

                # get the bias of current prediction
                weight = abs(pre_target[0][action] - total_reward)
                sample_weight.append(weight)

            priority = self._cal_priority(sample_weight)
            p_indices = np.random.choice(range(len(priority)), size=sample_size, p=priority, replace=False)
            sampled_memory = [memory[i] for i in p_indices]
        else:
            sampled_memory = random.sample(memory, sample_size)

        return sampled_memory

    @staticmethod
    def _cal_priority(sample_weight):
        pos_constant = 0.0001
        alpha = 1
        sample_weight_np = np.array(sample_weight)
        sample_weight_np_sum = sample_weight_np.sum()
        if sample_weight_np_sum == 0:
            # All weights are zero, use uniform probability
            return np.full(len(sample_weight), 1.0/len(sample_weight))

        sample_weight_np = np.power(sample_weight_np + pos_constant, alpha) / (sample_weight_np_sum + 1e-6) # Added epsilon for stability
        # Re-normalize just in case
        return sample_weight_np / (sample_weight_np.sum() + 1e-6)