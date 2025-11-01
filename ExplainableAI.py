# -*- coding: utf-8 -*-

'''
This module implements the Explainable AI (XAI) layer for FlowSync,
as described in the presentation.

It uses LIME (Local Interpretable Model-agnostic Explanations)
to explain why a specific decision (action) was taken given a state.

Installation required:
pip install lime
'''

import lime
import lime.lime_tabular
import numpy as np
import os

class FlowSyncExplainer:

    def __init__(self, agent, path_set):
        '''
        Initializes the explainer.

        :param agent: The DeeplightAgent (FlowSync) to explain.
        :param path_set: The experiment path set for saving outputs.
        '''

        print("Initializing FlowSync XAI Explainer...")
        self.agent = agent
        self.path_set = path_set
        self.explainer = None
        self.feature_names = agent.para_set.LIST_STATE_FEATURE
        self.output_dir = os.path.join(self.path_set.PATH_TO_OUTPUT, "xai_explanations")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _get_training_data_summary(self):
        '''
        Creates a data summary from the agent's memory for LIME.
        This is needed to establish feature distributions.
        '''
        memory = self.agent.memory
        if len(memory) < 100:
            print("Warning: XAI explainer has very little memory to build a summary.")
            if len(memory) == 0:
                return None

        # Sample up to 1000 points from memory
        sample_size = min(1000, len(memory))
        samples = np.random.choice(memory, sample_size, replace=False)

        # Convert states to a flat numpy array
        flat_data = []
        for state, _, _, _ in samples:
            flat_state = self._flatten_state(state)
            flat_data.append(flat_state)

        return np.array(flat_data)

    def _flatten_state(self, state):
        '''
        Flattens a complex State object into a 1D numpy array for LIME.
        '''
        flat_state_list = []
        for feature_name in self.feature_names:
            feature_data = getattr(state, feature_name)
            flat_state_list.append(feature_data.flatten())

        return np.concatenate(flat_state_list)

    def _get_model_prediction_function(self):
        '''
        Creates a wrapper function for LIME that takes a flat 1D array
        and returns the model's Q-value predictions.
        '''

        def predict_fn(flat_states_1d):
            # flat_states_1d is (num_samples, num_flat_features)

            q_values_list = []

            for flat_state in flat_states_1d:
                # 1. Reconstruct the State object
                # This is complex. We must reconstruct the list of inputs for model.predict
                model_inputs = []
                current_idx = 0
                for feature_name in self.feature_names:
                    feature_shape = getattr(State, "D_" + feature_name.UPPER())
                    feature_size = np.prod(feature_shape)

                    # Get the data chunk for this feature
                    data_chunk = flat_state[current_idx : current_idx + feature_size]

                    # Reshape to (1, *feature_shape) for model.predict
                    model_input = data_chunk.reshape((1,) + feature_shape)
                    model_inputs.append(model_input)

                    current_idx += feature_size

                # 2. Get Q-values
                q_values = self.agent.q_network.predict(model_inputs)
                q_values_list.append(q_values[0]) # q_values is (1, num_actions)

            return np.array(q_values_list) # (num_samples, num_actions)

        return predict_fn

    def explain_decision(self, state, action, current_time):
        '''
        Generates and saves an explanation for a single decision.
        '''

        print(f"XAI: Generating explanation for time {current_time}...")

        training_summary = self._get_training_data_summary()
        if training_summary is None:
            print("XAI: Cannot generate explanation, no training data summary.")
            return

        # Initialize LIME explainer
        # We create a new explainer each time, as the data summary might change
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_summary,
            feature_names=self._get_flat_feature_names(),
            class_names=["Keep_Phase", "Change_Phase"], # Assumes 2 actions
            verbose=False,
            mode='regression' # We are explaining Q-values (continuous)
        )

        flat_state_to_explain = self._flatten_state(state)
        prediction_fn = self._get_model_prediction_function()

        try:
            # Explain the Q-value for the action that was *taken*
            explanation = self.explainer.explain_instance(
                data_row=flat_state_to_explain,
                predict_fn=prediction_fn,
                num_features=10, # Show top 10 features
                top_labels=None, # Explain all classes
                num_samples=1000 # Number of perturbations
            )

            # Save the explanation to an HTML file
            output_file = os.path.join(self.output_dir, f"explanation_t{current_time}_action{action}.html")
            explanation.save_to_file(output_file)
            print(f"XAI: Explanation saved to {output_file}")

        except Exception as e:
            print(f"XAI: Error during explanation generation: {e}")

    def _get_flat_feature_names(self):
        ''' Helper to create a 1D list of feature names for LIME '''
        flat_names = []
        for feature_name in self.feature_names:
            shape = getattr(State, "D_" + feature_name.UPPER())
            size = np.prod(shape)
            if size == 1:
                flat_names.append(feature_name)
            else:
                # Create indexed names, e.g., "queue_length_0", "queue_length_1"
                for i in range(size):
                    flat_names.append(f"{feature_name}_{i}")
        return flat_names