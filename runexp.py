'''
@author: hzw77, gjz5038

python runexp.py

--- FLOWSYNC MODIFIED ---
This script is now a simple launcher for the multi-agent
traffic_light_dqn.py coordinator.
'''

import random
import numpy as np
import json
import os
import traffic_light_dqn  # This is our new coordinator
import time

# ================== CONFIGURATION ==================
SEED = 31200
setting_memo = "one_run" # This points to conf/one_run and data/one_run
# ===================================================

# --- Set random seeds ---
random.seed(SEED)
np.random.seed(SEED)
try:
    from tensorflow import set_random_seed
    set_random_seed(SEED)
except ImportError:
    import tensorflow as tf
    tf.random.set_seed(SEED)

# --- Get base paths ---
PATH_TO_CONF = os.path.join("conf", setting_memo)
base_path = os.path.split(os.path.realpath(__file__))[0]
data_path = os.path.join(base_path, 'data', setting_memo)

# --- Define SUMO commands ---
# Make sure the paths to sumo/sumo-gui are correct for your system
# --- Define SUMO commands ---
# Make sure the paths to sumo/sumo-gui are correct for your system
# --- Define SUMO commands ---
# Make sure the paths to sumo/sumo-gui are correct for your system
sumoBinary_nogui = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
sumoBinary_gui = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"

# Use GUI for testing to see what's happening
sumoCmd = [sumoBinary_gui,
           '-c',
           os.path.join(data_path, 'cross.sumocfg')]

sumoCmd_nogui = [sumoBinary_nogui,
                 '-c',
                 os.path.join(data_path, 'cross.sumocfg')]

# The pretrain config is also needed
sumoCmd_nogui_pretrain = [sumoBinary_nogui,
                          '-c',
                          os.path.join(data_path, 'cross_pretrain.sumocfg')]


# --- Load and set experiment config ---
# This ensures the correct model and traffic files are referenced
try:
    dic_exp = json.load(open(os.path.join(PATH_TO_CONF, "exp.conf"), "r"))
    dic_exp["MODEL_NAME"] = "Deeplight" # This is our FlowSync model

    # Use one of the traffic files
    dic_exp["TRAFFIC_FILE"] = ["cross.all_synthetic.rou.xml"]
    dic_exp["TRAFFIC_FILE_PRETRAIN"] = ["cross.all_synthetic.rou.xml"]

    json.dump(dic_exp, open(os.path.join(PATH_TO_CONF, "exp.conf"), "w"), indent=4)
except FileNotFoundError:
    print(f"Error: Could not find config files in {PATH_TO_CONF}")
    exit(1)


# --- Define a unique prefix for this run ---
prefix = "{0}_{1}_{2}".format(
    dic_exp["MODEL_NAME"],
    dic_exp["TRAFFIC_FILE"][0].split('.')[0],
    time.strftime('%m_%d_%H_%M_%S_', time.localtime(time.time())) + "seed_%d" % SEED
)

# --- Start the main coordinator ---
print(f"Starting FlowSync experiment: {prefix}")
print(f"Using SUMO config: {sumoCmd[2]}")

# We pass the GUI command for testing, and the nogui command for pretraining
# The new main function will handle the rest
traffic_light_dqn.main(
    memo=setting_memo,
    f_prefix=prefix,
    sumo_cmd_str=sumoCmd_nogui,  # Use GUI to watch
    sumo_cmd_pretrain_str=sumoCmd_nogui_pretrain
)

print(f"Experiment {prefix} finished.")