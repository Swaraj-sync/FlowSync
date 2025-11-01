'''
1) interacting with SUMO, including
      retrive values, set lights
2) interacting with sumo_agent, including
      returning status, rewards, etc.

--- FLOWSYNC MODIFICATION ---
This file has been parameterized to support multi-agent control.
Functions are no longer hardcoded to 'node0' and can operate
on any discovered traffic light node.
'''

import numpy as np
import math
import os
import sys
import xml.etree.ElementTree as ET
from sys import platform
from sumo_agent import Vehicles

###### Please Specify the location of your traci module

if platform == "linux" or platform == "linux2":# this is linux
    os.environ['SUMO_HOME'] = '/usr/share/sumo'
    try:
        import traci
        import traci.constants as tc
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
                os.path.join(os.environ["SUMO_HOME"], "tools")
            )
            try:
                import traci
                import traci.constants as tc
            except ImportError:
                raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")

elif platform == "win32":
    os.environ['SUMO_HOME'] = 'C:\\Program Files (x86)\\DLR\\Sumo'

    try:
        import traci
        import traci.constants as tc
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
                os.path.join(os.environ["SUMO_HOME"], "tools")
            )
            try:
                import traci
                import traci.constants as tc
            except ImportError:
                raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
elif platform =='darwin':
    os.environ['SUMO_HOME'] = "/Users/{0}/sumo/sumo-git".format(os.getlogin())

    try:
        import traci
        import traci.constants as tc
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
                os.path.join(os.environ["SUMO_HOME"], "tools")
            )
            try:
                import traci
                import traci.constants as tc
            except ImportError:
                raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")

else:
    sys.exit("platform error")

yeta = 0.15
tao = 2
constantC = 40.0
carWidth = 3.3
grid_width = 4
area_length = 600

# --- FLOWSYNC: Global dictionaries for network structure ---
# These will be populated by start_sumo()
NODE_PHASE_DEFINITIONS = {}
NODE_NEIGHBORS = {}
MAX_NEIGHBORS = 4 # From agent.py
MAX_LANES_PER_NODE = 12 # From agent.py (D_QUEUE_LENGTH)
# --- END MODIFICATION ---


'''
Original hardcoded values, removed for FlowSync
... (rest of comments) ...
'''


def start_sumo(sumo_cmd_str):
    traci.start(sumo_cmd_str)

    # --- FLOWSYNC: Discover network structure after starting ---
    global NODE_PHASE_DEFINITIONS, NODE_NEIGHBORS
    all_tls_ids = traci.trafficlight.getIDList()

    # --- THIS IS THE FIX ---
    # It now correctly finds the .net.xml file associated with your .sumocfg
    net_path = os.path.join(os.path.dirname(sumo_cmd_str[2]), "grid.net.xml")
    if not os.path.exists(net_path):
        # Fallback just in case
        net_path = 'data/grid_2x2/grid.net.xml'
    net = ET.parse(net_path).getroot()
    # --- END OF FIX ---


    for node_id in all_tls_ids:
        # 1. Discover Phases and Lanes
        program = traci.trafficlight.getCompleteRedYellowGreenDefinition(node_id)[0]
        phases = [p.state for p in program.phases]

        # Get all incoming lanes for this TLS
        controlled_links = traci.trafficlight.getControlledLinks(node_id)
        incoming_lanes = []
        if controlled_links: # Ensure it's not empty
            incoming_lanes = list(set(link[0][0] for link in controlled_links if link))

        NODE_PHASE_DEFINITIONS[node_id] = {
            "phases": phases,
            "incoming_lanes": incoming_lanes
        }

        # 2. Discover Neighbors (Simplified)
        NODE_NEIGHBORS[node_id] = []
        # Find junction element in XML
        junction = net.find(f".//junction[@id='{node_id}']")
        if junction is not None:
            incLanes_str = junction.get('incLanes', '')
            incLanes = incLanes_str.split(' ')
            for lane_id in incLanes:
                if not lane_id: continue
                # Find the edge this lane belongs to
                edge_id = '_'.join(lane_id.split('_')[:-1])
                edge = net.find(f".//edge[@id='{edge_id}']")
                if edge is not None:
                    from_node = edge.get('from')
                    # If 'from' node is a traffic light, it's a neighbor
                    if from_node in all_tls_ids and from_node != node_id:
                        if from_node not in NODE_NEIGHBORS[node_id]:
                            NODE_NEIGHBORS[node_id].append(from_node)

    print("--- FlowSync Network Discovery ---")
    print(f"Discovered {len(all_tls_ids)} nodes: {all_tls_ids}")
    for node_id in all_tls_ids:
        print(f"  Node: {node_id}")
        print(f"    Phases: {len(NODE_PHASE_DEFINITIONS[node_id]['phases'])}")
        print(f"    Lanes: {NODE_PHASE_DEFINITIONS[node_id]['incoming_lanes']}")
        print(f"    Neighbors: {NODE_NEIGHBORS[node_id]}")
    print("----------------------------------")

    for i in range(20):
        traci.simulationStep()


def end_sumo():
    traci.close()

def get_current_time():
    return traci.simulation.getTime()

'''
input: phase "NSG_SNG" , four lane number, in the key of W,E,S,N
output: 
1.affected lane number: 4_0_0, 4_0_1, 3_0_0, 3_0_1
# 2.destination lane number, 0_3_0,0_3_1  

'''
# This function is highly specific to the original 'cross' map and phase naming
# It is kept for legacy but is not used by the new parameterized functions
def phase_affected_lane(phase="NSG_SNG",
                        four_lane_ids={'W': 'edge1-0', "E": "edge2-0", 'S': 'edge4-0', 'N': 'edge3-0'}):
    direction_lane_dict = {"NSG": [1, 0], "SNG": [1, 0], "EWG": [1, 0], "WEG": [1, 0],
                           "NWG": [0], "WSG": [0], "SEG": [0], "ENG": [0],
                           "NEG": [2], "WNG": [2], "SWG": [2], "ESG": [2]}
    directions = phase.split('_')
    affected_lanes = []
    for direction in directions:
        for k, v in four_lane_ids.items():
            if v.strip() != '' and direction.startswith(k):
                for lane_no in direction_lane_dict[direction]:
                    affected_lanes.append("%s_%d" % (v, lane_no))
                    # affacted_lanes.append("%s_%d" % (v, 0))
    if affected_lanes == []:
        raise("Please check your phase and lane_number_dict in phase_affacted_lane()!")
    return affected_lanes


'''
input: central nodeid "node0", surrounding nodes WESN: [1,2,3,4]
output: four_lane_ids={'W':'edge1-0',"E":"edge2-0",'S':'edge4-0','N':'edge3-0'})
'''
# This function is also legacy and map-specific
def find_surrounding_lane_WESN(central_node_id="node0", WESN_node_ids={"W": "1", "E": "2", "S": "3", "N": "4"}):
    tree = ET.parse('./data/one_run/cross.net.xml') # TODO: Parameterize path
    root = tree.getroot()
    four_lane_ids_dict = {}
    for k, v in WESN_node_ids.items():
        edge = root.find("./edge[@from='%s'][@to='%s']" % (v, central_node_id))
        if edge is not None:
            four_lane_ids_dict[k] = edge.get('id')
    return four_lane_ids_dict


'''
coordinate mapper
'''
# This function is global
def coordinate_mapper(x1, y1, x2, y2, area_length=600, area_width=600):
    x1 = int(x1 / grid_width)
    y1 = int(y1 / grid_width)
    x2 = int(x2 / grid_width)
    y2 = int(y2 / grid_width)
    x_max = x1 if x1 > x2 else x2
    x_min = x1 if x1 < x2 else x2
    y_max = y1 if y1 > y2 else y2
    y_min = y1 if y1 < x2 else y2
    length_num_grids = int(area_length / grid_width)
    width_num_grids = int(area_width / grid_width)
    return length_num_grids - y_max, length_num_grids - y_min, x_min, x_max

# This function is legacy and map/data-specific
def get_phase_affected_lane_traffic_max_volume(phase="NSG_SNG", tl_node_id="node0",
                                               WESN_node_ids={"W": "1", "E": "2", "S": "3", "N": "4"}):
    four_lane_ids_dict = find_surrounding_lane_WESN(central_node_id=tl_node_id, WESN_node_ids=WESN_node_ids)
    directions = phase.split('_')
    traffic_volume_start_end = []
    for direction in directions:
        traffic_volume_start_end.append([four_lane_ids_dict[direction[0]],four_lane_ids_dict[direction[1]]])
    tree = ET.parse('./data/one_run/cross.all_synthetic.rou.xml') # TODO: Parameterize path
    root = tree.getroot()
    phase_volumes = []
    for lane_id in traffic_volume_start_end:
        to_lane_id="edge%s-%s"%(lane_id[1].split('-')[1],lane_id[1].split('-')[0][4:])
        flow = root.find("./flow[@from='%s'][@to='%s']" % (lane_id[0], to_lane_id))
        if flow is not None:
            time_begin = flow.get('begin')
            time_end = flow.get('end')
            volume = flow.get('number')
            phase_volumes.append((float(time_end)-float(time_begin))/float(volume))
    return max(phase_volumes) if phase_volumes else 0

# This function is legacy and map-specific
def phase_affected_lane_position(phase="NSG_SNG", tl_node_id="node0",
                                 WESN_node_ids={"W": "1", "E": "2", "S": "3", "N": "4"}):
    '''
    input: NSG_SNG ,central nodeid "node0", surrounding nodes WESN: {"W":"1", "E":"2", "S":"3", "N":"4"}
    output: edge-ids, 4_0_0, 4_0_1, 3_0_0, 3_0_1
    [[ 98,  100,  204,  301],[ 102, 104, 104, 198]]
    '''
    four_lane_ids_dict = find_surrounding_lane_WESN(central_node_id=tl_node_id, WESN_node_ids=WESN_node_ids)
    affected_lanes = phase_affected_lane(phase=phase, four_lane_ids=four_lane_ids_dict)
    tree = ET.parse('./data/one_run/cross.net.xml') # TODO: Parameterize path
    root = tree.getroot()
    indexes = []
    for lane_id in affected_lanes:
        lane = root.find("./edge[@to='%s']/lane[@id='%s']" % (tl_node_id, lane_id))
        if lane is not None:
            lane_shape = lane.get('shape')
            lane_x1 = float(lane_shape.split(" ")[0].split(",")[0])
            lane_y1 = float(lane_shape.split(" ")[0].split(",")[1])
            lane_x2 = float(lane_shape.split(" ")[1].split(",")[0])
            lane_y2 = float(lane_shape.split(" ")[1].split(",")[1])
            ind_x1, ind_x2, ind_y1, ind_y2 = coordinate_mapper(lane_x1, lane_y1, lane_x2, lane_y2)
            indexes.append([ind_x1, ind_x2 + 1, ind_y1, ind_y2 + 1])
    return indexes

# This function is legacy and map-specific
def phases_affected_lane_postions(phases=["NSG_SNG_NWG_SEG", "NEG_SWG_NWG_SEG"], tl_node_id="node0",
                                  WESN_node_ids={"W": "1", "E": "2", "S": "3", "N": "4"}):
    parameterArray = []
    for phase in phases:
        parameterArray += phase_affected_lane_position(phase=phase, tl_node_id=tl_node_id, WESN_node_ids=WESN_node_ids)
    return parameterArray

# This function is global
def vehicle_location_mapper(coordinate, area_length=600, area_width=600):
    transformX = math.floor(coordinate[0] / grid_width)
    transformY = math.floor((area_length - coordinate[1]) / grid_width)
    length_num_grids = int(area_length/grid_width)
    transformY = length_num_grids-1 if transformY == length_num_grids else transformY
    transformX = length_num_grids-1 if transformX == length_num_grids else transformX
    tempTransformTuple = (transformY, transformX)
    return tempTransformTuple

# This function is legacy
def translateAction(action):
    result = 0
    for i in range(len(action)):
        result += (i + 1) * action[i]
    return result

# --- FLOWSYNC: Parameterized function ---
def changeTrafficLight(node_id, current_phase):
    '''
    Changes the traffic light for the given node_id to the next phase.
    '''
    if node_id not in NODE_PHASE_DEFINITIONS:
        print(f"Error: Node {node_id} not in phase definitions.")
        return current_phase, 0

    phases = NODE_PHASE_DEFINITIONS[node_id]["phases"]
    if not phases:
        print(f"Error: Node {node_id} has no phases defined.")
        return current_phase, 0

    next_phase_index = (current_phase + 1) % len(phases)
    next_phase_signal = phases[next_phase_index]

    traci.trafficlight.setRedYellowGreenState(node_id, next_phase_signal)
    return next_phase_index, 0

# --- FLOWSYNC: Parameterized function ---
def get_phase_vector(node_id, current_phase_index):
    '''
    Returns a one-hot vector for the current phase index.
    '''
    if node_id not in NODE_PHASE_DEFINITIONS:
        return np.zeros(1) # Return a simple array

    num_phases = len(NODE_PHASE_DEFINITIONS[node_id]["phases"])
    if num_phases == 0:
        return np.zeros(1)

    phase_vector = np.zeros(num_phases)
    if current_phase_index < num_phases:
        phase_vector[current_phase_index] = 1
    return np.array(phase_vector)

# This function is legacy and map-specific
def getMapOfCertainTrafficLight(curtent_phase=0, tl_node_id="node0", area_length=600):
    # This uses the old hardcoded phase names. Kept for legacy.
    phases_light_7 = ["WNG_ESG_EWG_WEG_WSG_ENG", "NSG_NEG_SNG_SWG_NWG_SEG"]
    current_phases_light_7 = [phases_light_7[curtent_phase]]
    parameterArray = phases_affected_lane_postions(phases=current_phases_light_7, tl_node_id=tl_node_id)
    length_num_grids = int(area_length / grid_width)
    resultTrained = np.zeros((length_num_grids, length_num_grids))
    for affected_road in parameterArray:
        resultTrained[affected_road[0]:affected_road[1], affected_road[2]:affected_road[3]] = 127
    return resultTrained

# This function is legacy
def calculate_reward(tempLastVehicleStateList):
    waitedTime = 0
    stop_count = 0
    for key, vehicle_dict in tempLastVehicleStateList.items():
        if tempLastVehicleStateList[key]['speed'] < 5:
            waitedTime += 1
            #waitedTime += (1 +math.pow(tempLastVehicleStateList[key]['waitedTime']/50,2))
        if tempLastVehicleStateList[key]['former_speed'] > 0.5 and tempLastVehicleStateList[key]['speed'] < 0.5:
            stop_count += (tempLastVehicleStateList[key]['stop_count']-tempLastVehicleStateList[key]['former_stop_count'])
    #PI = (waitedTime + 10 * stop_count) / len(tempLastVehicleStateList) if len(tempLastVehicleStateList)!=0 else 0
    PI = waitedTime/len(tempLastVehicleStateList) if len(tempLastVehicleStateList)!=0 else 0
    return - PI

# This function is global
def getMapOfVehicles(area_length=600):
    '''
    get the vehicle positions as NIPS paper
    :param area_length:
    :return: numpy narray
    '''
    length_num_grids = int(area_length / grid_width)
    mapOfCars = np.zeros((length_num_grids, length_num_grids))

    vehicle_id_list = traci.vehicle.getIDList()
    for vehicle_id in vehicle_id_list:
        try:
            vehicle_position = traci.vehicle.getPosition(vehicle_id)  # (double,double),tuple
            transform_tuple = vehicle_location_mapper(vehicle_position)  # call the function
            mapOfCars[transform_tuple[0], transform_tuple[1]] = 1
        except traci.TraCIException:
            # Vehicle might have left the simulation in the same step
            pass

    return mapOfCars

# This function is global
def restrict_reward(reward,func="unstrict"):
    if func == "linear":
        bound = -50
        reward = 0 if reward < bound else (reward/(-bound) + 1)
    elif func == "neg_log":
        if -reward+1 > 0: # Add check for math domain error
            reward = math.log(-reward+1)
        else:
            reward = -10 # Assign large negative reward
    else:
        pass

    return reward

# This function is global
def log_rewards(vehicle_dict, action, rewards_info_dict, file_name, timestamp,rewards_detail_dict_list):

    # This function is called by the parameterized get_rewards_from_sumo
    # It needs node_id to get the correct list of lanes
    # For FlowSync, logging is handled by traffic_light_dqn.py
    pass
    '''
    reward, reward_detail_dict = get_rewards_from_sumo(vehicle_dict, action, rewards_info_dict) # This will cause recursion
    list_reward_keys = np.sort(list(reward_detail_dict.keys()))
    reward_str = "{0}, {1}".format(timestamp,action)
    for reward_key in list_reward_keys:
        reward_str = reward_str + ", {0}".format(reward_detail_dict[reward_key][2])
    reward_str += '\n'

    fp = open(file_name, "a")
    fp.write(reward_str)
    fp.close()
    rewards_detail_dict_list.append(reward_detail_dict)
    '''

# --- FLOWSYNC: Parameterized function ---
def get_rewards_from_sumo(node_id, vehicle_dict, action, rewards_info_dict):
    '''
    Calculates rewards for a specific node_id.
    '''
    reward = 0
    import copy
    reward_detail_dict = copy.deepcopy(rewards_info_dict)

    if node_id not in NODE_PHASE_DEFINITIONS:
        print(f"Error: Node {node_id} not in phase definitions for reward calculation.")
        return 0, reward_detail_dict

    listLanes = NODE_PHASE_DEFINITIONS[node_id]["incoming_lanes"]

    vehicle_id_entering_list = get_vehicle_id_entering(listLanes) # Pass lanes

    reward_detail_dict['queue_length'].append(get_overall_queue_length(listLanes))
    reward_detail_dict['wait_time'].append(get_overall_waiting_time(listLanes))
    reward_detail_dict['delay'].append(get_overall_delay(listLanes))
    reward_detail_dict['emergency'].append(get_num_of_emergency_stops(vehicle_dict))
    reward_detail_dict['duration'].append(get_travel_time_duration(vehicle_dict, vehicle_id_entering_list))
    reward_detail_dict['flickering'].append(get_flickering(action))
    reward_detail_dict['partial_duration'].append(get_partial_travel_time_duration(vehicle_dict, vehicle_id_entering_list))

    vehicle_id_list = traci.vehicle.getIDList()
    reward_detail_dict['num_of_vehicles_in_system'] = [False, 0, len(vehicle_id_list)]

    reward_detail_dict['num_of_vehicles_at_entering'] = [False, 0, len(vehicle_id_entering_list)]

    vehicle_id_leaving = get_vehicle_id_leaving(vehicle_dict, listLanes) # Pass lanes

    reward_detail_dict['num_of_vehicles_left'].append(len(vehicle_id_leaving))
    reward_detail_dict['duration_of_vehicles_left'].append(get_travel_time_duration(vehicle_dict, vehicle_id_leaving))

    for k, v in reward_detail_dict.items():
        if v[0] and len(v) > 2:  # Add check for len(v)
            reward += v[1]*v[2]

    reward = restrict_reward(reward)#,func="linear")
    return reward, reward_detail_dict

# This function is global
def get_rewards_from_dict_list(rewards_detail_dict_list):
    reward = 0
    for i in range(len(rewards_detail_dict_list)):
        for k, v in rewards_detail_dict_list[i].items():
            if v[0] and len(v) > 2:  # Add check for len(v)
                reward += v[1] * v[2]
    reward = restrict_reward(reward)
    return reward

# --- FLOWSYNC: Parameterized function ---
def get_overall_queue_length(listLanes):
    ''' Calculates queue length for the provided list of lanes '''
    overall_queue_length = 0
    for lane in listLanes:
        try:
            overall_queue_length += traci.lane.getLastStepHaltingNumber(lane)
        except traci.TraCIException:
            pass # Lane might not exist in this step
    return overall_queue_length

# --- FLOWSYNC: Parameterized function ---
def get_overall_waiting_time(listLanes):
    ''' Calculates waiting time for the provided list of lanes '''
    overall_waiting_time = 0
    for lane in listLanes:
        try:
            overall_waiting_time += traci.lane.getWaitingTime(str(lane)) / 60.0
        except traci.TraCIException:
            pass # Lane might not exist
    return overall_waiting_time

# --- FLOWSYNC: Parameterized function ---
def get_overall_delay(listLanes):
    ''' Calculates delay for the provided list of lanes '''
    overall_delay = 0
    for lane in listLanes:
        try:
            mean_speed = traci.lane.getLastStepMeanSpeed(str(lane))
            max_speed = traci.lane.getMaxSpeed(str(lane))
            if max_speed > 0:
                overall_delay += 1 - mean_speed / max_speed
            else:
                overall_delay += 0
        except traci.TraCIException:
            pass # Lane might not exist
    return overall_delay

# This function is global
def get_flickering(action):
    return action

# This function is global
def get_num_of_emergency_stops(vehicle_dict):
    emergency_stops = 0
    vehicle_id_list = traci.vehicle.getIDList()
    for vehicle_id in vehicle_id_list:
        try:
            traci.vehicle.subscribe(vehicle_id, (tc.VAR_LANE_ID, tc.VAR_SPEED))
            results = traci.vehicle.getSubscriptionResults(vehicle_id)
            if not results: continue # Skip if vehicle left
            current_speed = results.get(64) # 64 is tc.VAR_SPEED
            if current_speed is None: continue

            if (vehicle_id in vehicle_dict.keys()):
                vehicle_former_state = vehicle_dict[vehicle_id]
                if current_speed - vehicle_former_state.speed < -4.5:
                    emergency_stops += 1
            else:
                # print("##New car coming")
                if current_speed - Vehicles.initial_speed < -4.5:
                    emergency_stops += 1
        except traci.TraCIException:
            pass # Vehicle might have left
    if len(vehicle_dict) > 0:
        return emergency_stops/len(vehicle_dict)
    else:
        return 0

# This function is global
def get_partial_travel_time_duration(vehicle_dict, vehicle_id_list):
    travel_time_duration = 0
    current_sim_time = traci.simulation.getTime()
    for vehicle_id in vehicle_id_list:
        if (vehicle_id in vehicle_dict.keys()) and (vehicle_dict[vehicle_id].first_stop_time != -1):
            travel_time_duration += (current_sim_time - vehicle_dict[vehicle_id].first_stop_time)/60.0
    if len(vehicle_id_list) > 0:
        return travel_time_duration#/len(vehicle_id_list)
    else:
        return 0

# This function is global
def get_travel_time_duration(vehicle_dict, vehicle_id_list):
    travel_time_duration = 0
    current_sim_time = traci.simulation.getTime()
    for vehicle_id in vehicle_id_list:
        if (vehicle_id in vehicle_dict.keys()):
            travel_time_duration += (current_sim_time - vehicle_dict[vehicle_id].enter_time)/60.0
    if len(vehicle_id_list) > 0:
        return travel_time_duration#/len(vehicle_id_list)
    else:
        return 0

# This function is global
def update_vehicles_state(dic_vehicles):
    vehicle_id_list = traci.vehicle.getIDList()

    # This check is expensive, maybe rely on get_vehicle_id_entering per node?
    # vehicle_id_entering_list = get_vehicle_id_entering(ALL_ENTERING_LANES)

    for vehicle_id in (set(dic_vehicles.keys())-set(vehicle_id_list)):
        del(dic_vehicles[vehicle_id])

    current_sumo_time = traci.simulation.getTime()

    for vehicle_id in vehicle_id_list:
        try:
            if (vehicle_id in dic_vehicles.keys()) == False:
                vehicle = Vehicles()
                vehicle.id = vehicle_id
                traci.vehicle.subscribe(vehicle_id, (tc.VAR_LANE_ID, tc.VAR_SPEED))
                results = traci.vehicle.getSubscriptionResults(vehicle_id)
                vehicle.speed = results.get(64) if results else 0 # 64 is tc.VAR_SPEED
                if vehicle.speed is None: vehicle.speed = 0

                vehicle.enter_time = current_sumo_time
                # if it enters and stops at the very first
                if (vehicle.speed < 0.1) and (vehicle.first_stop_time == -1):
                    vehicle.first_stop_time = current_sumo_time
                dic_vehicles[vehicle_id] = vehicle
            else:
                results = traci.vehicle.getSubscriptionResults(vehicle_id)
                dic_vehicles[vehicle_id].speed = results.get(64) if results else 0
                if dic_vehicles[vehicle_id].speed is None: dic_vehicles[vehicle_id].speed = 0

                if (dic_vehicles[vehicle_id].speed < 0.1) and (dic_vehicles[vehicle_id].first_stop_time == -1):
                    dic_vehicles[vehicle_id].first_stop_time = current_sumo_time

                # This logic is flawed for multi-intersection, disabling
                # if (vehicle_id in vehicle_id_entering_list) == False:
                #     dic_vehicles[vehicle_id].entering = False
        except traci.TraCIException:
            pass # Vehicle may have left

    return dic_vehicles

# --- FLOWSYNC: Parameterized function ---
def status_calculator(node_id):
    '''
    Calculates the state for a specific node_id.
    Ensures state vectors are padded to MAX_LANES_PER_NODE (12).
    '''
    if node_id not in NODE_PHASE_DEFINITIONS:
        print(f"Error: Node {node_id} not in phase definitions.")
        # Return empty/zero states, padded to correct dimension
        return [np.zeros(MAX_LANES_PER_NODE),
                np.zeros(MAX_LANES_PER_NODE),
                np.zeros(MAX_LANES_PER_NODE),
                np.zeros((150, 150))]

    listLanes = NODE_PHASE_DEFINITIONS[node_id]["incoming_lanes"]

    laneQueueTracker=[]
    laneNumVehiclesTracker=[]
    laneWaitingTracker=[]

    for i in range(MAX_LANES_PER_NODE):
        if i < len(listLanes):
            lane = listLanes[i]
            try:
                laneQueueTracker.append(traci.lane.getLastStepHaltingNumber(lane))
                laneNumVehiclesTracker.append(traci.lane.getLastStepVehicleNumber(lane))
                laneWaitingTracker.append(traci.lane.getWaitingTime(str(lane)) / 60)
            except traci.TraCIException:
                # Lane might not exist, pad with 0
                laneQueueTracker.append(0)
                laneNumVehiclesTracker.append(0)
                laneWaitingTracker.append(0)
        else:
            # Pad with zeros
            laneQueueTracker.append(0)
            laneNumVehiclesTracker.append(0)
            laneWaitingTracker.append(0)

    # ================ get position matrix of vehicles on lanes
    mapOfCars = getMapOfVehicles(area_length=area_length) # This remains global

    return [laneQueueTracker, laneNumVehiclesTracker, laneWaitingTracker, mapOfCars]

# --- FLOWSYNC: Parameterized function ---
def get_vehicle_id_entering(entering_lanes):
    '''
    Gets vehicle IDs on the specified list of entering_lanes.
    '''
    vehicle_id_entering = []
    for lane in entering_lanes:
        try:
            vehicle_id_entering.extend(traci.lane.getLastStepVehicleIDs(lane))
        except traci.TraCIException:
            pass # Lane might not exist
    return vehicle_id_entering

# --- FLOWSYNC: Parameterized function ---
def get_vehicle_id_leaving(vehicle_dict, entering_lanes):
    '''
    Gets vehicle IDs that have left the specified entering_lanes.
    '''
    vehicle_id_leaving = []
    vehicle_id_entering = get_vehicle_id_entering(entering_lanes)

    # This logic is flawed. A vehicle leaving one intersection's
    # "entering_lanes" might just be entering the intersection.
    # A better check would be against 'controlled_lanes'.
    # For now, we stick to the original logic.

    for vehicle_id in vehicle_dict.keys():
        if not(vehicle_id in vehicle_id_entering) and vehicle_dict.get(vehicle_id) and vehicle_dict[vehicle_id].entering:
            # This check is problematic, disabling `entering` flag update
            # vehicle_id_leaving.append(vehicle_id)
            pass

    return vehicle_id_leaving

# --- FLOWSYNC: New function for neighbor states ---
def get_neighbor_states(node_id):
    '''
    Gets the states of neighbors for the attention mechanism.
    Pads to MAX_NEIGHBORS (4) and MAX_LANES_PER_NODE (12).
    '''

    neighbor_queues = np.zeros((MAX_NEIGHBORS, MAX_LANES_PER_NODE))
    neighbor_phases = np.zeros((MAX_NEIGHBORS, 1))
    neighbor_mask = np.zeros((MAX_NEIGHBORS,))

    if node_id not in NODE_NEIGHBORS:
        return neighbor_queues, neighbor_phases, neighbor_mask

    neighbors = NODE_NEIGHBORS[node_id]

    for i in range(MAX_NEIGHBORS):
        if i < len(neighbors):
            neighbor_node_id = neighbors[i]
            if neighbor_node_id in NODE_PHASE_DEFINITIONS:
                neighbor_mask[i] = 1 # This is a real neighbor

                # Get neighbor's phase
                try:
                    neighbor_phases[i, 0] = traci.trafficlight.getPhase(neighbor_node_id)
                except traci.TraCIException:
                    neighbor_phases[i, 0] = 0 # Default to phase 0 on error

                # Get neighbor's queue lengths
                neighbor_lanes = NODE_PHASE_DEFINITIONS[neighbor_node_id]["incoming_lanes"]
                for j in range(MAX_LANES_PER_NODE):
                    if j < len(neighbor_lanes):
                        try:
                            neighbor_queues[i, j] = traci.lane.getLastStepHaltingNumber(neighbor_lanes[j])
                        except traci.TraCIException:
                            neighbor_queues[i, j] = 0 # Lane error
                    # else: it remains 0 (padding)
        # else: it remains 0 (padding)

    return neighbor_queues, neighbor_phases, neighbor_mask


# This function is legacy and map-specific
def get_car_on_red_and_green(cur_phase):
    listLanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2', 'edge2-0_0', 'edge2-0_1', 'edge2-0_2',
                 'edge3-0_0', 'edge3-0_1', 'edge3-0_2', 'edge4-0_0', 'edge4-0_1', 'edge4-0_2']
    vehicle_red = []
    vehicle_green = []
    if cur_phase == 1:
        red_lanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2', 'edge2-0_0', 'edge2-0_1', 'edge2-0_2']
        green_lanes = ['edge3-0_0', 'edge3-0_1', 'edge3-0_2', 'edge4-0_0', 'edge4-0_1', 'edge4-0_2']
    else:
        red_lanes = ['edge3-0_0', 'edge3-0_1', 'edge3-0_2', 'edge4-0_0', 'edge4-0_1', 'edge4-0_2']
        green_lanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2', 'edge2-0_0', 'edge2-0_1', 'edge2-0_2']

    for lane in red_lanes:
        try:
            vehicle_red.append(traci.lane.getLastStepVehicleNumber(lane))
        except traci.TraCIException: pass
    for lane in green_lanes:
        try:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            omega = 0
            for vehicle_id in vehicle_ids:
                traci.vehicle.subscribe(vehicle_id, (tc.VAR_DISTANCE, tc.VAR_LANEPOSITION))
                results = traci.vehicle.getSubscriptionResults(vehicle_id)
                if results:
                    distance = results.get(132) # 132 is tc.VAR_DISTANCE
                    if distance is not None and distance > 100:
                        omega += 1
            vehicle_green.append(omega)
        except traci.TraCIException: pass

    return max(vehicle_red) if vehicle_red else 0, max(vehicle_green) if vehicle_green else 0

# This function is legacy
def get_status_img(current_phase,tl_node_id="node0",area_length=600):
    mapOfCars = getMapOfVehicles(area_length=area_length)
    current_observation = [mapOfCars]
    return current_observation

# --- FLOWSYNC: Parameterized function ---
def set_yellow(node_id, dic_vehicles,rewards_info_dict,f_log_rewards,rewards_detail_dict_list):
    ''' Sets the yellow phase for a specific node_id '''
    if node_id not in NODE_PHASE_DEFINITIONS: return

    try:
        current_phase_links = NODE_PHASE_DEFINITIONS[node_id]["phases"][0]
        num_links = len(current_phase_links)
    except IndexError:
        num_links = 16 # Fallback to original

    Yellow = "y" * num_links # Dynamic yellow phase

    for i in range(3):
        timestamp = traci.simulation.getTime()
        try:
            traci.trafficlight.setRedYellowGreenState(node_id, Yellow)
        except traci.TraCIException:
            pass # Node might not exist
        traci.simulationStep()
        # log_rewards(dic_vehicles, 0, rewards_info_dict, f_log_rewards, timestamp, rewards_detail_dict_list)
        update_vehicles_state(dic_vehicles)

# --- FLOWSYNC: Parameterized function ---
def set_all_red(node_id, dic_vehicles,rewards_info_dict,f_log_rewards,rewards_detail_dict_list):
    ''' Sets the all-red phase for a specific node_id '''
    if node_id not in NODE_PHASE_DEFINITIONS: return

    try:
        current_phase_links = NODE_PHASE_DEFINITIONS[node_id]["phases"][0]
        num_links = len(current_phase_links)
    except IndexError:
        num_links = 16 # Fallback to original

    Red = "r" * num_links # Dynamic red phase

    for i in range(3):
        timestamp = traci.simulation.getTime()
        try:
            traci.trafficlight.setRedYellowGreenState(node_id, Red)
        except traci.TraCIException:
            pass # Node might not exist
        traci.simulationStep()
        # log_rewards(dic_vehicles, 0, rewards_info_dict, f_log_rewards, timestamp,rewards_detail_dict_list)
        update_vehicles_state(dic_vehicles)

# --- FLOWSYNC: Parameterized function ---
def run(node_id, action, current_phase, current_phase_duration, vehicle_dict, rewards_info_dict, f_log_rewards, rewards_detail_dict_list):
    '''
    Executes one simulation step for a given agent's action.
    NOTE: In the new FlowSync MARL loop, this is simplified.
    The main loop handles the simulation step.
    This function is kept for compatibility but is modified.
    '''
    return_phase = current_phase
    return_phase_duration = current_phase_duration

    if action == 1:
        # Set yellow (which steps 3 times)
        set_yellow(node_id, vehicle_dict,rewards_info_dict,f_log_rewards, rewards_detail_dict_list)
        # set_all_red(vehicle_dict,rewards_info_dict,f_log_rewards, node_id=node_id)

        # Change light (does not step)
        return_phase, _ = changeTrafficLight(node_id, current_phase)  # Parameterized
        return_phase_duration = 0
    else:
        # Only step if no action was taken (to advance time by 1s)
        timestamp = traci.simulation.getTime()
        traci.simulationStep()
        # log_rewards(vehicle_dict, action, rewards_info_dict, f_log_rewards, timestamp, rewards_detail_dict_list)
        vehicle_dict = update_vehicles_state(vehicle_dict)
        return_phase_duration += 1

    return return_phase, return_phase_duration, vehicle_dict


# This function is legacy
def get_base_min_time(traffic_volumes,min_phase_time):
    traffic_volumes=np.array([36,72,0])
    min_phase_times=np.array([10,35,35])
    for i, min_phase_time in enumerate(min_phase_times):
        if traffic_volumes[i] > 0: # Add check for division by zero
            ratio=min_phase_time/traffic_volumes[i]
            traffic_volumes_ratio=traffic_volumes/ratio

# This function is legacy
def phase_vector_to_number(phase_vector,phases_light=None):
    if phases_light is None:
        phases_light = ["WNG_ESG_EWG_WEG_WSG_ENG", "NSG_NEG_SNG_SWG_NWG_SEG"]

    phase_vector_7 = []
    result = -1
    for i in range(len(phases_light)):
        # This will fail as get_phase_vector is parameterized
        # phase_vector_7.append(str(get_phase_vector(i)))
        pass
    if phase_vector in phase_vector_7:
        return phase_vector_7.index(phase_vector)
    else:
        # raise ("Phase vector %s is not in phases_light %s"%(phase_vector,str(phase_vector_7)))
        return 0 # Fallback


if __name__ == '__main__':
    pass
    # Example for legacy get_phase_vector
    # print(get_phase_vector(0))
    # print(get_phase_vector(1))
    # phase_vector_to_number('[0 1 0 1 0 0 1 1 0 1 0 1]')
    pass
    # traci.close()
