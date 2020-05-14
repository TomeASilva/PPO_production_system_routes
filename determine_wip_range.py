from production_system import ProductionSystem
import simpy
from buffer import EpBuffer
from agents import FixedPolicy
from agents import RandomPolicy
import sys  
import numpy as np
import datetime
import os
def model_performance_data(parameters_to_store, path):
    """"Stores information about the simulation in a csv file
    Inputs:
    simulation Number: Integer
    parameters_to_store: tuple (lead_time, flow_time, wip, throughput, parts_produced)
    path: string - directory + name of the file where to store the information

    Returns: 
    None: Creates a file where each line represents a simulation as: 

        line 1: Simulation number;Average Lead-time; Average Flow time, Average WIP, Average throughput, Number of Parts produced
    """
    with open(path, 'a') as file:

        for parameter in parameters_to_store:
            try:
                for element in parameter: 
                    file.write(str(element) + ";".rstrip('\n'))
            except:
                file.write(str(parameter) + ";".rstrip('\n'))
        file.write("\n")
        
        
        
def aggregate_data(path, number_runners, file_suffixes):
    for suffix in file_suffixes:
        number_simulations = 1
        for r in range(number_runners):
            with open(f"{path}Runner_{r}_{suffix}", "r") as fr, open(f"{path}{suffix}", 'a') as fw:
                for l in fr:
                    fw.writelines(f"{number_simulations};{l}")
                    number_simulations +=1 

            os.remove(f"{path}Runner_{r}_{suffix}")
            
def summarize_performance(path):
 

    wip = np.loadtxt (f"{path}/WIP.csv", delimiter=";", unpack=False)
    
    wip = np.mean(wip[:,[1, 3, 5]], axis=0)
    wip_total = np.sum(wip) # To determine the action range for our study we need to know 
    # a value o WIP that is hardly reached 
    
    parts_produced = np.loadtxt(f"{path}/PartsProduced.csv", delimiter=";", unpack=False)
    parts_produced = parts_produced[-1, 1:]
    
    flow_time = np.loadtxt (f"{path}/flow_time.csv", delimiter=";", unpack=False)
    mean_flow_time = np.mean(flow_time[:, -1])
    mean_cycle_time = np.mean(flow_time[:, -2])

    part_0 = flow_time[flow_time[:, 0] == 0 ][:, 2:]
    part_1 = flow_time[flow_time[:, 0] == 1 ][:, 2:]  
    part_2 = flow_time[flow_time[:, 0] == 2 ][:, 2:]  
    
    flow_time_parts = [np.mean(part[: , -1], axis=0) for part in [part_0, part_1, part_2]]
    cycle_time_parts = [np.mean(part[:, -2], axis=0) for part in [part_0, part_1, part_2]]
    """
    return : (wip_per_route-> List length 3, average total wip,
    parts_produced per route -> List of lenght 3, sum of parts produced from all routes
    Average cycle time per each Part -> List of lenght 3, average cycle time of all the parts
    Average Flow time per each Part -> List of lenght 3, average Flow time of all the parts
    """

    return (wip, wip_total, parts_produced[1:4], parts_produced[0], cycle_time_parts, mean_cycle_time, flow_time_parts, mean_flow_time) 

parts = {
    "0": {"machine_0": [0, 10], "machine_1": [0, 10], "demand": [5, 5]},
    "1": {"machine_0": [0, 5], "machine_2": [0, 20], "machine_1": [0, 20], "demand": [10, 10]},
    "2": {"machine_2": [0, 10], "machine_3": [0, 30], "demand": [15, 15]}}


production_system_config = {
    "decision_epoch_interval": 30,
    "track_state_interval": 5,
    "run_length": 3000, 
    "beta_state_weighted_average": 0.9,
    "parts_info": parts 
} 
policy = FixedPolicy(1000, 1000, 1000, action_range=[0, 1000])

storing_path = f"./model_performance{datetime.datetime.now().strftime('%Y_%m_%d %H_%M_%S')}/"
os.mkdir(storing_path)
model_name = "no_wip_cap"

for i in range (100):
    ep_buffer = EpBuffer()
    env = simpy.Environment()
    random_seeds = [i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7]
    
    my_production_system = ProductionSystem(**production_system_config,
                                            env=env,
                                            ep_buffer=ep_buffer,
                                            policy=policy,
                                            random_seeds= random_seeds,
                                            number_of_different_parts=3, 
                                            number_routes=3,
                                            number_workstations=4,
                                            use_seeds=True,
                                            twin_system=None,
                                            logging=False,
                                            files=True) 
                                        
    env.run(until=3000)
    data = summarize_performance(my_production_system.path)
    model_performance_data(data, f"{storing_path}{model_name}")
    
    
file = np.loadtxt(f"{storing_path}{model_name}", delimiter=";", usecols=range(16))
max_wip = np.max(file, axis=0)[3]

print(f"The sum of all WIP never went above: {max_wip} ")