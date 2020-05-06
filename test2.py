from production_system import ProductionSystem
import simpy
from buffer import EpBuffer
from agents import FixedPolicy
import sys  

parts = {
    "0": {"machine_0": [0, 10], "machine_1": [0, 10], "demand": [0, 10]},
    "1": {"machine_0": [0, 10], "machine_2": [0, 10], "machine_1": [0, 10], "demand": [0, 10]},
    "2": {"machine_2": [0, 10], "machine_3": [0, 10], "demand": [0, 10]}}

sys.setrecursionlimit(1000)

production_system_config = {
    "decision_epoch_interval": 30, 
    "track_state_interval": 5,
    "run_length": 7000, 
    "beta_state_weighted_average": 0.9,
    "parts_info": parts 
} 
ep_buffer = EpBuffer()
env = simpy.Environment()
policy = FixedPolicy(3, 3, 3, action_range=[0, 1000])

my_production_system = ProductionSystem(**production_system_config,
                                        env=env,
                                        ep_buffer=ep_buffer,
                                        policy=policy,
                                        random_seeds=[1, 2, 3, 4, 5, 6, 7], 
                                        number_of_different_parts=3, 
                                        number_routes=3,
                                        number_workstations=4,
                                        use_seeds=True,
                                        twin_system=None,
                                        logging=True,
                                        files=True) 

                                        
env.run(until=3000)