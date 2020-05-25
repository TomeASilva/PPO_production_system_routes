import numpy
import simpy
from collections import deque

parts_info = {
    "0": {"machine_0": [0, 10], "machine_1": [0, 10], "demand": [5, 5]},
    "1": {"machine_0": [0, 5], "machine_2": [0, 20], "machine_1": [0, 20], "demand": [10, 10]},
    "2": {"machine_2": [0, 10], "machine_3": [0, 30], "demand": [15, 15]}}


class Route:
    def __init__(self, env, index, machines, buffers):

        self.name = f"Route_{index}"
        self.index = index
        self.wip_limit = 0
        self.number_auth_onroute = 0
        self.number_auth_requested = 0
        self.machines = machines
        self.queue = simpy.Store(env)
        self.number_auth = 0 
        self.number_parts_produced = 0 
        self.buffers = buffers 
        
class ProductionControl:
    def __init__(self, env):
        self.parts_info = parts_info
        self.product_list = deque()
        self.env = env
        
        self.routes = []
    
    def generate_demand(parts_info, part_index) :
        """ 
        Process that creates parts of a given type designated by the part index
        """
        count = 0 # counter for the number of parts generated of each type
        while True: 

            interarrival_time_parameters = parts_info[str(part_index)]["demand"] # the parameters to be fed into the distribution
            
            interarrival_time = max(0, self.random_states_parts[part_index].normal(interarrival_time_parameters[0], interarrival_time_parameters[1]))

            

            yield env.timeout(interarrival_time)
    
    def create_routes(self, number_routes, machine_routes_index):
        for i in range(self.number_routes):
            machines = [self.machines[machine_index] for machine_index in machine_routes_index[i]]
            buffers = [self.buffers[buffer_index] for buffer_index in machine_routes_index[i]] 
            
            self.routes.append(Route(self.env, i, machines, buffers))
            
    
    def change_wip_limit(self, wip_limits):
        
        
        for route in self.routes:
            #Change the new_wip
            route.wip_limit = wip_limits[route.index]
            
            if route.number_auth > route.wip_limit:
                while route.number_auth > route.wip_limit and len(route.queue.items) > 0:
                    route.number_auth -= 1
                    auth = route.queue.get()
                
     
            #The number of authorizations stays the same 
            if route.number_auth == route.wip_limit:
                pass
            
            if route.number_auth < route.wip_limit:
               
                while route.number_auth < route.wip_limit:
                    
                    self.auth_index += 1
                    auth = Auth(self.auth_index)
                    
                    route.queue.put(auth) 
                    route.number_auth += 1
       
    

            
        

class Part:
    def __init__(self, _id, _type, production_control):
        self.id = _id
        self.type = _type
    
    def processing():
        
        yield
 

