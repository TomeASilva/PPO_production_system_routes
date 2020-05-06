import numpy as np
import simpy
import sys
import logging
import datetime
import os
#logging the Flow of Parts Controlled by the PPO algorithn
sys.setrecursionlimit(5000)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
file_handler = logging.FileHandler("./logs/PPO_system.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

#logging the Flow of Parts Controlled by The Pure Push System
loggerTwin = logging.getLogger(__name__ + "Twin")
loggerTwin.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
file_handler = logging.FileHandler("./logs/Twin_system.log")
file_handler.setFormatter(formatter)
loggerTwin.addHandler(file_handler)

#
# parts= {"Part": {
    # "machine": [uniforme_dist  for each machine need to produce part],
    #  "demand": [gauss distribution for interarrival time]
    # }


class IventoryBuffer ():
    """Creates a buffer that holds parts"""
    
    def __init__(self, env, index):
        self.name = f"buffer_{index}" 
        self.index = index # number of buffer
        self.queue = simpy.Store(env)

class Machine:
    """Creates a resource to be used"""
    def __init__(self, env, index):
        self.name = f"machine_{index}"
        self.index = index # number of machine
        self.resource = simpy.Resource(env, capacity=1)
        
class ProductionSystem: 
    def __init__(self,
                 env, 
                 decision_epoch_interval, 
                 track_state_interval, # < decision_epoch_inteval -> interval of time to check and compute continues variables
                 run_length,
                 random_seeds, # list [1, 2, 3, 4, 5, 6, 7] # the first n elements are the random seeds for each machine the following k the random seeds for part generation
                 beta_state_weighted_average,
                 number_of_different_parts,
                 number_routes,
                 number_workstations,
                 parts_info,
                 policy,
                 ep_buffer,
                 files=False, # if true creates 3 files: Flow Time.csv, Parts Produced.csv,
                 use_seeds=False, # use seeds to generate processing times and interval arrival times
                 twin_system=None, # compare current policy with another policy
                 logging=False):  # produces logs about the flow of parts through the system for current system and its twin

        self.env = env
        self.decision_epoch_interval = decision_epoch_interval
        self.track_state_interval = track_state_interval
        self.run_length = run_length
        self.random_seeds_machines = [random_seeds[i] for i in range(number_workstations)]
        self.random_seeds_part_gen = [random_seeds[i] for i in range(number_workstations - 1, len(random_seeds), 1)]
        self.beta_state_weighted_average = beta_state_weighted_average
        self.number_of_different_parts = number_of_different_parts
        self.number_routes = number_routes
        self.number_workstations = number_workstations
        self.parts_info = parts_info
        self.policy = policy
        self.ep_buffer = ep_buffer
        self.files = files
        self.use_seeds = use_seeds
        self.twin_system = twin_system
        self.logging = logging
        
        self.previous_exit_time = 0 #time last part exited the system
        #state representation
        #0- Wip route 1
        #1- Wip route 2
        #2- Wip route 3 
        #3- Average Interarrival time Part 1 
        #4- Average Interarrival time Part 2
        #5- Average Interarrival time Part 3 
        #6- std interarrival time Part 1 
        #7- std interarrival time Part 2
        #8- std interarrival time Part 3
        #9- u machine 0
        #10- u machine 1
        #11- u machine 2
        #12- u machine 3
        #13- Average Processing time machine 0 
        #14- =machine 1 
        #15- =machine 2
        #16- =machine 3
        
        self.state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # used to compute averages and std
        self.state_element_number_updates = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.env = env # simpy object that controls the clock of the simulation
        self.machines = [] # List of machine for this production system to be filled up create_workstations method
        self.buffers = [] #List of buffers for every machine to be filled up in create_workstations method
        self.order_buffer = None # Preshop floor buffer to be created in create_workstations method
        self.parts_produced = 0 # Counter for the total number of parts produced
        # Counter for the number of parts of each type
        self.parts_produced_type = [0, 0, 0]  #[number parts A, number parts B, number parts C] 
        self.parts_produced_epoch = 0 # counter for the total number of parts produced per epoch
        self.parts_produced_epoch_type = [0, 0, 0] # counter  number of parts produced per epoch of each type
        self.random_states_machines = [np.random.RandomState(seed) for seed in self.random_seeds_machines]
        self.random_states_parts = [np.random.RandomState(seed) for seed in self.random_seeds_part_gen]

        self.create_workstations() #create_workstations 
        self.route_0_machines = [self.machines[0], self.machines[1]]
        self.route_1_machines = [self.machines[0], self.machines[2], self.machines[1]]
        self.route_2_machines = [self.machines[2], self.machines[3]]
        self.routes_machines = [self.route_0_machines, self.route_1_machines, self.route_2_machines]
        
        self.route_0_buffers = [self.buffers[0], self.buffers[1]]
        self.route_1_buffers = [self.buffers[0], self.buffers[2], self.buffers[1]]
        self.route_2_buffers = [self.buffers[2], self.buffers[3]]
        self.routes_buffers = [self.route_0_buffers, self.route_1_buffers, self.route_2_buffers]
        
        # Creates a process that controls the decision making process and WIP control
        self.env.process(self.decision_epoch_interval_process_control())
        # Elements of the state vector that are continous need a process that checks those elements value and 
        # a computes an average
        self.env.process(self.cyclical_state_elements_process())
        #generates parts
        for part, config in self.parts_info.items(): 
            self.env.process(self.generate_demand(int(part)))
       
        if self.files:
            self.path = "./" + datetime.datetime.now().strftime("%Y_%m_%d %H_%M_%S") + " " + policy.name
            os.mkdir(self.path)
            # Periodically tracks the WIP in each route, and The number parts produced in Total and by each type of part
            self.env.process(self.tracking_files_WIP_TH())
        
        

    def create_workstations (self):
        """Creates a workstation, each workstation is comprised of machine and a buffer that precedes it"""
        self.order_buffer = IventoryBuffer(self.env, index="Order")
        for i in range(self.number_workstations):
            self.machines.append(Machine(self.env, i))
            self.buffers.append(IventoryBuffer(self.env, i))
            
    def generate_demand (self, part_index):
        """ 
        Process that creates parts of a given type designated by the part index
        """
        count = 0 # counter for the number of parts generated of each type
        while True: 
            interarrival_time_parameters = self.parts_info[str(part_index)]["demand"] # the parameters to be fed into the distribution
            
            if self.use_seeds:
                interarrival_time = max(0, self.random_states_parts[part_index].normal(interarrival_time_parameters[0], interarrival_time_parameters[1]))
            else:
                interarrival_time = max(0, np.random.normal(interarrival_time_parameters[0], interarrival_time_parameters[1])) 

            
            Part (count, part_index, self)
            count += 1
            
            yield self.env.timeout(interarrival_time) 
        
            
    def decision_epoch_interval_process_control(self):
        """
        Defines a process that after a given number of time units, will observe the current state
        of the production system and receive an action (wip level for each production route)
        """
        
        while True:
            # For state elements that need an event to be updated, like processing time and interarrival time, it may happen that
            # during the time in between decision making, there was no event tha led to update to that variable, this will cause an error
            #because we use an exponential moving avg with bias correction, no updates means division by zero, 
            #also it would mean avg processing time and interarrival time > than decision making interval
            for i in range(len(self.state)):
                if self.state_element_number_updates[i] == 0:
                    self.state_element_number_updates[i] = 1
                    self.state[i] = self.decision_epoch_interval
                    
            #applies bias correction to the state vector 
            self.previous_state = np.array(self.state, dtype=np.float32) / (1 - self.beta_state_weighted_average ** np.array(self.state_element_number_updates, dtype=np.float32))   
            self.previous_state = np.reshape(self.previous_state, (1, -1))
            self.action = self.policy.get_action(self.previous_state) # action is a list with Wip caps one for each route
            self.wip_cap = self.action 
            
            #reset state vector, number of updates and number of parts
            self.state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.state_element_number_updates =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.parts_produced_epoch = 0
            self.parts_produced_epoch_type = [0, 0, 0]

            yield self.env.timeout(self.decision_epoch_interval)
            
            for i in range(len(self.state)):
                if self.state_element_number_updates[i] == 0:
                    self.state_element_number_updates[i] = 1
                    self.state[i] = self.decision_epoch_interval
            
            self.next_state = np.array(self.state, dtype=np.float32) / (1 - self.beta_state_weighted_average ** np.array(self.state_element_number_updates, dtype=np.float32))
            self.next_state = np.reshape(self.next_state, (1, -1))
            
            if self.twin_system != None:
                self.reward = self.compute_reward(self.next_state)

                sample = (self.previous_state.reshape(self.previous_state.shape[1],), np.array(self.action), self.next_state.reshape(self.next_state.shape[1],), self.reward)
                self.ep_buffer.add_transition(sample)
           
        
            
    def cyclical_state_elements_process(self):
        """
        Orders the update of variables that need to be tracked cyclically, for variables like interarrival time there is an 
        event that triggers the update, in this case the arrival of another part. For variables like Avg utilization and Average WIP, there is no
        even to trigger its update, therefore we have to create a process to update those 2 types of variables
        """
        while True:
            yield self.env.timeout(self.track_state_interval)
            self.average_wip_tracking()
            self.utilization_tracking()
              
    def average_wip_tracking(self):
        """
        Calculates the exponential average for the wip of each production route 
        Updates the self.state vector and self.state_element_number_updates
        """ 
        #0- Wip route 1
        #1- Wip route 2
        #2- Wip route 3 
        routes = [self.route_0_machines, self.route_1_machines, self.route_2_machines]
        
        for i in range(3):
            self.state_element_number_updates[i] += 1
            self.state[0] = (self.state[i] * self.beta_state_weighted_average + self.compute_wip_level(routes[i]) * (1 - self.beta_state_weighted_average))

    def utilization_tracking(self):
        """ 
        Calculates the exponential average for the utilzation of each machine
        Updates the self.state vector and self.state_element_number_updates
        """
        #state representation
        #9- u machine 0
        #10- u machine 1
        #11- u machine 2
        #12- u machine 3

        utilization = self.compute_utilization()
        for i in range (len(utilization)):
            utilization[i] = max(1e-8, utilization[i])
        
        machine_index = 0
        for i in range(9, 13):
            self.state_element_number_updates[i] += 1
            
            self.state[i] = self.state[i] * self.beta_state_weighted_average + utilization[machine_index] * (1 - self.beta_state_weighted_average)
            machine_index += 1

    def interarrival_time_tracking(self, interarrival_time, part_type_index):
        """
        Updates the state vector with the new values of interarrival times for a given part
        
        Input: 
        interarrival_time: int
        part_type_index : int -> from 0 to the number of parts - 1
        """
        i = part_type_index 
        #Increment the number of updates for interarrival state variables
        
        self.state_element_number_updates[i + 3] += 1 # average
        self.state_element_number_updates[i + 6] += 1 # std
        #Compute exponential moving average for interarrival time of the different parts
        state_temp = self.state
        #average interarrival time 
        self.state[i + 3] = self.state[i + 3] * self.beta_state_weighted_average + interarrival_time * (1 - self.beta_state_weighted_average)
        #std 
        self.state[i + 6] = self.state[i + 6] * self.beta_state_weighted_average + (interarrival_time - self.state[i + 3]) * (1 - self.beta_state_weighted_average)
    
    def processing_time_tracking(self, processing_time, machine_index):
        """
        Updates the state vector with the new values of processing_times for a given machine
        Inputs:
        processing_time: int -> the processing_time of a given machine
        machine_index : int [0, 3] where and index incates a machine
        
         """
        #state representation
        #0- Wip route 1
        #1- Wip route 2
        #2- Wip route 3 
        #3- Average Interarrival time Part 1 
        #4- Average Interarrival time Part 2
        #5- Average Interarrival time Part 3 
        #6- std interarrival time Part 1 
        #7- std interarrival time Part 2
        #8- std interarrival time Part 3
        #9- u machine 0
        #10- u machine 1
        #11- u machine 2
        #12- u machine 3
        #13- Average Processing time machine 0 
        #14- =machine 1 
        #15- =machine 2
        #16- =machine 
        i = machine_index
        self.state_element_number_updates[i + 13] += 1
        self.state[i + 13] = self.state[i + 13] * self.beta_state_weighted_average + processing_time * (1 - self.beta_state_weighted_average)
         
            
    def compute_utilization(self):
        """
        Returns:
        utilization: list of 1's and 0's, where 1 means that the machine is being used at the time 
        of function call, or 0 if the machine is not being used
        """
        utilization = []
        for machine in self.machines:
            utilization.append(machine.resource.count)
        
        return utilization
        

    def compute_wip_level(self, route):
        """
        Computes the level of Wip at route
        Input: 
        route: a list of Machine objects 

        Return:
        wip: int with the number of parts in route
        """
        wip = 0
        for machine in route:
            wip += machine.resource.count + len(machine.resource.queue)
        return wip

    def tracking_files_WIP_TH(self):
        """Writes 2 files WIP.cvs and Parts Produced.cvs, the first tracks the WIP along the simulation 
           the second tracks the number o parts that already exited the system along the simulation
        """

        while True:
            yield self.env.timeout(self.track_state_interval)
            
            #Time;WIP_route_0;WIP_CAP_route_0;WIP_route_1; WIP_Cap_route_1;WIP route_2; WIP Cap route_2
            with open(f"{self.path}/WIP.csv", 'a') as f:
                f.write(f"{str(self.env.now)};")
                f.write(f"{self.compute_wip_level(self.route_0_machines)};{self.wip_cap[0]};") #route_0
                f.write(f"{self.compute_wip_level(self.route_1_machines)};{self.wip_cap[1]};") #route_1
                f.write(f"{self.compute_wip_level(self.route_2_machines)};{self.wip_cap[2]}\n") #route_2
            
            #Time;Total parts Produced;Part0; Part1; Part2            
            with open(f"{self.path}/PartsProduced.csv", 'a') as f:
                f.write (f"{str(self.env.now)};")
                f.write (f"{self.parts_produced};")
                f.write (f"{self.parts_produced_type[0]};{self.parts_produced_type[1]};{self.parts_produced_type[2]}\n")
    
    def tracking_files_flow_times(self, part):
        
        with open(f"{self.path}/flow_time.csv", 'a') as f:
            #Part_0_0;time spent order buffer, cycle time, Flow Time
             
            f.write(f"{part.type};{part.id};{part.inf_e-part.inf_s};{part.shop_e - part.shop_s};{part.shop_e - part.inf_s}\n")
        
    def compute_reward(self, state):
        """
        Computes the Reward of a state transition and action R(s, a)
        
        Input: 
        State: an array shape(1, 17) -> the current vector representation of the system 
        
        Returns:
        reward: int
        """
        #state representation
        #9- u machine 0
        #10- u machine 1
        #11- u machine 2
        #12- u machine 3
        state = list(state[0, :])
        if self.twin_system != None:
            #Production system
            utilization = state[9:13]
            route_0_bottleneck = max([utilization[machine.index] for machine in self.route_0_machines])
            route_1_bottleneck = max([utilization[machine.index] for machine in self.route_1_machines])
            route_2_bottleneck = max([utilization[machine.index] for machine in self.route_2_machines]) 

            wip_route_0 = state[0]
            wip_route_1 = state[1]
            wip_route_2 = state[2]

            # Twin system
            twin_state = self.twin_system.next_state[0, :]
            utilization_twin = twin_state[9:13]
            
            route_0_bottleneck_twin = max([utilization_twin[machine.index] for machine in self.route_0_machines])
            route_1_bottleneck_twin = max([utilization_twin[machine.index] for machine in self.route_1_machines])
            route_2_bottleneck_twin = max([utilization_twin[machine.index] for machine in self.route_2_machines])
            
            wip_route_0_twin = twin_state[0]
            wip_route_1_twin = twin_state[1]
            wip_route_2_twin = twin_state[2]
            
            route_0 = (route_0_bottleneck >= route_0_bottleneck_twin) and (wip_route_0 <= wip_route_0_twin)
            route_1 = (route_1_bottleneck >= route_1_bottleneck_twin) and (wip_route_1 <= wip_route_1_twin)
            route_2 = (route_2_bottleneck >= route_2_bottleneck_twin) and (wip_route_2 <= wip_route_2_twin)
            
            if route_0 and route_1 and route_2:
                reward = 1
            else:
                reward = 0
            
            return reward 
            
        else: 
            return 0
        

class Part:
    def __init__(self, _id, _type, production_system):
        self.env = production_system.env
        self.production_system = production_system
        self.type = _type
        self.id = _id
        self.machines = production_system.machines
        self.buffers = production_system.buffers
        self.order_buffer = production_system.order_buffer
        self.inf_s = None # time it starts as information
        self.inf_e = None # time it starts as part in shop-floor(shop-floor release)
        # time it starts as a part in shop-floor (shop-floor release)
        self.shop_s = None 
        # time it exits shop-floor (shop-floor exit)
        self.shop_e = None
        self.done = False
        
        self.env.process(self.processing())

    def processing(self):
        
        """Sets a part into its production route """
        self.order_buffer.queue.put(self)
        self.inf_s = self.env.now # Entered the plant as information
        
        if self.production_system.twin_system == None:
            loggerTwin.debug(f"Part_{self.type}_{self.id} entered the system as information at {self.inf_s}")
        else:
            logger.debug(f"{self.type}_{self.id} entered the system as information at {self.inf_s}")

        # mantains env cycling until condition is met otherwise the simpy will stop cycling
        # yield self.env.event()
        
        while not self.done:
            yield self.env.timeout(0)
            
            if self.production_system.compute_wip_level(self.production_system.routes_machines[self.type]) < self.production_system.wip_cap[self.type] and (f"{self.production_system.order_buffer.queue.items[0].type}_{self.production_system.order_buffer.queue.items[0].id}" == f"{self.type}_{self.id}" or len(self.production_system.order_buffer.queue.items) == 0):
                self.order_buffer.queue.get()
                #order entered production system
                
                if self.production_system.twin_system == None:
                    loggerTwin.debug(f"Part_{self.type}_{self.id}, released into production at {self.env.now}")
                else:
                    logger.debug(f"{self.type}_{self.id}, released into production at {self.inf_s}")
                
                self.production_system.routes_buffers[self.type][0].queue.put(self)
                
                if self.production_system.twin_system == None:
                    loggerTwin.debug(f"Part_{self.type}_{self.id}, entered {self.production_system.routes_buffers[self.type][0].name} at {self.env.now}")
                else:
                    logger.debug(f"{self.type}_{self.id}, entered {self.production_system.routes_buffers[self.type][0].name} at {self.env.now}")
                
                self.inf_e = self.env.now
                self.shop_s = self.env.now
                #Request  machine on route and process part 
                for i in range (len(self.production_system.routes_machines[self.type])):
                        
                    with self.production_system.routes_machines[self.type][i].resource.request() as request:
                        yield request
                        self.production_system.routes_buffers[self.type][i].queue.get()
                        
                        if self.production_system.twin_system == None:
                            loggerTwin.debug(f"Part_{self.type}_{self.id} started being processed on {self.production_system.routes_machines[self.type][i].name} at {self.env.now}")
                        else:
                            logger.debug(f"{self.type}_{self.id}, started being processed on {self.production_system.routes_machines[self.type][i].name} at {self.env.now}")
                        
                        distribution_parameters = self.production_system.parts_info[str(self.type)][self.production_system.routes_machines[self.type][i].name]
                        if self.production_system.use_seeds:
                            time = self.production_system.random_states_machines[self.production_system.routes_machines[self.type][i].index].uniform(distribution_parameters[0], distribution_parameters[1])
                        else:
                            time = np.random.uniform(distribution_parameters[0], distribution_parameters[1])
                        
                        #update processing time for first machine on route
                        self.production_system.processing_time_tracking(time, self.production_system.routes_machines[self.type][i].index)
                        yield self.env.timeout(time)

                    #Put part on the  next buffer of the product route
                    if i != ( len(self.production_system.routes_machines[self.type]) - 1 ):
                        self.production_system.routes_buffers[self.type][i + 1].queue.put(self)
                        if self.production_system.twin_system == None:
                            loggerTwin.debug(f"Part_{self.type}_{self.id}, entered {self.production_system.routes_buffers[self.type][i + 1].name} at {self.env.now}")
                        else:
                            logger.debug(f"{self.type}_{self.id}, entered {self.production_system.routes_buffers[self.type][i + 1].name} at {self.env.now}")
                                
                
                self.shop_e = self.env.now
                
                if self.production_system.twin_system == None:
                    loggerTwin.debug(f"Part_{self.type}_{self.id}, Exited the production system at {self.env.now}")
                else:
                    logger.debug(f"{self.type}_{self.id}, Exited the production system at {self.env.now}")
                
                #Increase the count of parts that exited the system     
                self.production_system.parts_produced += 1
                self.production_system.parts_produced_type[self.type] += 1
                self.production_system.parts_produced_epoch += 1
                self.production_system.parts_produced_epoch_type[self.type] += 1
                
                if self.production_system.files:
                    self.production_system.tracking_files_flow_times(self)
                
                self.done = True
            else:
                #If WIP still above WIP cap wait until WIP Drops
                self.env.step()
                
                
