import simpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from buffer import EpBuffer
from production_system import ProductionSystem
from collections import deque
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from multiprocessing import Manager, Queue, Process
from multiprocessing.queues import Full
import multiprocessing
import csv
import datetime
import logging
import pickle

gradient_logger = logging.getLogger(__name__)
gradient_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
file_handler = logging.FileHandler("./logs/gradient.log")
file_handler.setFormatter(formatter)
gradient_logger.addHandler(file_handler)

def summarize_performance(path):

    wip = np.loadtxt (f"{path}/WIP.csv", delimiter=";", unpack=False)
    
    wip = np.mean(wip[:,[1, 3, 5]], axis=0)
    wip_total = np.mean(wip)
    
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
    
    
    print(f"\033[0;31mAverage WIP: {wip_total}\033[0m")
    for i in range(3):
        print(f"Route_{i}: {wip[i]}")

    print(f"\033[0;31mParts Produced Total: {parts_produced[0]}\033[0m") 
    for i in range(1, 4):
        print(f"Part_{i-1}: {parts_produced[i]}")
    
    print(f"\033[0;31mParts Cycle Time: {mean_cycle_time}\033[0m")
    for i in range (3):
        print(f"Part{i}: {cycle_time_parts[i]}")
        
    print(f"\033[0;31mParts Flow Time: {mean_flow_time}\033[0m")
    for i in range (3):
        print(f"Part{i}: {flow_time_parts[i]}")

    
class RandomPolicy():
    def __init__(self, action_range, number_actions):
        self.action_range = action_range
        self.number_actions = number_actions
        self.name = f"WIP set by a random uniform dist" 
    def get_action(self, state):
        return [np.random.randint(self.action_range[0], self.action_range[1]) for i in range(self.number_actions)]

class FixedPolicy():
    def __init__ (self, *argswip,  action_range):
        """
        Input:
        argswip: n number of ints according to the number of production routes to be controled by the agent
        action_range: a list [lower limit to the action, upperlimit to the action not used at the moment
        """
        self.action_range = action_range    
        self.wip = [wip for wip in argswip]
        self.name = f"WIP set to {self.wip}"
    
    def get_action(self, state):
        return self.wip

def build_networks (layer_sizes, activations, input, istrunk=False):
    num_layers = len(layer_sizes)
    output = keras.layers.Dense(units=layer_sizes[0], kernel_initializer='glorot_normal')(input)
    output = keras.layers.ELU(alpha=3)(output)
    for i in range(1, num_layers):
        if i == num_layers - 1 and not istrunk:
            output = keras.layers.Dense(units=layer_sizes[i], activation=activations[i], kernel_initializer='glorot_normal')(output)
        else:
            output = keras.layers.Dense(units=layer_sizes[i], kernel_initializer='glorot_normal')(output)
            output = keras.layers.ELU(alpha=3)(output)
    
    return output

def build_model(input, output, name):
    return keras.Model(input, output, name=name)

class PPO:
    def __init__(self,
                 agent_name,
                 input_layer_size,
                 trunk_config,
                 mu_head_config,
                 cov_head_config,
                 critic_net_config,
                 action_range, 
                 summary_writer=None):
        """
        Inputs:
        input_layer_size -> int
        trunk_config -> dict {"layer_sizes": [123, 123, 123 .....]}
        mu_head_config -> dict {"layer_sizes": [123, 123, 123 .....]}
        critic_net_config -> dict {"layer_sizes": [123, 123, 123 .....]}
        action_range -> tuple (0, 100)
        """

        self.action_range = action_range
        self.input_layer_size = input_layer_size
        self.trunk_config = trunk_config
        self.mu_head_config = mu_head_config
        self.cov_head_config = cov_head_config
        self.critic_net_config = critic_net_config
        self.name = f"WIP set by PPO"
        self.action_range = action_range
        self.agent_name = agent_name
        self.summary_writer = summary_writer
        self.number_action_calls = 0

    def build_models(self):

        self.input = keras.Input(shape=(self.input_layer_size), name="state", dtype=tf.float32)
        self.trunk = build_networks(**self.trunk_config, input=self.input, istrunk=True)
        mu_head = build_networks(**self.mu_head_config, input=self.trunk)
        cov_head = build_networks(**self.cov_head_config, input=self.trunk)
        critic = build_networks(**self.critic_net_config, input=self.input)
        
        # Creates a model for mu cov and critic
        self.actor_mu = build_model(self.input, mu_head, "actor_mu")
        self.actor_cov = build_model(self.input, cov_head, "actor_cov")
        self.critic = build_model(self.input, critic, "critic")
        #---Start Find the variables of the actor cov model
        actor_cov_n_layers_head = len(self.cov_head_config["layer_sizes"]) * 2 - 1
        # print(f"Number cov head {actor_cov_n_layers_head}")
        actor_cov_total_n_layers = len(self.actor_cov.layers)
        # print(f"Number cov Layers: {actor_cov_total_n_layers}")
        # exit()
        cov_head_variables = []
        
        for i in range (actor_cov_total_n_layers - actor_cov_n_layers_head, actor_cov_total_n_layers, 1):
            for variable in self.actor_cov.get_layer(index = i).trainable_variables:
                cov_head_variables.append(variable)
        #---END Find the variables of the actor cov model
        self.cov_head_variables = cov_head_variables
        # ---START Creates a way to access the  current value of the weights of the networks
        # actor_mu and actor_cov will be identical with exception of the output layer
        self.current_parameters = {"mu": [variable.numpy() for variable in self.actor_mu.trainable_variables],
                        "cov": [variable.numpy() for variable in cov_head_variables],
                        "critic": [variable.numpy() for variable in self.critic.trainable_variables]
                        }
        # for key, variables in self.current_parameters.items():
        #     print(key)
        #     for var in variables:
        #         print(var.shape) 

        # ---END Creates a way to access the  current value of the weights of the networks
        # ---START Creates a way to acess the variables of the models (used to apply the gradients)
        self.variables = {"mu": self.actor_mu.trainable_variables,
                        "cov": cov_head_variables,
                        "critic": self.critic.trainable_variables}
        # ---END Creates a way to acess the variables of the models (used to apply the gradients)

        if self.agent_name == "Global Agent":
        
            self.trunk_old = build_networks(**self.trunk_config, input=self.input, istrunk=True)
            mu_head_old = build_networks(**self.mu_head_config, input=self.trunk_old)
            cov_head_old = build_networks(**self.cov_head_config, input=self.trunk_old)
            self.actor_mu_old = build_model(self.input, mu_head_old, "actor_mu_old")
            self.actor_cov_old = build_model(self.input, cov_head_old, "actor_cov_old")

            #---Start Find the variables of the actor cov model
            actor_cov_n_layers_head_old = len(self.cov_head_config["layer_sizes"]) * 2  - 1
            actor_cov_total_n_layers_old = len(self.actor_cov_old.layers)
            cov_head_variables_old = []
            for i in range (actor_cov_total_n_layers_old - actor_cov_n_layers_head_old, actor_cov_total_n_layers_old, 1):
                for variable in self.actor_cov_old.get_layer(index = i).trainable_variables:
                    cov_head_variables_old.append(variable)
        #---END Find the variables of the actor cov model
            self.cov_head_variables_old = cov_head_variables_old

            self.variables_old =  {"mu": self.actor_mu_old.trainable_variables,
                            "cov": cov_head_variables_old}
            
            self.current_parameters_old =  {"mu": [variable.numpy() for variable in self.actor_mu_old.trainable_variables],
                        "cov": [variable.numpy() for variable in cov_head_variables_old],
                        "critic": [variable.numpy() for variable in self.critic.trainable_variables]
                        }

    def get_action(self, state):
        """
        Inputs:
        state -> numpy.ndarray

        Returns:
        action -> array([[0.433598 0.12343 1.12341234]], shape=(1, 3), dtype=float32)

        Intermediate variables: 
        mu -> tf.Tensor([[0.433598 0.12343 1.12341234]], shape=(1, 3), dtype=float32)
        cov -> tf.Tensor([[0.39956307 0.1234 2.12341234]], shape=(1, 3), dtype=float32)

       """
        assert state.shape == (1, self.input_layer_size), "Check the dimensions of State"
        # ---START Generates the average and standard deviation for the wip at the given stage
        mu = self.actor_mu(state)
        cov = self.actor_cov(state)
        # print(f"Average: {mu}")
        # print(f"Std: {cov}")
        # ---END Generates the average and standard deviation for the wip at the given stage

        # Computes a noram distribution
        probability_density_func = tfp.distributions.Normal(mu, cov)
        # Samples a WIP from the distribution
        action = probability_density_func.sample()

        if self.summary_writer != None: 
            with self.summary_writer.as_default():
                tf.summary.histogram("Mu dist_" + self.agent_name, mu, self.number_action_calls)
                tf.summary.histogram("Cov dist_" + self.agent_name, cov, self.number_action_calls)
                tf.summary.histogram("Action_" + self.agent_name, action, self.number_action_calls)
                
        # print(f"Action: {action}")
        self.number_action_calls += 1
        action = action.numpy()
        action = list(action[0, : ])
        wip_cap = []
        for wip in action:
            wip_cap.append(int(max(min(wip, self.action_range[1]), self.action_range[0])))
        return wip_cap 

    def get_state_value(self, state):
        state_value = self.critic(state)
        return float(state_value)          
            

class Agent:
    def __init__(self,
                 name,
                 action_range,
                 conwip,
                 production_system_configuration,
                 ppo_networks_configuration,
                 gamma,
                 current_number_episodes,
                 total_number_episodes,
                 episode_queue,
                 run_length,
                 epsilon,
                 record_statistics,
                 gradient_steps_per_episode_actor,
                 gradient_steps_per_episode_critic,
                 gradient_clipping_actor,
                 gradient_clipping_critic,
                 parameters_queue,
                 actor_optimizer_mu,
                 actor_optimizer_cov,
                 critic_optimizer,
                 entropy,
                 number_episodes_worker,
                 log_gradient_descent,
                 n_reward_returns
                 
                 ):
        
        self.name = name
        self.action_range = action_range
        self.conwip = conwip
        self.buffer1 = EpBuffer()
        self.buffer2 = EpBuffer()
        self.production_system_configuration = production_system_configuration
        self.current_number_episodes = current_number_episodes  # number of episodes already run
        self.total_number_episodes = total_number_episodes  # Total number of episodes to be run
        self.episode_queue = episode_queue
        self.run_length= run_length
        self.ppo_networks_configuration = ppo_networks_configuration
        self.gamma = gamma
        self.action_range = action_range
        self.episode_queue = episode_queue
        self.epsilon = epsilon
        self.record_statistics = record_statistics
        self.gradient_steps_per_episode_actor = gradient_steps_per_episode_actor
        self.gradient_steps_per_episode_critic = gradient_steps_per_episode_critic
        self.gradient_clipping_actor = gradient_clipping_actor
        self.gradient_clipping_critic = gradient_clipping_critic
        self.parameters_queue = parameters_queue
        self.actor_optimizer_mu = actor_optimizer_mu
        self.actor_optimizer_cov = actor_optimizer_cov
        self.critic_optimizer = critic_optimizer
        self.entropy = entropy
        self.number_episodes_worker = number_episodes_worker
        self.log_gradient_descent = log_gradient_descent
        self.n_reward_returns = n_reward_returns
        
    def collect_episodes_training(self,
                        number_ep,
                        files,
                        logging):

        for ep in range(number_ep):
            random_seeds = [ep + 1, ep + 2, ep + 3, ep + 4, ep + 5, ep + 6, ep + 7]
            env = simpy.Environment()

            self.production_system_1 = ProductionSystem(env=env,
                             **self.production_system_configuration,
                             ep_buffer=self.buffer1,
                             policy=self.FixedPolicy, # will be defined only after instatiation of child classes
                             use_seeds=True,
                             files=files,
                             random_seeds= random_seeds,
                             logging=logging,
                             run_length=self.run_length,

                             
                             
                             )
            self.production_system_2 = ProductionSystem(env=env,
                                                   **self.production_system_configuration,
                                                   ep_buffer=self.buffer2,
                                                   twin_system=self.production_system_1,
                                                   policy=self.PPO, # will be defined only after instatiation of child classes
                                                   use_seeds=True,
                                                   files=files,
                                                   random_seeds=random_seeds,
                                                   logging=logging,
                                                   run_length=self.run_length,

                                                  )

            env.run(self.run_length)
        if files:
            return (self.production_system_1.path, self.production_system_2.path)
        

class GlobalAgent(Agent):
    def __init__(self,
                 name,
                 action_range,
                 conwip,
                 production_system_configuration,
                 ppo_networks_configuration,
                 gamma,
                 current_number_episodes,
                 total_number_episodes,
                 episode_queue,
                 run_length,
                 epsilon,
                 record_statistics,
                 number_of_child_agents,
                 number_episodes_worker,
                 gradient_steps_per_episode_actor,
                 gradient_steps_per_episode_critic,
                 save_checkpoints,
                 gradient_clipping_actor,
                 gradient_clipping_critic,
                 parameters_queue,
                 average_reward_queue,
                 actor_optimizer_mu,
                 actor_optimizer_cov,
                 critic_optimizer,
                 entropy,
                 log_gradient_descent,
                 n_reward_returns,
                 ):

                
        Agent.__init__(self,
                 name=name,
                 action_range = action_range,
                 conwip = conwip,
                 production_system_configuration = production_system_configuration,
                 current_number_episodes = current_number_episodes,
                 total_number_episodes=total_number_episodes,
                 episode_queue = episode_queue,
                 ppo_networks_configuration=ppo_networks_configuration,
                 epsilon=epsilon,
                 record_statistics=record_statistics,
                 run_length=run_length,
                 gamma=gamma,
                 gradient_steps_per_episode_actor = gradient_steps_per_episode_actor,
                 gradient_steps_per_episode_critic = gradient_steps_per_episode_critic,
                 gradient_clipping_actor=gradient_clipping_actor,
                 gradient_clipping_critic=gradient_clipping_critic,
                 parameters_queue=parameters_queue,
                 actor_optimizer_mu=actor_optimizer_mu,
                 actor_optimizer_cov=actor_optimizer_cov,
                 critic_optimizer=critic_optimizer,
                 entropy=entropy,
                 number_episodes_worker = number_episodes_worker,
                 log_gradient_descent=log_gradient_descent,
                 n_reward_returns = n_reward_returns)

        self.number_of_child_agents = number_of_child_agents
        self.save_checkpoints = save_checkpoints
        self.average_reward_queue = average_reward_queue
        self.rewards = deque(maxlen=100)

    
    def restore_old_models(self, ppo_config):
        
        
        try:
            self.PPO.actor_mu_old.load_weights("./saved_checkpoints/actor_mu/")
            self.PPO.actor_cov_old.load_weights("./saved_checkpoints/actor_cov/")
            self.PPO.critic.load_weights("./saved_checkpoints/critic/")
            
            self.PPO.actor_mu.load_weights("./saved_checkpoints/actor_mu/")
            self.PPO.actor_cov.load_weights("./saved_checkpoints/actor_cov/")
            
            actor_cov_n_layers_head = len(ppo_config["mu_head_config"]["layer_sizes"]) * 2 - 1
            actor_cov_total_n_layers = len(self.PPO.actor_cov.layers)
            cov_head_variables = []

            for i in range (actor_cov_total_n_layers - actor_cov_n_layers_head, actor_cov_total_n_layers, 1):
                for variable in self.PPO.actor_cov.get_layer(index = i).trainable_variables:
                    cov_head_variables.append(variable)

            self.PPO.cov_head_variables = cov_head_variables
            
            self.PPO.current_parameters = {"mu": [variable.numpy() for variable in self.PPO.actor_mu.trainable_variables],
                           "cov": [variable.numpy() for variable in cov_head_variables],
                           "critic": [variable.numpy() for variable in self.PPO.critic.trainable_variables]
                           }

            #check if variables in fact were loaded
            # for key, model in self.PPO.current_parameters.items():
            #     for variable in model:
            #         tf.print(variable)        
            
            cov_head_variables_old = []

            for i in range (actor_cov_total_n_layers - actor_cov_n_layers_head, actor_cov_total_n_layers, 1):
                for variable in self.PPO.actor_cov_old.get_layer(index = i).trainable_variables:
                    cov_head_variables_old.append(variable)

            self.PPO.cov_head_variables_old = cov_head_variables_old
            
            self.PPO.current_parameters_old = {"mu": [variable.numpy() for variable in self.PPO.actor_mu_old.trainable_variables],
                           "cov": [variable.numpy() for variable in cov_head_variables_old],
                           "critic": [variable.numpy() for variable in self.PPO.critic.trainable_variables]
                           }

            return f"Loading models for the training policy of {self.name} was succefull"
        except Exception as message:
            return (str(message) + "\n" + f"The weights of training policy for {self.name} will be randomly generated")
    def training_loop(self):
        try:
            #---START Create a summary writer
            if self.record_statistics: 
                self.writer = tf.summary.create_file_writer(f"./summaries/global/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
            self.current_pass = 0
            #---END of Create summary writer
            
            self.FixedPolicy= FixedPolicy(self.conwip, self.conwip, self.conwip, action_range=self.action_range)
            # Creates the networks for the fixed policy
            # Creates Fixed Policy
        
            self.PPO = PPO(**self.ppo_networks_configuration, action_range=self.action_range, agent_name=self.name, summary_writer= self.writer)
            self.PPO.build_models() #build models (Critic, Actor Networks) 
            # print(f"PPO")
            # print(f"Mu")
            # self.PPO.actor_mu.summary()
            # print(f"Cov")
            # self.PPO.actor_cov.summary()
            # exit()

            print(f"1 optimization cycle corresponds to {self.number_of_child_agents * self.number_episodes_worker} episodes and {self.gradient_steps_per_episode_actor} gradient steps")
            
            self.number_episodes_extracted = 0     
            #---START Load variable weights if self.save_checkpoints is activated
            if self.save_checkpoints:
                try:
                    with open("./saved_checkpoints/optimizers/actor/actor_optimizer_mu.pkl", "rb") as file:
                        self.actor_optimizer_mu = pickle.load(file)
                        self.actor_optimizer_mu.learning_rate.assign(0.00001)
                    
                    with open("./saved_checkpoints/optimizers/actor/actor_optimizer_cov.pkl", "rb") as file:
                        self.actor_optimizer_cov = pickle.load(file)
                        self.actor_optimizer_cov.learning_rate.assign(0.00001)  

                    with open("./saved_checkpoints/optimizers/critic/critic_optimizer.pkl", "rb") as file:
                        self.critic_optimizer = pickle.load(file)
                        # self.critic_optimizer.learning_rate.assign(0.00001)
                
                except Exception as e :    
                    print(e)
                    print("We couldn't load the optimizer state")
                    print("The optimizers state will be reseted")
                    
                
                message = self.restore_old_models(self.ppo_networks_configuration)
                print(message)
                
            #---END Load variable weights if self.save_checkpoints is activated
            self.number_optimization_cycles = 0
            self.number_of_gradient_descent_steps_actor = 0
            self.number_of_gradient_descent_steps_critic = 0
            # for every optimizatin cycle there will be un x episodes and y number of gradient descent steps
            while self.current_number_episodes.value < self.total_number_episodes:
                # because workers run in pararell the number of episodes 
                #would change within a cycle 
                
                for i in range(self.number_of_child_agents):
                    #Put enough parameters for all workers
                    self.parameters_queue.put(self.PPO.current_parameters_old, block=True, timeout=30)

                #---START Record the values for the weights of policy gradient NN
                if self.record_statistics:
                    with self.writer.as_default():
                        for key, parameters in self.PPO.variables.items():
                            for variable in parameters:
                                tf.summary.histogram(f"Params_{self.name}_{str(key)}_{variable.name}", variable, self.number_optimization_cycles)
                #---END Record the values for the weights of policy gradient NN 
                
    
                # Collect all the episodes waiting on the queue
                for i in range(self.number_of_child_agents * self.number_episodes_worker):
                    episode = self.episode_queue.get(block=True, timeout=1000)

                    if i == 0:
                        states, actions, next_states, rewards, qsa = episode

                        if self.record_statistics:
                            episode_reward = np.sum(rewards)
                            tf.summary.scalar(f"Rewards", episode_reward, self.number_episodes_extracted)
                            
                    else:
                        states_temp, actions_temp, next_states_temp, rewards_temp, qsa_temp = episode
                        
                        states = np.vstack((states, states_temp))
                        actions = np.vstack((actions, actions_temp))
                        qsa = np.vstack((qsa, qsa_temp))
                        if self.record_statistics:
                            episode_reward = np.sum(rewards_temp)
                            tf.summary.scalar(f"Rewards", episode_reward, self.number_episodes_extracted)
                        
                    self.number_episodes_extracted += 1
                #---END collect episodes available from all worker

                for j in range(self.gradient_steps_per_episode_critic): 
                    #---START gradient descent for critic
                    critic_gradient, critic_loss = self.gradient_critic(states, qsa)
                    #---START Gradient Clipping critic
                    critic_gradient = [tf.clip_by_norm(value, self.gradient_clipping_critic) for value in critic_gradient]
                    #---
                    if self.record_statistics:
                        with self.writer.as_default():
                                tf.summary.scalar("Critic_Loss", critic_loss, self.number_of_gradient_descent_steps_critic) 
                    #---START Record gradient to summaries
                    if self.record_statistics:
                        with self.writer.as_default():
                            for gradient, variable in zip(critic_gradient, self.PPO.variables["critic"]):
                                tf.summary.histogram(f"Gradients_{self.name}_critic_{variable.name}", gradient, self.number_of_gradient_descent_steps_critic)

                    #---END Record gradient to summaries
                    #---START Aplly Critic Gradients
                    self.critic_optimizer.apply_gradients(zip(critic_gradient, self.PPO.variables["critic"]))
                    #---
                    #---START Observe the value of given state to see convergence
                    state = np.array([4, 5, 10, 4, 3, 2, 1, 1, 1, 0.4, 0.9, 0.7, 0.32, 10, 11, 4, 10])
                    value = self.state_value(state.reshape(1, -1))
                    value = float(value.numpy()[0])
                
                    if self.record_statistics:
                        with self.writer.as_default():
                            tf.summary.scalar(f"State_value", value, self.number_of_gradient_descent_steps_critic)
                    self.number_of_gradient_descent_steps_critic += 1
                    #---END Observe the value of given state to see convergence
                #---END gradient descent for critic                
            
                #---START gradient descent for actor Nets
                for i in range(self.gradient_steps_per_episode_actor):
                    
                    
                    #---START Record the values for the weights of policy gradient NN
                    if self.record_statistics:
                        with self.writer.as_default():
                            for key, parameters in self.PPO.variables.items():
                                for variable in parameters:
                                    tf.summary.histogram(f"Params_{self.name}_{str(key)}_{variable.name}", variable, self.number_of_gradient_descent_steps_actor)
                    #---END Record the values for the weights of policy gradient NN 
                
                    gradients, entropy, actor_loss = self.gradient_actor(states, actions, qsa)

                    if self.record_statistics:
                        with self.writer.as_default():
                            tf.summary.scalar("Action_Loss", actor_loss, self.number_of_gradient_descent_steps_actor)  

                    #---START Cliping gradients
                    for key, gradient in gradients.items():
                        gradients[key] = [tf.clip_by_value(value, -self.gradient_clipping_actor, self.gradient_clipping_actor) for value in gradient]
                    #---END Clipping Gradients
                    
                    #---START Record gradient to summaries
                    if self.record_statistics:
                        with self.writer.as_default():
                                for key, gradient_list in gradients.items():
                                    for gradient, variable in zip(gradient_list, self.PPO.variables[key]):
                                
                                        tf.summary.histogram(f"Gradients_{self.name}_{str(key)}_{variable.name}", gradient, self.number_of_gradient_descent_steps_actor)
                    #---END Record gradient to summaries
                    
                    #---Start Record average entropy for episodes
                    if self.record_statistics:
                        with self.writer.as_default():
                            tf.summary.scalar(f"Entropy", entropy, self.number_of_gradient_descent_steps_actor)
                    #---End Record average entropy'for episodes
                    
                    #---START apply gradients for actor
                    for key, value in gradients.items():
                        if key == "mu":
                            self.actor_optimizer_mu.apply_gradients(zip(value, self.PPO.variables[key]))
                        if key == "cov":
                            self.actor_optimizer_cov.apply_gradients(zip(value, self.PPO.variables[key]))
                    self.number_of_gradient_descent_steps_actor += 1
                    #---END apply gradients for actor
                
                    #---END gradient descent for actor Nets
                    
                #---START update self.current_parameter with the parameters resulting from n steps of gradient descent
                self.current_parameters = {"mu": [variable.numpy() for variable in self.PPO.actor_mu.trainable_variables],
                            "cov": [variable.numpy() for variable in self.PPO.cov_head_variables],
                            "critic": [variable.numpy() for variable in self.PPO.critic.trainable_variables]
                            }
                #---END update self.current_parameter with the parameters resulting from n steps of gradient descent
                #---START Update Old policy seting theta_old = theta
                for key, value in self.PPO.current_parameters.items():
                    if key != "critic":
                        for n, variable in enumerate(self.PPO.variables_old[key]):
                            variable.assign(value[n])
                #---END Update Old policy seting theta_old = theta
   
                #---START updattrigger = True self.current_parameter_old with 
                self.current_parameters_old = self.current_parameters
                
                self.number_optimization_cycles += 1
            
                #---START after n iterations RUN EPISODE and PRINT REWARD
                if self.number_optimization_cycles % 1 == 0:
                    rewards_volatile = []
                for i in range (1):
                    path_conwip, path_PPO = self.collect_episodes_training(1, True, False)
                    reward_fixed = self.production_system_1.sum_rewards 
                    reward_PPO = self.production_system_2.sum_rewards 

                    print(f"\n Ep number: {self.current_number_episodes.value}")

                    print("--------------Conwip---------------")
                    summarize_performance(path_conwip)
                    print("--------------PPO Policy---------------")
                    summarize_performance(path_PPO)
                    print(f"\033[0;31mParts that entered the system: {sum(self.production_system_2.parts_in_system)}\033[m")
                    for i in range (3):
                        print(f"Part_{i}: {self.production_system_1.parts_in_system[i]}")

                    print(f"\033[0;32mReward {reward_PPO}\033[m")
                     

                    # av_reward = sum(rewards_volatile) / len(rewards_volatile)
                    # max_reward = max(rewards_volatile)
                    # min_reward = min(rewards_volatile)
                    
                    self.number_episodes_extracted += 1 # Counts the number of episodes extraceted from episode queue and not the number o episodes already run, still yet to be extracted
                    
                    # print(f"Ep number: {self.self.current}: Average: {av_reward} -- Max: {max_reward} -- Min {min_reward} ")
                    if self.record_statistics:
                        with self.writer.as_default():
                            tf.summary.scalar(f"Rewards", reward_PPO, self.number_episodes_extracted)
                            # tf.summary.scalar(f"Max_Reward", max_reward, self.self.current_number_episodes.value)
                            # tf.summary.scalar(f"Min_Reward", min_reward, self.self.current_number_episodes.value)
                    # self.rewards.append(av_reward)
                #---END after n iterations of the loop run episode and print reward
                
                #---START Save weights at current iter
            
                if self.save_checkpoints:
                    
                    with open("./saved_checkpoints/optimizers/actor/actor_optimizer_mu.pkl", "wb") as file:
                        pickle.dump(self.actor_optimizer_mu, file)
                    
                    with open("./saved_checkpoints/optimizers/actor/actor_optimizer_cov.pkl", "wb") as file:
                        pickle.dump(self.actor_optimizer_cov, file)
                    
                    
                    with open("./saved_checkpoints/optimizers/critic/critic_optimizer.pkl", "wb") as file:
                        pickle.dump(self.critic_optimizer, file)
    
                        
                    self.PPO.actor_mu.save_weights("./saved_checkpoints/actor_mu/")
                    self.PPO.actor_cov.save_weights("./saved_checkpoints/actor_cov/")
                    self.PPO.critic.save_weights("./saved_checkpoints/critic/")
                    
            
            #--- END Main RL loop
            #---After all cycles run a summary of the training session and write it on a file
            with open("Running_Log.csv", "a") as file:
                    
                writer = csv.writer(file, delimiter=",")
                writer.writerow(["Run", self.current_number_episodes.value])
            rewards_volatile = []
            for i in range (1):
                
                path_conwip, path_PPO = self.collect_episodes_training(1, True, False)
                reward_fixed = self.production_system_1.sum_rewards 
                reward_PPO = self.production_system_2.sum_rewards 

                print(f"\n Ep number: {self.current_number_episodes.value}")

                print("--------------Conwip---------------")
                summarize_performance(path_conwip)
                print("--------------PPO Policy---------------")
                summarize_performance(path_PPO)
                print(f"\033[0;31mParts that entered the system: {sum(self.production_system_1.parts_in_system)}\033[m")
                
                for i in range (3):
                    
                    print(f"Part_{i}: {self.production_system_1.parts_in_system[i]}")

                print(f"\033[0;32mReward {reward_PPO}\033[m")
                     
            print(f"Exited Global Agent")
                
        except KeyboardInterrupt:
                # rewards_volatile = []
            for i in range (1):
                path_conwip, path_PPO = self.collect_episodes_training(1, True, False)
                reward_fixed = self.production_system_1.sum_rewards 
                reward_PPO = self.production_system_2.sum_rewards 

                print(f"\n Ep number: {self.current_number_episodes.value}")

                print("--------------Conwip---------------")
                summarize_performance(path_conwip)
                print("--------------PPO Policy---------------")
                summarize_performance(path_PPO)
                print(f"\033[0;31mParts that entered the system: {sum(self.production_system_1.parts_in_system)}\033[m")
                for i in range (3):
                    
                    print(f"Part_{i}: {self.production_system_1.parts_in_system[i]}")      
                
                print(f"\033[0;32mReward {reward_PPO}\033[m")
                
            # rewards_volatile.append(reward)
            # av_reward = sum(rewards_volatile) / len(rewards_volatile)
            # max_reward = max(rewards_volatile)
            # min_reward = min(rewards_volatile)

            print("Press ctr + C one last time. Summary has be saved!")
            # self.average_reward_queue.put(sum(self.rewards) / len(self.rewards), block=True, timeout=30)

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, 17]),))
    def state_value(self, state):
        value = self.PPO.critic(state)
        return value
    
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, 17]), tf.TensorSpec(shape=[None, 3]), tf.TensorSpec(shape=[None, 1])))  
    def gradient_actor(self, states, actions, Qsa):
        with tf.GradientTape(persistent=True) as tape:
            if self.log_gradient_descent:       
                gradient_logger.debug(f"Cycle {self.number_optimization_cycles}")
                
            #---START Actor gradient calculation
            #---START Get the parameters for the Normal dist
            
            if self.log_gradient_descent:        
                gradient_logger.debug(f"Actions")
                for value in actions:
                    gradient_logger.debug(value)

            mu = self.PPO.actor_mu(states)
            
            if self.log_gradient_descent:        
                gradient_logger.debug(f"MU")
                for value in mu:
                    gradient_logger.debug (value)
                
            cov = self.PPO.actor_cov(states)
            
            if self.log_gradient_descent:        
                gradient_logger.debug(f"Cov")
                for value in cov: 
                    gradient_logger.debug(value)
                
            mu_old = tf.stop_gradient(self.PPO.actor_mu_old(states))
            cov_old = tf.stop_gradient(self.PPO.actor_cov_old(states))
            
            if self.log_gradient_descent:        
                gradient_logger.debug(f"MU_old")
                for value in mu_old:
                    gradient_logger.debug (value)
            
            if self.log_gradient_descent:        
                gradient_logger.debug(f"Cov_old")
                for value in cov_old:
                    gradient_logger.debug (value)
                
            #---END Get the parameters for the Normal dist
            #---START Advantage function computation and normalization
            advantage_function = Qsa - self.PPO.critic(states)
            advantage_function_mean = tf.math.reduce_mean(advantage_function)
            advantage_function_std = tf.math.reduce_std(advantage_function)
            advantage_function = tf.math.divide(advantage_function - advantage_function_mean, (advantage_function_std + 1.0e-11))
            #---END Advantage function computation and normalization
            #---START compute the Normal distributions
            self.probability_density_func = tfp.distributions.Normal(mu, cov + 1.0e-11)
            self.probability_density_func_old = tfp.distributions.Normal(mu_old, cov_old + 1.0e-11)
            #---END compute the Normal distributions
            #---Entropy
            entropy = self.probability_density_func.entropy()
            entropy_average = tf.reduce_mean(entropy)
            entropy_average = tf.stop_gradient(entropy_average)
            #---
            #---START compute the probability of the actions taken at the current episode
            probs = self.probability_density_func.prob(actions)
            
            if self.log_gradient_descent:        
                gradient_logger.debug(f"Probs")
                for value in probs:
                    gradient_logger.debug(value)
                
            probs_old = tf.stop_gradient(self.probability_density_func_old.prob(actions))
            
            if self.log_gradient_descent:        
                gradient_logger.debug(f"Probs_old")
                for value in probs_old:
                    gradient_logger.debug(value)
                
            #---START Ensemble Actor loss function
            self.probability_ratio = tf.math.divide(probs + 1e-11, probs_old + 1e-11)
            
            if self.log_gradient_descent:        
                gradient_logger.debug(f"Probability ratio")
                for value in self.probability_ratio:
                    gradient_logger.debug(value)
                    
            cpi = tf.math.multiply(self.probability_ratio, tf.stop_gradient(advantage_function))
            clip = tf.math.minimum(cpi, tf.multiply(tf.clip_by_value(self.probability_ratio, 1 - self.epsilon, 1 + self.epsilon), tf.stop_gradient(advantage_function)))
            actor_loss = -tf.reduce_mean(clip) - entropy_average * self.entropy
            gradient_logger.debug(f"Action Loss: {actor_loss}")
            
            #---END Ensemble Actor loss function
        
        #---START Compute gradients for average
        gradients_mu = tape.gradient(actor_loss, self.PPO.actor_mu.trainable_variables)
        #---
        if self.log_gradient_descent:        
            gradient_logger.debug(f"Gradients Mu")
            for value in gradients_mu:
                gradient_logger.debug(value)
            
        #---START Compute gradients for the covariance

        gradients_cov = tape.gradient(actor_loss, self.PPO.cov_head_variables)
        # END Compute gradients for the covariance
        
        if self.log_gradient_descent:        
            gradient_logger.debug("Gradient Cov")
            for value in gradients_cov:
                gradient_logger.debug(value)
                
        gradients = {"mu": gradients_mu,
                     "cov": gradients_cov,}
        #---END Actor gradient calculation
        
          
        return gradients, entropy_average, actor_loss
            
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, 17]), tf.TensorSpec(shape=[None, 1])))
    def gradient_critic(self, states, Qsa):
        with tf.GradientTape(persistent=True) as tape:
        
            critic_cost = tf.reduce_sum(tf.losses.mean_squared_error(Qsa, self.PPO.critic(states)))
    
        gradients_critic = tape.gradient(critic_cost, self.PPO.critic.trainable_variables)
    
        return gradients_critic, critic_cost
            
class WorkerAgent(Agent):
    def __init__(self,
                 name,
                 action_range, 
                 conwip, 
                 production_system_configuration,
                 ppo_networks_configuration,
                 gamma,
                 current_number_episodes, 
                 total_number_episodes,
                 episode_queue,
                 number_episodes_worker,
                 run_length,
                 epsilon,
                 record_statistics,
                 gradient_clipping_actor,
                 gradient_clipping_critic,
                 parameters_queue,
                 actor_optimizer_mu,
                 actor_optimizer_cov,
                 critic_optimizer,
                 entropy,
                 gradient_steps_per_episode_actor,
                 gradient_steps_per_episode_critic,
                 log_gradient_descent,
                 n_reward_returns,
                 ):
        Agent.__init__(self,
                        name=name,
                        action_range=action_range, 
                        conwip=conwip, 
                        production_system_configuration=production_system_configuration,
                        current_number_episodes=current_number_episodes, 
                        total_number_episodes=total_number_episodes,
                        episode_queue=episode_queue,
                        run_length=run_length,
                        epsilon=epsilon,
                        gamma=gamma,
                        record_statistics=record_statistics,
                        ppo_networks_configuration=ppo_networks_configuration,
                        gradient_clipping_actor=gradient_clipping_actor,
                        gradient_clipping_critic=gradient_clipping_critic,
                        parameters_queue=parameters_queue,
                        actor_optimizer_mu=actor_optimizer_mu,
                        actor_optimizer_cov=actor_optimizer_cov,
                        critic_optimizer=critic_optimizer,
                        entropy=entropy,
                        number_episodes_worker=number_episodes_worker,
                        gradient_steps_per_episode_actor = gradient_steps_per_episode_actor,
                        gradient_steps_per_episode_critic = gradient_steps_per_episode_critic,
                        log_gradient_descent=log_gradient_descent,
                        n_reward_returns=n_reward_returns)

    def training_loop(self):
         # ---START build NETworks for critic and actor
        if self.record_statistics:
            self.writer = tf.summary.create_file_writer(f"./summaries/global/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{self.name}")
        
        #define fixed policy
        self.FixedPolicy = FixedPolicy(self.conwip, self.conwip, self.conwip, action_range = self.action_range) 
        self.PPO = PPO(**self.ppo_networks_configuration, action_range=self.action_range, agent_name=self.name,summary_writer=self.writer)
        self.PPO.build_models() #build models (Critic, Actor Networks)
        # ---

    
        while self.current_number_episodes.value < self.total_number_episodes:
            self.number_episodes_run_upuntil = self.current_number_episodes.value
            # ---Update variables with information coming from gradient descent
            self.update_variables()
            # ---
            
            # ---START Collect n episodes from this worker
            for ep in range(self.number_episodes_worker): # Run more than episode per iteration of PPO
                if self.current_number_episodes.value < self.total_number_episodes:
                    self.collect_episodes_training(1, False, False)
                    
                    states, actions, next_states, rewards, qsa = self.buffer2.unroll_memory(self.gamma, self.n_reward_returns, self.PPO)
                    rollout = (states, actions, next_states, rewards, qsa)
                    try:
                        self.episode_queue.put(rollout, block=False)
                    except Full:
                        print("Queue Was full")
                        break
                        
                    self.current_number_episodes.value += 1
                else: break
                
            #---END Collect n episodes from this worker   
              
        print(f"Exited {self.name}")
            
    def update_variables(self):
         #---Get current parameters from queue(put in by the global agent)
        try:
            self.new_params = self.parameters_queue.get(block=True)
        except Exception as e:
            print(e)
        
        
        # for key, value in self.new_params.items():
        #     print(key)
        #     for n, variable in enumerate(self.PPO.variables[key]):
        #         print(f"Variable to be passed {value[n].shape}")
        #         print(f"Variable to be updated{variable}")
        
        # exit()

        #---
        #---START assign the variables of the worker with the variable values from global
        for key, value in self.new_params.items():
            for n, variable in enumerate(self.PPO.variables[key]):
                variable.assign(value[n])
        #END assign the variables of the worker with the variable values from global

        
        #---START update variable current_parameters to reflect the information provided by global
        self.current_parameters = {"mu": [variable.numpy() for variable in self.PPO.actor_mu.trainable_variables],
                        "cov": [variable.numpy() for variable in self.PPO.cov_head_variables],
                        "critic": [variable.numpy() for variable in self.PPO.critic.trainable_variables]
    
                        }
        #---END update variable current_parameters to reflect the information provided by global
    



ppo_networks_configuration = {"trunk_config": {"layer_sizes": [100, 80, 70],
                                      "activations": ["elu",   "elu", "elu"]},

                     "mu_head_config": {"layer_sizes": [60, 50, 3],
                                        "activations": ["elu", "elu", "relu"]},
                     "cov_head_config": {"layer_sizes": [60, 50, 3],
                                        "activations": ["elu","elu", "relu"]},
                     "critic_net_config": {"layer_sizes": [100, 100, 100, 80, 40, 1],
                                            "activations": ["elu", "elu","elu", "elu", "elu", "linear"]},
                     "input_layer_size": 17
                       }


hyperparameters = {"ppo_networks_configuration" : ppo_networks_configuration,
                   "actor_optimizer_mu": tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9),
                   "actor_optimizer_cov": tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9),
                    "critic_optimizer": tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9),
                    "entropy": 1,  
                    "gamma":0.999,
                    "gradient_clipping_actor": 1.0, 
                    "gradient_clipping_critic": 1.0, 
                    "gradient_steps_per_episode_critic": 5,
                    "gradient_steps_per_episode_actor": 10,
                    "epsilon": 0.2,
                    "number_episodes_worker": 1,
                    "n_reward_returns": 5
                    }
agent_config = {
    "action_range": (0, 500),
    "total_number_episodes" : 100000,
    "conwip": 1000,
    "run_length": 3000
    
}
 
parts = {
    "0": {"machine_0": [0, 10], "machine_1": [0, 10], "demand": [5, 5]},
    "1": {"machine_0": [0, 5], "machine_2": [0, 20], "machine_1": [0, 20], "demand": [10, 10]},
    "2": {"machine_2": [0, 10], "machine_3": [0, 30], "demand": [15, 15]}}


production_system_config = {
    "decision_epoch_interval": 30, 
    "track_state_interval": 5,
    "beta_state_weighted_average": 0.9,
    "parts_info": parts, 
    "number_routes": 3, 
    "number_workstations": 4,
    "number_of_different_parts": 3,
} 
if __name__ == "__main__":
    
    multiprocessing.set_start_method('spawn')
    number_of_workers = 4 

    params_queue = Manager().Queue(number_of_workers)
    current_number_episodes = Manager().Value("i", 0)    
    episode_queue = Manager().Queue(number_of_workers*hyperparameters["number_episodes_worker"])
    average_reward_queue = Queue(1)
    global_agent = GlobalAgent(**hyperparameters,
                               **agent_config,
                               production_system_configuration = production_system_config,
                               name="Global Agent",
                               record_statistics=True,
                               average_reward_queue=average_reward_queue,
                               number_of_child_agents=number_of_workers,
                               save_checkpoints=True,
                               episode_queue=episode_queue,
                               current_number_episodes=current_number_episodes,
                               parameters_queue=params_queue,
                               log_gradient_descent=False)
    workers = []
    for _ in range(number_of_workers):
        print("worker created")
        myWorker = WorkerAgent(**hyperparameters,
                        **agent_config,
                        production_system_configuration=production_system_config,
                        name=f"Worker_{_}",
                        record_statistics=True,
                        episode_queue=episode_queue,
                        current_number_episodes=current_number_episodes,
                        parameters_queue=params_queue,
                        log_gradient_descent=False) 
        
        workers.append(myWorker)
   


    processes = []
    p1 = Process(target=global_agent.training_loop)
    processes.append(p1)
    p1.start()
    for worker in workers:
        
        p = Process(target=worker.training_loop)
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    print(average_reward_queue.get())
    print("Simulation Run")