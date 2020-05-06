from collections import deque
import numpy as np

class EpBuffer:
    """
    Class that stores the state transition information of an episode
    """

    def __init__(self):
        self.memory = deque()
    def add_transition(self, transition):
        """
        Arguments:
        transition -> Tuple (s, a, s', reward) (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.float64 )
        """
        self.memory.append(transition)

    @staticmethod
    def compute_Qsa(rewards, states, gamma, n_steps, policy):  # O(n^2)
        """
        Computes the sample value function (ground truth) for every single state action pair of an episode

        Arguments:
        rewards -> object that contain all the rewards from the episode from t = 0 to t = len(rewards)
        gamma -> float, discount factor for the rewards

        Returns:
        Qsa -> List

        """
        Qsa = []
        # print(f"states\n {states}")
        # print(f"Rewards\n {rewards}")
        for i in range(len(rewards)):

            partial_Qsa = 0
            t = 0
            for j in range(n_steps):
                if (i + j) >= (len(rewards) - 1):
                    
                    break
                else:
                    partial_Qsa += rewards[i + j] * (gamma ** t)
                    t += 1
            # print(f"T : {t}")
            state = np.reshape(states[i + t], (1, -1))
            # print(f"State {state}")
            state_value = policy.get_state_value(state)
            partial_Qsa += (gamma ** t) * state_value 
            Qsa.append(partial_Qsa)
            # print(f"State value : {state_value}")
            # print(f"Qsa\n {Qsa}")
        
        # print(f"Qsa\n {Qsa}")
        # exit()
        return Qsa
    
    def unroll_memory(self, gamma, n_reward_returns, policy):
        """
        Unrolls the states transitions information so that states , actions, next_states, rewards and Qsa's
        are separeted into different numpy arrays

        Returns:
        states -> numpy array (state dimension, num of state transitions)
        actions -> numpy array (action dimension, num of state transitions)
        next_states -> numpy array (state dimension, num of state transitions)
        rewards -> numpy array (num of state transitions, )
        qsa -> numpy array (num of state transitions, )
        """

        states, actions, next_states, rewards = zip(*self.memory)
        qsa = self.compute_Qsa(rewards, states, gamma, n_reward_returns, policy)
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32).reshape(-1, 1)
        next_states = np.asarray(next_states)
        rewards = np.asarray(rewards, dtype=np.float32)
        qsa = np.asarray(qsa, dtype=np.float32).reshape(-1, 1)

        # print(f"States: {states.shape}")
        # print(f"actions: {actions.shape}")
        # print(f"next_states: {next_states.shape}")
        # print(f"rewards: {rewards.shape}")
        # print(f"qsa: {qsa.shape}")
        self.memory = deque()
        return states, actions, next_states, rewards, qsa
