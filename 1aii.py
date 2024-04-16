import numpy as np

# Define the states, actions, cost function, and transition probabilities
states = ['G', 'B']
actions = [0, 1]  # 0: not use the channel, 1: use the channel
eta_values = [0.9, 0.7, 0.01]
beta = 0.75  # Discount factor

# Transition matrix for action 'use the channel' (u=1)
P_use = np.array([[0.1, 0.9],  # Probability of going to [G, B] from G
                  [0.8, 0.2]]) # Probability of going to [G, B] from B

# Transition matrix for action 'not use the channel' (u=0)
P_not_use = np.array([[0.9, 0.1],  # Probability of going to [G, B] from G
                      [0.5, 0.5]]) # Probability of going to [G, B] from B

# Function to calculate the cost
def cost_function(state, action, eta):
    cost = 0
    if state == 'G' and action == 1:
        cost = -1
    cost += (eta*action)
    return cost

# Policy iteration algorithm
def policy_iteration(states, actions, P_use, P_not_use, beta, eta, threshold=1e-6):
    policy = np.zeros(len(states), dtype=int)
    stable = False
    V = np.zeros(len(states))
    
    while not stable:
        # Policy evaluation
        while True:
            delta = 0
            for state_index, state in enumerate(states):
                v = V[state_index]
                action = policy[state_index]
                # Update value for the current policy
                if action == 1:
                    V[state_index] = cost_function(state, action, eta) + beta * P_use[state_index] @ V
                else:
                    V[state_index] = cost_function(state, action, eta) + beta * P_not_use[state_index] @ V
                delta = max(delta, abs(v - V[state_index]))
            if delta < threshold:
                break
        
        # Policy improvement
        stable = True
        for state_index, state in enumerate(states):
            old_action = policy[state_index]
            # Look for the action that minimizes the cost function
            action_values = [cost_function(state, action, eta) + beta * P_not_use[state_index] @ V if action == 0 else cost_function(state, action, eta) + beta * P_use[state_index] @ V for action in actions]
            policy[state_index] = np.argmin(action_values)
            if old_action != policy[state_index]:
                stable = False
    
    return V

# Perform policy iteration for each eta value
policies = {}
for eta in eta_values:
    policies[eta] = policy_iteration(states, actions, P_use, P_not_use, beta, eta)
    print(policies[eta])


policies
