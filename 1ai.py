import numpy as np

# Define the states, actions, cost function, and transition probabilities
states = ['G', 'B']
actions = [0, 1]  # 0: not use the channel, 1: use the channel
eta_values = [0.9, 0.7, 0.01]
beta = 0.7  # Discount factor

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

# Initialize the value function
V = np.zeros(len(states))

# Value iteration algorithm
def value_iteration(states, actions, P_use, P_not_use, V, beta, eta, threshold=1e-6):
    delta = float('inf')
    while delta > threshold:
    # for i in range(3):
        delta = 0
        for state_index, state in enumerate(states):
            v = V[state_index]
            # Calculate the value for each action and choose the minimum one
            V[state_index] = min(cost_function(state, 0, eta) + beta * P_not_use[state_index] @ V,
                                 cost_function(state, 1, eta) + beta * P_use[state_index] @ V)
            delta = max(delta, abs(v - V[state_index]))
    return V

# Perform value iteration for each eta value 
optimal_values = {}
for eta in eta_values:
    optimal_values[eta] = value_iteration(states, actions, P_use, P_not_use, V.copy(), beta, eta)
    print(optimal_values[eta])

