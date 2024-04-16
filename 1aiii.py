import numpy as np

# Define the states, actions, cost function, and transition probabilities
states = ['G', 'B']
actions = [0, 1]  # 0: not use the channel, 1: use the channel
eta = 0.7  # The given eta value for the cost function
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

# Initialize the Q-table
Q_table = np.zeros((len(states), len(actions)))

# Initialize the counters for state-action pairs for alpha calculation
state_action_counts = np.zeros((len(states), len(actions)))

# Define the alpha function as per the given formula
def alpha_t(state_index, action_index, count):
    return 1 / (1 + count)

# Learning parameters
episodes = 10000  # Total number of episodes for training

# Q-learning update function
def q_learning_update(state_index, action, reward, next_state_index, alpha):
    best_future_q = min(Q_table[next_state_index, :])
    Q_table[state_index, action] += alpha * (reward + beta * best_future_q - Q_table[state_index, action])

# Simulate the Q-learning process
for episode in range(episodes):
    for state_index, state in enumerate(states):
        for action in actions:
            # Simulate the transition
            if action == 1:
                next_state_index = np.random.choice([0, 1], p=P_use[state_index])
            else:
                next_state_index = np.random.choice([0, 1], p=P_not_use[state_index])

            # Get the reward for the transition
            reward = cost_function(state, action, eta)

            # Update the count for the current state-action pair
            state_action_counts[state_index, action] += 1

            # Calculate the learning rate
            alpha = alpha_t(state_index, action, state_action_counts[state_index, action])

            # Perform the Q-learning update
            q_learning_update(state_index, action, reward, next_state_index, alpha)

# After training, we can determine the policy
policy_from_q = np.argmax(Q_table, axis=1)

# Print the results
print("Q-table:")
print(Q_table)
print("\nDerived policy (0: not use the channel, 1: use the channel):")
print("Policy for state G:", "Use the channel" if policy_from_q[0] else "Do not use the channel")
print("Policy for state B:", "Use the channel" if policy_from_q[1] else "Do not use the channel")
