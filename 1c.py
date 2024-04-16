import numpy as np

# Problem setup
states = [0, 1]  # 0 for good state, 1 for bad state
actions = [0, 1]  # 0 for do nothing, 1 for repair
eta = 0.7  # The cost of repairing
gamma = 0.95  # Discount factor

# Transition probabilities
p = 0.9  # Probability of staying in the same state when taking the 'do nothing' action
alpha = 0.1  # Probability of staying in the bad state even after repair

# Cost function as given in the problem statement
def cost(x, u, eta):
    return -x*u + eta*u

# Function to determine the next state
def next_state(current_state, action, p, alpha):
    if action == 1:
        return np.random.choice(states, p=[1-alpha, alpha])
    else:
        return np.random.choice(states, p=[p, 1-p])

# Q-Learning parameters
num_episodes = 5000
window_size = 5  # Window size for partial observability
learning_rate = 0.1  # Initial learning rate

# Initialize Q-table
Q_table = np.zeros((2, 2))

# Function to update the Q-table
def update_Q(Q, state, action, reward, next_state, alpha, gamma):
    best_next_action = np.argmax(Q[next_state])
    Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * Q[next_state, best_next_action])
    return Q

# Simulate Q-learning process
for episode in range(num_episodes):
    # Initialize the state (randomly choosing a good or bad state)
    current_state = np.random.choice(states)
    state_memory = [current_state] * window_size  # Memory of the last 'window_size' states

    for t in range(100):  # Limit the number of steps in each episode
        # Choose action based on policy derived from Q (epsilon-greedy)
        if np.random.uniform(0, 1) < (1 - learning_rate):
            action = np.random.choice(actions)  # Explore
        else:
            action = np.argmax(Q_table[current_state])  # Exploit

        # Take action and observe new state and reward
        next_true_state = next_state(current_state, action, p, alpha)
        reward = cost(current_state, action, eta)
        state_memory.append(next_true_state)
        state_memory.pop(0)

        # Update Q-table
        Q_table = update_Q(Q_table, current_state, action, reward, next_true_state, learning_rate, gamma)

        # Update current state
        current_state = int(np.mean(state_memory) >= 0.5)  # Estimate the current state based on the windowed memory

        # Decrease learning rate over time
        learning_rate *= 0.9999

# Extract policy from Q-table
policy = np.argmax(Q_table, axis=1)

print("Q-table:")
print(Q_table)
print("\nDerived policy:")
print("State 0 (Good):", "Repair" if policy[0] else "Do nothing")
print("State 1 (Bad):", "Repair" if policy[1] else "Do nothing")
