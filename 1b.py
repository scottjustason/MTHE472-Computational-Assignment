import numpy as np

def quantize_state_space(num_states):
    """Quantize the state space into discrete levels."""
    return np.linspace(0, 1, num_states)

def transition_probability(current_state_index, action, states):
    """Calculate the transition probabilities for quantized states."""
    current_state = states[current_state_index]
    if action == 1:  # Using the channel
        return 2 * (1 - current_state)
    else:  # Not using the channel
        return 2 * current_state

def cost_function(state, action, eta):
    """Calculate the cost for the state-action pair."""
    return -state * action + eta * action

# Quantize the state space and action space
num_states = [4, 8, 16, 32, 64]  # Different levels of quantization
actions = [0, 1]  # Not use or use the channel
eta = 0.7
gamma = 0.5  # Discount factor

# Q-learning for different quantization levels
for n in num_states:
    states = quantize_state_space(n)
    Q_table = np.zeros((n, len(actions)))
    state_action_counts = np.zeros((n, len(actions)))

    episodes = 5000
    for episode in range(episodes):
        for state_index in range(n):
            for action in actions:
                # Simulate the transition based on the action
                # Transition only to adjacent states or stay in the same state
                if state_index < n - 1:
                    next_state_probs = [
                        transition_probability(state_index, action, states),
                        transition_probability(state_index + 1, action, states)
                    ]
                    next_state_probs /= np.sum(next_state_probs)  # Normalize
                    next_state_choices = [state_index, state_index + 1]
                else:
                    next_state_probs = [1.0]  # Stay in the last state
                    next_state_choices = [state_index]

                next_state_index = np.random.choice(next_state_choices, p=next_state_probs)
                reward = cost_function(states[state_index], action, eta)

                # Update the count for the current state-action pair
                state_action_counts[state_index, action] += 1

                # Calculate the learning rate
                alpha = 1.0 / (1 + state_action_counts[state_index, action])

                # Perform the Q-learning update
                Q_table[state_index, action] += alpha * (reward + gamma * np.min(Q_table[next_state_index]) - Q_table[state_index, action])

    # Determine the policy from the Q-table
    policy_from_q = np.argmax(Q_table, axis=1)
    print(f"Quantization level: {n} states")
    print("Derived policy:", policy_from_q)
    print("Q-table:", Q_table, "\n")
