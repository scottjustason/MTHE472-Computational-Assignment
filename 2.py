import pulp

# Define the transition probabilities and cost for actions
p = 0.9  # Probability of staying in the same state when 'do_nothing' action is taken
alpha = 0.1  # Probability of staying in the bad state even after 'repair' action is taken
eta = 0.7  # Cost when an action 'repair' is taken

# Assuming transition probabilities and cost are given as follows
transition_probabilities = {
    'G': {'G': {'do_nothing': p, 'repair': 1-alpha},
          'B': {'do_nothing': 1-p, 'repair': alpha}},
    'B': {'G': {'do_nothing': 0, 'repair': 1-alpha},
          'B': {'do_nothing': 1, 'repair': alpha}}
}
costs = {
    'G': {'do_nothing': -1, 'repair': eta},
    'B': {'do_nothing': 0, 'repair': eta}
}
states = ['G', 'B']
actions = ['do_nothing', 'repair']

# Initialize the LP problem
lp_problem = pulp.LpProblem('Average_Cost_Control', pulp.LpMinimize)

# Define the decision variables
v_vars = pulp.LpVariable.dicts("v", states, cat='Continuous')
d = pulp.LpVariable('d', cat='Continuous')

# Objective function: Minimize the average cost
lp_problem += d, "Minimize the average cost"

# Constraints
for x in states:
    for u in actions:
        next_state_cost = sum(transition_probabilities[x][next_x][u] * v_vars[next_x] for next_x in states)
        lp_problem += v_vars[x] >= next_state_cost + costs[x][u] - d, f"Constraint for state {x} and action {u}"

# Solve the problem
lp_problem.solve()

# Print the optimal value function and the average cost per stage
for x in states:
    print(f"The long-term expected cost for state {x} is: {v_vars[x].value()}")
print(f"The minimized average cost per stage is: {d.value()}")
