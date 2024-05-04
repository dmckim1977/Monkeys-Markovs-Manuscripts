import pandas as pd
import numpy as np

NUM_DAYS = 15
NUM_TICKETS = 10
PROBS = np.array([0.25, 0.5, 0.25]) # P(X = x) = {1/4 for x=0, 1/2 for x=1, 1/4 for x=2} 
EXPENSIVE_TICKET_PROB = 0.4
CHEAP_TICKET_PROB = 0.6
CHEAP_PRICE = 100
EXPENSIVE_PRICE = 200
ACTIONS = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]

g_00 = 0

# (1, 0) -> 1 cheap ticket
g_10 = PROBS[1] * CHEAP_PRICE + PROBS[2] * CHEAP_PRICE

# (0, 1) -> 1 expensive ticket
g_01 = (PROBS[1] + PROBS[2]) * EXPENSIVE_TICKET_PROB * EXPENSIVE_PRICE

# (1, 1) -> 1 cheap ticket, 1 expensive ticket
g_11 = PROBS[1] * CHEAP_PRICE + PROBS[2] * (CHEAP_PRICE + EXPENSIVE_TICKET_PROB * EXPENSIVE_PRICE)

# (2, 0) -> 2 cheap tickets
g_20 = PROBS[1] * CHEAP_PRICE + PROBS[2] * 2 * CHEAP_PRICE

# (0, 2) -> 2 expensive tickets
g_02 = PROBS[1] * EXPENSIVE_TICKET_PROB * EXPENSIVE_PRICE + PROBS[2] * 2 * EXPENSIVE_TICKET_PROB * EXPENSIVE_PRICE

G_rewards = g_00, g_10, g_01, g_20, g_11, g_02
print(G_rewards)


P_matrices = {
    action: np.zeros((NUM_TICKETS + 1, NUM_TICKETS + 1)) for action in ACTIONS
}
P_matrices[(0, 0)] = np.eye(NUM_TICKETS + 1)


def fill_matrix(action):
    if action == (1, 0):
        for i in range(1, NUM_TICKETS + 1):
            P_matrices[action][i, i] = PROBS[0]
            P_matrices[action][i, i - 1] = PROBS[1] + PROBS[2]
    elif action == (0, 1):
        for i in range(1, NUM_TICKETS + 1):
            P_matrices[action][i, i] = (
                PROBS[0] + PROBS[1] * CHEAP_TICKET_PROB + PROBS[2] 
                    * CHEAP_TICKET_PROB
            )
            P_matrices[action][i, i - 1] = (
                PROBS[1] * EXPENSIVE_TICKET_PROB + PROBS[2] * EXPENSIVE_TICKET_PROB
            )
    elif action == (2, 0):
        for i in range(1, NUM_TICKETS + 1):
            P_matrices[action][i, max(i - 2, 0)] = PROBS[2]
            if i == 1:
                P_matrices[action][i, i - 1] = PROBS[1] + PROBS[2]
            if i > 1:
                P_matrices[action][i, i - 1] = PROBS[1]
            P_matrices[action][i, i] = PROBS[0]
    elif action == (1, 1):
        for i in range(2, NUM_TICKETS + 1):
            P_matrices[action][i, i - 2] = PROBS[2] * EXPENSIVE_TICKET_PROB
            P_matrices[action][i, i - 1] = PROBS[1] + PROBS[2] * CHEAP_TICKET_PROB
            P_matrices[action][i, i] = PROBS[0]
        P_matrices[action][1, 0] = PROBS[1] + PROBS[2]  # Special case for i=1
        P_matrices[action][1, 1] = PROBS[0]
    elif action == (0, 2):
        for i in range(2, NUM_TICKETS + 1):
            P_matrices[action][i, i - 2] = PROBS[2] * EXPENSIVE_TICKET_PROB**2
            P_matrices[action][i, i - 1] = (
                PROBS[1] * EXPENSIVE_TICKET_PROB
                + 2 * PROBS[2] * EXPENSIVE_TICKET_PROB * CHEAP_TICKET_PROB
            )
            P_matrices[action][i, i] = (
                PROBS[0]
                + PROBS[1] * CHEAP_TICKET_PROB
                + PROBS[2] * CHEAP_TICKET_PROB**2
            )
        P_matrices[action][1, 0] = (
            PROBS[1] * EXPENSIVE_TICKET_PROB + PROBS[2] * EXPENSIVE_TICKET_PROB
        )  # Special case for i=1
        P_matrices[action][1, 1] = (
            PROBS[0] + PROBS[1] * CHEAP_TICKET_PROB + PROBS[2] * CHEAP_TICKET_PROB
        )

    P_matrices[action][0, 0] = 1


[fill_matrix(action) for action in ACTIONS]

for action in ACTIONS:
    print(f"Transition Probabilities for Action: {action}:")
    print(P_matrices[action])
    print("\n ---------------------------------------------------------------- \n")
    
    
optimal_ticket_sale_policy = {}

v_star = np.zeros(NUM_TICKETS + 1)

for day in range(NUM_DAYS):
    v = v_star.copy()
    a_star_list = []
    v_star = np.zeros(NUM_TICKETS + 1)
    for i in range(NUM_TICKETS + 1):
        v_i = np.zeros(len(ACTIONS))
        for j, action in enumerate(ACTIONS):
            if sum(action) <= i:
                v_i[j] = G_rewards[j] + np.dot(P_matrices[action][i], v)
        
        v_star[i] = np.max(v_i)
        a_star = np.argmax(v_i)
        a_star_list.append(ACTIONS[a_star])
    
    optimal_ticket_sale_policy[day] = (a_star_list, v_star.tolist())
    
optimal_action_df = pd.DataFrame({key: a[0] 
                                    for key, a in optimal_ticket_sale_policy.items()
                                }).T
optimal_action_df = optimal_action_df.iloc[::-1]
optimal_action_df.index.name = 'm:'
optimal_action_df = optimal_action_df.round(1)
optimal_action_df = optimal_action_df.astype(str)
# optimal_action_df = optimal_action_df.style.set_caption(
#     "Optimal Ticket Sale Policy Action Matrix")

print(optimal_action_df)


optimal_value_df = pd.DataFrame({key: a[1] 
                                    for key, a in optimal_ticket_sale_policy.items()
                                }).T
optimal_value_df = optimal_value_df.iloc[::-1]
optimal_value_df.index.name = 'm:'
optimal_value_df = optimal_value_df.round(2)
optimal_value_df = optimal_value_df.astype(str)
# optimal_value_df = optimal_value_df.style.set_caption(
#     "Optimal Ticket Sale Policy Value Matrix")

print(optimal_value_df)