import random as rnd
import matplotlib.pyplot as plt
GLASSES_COUNT = 5
ORDER = 10
NUM_EPISODES = 2000
ALPHA = 0.8
GAMMA = 0.95
Q = {}
#Une liste g est créée, qui représentera la répartition initiale de la compote dans les verres. Ajoutons une condition : 
# si la somme de tous les éléments de la liste g n'est pas divisible par le nombre de verres, 
# alors la valeur manquante est ajoutée à l'un des verres pour que la somme devienne un multiple du nombre de verres.

def init_glasses():
    g = []
    for _ in range(GLASSES_COUNT):
        g.append(rnd.randint(1, ORDER))
    if sum(g) % GLASSES_COUNT != 0:
        g[rnd.randint(0, GLASSES_COUNT-1)] += GLASSES_COUNT - (sum(g) % GLASSES_COUNT)
    return g

INITIAL_GLASSES = init_glasses()

print("Изначальное состояние стаканов:", INITIAL_GLASSES)
glasses = INITIAL_GLASSES

def state_to_string(state):
    _t = ""
    for s in state:
        _t += f'-{s}'
    return _t[1:]

def action_to_string(action):
    return f'{action[0]}/{action[1]}/{action[2]}'

def describe_action(action):
    return f'Агент переливает {action[2]} мл. компота из стакана № {action[0]+1} в стакан № {action[1]+1}'

def is_equal(glasses):
    for i in range(len(glasses)-1):
        if glasses[i] != glasses[i+1]:
            return False
    return True

def choose_random_action(glasses):
    glass_from = rnd.randint(0, GLASSES_COUNT-1)
    if glasses[glass_from] == 0:
        return choose_random_action(glasses)
    glass_to = rnd.randint(0, GLASSES_COUNT-2)
    if glass_to >= glass_from:
        glass_to += 1
    glass_compote = rnd.randint(1, glasses[glass_from])
    return (glass_from, glass_to, glass_compote)

def pour_compote(ogl, instruction):
    gl = ogl.copy()
    if gl[instruction[0]] >= instruction[2]:
        gl[instruction[0]] -= instruction[2]
        gl[instruction[1]] += instruction[2]
    return gl

def get_all_actions(glasses):
    actions = []
    for i in range(0, len(glasses)):
        for j in range(0, len(glasses)):
            if i == j:
                continue
            for k in range(1, glasses[i]+1):
                actions.append(action_to_string((i, j, k)))
    return actions

def choose_action(glasses):
    state = state_to_string(glasses)
    if state not in Q:
        return choose_random_action(glasses)
    
    max_value = max(Q[state].values())
    best_actions = [action for action, value in Q[state].items() if value == max_value]
    choice = rnd.choice(best_actions)
    return (int(choice.split("/")[0]), int(choice.split("/")[1]), int(choice.split("/")[2]))

def update_Q(glasses, action, reward, next_glasses):
    state = state_to_string(glasses)
    next_state = state_to_string(next_glasses)
    if state not in Q:
        Q[state] = {}
        for a in get_all_actions(glasses):
            Q[state][a] = 0

    if next_state not in Q:
        Q[next_state] = {}
        for a in get_all_actions(next_glasses):
            Q[next_state][a] = 0

    Q[state][action] = Q[state][action] + ALPHA * (reward + GAMMA * max(Q[next_state].values()) - Q[state][action])

def perform_actions(describe = False, random_initial = False):
    glasses = []
    if random_initial:
        glasses = init_glasses()
    else:
        glasses = INITIAL_GLASSES.copy()
    at = 0
    while not is_equal(glasses):
        at += 1
        if describe:
            print(f"{at}. Состояние стаканов:", glasses)
        action = choose_action(glasses)
        if describe:
            print(describe_action(action))
        new_glasses = pour_compote(glasses, action)
        update_Q(glasses, action_to_string(action), -1, new_glasses)
        glasses = new_glasses.copy()
    if describe:
        print(f"{at+1}. Состояние стаканов:", glasses)
        print(f"Стаканы уравновешены")
    return at
rewards_history = []
for ep in range(NUM_EPISODES):
    at = perform_actions(False, False)
    if ep % 100 == 0:
        print(f"Прогоны {ep+1}-{ep+100}")
    rewards_history.append(-at)

plt.plot(rewards_history)
plt.xlabel('Эпизоды')
plt.ylabel('Штраф')
plt.title('Кривая обучения')
plt.grid(True)
plt.show()
#Let's take a closer look at the segment where the agent's training took place.

plt.plot(rewards_history[0:350])
plt.xlabel('Эпизоды')
plt.ylabel('Штраф')
plt.title('Кривая обучения')
plt.grid(True)
plt.show()
perform_actions(True, False)
#Let's consider what the average number, minimum and maximum of transfusions were for training an agent on the given initial data.

total_moves = 0
min_moves = 999999
max_moves = 0
experiments = 100
for i in range(experiments):
    moves = perform_actions(False, True)
    total_moves += moves
    if moves < min_moves:
        min_moves = moves
    if moves > max_moves:
        max_moves = moves
print(f"Среднее количество переливаний: {total_moves / experiments}")
print(f"Минимум переливаний: {min_moves}")
print(f"Максимум переливаний: {max_moves}")
#Repeat agent training on a new set of input data:

INITIAL_GLASSES = init_glasses()

rewards_history = []
for ep in range(NUM_EPISODES):
    at = perform_actions(False, False)
    rewards_history.append(-at)

plt.plot(rewards_history)
plt.xlabel('Эпизоды')
plt.ylabel('Штраф')
plt.title('Кривая обучения')
plt.grid(True)
plt.show()

perform_actions(True, False)

for i in range(experiments):
    moves = perform_actions(False, True)
    total_moves += moves
    if moves < min_moves:
        min_moves = moves
    if moves > max_moves:
        max_moves = moves
print(f"Среднее количество переливаний: {total_moves / experiments}")
print(f"Минимум переливаний: {min_moves}")
print(f"Максимум переливаний: {max_moves}")
#И запустим обучение на ещё одном наборе данных:
INITIAL_GLASSES = init_glasses()

rewards_history = []
for ep in range(NUM_EPISODES):
    at = perform_actions(False, False)
    rewards_history.append(-at)

plt.plot(rewards_history)
plt.xlabel('Эпизоды')
plt.ylabel('Штраф')
plt.title('Кривая обучения')
plt.grid(True)
plt.show()

perform_actions(True, False)

for i in range(experiments):
    moves = perform_actions(False, True)
    total_moves += moves
    if moves < min_moves:
        min_moves = moves
    if moves > max_moves:
        max_moves = moves
print(f"Среднее количество переливаний: {total_moves / experiments}")
print(f"Минимум переливаний: {min_moves}")
print(f"Максимум переливаний: {max_moves}")