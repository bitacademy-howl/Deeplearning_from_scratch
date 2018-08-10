import gym
import numpy as np

possible_actions = [[0, 1, 2], [0, 2], [1]]

def optimal_policy(state):
    return np.argmax(q_values[state])

class MDPEnvironment(object):
    def __init__(self, start_state = 0):
        self.start_state = start_state # 멤버변수를 정의 하지 않아도 되나?????
        self.reset()
    def reset(self):
        self.total_rewards = 0
        self.state = self.start_state
    def step(self, action):
        next_state = np.random.choice(range(3), p=transition_probabilities[self.state][action])
        reward = rewards[self.state][action][next_state]
        self.state = next_state
        self.total_rewards += reward
        return self.state, reward


def policy_random(state):
    return np.random.choice(possible_actions[state])

n_states = 3
n_actions = 3
n_steps = 20000
alpha = 0.01
gamma = 0.99

exploration_policy = policy_random

q_values = np.full((n_states, n_actions), -np.inf)

#############################################################################################
#     이 예제에서 이런 확률값을 조정하는 과정(Optimal Policy)이 포함되어야 Q-learning       #
#############################################################################################
transition_probabilities = [
    [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
    [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
    [None, [0.8, 0.1, 0.1], None]
]

rewards = [
    [[+10,0,0],[0,0,0],[0,0,0]],
    [[0,0,0],[0,0,0],[0,0,-50]],
    [[0,0,0],[+40,0,0],[0,0,0]]
]

def run_episode(policy, n_steps, start_state=0, display=True):
    env = MDPEnvironment()

    if display:
        print("상태 (+보상) : ", end=" ")
    for step in range(n_steps):
        if step == 10:
            print("...", end=" ")
        elif step < 10:
            print(env.state, end=" ")
    action = policy(env.state)
    state, reward = env.step(action)

    if display and step < 10:
        if reward:
            print("({})".format(reward), end=" ")

    if display:
        print("전체 보상 = ", env.total_rewards)
    return env.total_rewards


for state, actions in enumerate(possible_actions):
    q_values[state][actions] = 0


env = MDPEnvironment()

for step in range(n_steps):
    action = exploration_policy(env.state)
    state = env.state
    next_state, reward = env.step(action)
    next_value = np.max(q_values[next_state])
    q_values[state, action] = (1-alpha) * q_values[state, action] + alpha * (reward + gamma * next_value)

all_totals = []
for episode in range(1000):
    all_totals.append(run_episode(optimal_policy, n_steps=100, display=(episode < 5)))
print("요약 : 평균 = {:.1f}, 표준편차 = {:1f}, 최소={}, 최대={}".format(np.mean(all_totals), np.std(all_totals), np.min(all_totals), np.max(all_totals)))
print()