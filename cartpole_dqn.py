import torch
from torch import nn
from collections import deque
import copy
import random
import gym
from torch._C import device
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class DQN_Agent:
    def __init__(self, seed, layer_sizes, lr, sync_freq, exp_replay_size):
        torch.manual_seed(seed)
        self.q_net = self.build_nn(layer_sizes)
        self.target_net = copy.deepcopy(self.q_net)
        self.q_net.to(DEVICE)
        self.target_net.to(DEVICE)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        self.gamma = torch.tensor(0.95).float().to(DEVICE)
        self.experience_replay = deque(maxlen=exp_replay_size)
        return

    def build_nn(self, layer_sizes):
        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
            act = nn.Tanh() if index < len(layer_sizes) - 2 else nn.Identity()
            layers += (linear, act)
        return nn.Sequential(*layers)

    def get_action(self, state, action_space_len, epsilon):
        # We do not require gradient at this point, because this function will be used either
        # during experience collection or during inference
        with torch.no_grad():
            Qp = self.q_net(torch.from_numpy(state).float().to(DEVICE))
        Q, A = torch.max(Qp, axis=0)
        A = (
            A
            if torch.rand(
                1,
            ).item()
            > epsilon
            else torch.randint(0, action_space_len, (1,))
        )
        return A

    def get_q_next(self, state):
        with torch.no_grad():
            qp = self.target_net(state)
        q, _ = torch.max(qp, axis=1)
        return q

    def collect_experience(self, experience):
        self.experience_replay.append(experience)
        return

    def sample_from_experience(self, sample_size):
        if len(self.experience_replay) < sample_size:
            sample_size = len(self.experience_replay)
        sample = random.sample(self.experience_replay, sample_size)
        s = torch.tensor([exp[0] for exp in sample]).float()
        a = torch.tensor([exp[1] for exp in sample]).float()
        rn = torch.tensor([exp[2] for exp in sample]).float()
        sn = torch.tensor([exp[3] for exp in sample]).float()
        return s, a, rn, sn

    def train(self, batch_size):
        s, a, rn, sn = self.sample_from_experience(sample_size=batch_size)
        if self.network_sync_counter == self.network_sync_freq:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.network_sync_counter = 0

        # predict expected return of current state using main network
        qp = self.q_net(s.to(DEVICE))
        pred_return, _ = torch.max(qp, axis=1)

        # get target return using target network
        q_next = self.get_q_next(sn.to(DEVICE))
        target_return = rn.to(DEVICE) + self.gamma * q_next

        loss = self.loss_fn(pred_return, target_return)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.network_sync_counter += 1
        return loss.item()


env = gym.make("CartPole-v0")
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
exp_replay_size = 256
agent = DQN_Agent(
    seed=1423,
    layer_sizes=[input_dim, 64, output_dim],
    lr=1e-3,
    sync_freq=5,
    exp_replay_size=exp_replay_size,
)

# initiliaze experiance replay
index = 0
for i in range(exp_replay_size):
    obs = env.reset()
    done = False
    while done != True:
        A = agent.get_action(obs, env.action_space.n, epsilon=1)
        obs_next, reward, done, _ = env.step(A.item())
        agent.collect_experience([obs, A.item(), reward, obs_next])
        obs = obs_next
        index += 1
        if index > exp_replay_size:
            break

# Main training loop
previousCnt = []  # array of all scores over runs
metrics = {"ep": [], "avg": [], "min": [], "max": []}  # metrics recorded for graph

losses_list, reward_list, episode_len_list, epsilon_list = [], [], [], []
index = 128
episodes = 11000
epsilon = 1
updateEvery=100
showEvery=1000


for run in tqdm(range(episodes)):
    obs, done, losses, ep_len, rew = env.reset(), False, 0, 0, 0
    while done != True:
        if run % showEvery == 0:
                env.render()  # if running RL comment this out
        ep_len += 1
        A = agent.get_action(obs, env.action_space.n, epsilon)
        obs_next, reward, done, _ = env.step(A.item())
        agent.collect_experience([obs, A.item(), reward, obs_next])

        obs = obs_next
        rew += reward
        index += 1

        if index > 128:
            index = 0
            for j in range(4):
                loss = agent.train(batch_size=16)
                losses += loss
    previousCnt.append(ep_len)

    if epsilon > 0.05:
        epsilon -= 1 / 5000

    if run % updateEvery == 0:
            latestRuns = previousCnt[-updateEvery:]
            averageCnt = sum(latestRuns) / len(latestRuns)
            metrics["ep"].append(run)
            metrics["avg"].append(averageCnt)
            metrics["min"].append(min(latestRuns))
            metrics["max"].append(max(latestRuns))

    losses_list.append(losses / ep_len), reward_list.append(
        rew
    ), episode_len_list.append(ep_len), epsilon_list.append(epsilon)


plt.plot(metrics["ep"], metrics["avg"], label="average rewards")
plt.plot(metrics["ep"], metrics["min"], label="min rewards")
plt.plot(metrics["ep"], metrics["max"], label="max rewards")
plt.legend(loc=4)
plt.show()

env = gym.make("CartPole-v0")
for i in tqdm(range(2)):
    obs = env.reset()
    done = False
    rew = 0
    while done != True:
        A = agent.get_action(obs, env.action_space.n, epsilon=0)
        obs, reward, done, info = env.step(A.item())
        rew += reward
        sleep(0.03)
        env.render()
    print("episode : {}, reward : {}".format(i, rew))
