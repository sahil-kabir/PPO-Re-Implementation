import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import gymnasium as gym
import pandas as pd
import plotly.express as px

class FFNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, out_dim)
        )
    
    def forward(self, input):
        input = torch.tensor(input)
        if len(input.shape) == 1:
            input = input.unsqueeze(0)
        return self.sequential(input)

def create_trajectories(env, actor, timesteps):
    states = []
    state_terminal_indicator = []
    state_terminal_indicator.append(True)
    actions = []
    rewards = []
    logodds = []
    obs, _ = env.reset()
    while (timesteps >= 0) or (not terminated): # let the last episode finish
        timesteps = timesteps - 1
        states.append(obs)
        logits = actor(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()

        logodds.append(dist.log_prob(action).item())
        actions.append(action.item())
        obs, reward, terminated, _, _ = env.step(actions[-1])
        rewards.append(reward)
        state_terminal_indicator.append(terminated)
        if terminated:
            obs, _ = env.reset()

    return np.array(states), np.array(actions), np.array(logodds), np.array(rewards), np.array(state_terminal_indicator)

def validate(env, actor):
    seeds = [0, 42, 200, 1000, 9999]
    score = 0
    for seed in seeds:
        obs, _ = env.reset(seed=42)
        terminated = False
        while not terminated:
            obs, reward, terminated, _, _ = env.step(actor(obs).argmax().item())
            score += reward
    return score/5

def GAE(critic, states, starts, rewards, gamma=0.99, lbda=0.95):
    timesteps = len(states)
    A = np.zeros(timesteps)
    delta = np.zeros(timesteps)
    
    for t in range(timesteps - 1):
        delta[t] = rewards[t] + gamma * critic(states[t+1]).item() * int(not starts[t+1]) - critic(states[t]).item()
    
    for t in reversed(range(timesteps - 1)):
        A[t] = delta[t] + gamma * lbda * A[t+1] * int(not starts[t+1])

    return A

def UpdatePPO(actor, critic, optimizer_actor, optimizer_critic,
              states, advantages, actions, rewards, old_logodds,
              clip_ratio=0.2,
              epochs=10,
              batch_size=64):
    for _ in range(epochs):
        indeces = np.arange(len(states))
        np.random.shuffle(indeces)
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            idx = indeces[start:end]
            s = states[idx]
            a = torch.tensor(actions[idx], dtype=torch.int64)
            lo = torch.tensor(old_logodds[idx], dtype=torch.float32) 
            r = torch.tensor(rewards[idx], dtype=torch.float32) 
            adv = torch.tensor(advantages[idx], dtype=torch.float32)

            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()

            ratio = torch.exp(Categorical(logits=actor(s)).log_prob(a) - lo)
            loss1, loss2 = ratio * adv, torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
            a_loss = -1 * torch.min(loss1, loss2).mean()
            v_loss = ((critic(s) - r)**2).mean()

            a_loss.backward()
            v_loss.backward()
            optimizer_actor.step()
            optimizer_critic.step()


def train_ppo(env_name="CartPole-v1", render="human", iterations=200, steps_per_iter=4000, k_epochs=15, lr_actor=1e-6, lr_critic=1e-6, clip_ratio=0.2, batch_size=64, gamma=0.99, lbda=0.95, device="cpu"):
    env = gym.make(env_name, render_mode=render)

    actor = FFNetwork(env.observation_space.shape[0], env.action_space.n)
    critic = FFNetwork(env.observation_space.shape[0], 1)
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lr_critic)

    best_score = 0
    best_model = actor.state_dict()

    avg_rewards = []
    for i in range(iterations):
        states, actions, old_logodds, rewards, starts = create_trajectories(env, actor, steps_per_iter)
        advantages = GAE(critic, states, starts, rewards)
        UpdatePPO(actor, critic, optimizer_actor, optimizer_critic,
              states, advantages, actions, rewards, old_logodds,
              clip_ratio=clip_ratio,
              epochs=k_epochs,
              batch_size=batch_size)
        s = validate(env, actor)
        avg_rewards.append(s)
        print(f"Iteration {i} complete. score: {s}")
        if s >= best_score:
            best_score = s
            best_model = actor.state_dict()
    
    actor.load_state_dict(best_model)
    env.close()
    return actor, avg_rewards

if __name__ == '__main__':
    actor, avg_rewards = train_ppo(render=None)
    df = pd.DataFrame({'Iteration': range(1, len(avg_rewards) + 1), 'Score': avg_rewards})
    fig = px.line(df, x='Iteration', y='Score', title='Score per Iteration', markers=True)
    fig.show()
    
