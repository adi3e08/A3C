import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from dm_control import suite

def process_observation(time_step):
    o_1 = np.array([])
    for k in time_step.observation:
        if time_step.observation[k].shape:
            o_1 = np.concatenate((o_1, time_step.observation[k]))
        else :
            o_1 = np.concatenate((o_1, np.array([time_step.observation[k]])))
    r = time_step.reward
    done = time_step.last()

    return o_1, r, done

"""
def set_init(layers):
    for layer in layers:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
"""

# Fully Connected Joint Actor Critic network
class Pi_V_FC(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(Pi_V_FC, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.mu = torch.nn.Linear(64, action_size)
        self.sigma = torch.nn.Linear(64, action_size)
        self.v = torch.nn.Linear(64, 1)
        # set_init([self.fc1, self.fc2, self.mu, self.sigma, self.v])
        self.distribution = torch.distributions.Normal        
    def forward(self, x):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        mu = torch.tanh(self.mu(y2))
        sigma = torch.clamp(F.softplus(self.sigma(y2)),min=1e-10)      # avoid 0
        dist = self.distribution(mu,sigma)
        values = self.v(y2).view(-1)
        return dist, values

# Fully Connected Actor network
class Pi_FC(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(Pi_FC, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.mu = torch.nn.Linear(64, action_size)
        self.sigma = torch.nn.Linear(64, action_size)
        # set_init([self.fc1, self.fc2, self.mu, self.sigma])
        self.distribution = torch.distributions.Normal        
    def forward(self, x):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        mu = torch.tanh(self.mu(y2))
        sigma = torch.clamp(F.softplus(self.sigma(y2)),min=1e-10)      # avoid 0
        dist = self.distribution(mu,sigma)
        
        return dist

# Fully Connected Critic network
class V_FC(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(V_FC, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.v = torch.nn.Linear(64, 1)
        # set_init([self.fc1, self.fc2, self.v])
    def forward(self, x):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        values = self.v(y2).view(-1)
        return values

# Advantage Actor Critic Algorithm
class advantage_actor_critic:
    def __init__(self, arglist):

        self.arglist = arglist
        self.device = torch.device("cpu")
        # self.device = torch.device("cuda:0")
        self.exp_dir = os.path.join("./log", self.arglist.exp_name)
        self.model_dir = os.path.join(self.exp_dir, "models")
        self.tensorboard_dir = os.path.join(self.exp_dir, "tensorboard")
        if os.path.exists("./log"):
            pass            
        else:
            os.mkdir("./log")
        os.mkdir(self.exp_dir)
        os.mkdir(os.path.join(self.tensorboard_dir))
        os.mkdir(self.model_dir)

        self.env = make_env()
        if self.arglist.joint_actor_critic:
            self.model = Pi_V_FC(self.env.state_size,self.env.action_size).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.arglist.lr)
        else:
            self.actor = Pi_FC(self.env.state_size,self.env.action_size).to(self.device)
            self.critic = V_FC(self.env.state_size,self.env.action_size).to(self.device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.arglist.lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.arglist.lr)

    def save_checkpoint(self, name):
        if self.arglist.joint_actor_critic:
            checkpoint = {'model' : self.model.state_dict()}
        else:
            checkpoint = {'actor' : self.actor.state_dict()}
        torch.save(checkpoint, os.path.join(self.model_dir, name))

    def train(self):
        writer = SummaryWriter(log_dir=self.tensorboard_dir)
        log_probs, entropies, values, R = [], [], [], []

        for episode in range(self.arglist.episodes):
            t = 0
            ep_r = 0.0
            o, _, _ = process_observation(self.env.reset())
            while True:
                if self.arglist.joint_actor_critic:
                    dist, values_ = self.model(torch.tensor(o, dtype=torch.float, device=self.device).unsqueeze(0))
                else:
                    O = torch.tensor(o, dtype=torch.float, device=self.device).unsqueeze(0)
                    dist = self.actor(O)
                    values_ = self.critic(O)
                a = dist.sample()                                   
                log_probs.append(dist.log_prob(a).sum())
                entropies.append(dist.entropy().sum())
                values.append(values_[0])
                o_1, r, done = process_observation(self.env.step(a.detach().cpu().numpy()[0]))
                t += 1
                ep_r += r
                R.append(torch.tensor(r, dtype=torch.float, device=self.device))
                o = o_1
                if (t % self.arglist.update_every == 0 or done):
                    # if done:
                    #     v_n = 0.0 # terminal. Only for gym.
                    # else:
                    with torch.no_grad():
                        if self.arglist.joint_actor_critic: 
                            dist, next_values = self.model(torch.tensor(o_1, dtype=torch.float, device=self.device).unsqueeze(0))
                        else:
                            next_values = self.critic(torch.tensor(o_1, dtype=torch.float, device=self.device).unsqueeze(0))
                    v_n = next_values[0]
                    values.append(next_values[0])

                    critic_loss = 0
                    actor_loss = 0
                    gae = 0
                    for i in reversed(range(len(R))):
                        v_n = R[i] + self.arglist.gamma * v_n
                        advantage = v_n - values[i]
                        critic_loss = critic_loss + advantage.pow(2)

                        gae = self.arglist.gamma * self.arglist.gae_lambda * gae + R[i] + self.arglist.gamma * values[i+1] - values[i] 
                        actor_loss = actor_loss - log_probs[i] * gae.detach() - self.arglist.entropy_weightage * entropies[i]

                    # actor_loss = actor_loss / len(R)
                    critic_loss = critic_loss / len(R)

                    if self.arglist.joint_actor_critic:
                        loss = (actor_loss + self.arglist.critic_loss_weightage * critic_loss)
                        self.optimizer.zero_grad()
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arglist.clip_term)
                        self.optimizer.step()
                    else:
                        self.critic_optimizer.zero_grad()
                        critic_loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.arglist.clip_term)
                        self.critic_optimizer.step()

                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.arglist.clip_term)
                        self.actor_optimizer.step()

                    log_probs, entropies, values, R = [], [], [], []

                if done :
                    writer.add_scalar('ep_r', ep_r, episode)
                    if episode % self.arglist.eval_every == 0 or episode == self.arglist.episodes-1:
                        eval_ep_r_list = self.eval(self.arglist.eval_over)
                        writer.add_scalar('eval_ep_r', np.mean(eval_ep_r_list), episode)
                        self.save_checkpoint(str(episode)+".ckpt")
                    break

    def eval(self, episodes):
        ep_r_list = []
        for episode in range(episodes):
            o, _, _ = process_observation(self.env.reset())
            ep_r = 0
            while True:
                with torch.no_grad():
                    if self.arglist.joint_actor_critic:
                        dist, values_ = self.model(torch.tensor(o, dtype=torch.float, device=self.device).unsqueeze(0))
                    else:
                        dist = self.actor(torch.tensor(o, dtype=torch.float, device=self.device).unsqueeze(0))
                a = dist.sample().detach().cpu().numpy()[0]   
                o_1, r, done = process_observation(self.env.step(a))
                ep_r += r
                o = o_1
                if done:
                    ep_r_list.append(ep_r)
                    break
        return ep_r_list  

def parse_args():
    parser = argparse.ArgumentParser("Advantage Actor Critic")
    parser.add_argument("--exp-name", type=str, default="expt_cartpole_swingup_sep_64_n_20_mean", help="name of experiment")
    parser.add_argument("--episodes", type=int, default=5000, help="number of episodes")
    # Core training parameters
    parser.add_argument("--joint-actor-critic", action="store_true", default=False, help="joint / separate actor critic")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="learning rate")
    parser.add_argument("--clip-term", type=float, default=0.5, help="clip grad norm")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='lambda parameter for GAE')
    parser.add_argument('--entropy-weightage', type=float, default=0.01,help='entropy term coefficient')
    parser.add_argument('--critic-loss-weightage', type=float, default=0.5, help='value loss coefficient')
    parser.add_argument("--update-every", type=int, default=20, help="train after every _ steps")
    parser.add_argument("--eval-every", type=int, default=500, help="eval every _ episodes")
    parser.add_argument("--eval-over", type=int, default=100, help="eval over _ episodes")
    return parser.parse_args()

def make_env():
    # env = suite.load(domain_name="reacher", task_name="hard")
    # env.state_size = 6
    # env.action_size = 2

    env = suite.load(domain_name="cartpole", task_name="swingup")
    env.state_size = 5
    env.action_size = 1

    return env

if __name__ == '__main__':

    arglist = parse_args()
    ac = advantage_actor_critic(arglist)
    ac.train()