import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import gym

# Fully Connected Joint Actor Critic network
class Pi_V_FC(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(Pi_V_FC, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.pi = torch.nn.Linear(64, action_size)
        self.v = torch.nn.Linear(64, 1)
        self.distribution = torch.distributions.Categorical        
    def forward(self, x):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        probs = F.softmax(self.pi(y2), dim=1)
        values = self.v(y2).view(-1)
        return probs, values

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
        self.model = Pi_V_FC(self.env.state_size,self.env.action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.arglist.lr)

    def save_checkpoint(self, name):
        checkpoint = {'model' : self.model.state_dict()}
        torch.save(checkpoint, os.path.join(self.model_dir, name))

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])

    def train(self):
        writer = SummaryWriter(log_dir=self.tensorboard_dir)
        new_batch = True  
        for episode in range(self.arglist.episodes):
            t = 0
            ep_r = 0.0
            o = self.env.reset()
            while True:
                probs, values_ = self.model(torch.tensor(o, dtype=torch.float, device=self.device).unsqueeze(0))
                m = self.model.distribution(probs)
                index = m.sample()                 
                if new_batch:
                    log_probs = m.log_prob(index)
                    entrops = m.entropy()
                    values = values_
                    R = []
                    new_batch = False                    
                else:
                    log_probs = torch.cat((log_probs,m.log_prob(index)),dim=0)
                    entrops = torch.cat((entrops,m.entropy()),dim=0)
                    values = torch.cat((values,values_),dim=0)
                a = index[0].item()
                o_1, r, done, info = self.env.step(a)
                t += 1
                ep_r += r
                R.append(r)
                o = o_1
                if (t % self.arglist.update_every == 0 or done):
                # if done:
                    if done:
                        v_s_ = 0.0 # terminal
                    else:
                        with torch.set_grad_enabled(False): 
                            _, next_values = self.model(torch.tensor(o_1, dtype=torch.float, device=self.device).unsqueeze(0))
                        v_s_ = next_values[0].item()

                    v_targets = []
                    for r in R[::-1]: # reverse R
                        v_s_ = r + self.arglist.gamma * v_s_
                        v_targets.append(v_s_)
                    v_targets.reverse()
                    v_targets = torch.tensor(v_targets, dtype=torch.float, device=self.device)
                    td = v_targets - values
                    a_loss = - (log_probs * td.detach()).mean()
                    e_loss = - entrops.mean()                                
                    c_loss = td.pow(2).mean()

                    loss = (a_loss + 0.01 * e_loss + 0.5 * c_loss)
                    self.optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arglist.clip_term)
                    self.optimizer.step()                
                    new_batch = True

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
            o = self.env.reset()
            ep_r = 0
            while True:
                with torch.set_grad_enabled(False):
                    probs, values_ = self.model(torch.tensor(o, dtype=torch.float, device=self.device).unsqueeze(0))
                m = self.model.distribution(probs)
                index = m.sample()                 
                a = index[0].item()   
                o_1, r, done, info = self.env.step(a)
                ep_r += r
                o = o_1
                if done:
                    ep_r_list.append(ep_r)
                    break
        return ep_r_list  

def parse_args():
    parser = argparse.ArgumentParser("Advantage Actor Critic")
    parser.add_argument("--exp-name", type=str, default="expt_cartpole", help="name of experiment")
    parser.add_argument("--episodes", type=int, default=25000, help="number of episodes")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=2.5e-4, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    # parser.add_argument("--clip-term", type=float, default=0.5, help="gradient clipping parameter")
    parser.add_argument("--update-every", type=int, default=5, help="train after every _ steps")
    parser.add_argument("--eval-every", type=int, default=1000, help="eval every _ episodes")
    parser.add_argument("--eval-over", type=int, default=100, help="eval over _ episodes")
    return parser.parse_args()

def make_env():
    env = gym.make('CartPole-v1')
    env.state_size = 4
    env.action_size = 2
    # env = gym.make('MountainCar-v0')
    # env.state_size = 2
    # env.action_size = 3
    return env

if __name__ == '__main__':

    arglist = parse_args()
    ac = advantage_actor_critic(arglist)
    ac.train()