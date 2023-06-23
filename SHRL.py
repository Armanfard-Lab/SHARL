import torch
import torch.nn as nn
from torch.distributions import Bernoulli 
from torch.autograd import Variable
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class SpatialNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_joints):
        super(SpatialNet, self).__init__()
        self.hidden_size = hidden_size
        self.n_joints = n_joints
        # Episode policy and reward history
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=True)

        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.act1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(256, 30)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = x.float()
        h0 = torch.randn(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.randn(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        out, (h, c) = self.lstm(x, (h0, c0))
        out = self.fc1(torch.cat((out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]), 1))
        out = self.act1(out)
        out = self.fc2(out)
        out = self.act2(out)

        return out.to(device)

def spatial_select_action(spatial_net,state, J , D_f):
    state = state.type(torch.FloatTensor).to(device)
    pr = spatial_net(Variable(torch.cat([state,D_f.view([1,D_f.shape[0],D_f.shape[1]]).to(device)],dim=2)))
    m = Bernoulli(pr)
    c = m.sample()
    PR = pr.clone()
    PR[c == 0]=1-pr[c == 0]
    log_pr = torch.log(PR).to(device)
    c = c.to(device)
    LP_SUM = log_pr.sum().to(device).view(1)
    c = c.to('cpu').data.numpy().astype(int)
    c = np.reshape(c, [J.shape[0]])
    new_j = J.clone()
    new_j[np.where(c == 1)] = 1 - new_j[np.where(c == 1)] 
    return c, new_j, LP_SUM,pr

def spatial_update_policy(spatial_net,optimizer1,Re):

    rewards = torch.FloatTensor(spatial_net.reward_episode)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    loss1 = (torch.sum( Variable(rewards).mul(-1).to(device))).to(device)
    Loss2 = Re
    loss = loss1.clone() -(0.009*(Loss2))+(0.01*(Loss2-(int(np.ceil(30*0.5)))))
    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()
    spatial_net.loss_history.append(loss.item())
    spatial_net.reward_history.append(np.sum(spatial_net.reward_episode))
    spatial_net.policy_history = Variable(torch.Tensor())
    spatial_net.reward_episode = []

def spatial_update_policy(spatial_net,optimizer1,Re):
    R = 0
    rewards = []
    for r in spatial_net.reward_episode[::-1]:
        R = r + 0 * R
        rewards.insert(0, R)
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    Variable(rewards).mul(-1).to(device).get_device()
    loss1 = (torch.sum(torch.mul(spatial_net.policy_history.to(device), Variable(rewards).mul(-1).to(device)))).to(
        device)
    Loss2 = Re
    loss = loss1.clone() -(0.009*(Loss2))+(0.01*(Loss2-(int(np.ceil(30*0.5)))))
    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()

    # Save and intialize episode history counters
    spatial_net.loss_history.append(loss.item())
    # print('loss:', loss.item())
    spatial_net.reward_history.append(np.sum(spatial_net.reward_episode))
    # print('reward:', np.sum(spatial_net.reward_episode))
    spatial_net.policy_history = Variable(torch.Tensor())
    spatial_net.reward_episode = []

    
def calculate_reward(Probs, Probs_history, true_class):
    omega = 10  # a measure of how strong are the punishments and stimulations
    predicted_class = np.argmax(Probs)
    prev_predicted_class = np.argmax(Probs_history)

    if (predicted_class == true_class and not (prev_predicted_class == true_class)):
        reward1 = omega  ## stimulation
    elif (not (predicted_class == true_class) and (prev_predicted_class == true_class)):
        reward1 = - omega  ## punishment
    elif (not (predicted_class == true_class) and not (prev_predicted_class == true_class)):
        reward1 = np.sign(Probs[true_class] - Probs_history[true_class])
    else:
        reward1 = np.sign(Probs[true_class] - Probs_history[true_class])

    reward = reward1

    return reward, predicted_class