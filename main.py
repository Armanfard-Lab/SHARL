
######################### IMPORTS #########################
from google.colab import drive
drive.mount('/content/drive')
import os
import torch
import scipy.io
os.chdir('/content/drive/MyDrive/the_matlab_datasets/SBU')
import numpy as np

import pprint, pickle
import numpy as np
from numpy.random import default_rng
import copy
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Bernoulli
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from baseline import Model
from data_gen import da_g
from agent import SHRL
from data_proc import data_proc

######################### Parameters #########################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_layers = 3
hidden_size = 256
num_classes = 8
n_batches = 10
num_epochs = 4000
num_joints = 30
num_frames = 45
num_dimensions = 3
dropout_rate = 0.5
input_size = num_joints * num_dimensions

######################### Baseline model #########################

model = Model.bi_LSTM(input_size, hidden_size, num_layers, num_classes,dropout_rate).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()
fold_num = 5
for fold in tqdm(range(fold_num)):
  Data = da_g.data(fold)
  Data.train_test()
  train_data = Data.train_data
  train_labels = Data.train_labels
  test_data = Data.test_data
  test_labels = Data.test_labels
  print(np.shape(train_data))
  print(np.shape(train_labels))
  print(np.shape(test_data))
  print(np.shape(test_labels))
  train_data = torch.from_numpy(train_data)
  train_labels = torch.from_numpy(train_labels)
  test_data = torch.from_numpy(test_data)
  test_labels = torch.from_numpy(test_labels)
  indexes = torch.randperm(np.shape(train_data)[0])
  train_data = train_data[indexes,:,:]
  train_labels = train_labels[indexes]
  model.train()
  for epoch in range(4000):
    for i in range(int(np.ceil(train_data.shape[0] / n_batches))):
        local_X, local_y = train_data[i * n_batches:(i + 1) * n_batches, ], train_labels[i * n_batches:(i + 1) * n_batches, ]
        local_X = local_X.to(device)
        local_y = local_y.to(device).long()
        scores = model(local_X)
        loss = criterion(scores, local_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


  # checkpoint = torch.load('model_f.pt')
  # model.load_state_dict(checkpoint['model_state_dict'])
  # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  # epoch = checkpoint['epoch']
  # loss = checkpoint['loss']
  def check_accuracy(model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        scores = model(train_data.to(device))
        _, predictions = scores.max(1)
        num_correct += (predictions == train_labels.to(device)).sum()
        num_samples += predictions.size(0)
        # print(f'Got {num_correct} / {num_samples} with accuracy  \
              # {float(num_correct) / float(num_samples) * 100:.2f}')
  check_accuracy(model)
  spatial_net = SHRL.SpatialNet(6, hidden_size, num_layers, num_joints).to(device).float()
  optimizer1 = optim.Adam(spatial_net.parameters(), lr=0.001)

  def exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay_epoch):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer



  def spatial_select_action(state, J , D_f):
    state = state.type(torch.FloatTensor).to(device)
    p = spatial_net(Variable(torch.cat([state,D_f.view([1,D_f.shape[0],D_f.shape[1]]).to(device)],dim=2)))
    m = Bernoulli(p)
    c = m.sample()
    P = p.clone()
    P[c == 0]=1-p[c == 0]
    log_pr = torch.log(P).to(device)
    c = c.to(device)
    LP_SUM = log_pr.sum().to(device).view(1)
    c = c.to('cpu').data.numpy().astype(int)
    c = np.reshape(c, [J.shape[0]])
    new_j = J.clone()
    new_j[np.where(c == 1)] = 1 - new_j[np.where(c == 1)]
    return c, new_j, LP_SUM,p



  learning_rate = 0.001
  N = 30
  spatial_net = SHRL.SpatialNet(6, hidden_size, num_layers, N).to(device).float()
  optimizer1 = optim.Adam(spatial_net.parameters(), lr=learning_rate)


  def spatial_update_policy(re):
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in spatial_net.reward_episode[::-1]:
        R = r + 0 * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    Variable(rewards).mul(-1).to(device).get_device()
    loss1 = (torch.sum(torch.mul(spatial_net.policy_history.to(device), Variable(rewards).mul(-1).to(device)))).to(
        device)

    loss = loss1.clone() -(0.009*(re))+(0.01*(re-(int(np.ceil(30*0.5)))))
    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()

    # Save and intialize episode history counters
    spatial_net.loss_history.append(loss.item())
    spatial_net.reward_history.append(np.sum(spatial_net.reward_episode))
    spatial_net.policy_history = Variable(torch.Tensor())
    spatial_net.reward_episode = []


  def calculate_reward(Probs, Probs_history, true_class, n, ind, alpha, betha):
    ## Probs is the outcome of softmax layer from classifier CNN # Probs : N_classes * 1
    ## Probs_history i the output of previous iteration
    ## true_class is an integer from [1-10]
    ## iteration is the number of iterations passed from the beginning
    omega = 10  # a measure of how strong are the punishments and stimulations
    predicted_class = np.argmax(Probs)
    # print("predicted_class:", predicted_class)
    # print("true_class",true_class.shape)
    prev_predicted_class = np.argmax(Probs_history)

    if (predicted_class == true_class and not (prev_predicted_class == true_class)):
        reward1 = omega  ## stimulation
        # print('yes')
    elif (not (predicted_class == true_class) and (prev_predicted_class == true_class)):
        reward1 = - omega  ## punishment
        # print('No')
    elif (not (predicted_class == true_class) and not (prev_predicted_class == true_class)):
        reward1 = np.sign(Probs[true_class] - Probs_history[true_class])
        # print('sno')
    else:
        reward1 = np.sign(Probs[true_class] - Probs_history[true_class])
    reward2 = -np.abs(n - np.sum(ind))
    reward = (alpha * reward1) + (betha * reward2)

    return reward, predicted_class

  def main(x_f, labelss, N, T):
    device = 'cuda'
    state_pool = []
    action_pool = []
    reward_pool = []
    k3 = 1
    k1 = 4
    all_rewards = []
    all_rewards1 = []
    all_rewards2 = []
    count = 0
    count_up = 0
    new_x = []
    L=[]
    # all_pratio = []
    for p in range(25):
        for video in range(x_f.shape[0]):
            count += 1
            count_up +=1
            ind = video
            cur_f = x_f[ind]
            Diff = (data_proc.diff_cal(cur_f)*10)
            ll = labelss[ind]
            original_label = labelss.data.numpy()[ind]
            # print("original_label", original_label)
            x = cur_f.float().clone()
            # state=torch.cat((x.float(),torch.from_numpy(np.random.randint(2,size=[44,1])).float()), 1)
            state = x.float()
            or_state = state.clone()
            selected_frames = np.ones(x_f.shape[1])
            J = (torch.ones([x.shape[0], N]))
            d_joints = torch.Tensor(data_proc.three_d(x))
            Diff_d = torch.Tensor(data_proc.three_d(Diff))
            x = d_joints.clone()
            # print(np.shape(x))
            all_rewards_episode = []
            for i in range(k3):
                LOGP = []
                rewards1 = []
                LP = 0
                for j in range(k1):
                    x_main = d_joints.clone()
                    count1 = 0
                    A = 0
                    for frames in range(T):
                        if selected_frames[frames] == 1:
                            count1 += 1
                            # print(np.shape(x[frames].reshape([1, x.shape[1], x.shape[2]])))
                            # act,log_pr,new_state_J= spatial_net.take_step(x[frames].reshape([1,3,N]),J[frames])
                            D_f = Diff_d[frames]
                            action2, new_J, lp_sum,pr = spatial_select_action(
                                x[frames].reshape([1, x.shape[1], x.shape[2]]), J[frames],D_f)
                            A+=pr.mean()
                            # print('action:',action)
                            J[frames] = new_J
                            LP = LP + lp_sum
                            non_ind = np.where(J[frames] == 0)
                            x_main[frames, non_ind] = 0
                            x[frames] = x_main[frames]

                    if spatial_net.policy_history.dim() != 0:
                        spatial_net.policy_history = torch.cat(
                            [spatial_net.policy_history.to(device), LP / count1]) ### we can have frame level prediction
                    else:
                        spatial_net.policy_history = (LP / count1)
                    x_sf = x.clone()
                    x_tf = x_sf.clone()
                    # print(np.shape(x_tf))
                    # x_tf=x_sf.reshape([1,x_tf.shape[0],x_tf.shape[1]*x_tf.shape[2]])
                    x_tf = x_sf.reshape([1, x_tf.shape[0], x_tf.shape[1] * x_tf.shape[2]])
                    x_sf[np.where(selected_frames == 0)] = 0
                    x_sf = x_sf.reshape([1, x_sf.shape[0], x_sf.shape[1] * x_sf.shape[2]])
                    model.eval()
                    with torch.no_grad():
                        scores1 = model(x_sf.to(device))
                    prob1 = list(scores1.to('cpu').numpy())
                    prediction1 = np.argmax(prob1, axis=1)
                    prob1 = prob1[0]
                    if i == 0:
                        reward1 = 1 if prediction1 == original_label else -1
                    else:
                        reward1, p1 = calculate_reward(prob1, Probs_history1, original_label, 10, 10, 1,0)
                    Probs_history1 = prob1
                    rewards1.append(reward1)
                    spatial_net.reward_episode.append(reward1)
                # print(np.shape(J))
                # print('joint reduction:', (np.sum(J.numpy()) / (30 * 44)) * 100, '%')
                spatial_update_policy(A/(k1*count1*N))
                torch.cuda.empty_cache()

            state = x.float()
            new_x .append( x_tf.reshape([x_tf.shape[1],x_tf.shape[2]]).numpy())
            L.append(original_label)
            if count_up ==1.5:
                  model.train()
                  count_up =0
                  criterion = nn.CrossEntropyLoss()
                  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                  for epoch in range(200):
                          model.train()
                          # Local batches and labels
                          L = torch.from_numpy(np.array(L)).long()
                          # print(L)
                          # print(new_x)
                          local_X, local_y =torch.from_numpy(np.array(new_x)), L
                          local_X = local_X.to('cuda')
                          # print(local_X.shape)
                          local_y = local_y.to('cuda')
                          # forward
                          scores2 = model(local_X)
                          loss2 = criterion(scores2, local_y)
                          # backward
                          optimizer.zero_grad()
                          loss2.backward()
                          # gradient descent or adam step
                  new_x = []
                  L= []
    # torch.save(spatial_net, 'spatial_net.pt')

    return all_rewards, model, spatial_net


  def test(X_test, y_test, spatial_net, model, T, N):
    spatial_net.eval()
    J = (torch.ones([X_test.shape[1], N]))
    d_joints = torch.Tensor(data_proc.three_d(X_test.reshape([X_test.shape[1], X_test.shape[2]])))
    # print(X_test.shape)
    # print(d_joints.shape)
    Diff = (data_proc.diff_cal((X_test.reshape([X_test.shape[1], X_test.shape[2]])))*10)
    # print(Diff.shape)
    Diff_d = torch.Tensor(data_proc.three_d(Diff))
    x = d_joints.clone()
    with torch.no_grad():
        for frames in range(T):
            D_f = Diff_d[frames]
            state = x[frames].reshape([1, x.shape[1], x.shape[2]]).type(torch.FloatTensor).to(device)
            pr = spatial_net(Variable(torch.cat([state,D_f.view([1,D_f.shape[0],D_f.shape[1]]).to(device)],dim=2)))
            m = Bernoulli(pr)
            action = m.sample()
            action = action.to('cpu').data.numpy().astype(int)
            action = np.reshape(action, [J[frames].shape[0]])
            new_j = J[frames].clone()
            new_j[np.where(action == 1)] = 1 - new_j[np.where(action == 1)]
            J[frames] = new_j
            non_ind = np.where(J[frames] == 0)
            # print('removed joints:', non_ind)
            x[frames, non_ind] = 0
    x_sf = x.clone()
    x_tf = x_sf.reshape([1, x_sf.shape[0], x_sf.shape[1] * x_sf.shape[2]])
    x_tf = x_tf.float()
    model.eval()
    with torch.no_grad():
        scores = model(x_tf.to(device))
    _, prediction = scores.max(1)
    if prediction == y_test.to(device):
        acc = 1
    else:
        acc = 0

    model.train()
    spatial_net.train()
    # temporal_net.train()
    return acc


  # test_labels = torch.from_numpy(test_labels)
  all_rewards, model, spatial_net= main(train_data.cpu(), train_labels, N, 45)

  spatial_net.eval()
  # temporal_net.eval()
  xxx = train_data.clone()
  for y in range(train_data.shape[0]):

    xx = train_data[y].float()
    J = (torch.ones([xx.shape[0], N]))
    d_joints = torch.Tensor(data_proc.three_d(xx.reshape([train_data.shape[1], train_data.shape[2]])))
    Diff = (data_proc.diff_cal(xx)*10)
    Diff_d = torch.Tensor(data_proc.three_d(Diff))
    x = d_joints.clone()
    with torch.no_grad():
        for frames in range(45):
            D_f = Diff_d[frames]
            state = x[frames].reshape([1, x.shape[1], x.shape[2]]).type(torch.FloatTensor).to(device)
            pr = spatial_net(Variable(torch.cat([state,D_f.view([1,D_f.shape[0],D_f.shape[1]]).to(device)],dim=2)))
            m = Bernoulli(pr)
            action = m.sample()
            action = action.to('cpu').data.numpy().astype(int)
            # print(action.shape)
            action = np.reshape(action, [J[frames].shape[0]])
            new_j = J[frames].clone()
            new_j[np.where(action == 1)] = 1 - new_j[np.where(action == 1)]
            J[frames] = new_j
            non_ind = np.where(J[frames] == 0)
            x[frames, non_ind] = 0
    x_sf = x.clone()
    x_tf = x_sf.reshape([1, x_sf.shape[0], x_sf.shape[1] * x_sf.shape[2]])
    x_tf = x_tf.float()

    xxx[y] = x_tf

  model = Model.bi_LSTM(input_size, hidden_size, num_layers, num_classes,dropout_rate).to(device).float()

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.0001)
  n_batches=10
  # Train Network
  for epoch in range(4000):
    for i in range(int(np.ceil(xxx.shape[0] / n_batches))):
        # Local batches and labels
        local_X, local_y = xxx[i * n_batches:(i + 1) * n_batches, ], train_labels[
                                                                     i * n_batches:(i + 1) * n_batches, ]
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_X = local_X.to(device)
        local_y = local_y.to(device).long()
        scores = model(local_X)
        loss = criterion(scores, local_y)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

  check_accuracy(model)

  ACC = 0
  for i in range(test_data.shape[0]):
    acc = test(test_data[i].reshape([1, test_data.shape[1], test_data.shape[2]]).float(), test_labels[i], spatial_net,
                model, 45, N)
    ACC = ACC + acc
  print('test accuracy:', (ACC / test_data.shape[0]) * 100, '%')