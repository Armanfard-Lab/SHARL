import torch
import numpy as np
def diff_cal(X):
  diff=torch.zeros([X.shape[0],X.shape[1]])
  # print(diff)
  for i in range(X.shape[0]-1):
    diff[i+1]=X[i+1]-X[i]
  return diff


def three_d(data):
    d_data=[]
    for k in range(data.shape[0]):
        x_i=data[k]
        f_x=[]
        f_y=[]
        f_z=[]
        for i in range((data.shape[1])):
            if (i+1) % 3 == 1:
                f_x.append(x_i[i])
            elif (i+1) % 3 == 2:
                f_y.append(x_i[i])
            else:
                f_z.append(x_i[i])
        x = np.vstack((f_x, f_y, f_z))
        d_data.append(x.T)
    d_data = torch.from_numpy(np.array(d_data))
    return d_data