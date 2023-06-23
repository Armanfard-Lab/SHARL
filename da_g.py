import numpy as np
class data():
  def __init__(self, fold):
    self.fold = fold
    self.train_data = None
    self.test_data = None
    self.train_labels = None
    self.test_labels = None
  def train_test(self):
    all_f = np.array( np.arange(5)+1)
    self.fold = 2
    train_inds = all_f[np.where(all_f!= self.fold+1)]
    F = []
    F_L = []
    for i in train_inds:
      F.append(np.load('fold'+str(i)+'.npy'))
      F_L.append(np.load('labels_fold'+str(i)+'.npy'))
    self.train_data = np.concatenate([F[0] , F[1] , F[2] , F[3]],axis=0)
    train_labels = np.concatenate([F_L[0],F_L[1],F_L[2],F_L[3]],axis = 0)
    self.test_data = np.load('fold'+str(self.fold+1)+'.npy')
    test_labels = np.load('labels_fold'+str(self.fold+1)+'.npy')
    self.train_labels = train_labels - 1
    self.test_labels = test_labels - 1