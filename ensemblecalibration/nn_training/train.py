import torch
import numpy as np
import matplotlib.pyplot as plt

from ensemblecalibration.nn_training.distances import tv_distance
from ensemblecalibration.nn_training.model import MLPCalW
from ensemblecalibration.nn_training.losses import SKCEuqLoss
from ensemblecalibration.calibration.t1_t2_analysis import experiment_h0, experiment_h1



# define variables
N_ENSEMBLE = 10
N_EPOCHS = 100
LR = 0.0001

N=1000
M=10
K=10
U=0.01


model = MLPCalW(in_channels=10, hidden_dim=32)
loss_model = SKCEuqLoss(dist_fct=tv_distance)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
P, y = experiment_h1(N,M,K,U, random=True)
P_2, y_2 = experiment_h1(N,M,K,U, random=True)
P = torch.from_numpy(P).float()
y = torch.from_numpy(y)
P_2 = torch.from_numpy(P_2).float()
y_2 = torch.from_numpy(y_2)

loss_per_epoch_train = np.zeros(N_EPOCHS)
loss_per_epoch_val = np.zeros(N_EPOCHS)
for n in range(N_EPOCHS):
    weights_l = model(P)
    loss = loss_model(P, weights_l, y)
    print(f'epoch: {n} loss: {loss.item()}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_per_epoch_train[n] = loss.item()

    weights_val = model(P_2)
    loss_val = loss_model(P_2, weights_val, y_2)
    loss_per_epoch_val[n] = loss_val.item()

fig, ax = plt.subplots()
x = np.arange(0, N_EPOCHS, 1)
ax.plot(x, loss_per_epoch_train, label='train')
ax.plot(x, loss_per_epoch_val, label='val')
ax.set_xscale("log")
ax.set_xlabel("epochs")
ax.set_ylabel("loss: SKCE_ul")
min_loss = np.min(loss_per_epoch_train)
plt.title(f"Loss per epoch, uncalibrated setting, MLP model \n yielding weight matrix as output, \n minimum loss: {min_loss}")
plt.legend()
plt.show()




