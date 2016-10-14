import matplotlib.pyplot as plt
import numpy as np
import os


save_path = os.path.join('results', 'real_data')
npzfile = np.load(os.path.join(save_path, 'mnist.dat.npz'))
losses = npzfile['losses']
# plot
fig = plt.figure(num=1, figsize=(20, 12))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('iterations', fontsize=20)
plt.ylabel('loss', fontsize=20)
plt.plot(losses[0], lw=2, label='serial, gamma=0.5')
plt.plot(losses[1], lw=2, label='parallel random, gamma=1')
plt.plot(losses[2], lw=2, label='parallel correlation, gamma=1')
plt.tight_layout()
plt.legend()
# plt.show()
plt.savefig(os.path.join(save_path, 'mnist.pdf'))