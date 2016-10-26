import matplotlib.pyplot as plt
import numpy as np
import os


save_path = os.path.join('results', 'real_data_5000')
npzfile = np.load(os.path.join(save_path, 'mnist.dat.npz'))
losses = npzfile['losses']
learning_rates = npzfile['learning_rate']
# plot
fig = plt.figure(num=1, figsize=(20, 12))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('iterations', fontsize=20)
plt.ylabel('log(loss)', fontsize=20)
plt.plot(np.log(losses[0]), lw=2, label='serial, gamma=%f' % learning_rates[0])
plt.plot(np.log(losses[1]), lw=2, label='parallel random, gamma=%f' % learning_rates[1])
plt.plot(np.log(losses[2]), lw=2, label='parallel correlation, gamma=%f' % learning_rates[2])
plt.tight_layout()
plt.legend(fontsize=30)
# plt.show()
plt.savefig(os.path.join(save_path, 'mnist.pdf'))