import matplotlib.pyplot as plt
import numpy as np
def CyclicLR(num_itr,base_lr,max_lr,step_size):
  lrate =[]
  for itr in range(num_itr):
    cycle = np.floor(1+itr/(2*step_size))
    x = np.absolute(itr/step_size - 2*cycle + 1)
    lr = base_lr + (max_lr-base_lr)*(1-x)
    lrate.append(lr)  
  plt.xlabel('Training Iterations')
  plt.ylabel('Learning Rate')
  plt.plot(range(num_itr),lrate)