#packages
from matplotlib import legend
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import os
import numpy as np
#from torchdyn.numerics import odeint
from torchdiffeq import odeint
from icecream import ic
from tqdm import tqdm
import scipy
from itertools import chain
import cProfile
import pstats
from get_data import get_batch, generate_data, get_batch_RNN

def plot_data(data, time, plot_derive = 0):
    for i in range(data.size(dim=0)):
        plt.plot(time, data[i,:,plot_derive].detach().numpy())#, label="example_training_data")

    plt.grid()
    plt.xlabel("time")
    plt.ylabel("angle")

    plt.title('Samples of the training data with noise')
    #plt.legend()
    plt.show()

def plot_batch(batch_y0, batch_t, batch_y):
    for i in range(batch_y.size(dim=1)):
        plt.plot(batch_t, batch_y[:,i,0].detach().numpy(), label="batch for one trajectory")
    plt.grid()
    plt.legend()
    
    plt.show()

def plot_batch_rnn(data, batch_y0_rnn, batch_t_rnn, batch_y_rnn, t_end_batch, t_start, time_steps):
   # plt.plot(time, data[0,:,0].detach().numpy(), label="example_training_data", color="grey")
    for i in range(batch_y_rnn.size(dim=1)):
        plt.scatter(time[i*t_end_batch : (i+1)*t_end_batch], batch_y_rnn[:,i,0].detach().numpy(), label=f"{i} batch_rnn for one trajectory", marker="o", s=25)
    plt.grid()
    plt.plot(time, data[0,:,0].detach().numpy(), label="example_training_data", color="grey")
    plt.legend()
    plt.show()

t_start = 0
t_end = 30
time_steps = 200
num_of_inits = 10

t_end_batch = 100
batch_size = 20

data, time, initial_values = generate_data(x0 = np.pi/4, y0 = 0.1, use_fixed_init = False, t0=t_start, t1=t_end, 
                                           time_steps=time_steps, num_of_inits=num_of_inits, normalize=True, add_noise=False, u_option="sin")



#batch_y0, batch_t, batch_y = get_batch(data[0], time, t_end_batch=t_end_batch , batch_size=batch_size)
#batch_y0_rnn, batch_t_rnn, batch_y_rnn = get_batch_RNN(data[0], time, t_end_batch=t_end_batch , batch_size=batch_size)

plot_data(data, time, plot_derive = 0)
#plot_batch(batch_y0, batch_t, batch_y)
#plot_batch_rnn(data, batch_y0_rnn, batch_t_rnn, batch_y_rnn, t_end_batch, t_start, time_steps)
