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
from get_data import  get_data

def plot_data(data, time, plot_derive = 0, plot_all=True):

    greek_letterz=[chr(code) for code in range(945,970)] #7 theta 24 omega
    
    if not plot_all:
     for i in range(data.size(dim=0)):
        plt.plot(time, data[i,:,plot_derive].detach().numpy())#, label="example_training_data")
     
    #  plt.title("random walk input signal") 
    #  plt.xlabel('time [t]')
    #  plt.ylabel('u(t)')

     plt.tight_layout()
     plt.grid()
     plt.legend()
     plt.show()

    fig, axs = plt.subplots(3, 1, figsize=(16, 10))
     
    if plot_all: 
        for i in range(data.size(dim=0)):
            # Plot the first function
            axs[0].plot(time, data[i,:,0], color="green", label="pred")
            axs[0].set_xlabel('time [t]')
            axs[0].set_ylabel('u(t)')
            axs[0].set_title('damped pendulum with input')
            axs[0].legend()
            axs[0].grid()

            # Plot the second function
            axs[1].plot(time, data[i,:,1], color="red", label=f'{greek_letterz[7]}(t)')
            axs[1].set_xlabel('time')
            axs[1].set_ylabel(f'{greek_letterz[7]}(t)')
            axs[1].set_title('angle')
            axs[1].legend()
            axs[1].grid()

            axs[2].plot(time,  data[i,:,2], color="blue", label=f'{greek_letterz[24]}(t)')
            axs[2].set_xlabel('time')
            axs[2].set_ylabel(f'{greek_letterz[24]}(t)')
            axs[2].set_title('angular velocity')
            axs[2].legend()
            axs[3].grid()
            
            plt.tight_layout()
            plt.grid()
            plt.legend()
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
t_end = 60
time_steps = 200
num_of_inits = 1

t_end_batch = 100
batch_size = 3

input_data, test_data, time, initial_values, input_data_w_time = get_data(x0 = np.pi/4, y0 = 0.1, use_fixed_init = False, t0=t_start, t1=t_end, 
                                                                                   time_steps=time_steps, num_of_inits=num_of_inits, normalize=False,
                                                                                     add_noise=False, u_option="random_walk",  set_seed=True)


plot_data(input_data, time, plot_derive = 0, plot_all=True)


#plot_batch(batch_y0, batch_t, batch_y)
#plot_batch_rnn(data, batch_y0_rnn, batch_t_rnn, batch_y_rnn, t_end_batch, t_start, time_steps)
