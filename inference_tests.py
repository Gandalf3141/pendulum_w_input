#packages
from matplotlib import legend
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import os
import numpy as np
from torchdyn.numerics import odeint
from icecream import ic
from tqdm import tqdm
import scipy
from itertools import chain
import cProfile
import pstats
from get_data import get_data

# Defining the LSTM model with two hidden layers
torch.set_default_dtype(torch.float64)
device="cpu"

    
class LSTMmodel(nn.Module):
    
    def __init__(self,input_size , hidden_size , out_size, layers):
        
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.input_size = input_size
        
        self.lstm = nn.LSTM(input_size,hidden_size, num_layers=layers) #ohne bias ändern 0er in input sequence nichts.
                
        self.linear = nn.Linear(hidden_size, out_size)
        
        
    def forward(self,seq):


        lstm_out , hidden = self.lstm(seq)         

        pred = self.linear(lstm_out.view(len(seq),-1))
        
        return pred, hidden
 


def test(test_data, time, model, plot_opt=False, ws=1, steps=200, odestep=False, use_autograd=False,  plot_derivative=False, error_plot=False):

    model.eval() 
    loss_fn = nn.MSELoss()

    test_loss = 0 
    test_loss_deriv = 0

    for x in test_data: 
        with torch.inference_mode():

            pred = torch.zeros((steps,3))
            pred_next_step = torch.zeros((steps,3))

            if ws>1:
             pred[0:ws, :] = x[0:ws, :]
             pred[:, 0] = x[:, 0]
             pred_next_step[0:ws, :] = x[0:ws, :]
             pred_next_step[:, 0] = x[:, 0]
            else:
             pred[0, :] = x[0,:]
             pred[:, 0] = x[:, 0]
             pred_next_step[0, :] = x[0,:]
             pred_next_step[:, 0] = x[:, 0]
       
            for i in range(len(x)- ws):

                out, _ = model(pred[i:i+ws,:])  
                out2, _ = model(x[i:i+ws,:])  
               
                if odestep:
                    
                          # x[k+1] = x[k]              + NN(x[k])
                    pred[i+ws, 1:] = pred[i+ws-1 , 1:] + out[-1, :]
                    pred_next_step[i+ws, 1:] = x[i+ws-1 , 1:] + out2[-1, :]

                else:
                 pred[i+ws, 1:] = out[-1]
                 outt , _  = model(x[i:i+ws,:])
                 pred_next_step[i+ws, 1:] = outt[-1]

                if use_autograd:
                  print("not implemented yet")


            test_loss += loss_fn(pred[:,1], x[:,1]).detach().numpy()
            test_loss_deriv += loss_fn(pred[:,2], x[:,2]).detach().numpy()

            if plot_opt:

                greek_letterz=[chr(code) for code in range(945,970)] #7 theta 24 omega

                fig, axs = plt.subplots(2, 1, figsize=(16, 9))

                # Plot the first function
                k = 1 if plot_derivative else 0
                axs[0].plot(time, pred.detach().numpy()[:,1+k], color="red", label="pred")
                axs[0].plot(time, pred_next_step.detach().numpy()[:,1+k], color="green", label="next_step_pred", linestyle="dashed")
                axs[0].plot(time, x.detach().numpy()[:,1+k], color="blue", label="true")
                axs[0].set_xlabel('time [t]')
                axs[0].set_ylabel(f"{greek_letterz[7]}(t)")
                axs[0].set_title('damped pendulum with input')
                axs[0].legend()
                axs[0].grid()

                # Plot the second function
                axs[1].plot(time, pred.detach().numpy()[:,0], color="green", label="input")
                axs[1].set_xlabel('time')
                axs[1].set_ylabel('u(t)')
                axs[1].set_title('input signal')
                axs[1].legend()
                plt.tight_layout()
                plt.grid()

                

            if error_plot:

                error = torch.zeros((steps,4))
                cumulative_error = torch.zeros((steps,4))
             
                for j in range(steps):
                    error[j,0] = abs(x[j,1] - pred[j,1])
                    error[j,1] = abs(x[j,2] - pred[j,2])
                    error[j,2] = abs(x[j,1] - pred_next_step[j,1])
                    error[j,3] = abs(x[j,2] - pred_next_step[j,2])
                
                for k in range(4):
                 cumulative_error[:,k] = np.cumsum(error[:,k])

                fig, axs = plt.subplots(3, 1, figsize=(16, 9))

                # Plot the absolute errors
                axs[0].plot(time, error[:,0], color="red", label="abs_error")
                axs[0].plot(time, error[:,2], color="blue", label="abs_error")
                axs[0].set_xlabel('time [t]')
                axs[0].set_ylabel(f"e(t) = {greek_letterz[7]}(t) - {greek_letterz[7]}_pred(t) ")
                axs[0].set_title('absolute error')
                axs[0].legend()
                axs[0].grid()

                axs[1].plot(time, error[:,1], color="red", label="abs_error")
                axs[1].plot(time, error[:,3], color="blue", label="abs_error")
                axs[1].set_xlabel('time')
                axs[1].set_ylabel(f"e(t) = {greek_letterz[24]}(t) - {greek_letterz[24]}_pred(t) ")
                axs[1].set_title('absolute error')
                axs[1].legend()

                #plot the cumulative errors
                axs[2].plot(time, cumulative_error[:,0], color="red", label="angle")
                axs[2].plot(time, cumulative_error[:,1], color="blue", label="angular velocity")
                axs[2].plot(time, cumulative_error[:,2], color="green", label="angle_nextstep")
                axs[2].plot(time, cumulative_error[:,3], color="yellow", label="angular velocity nextstep")

                axs[2].set_xlabel('time')
                axs[2].set_title('cumulative_error')
                axs[2].legend()
                # Adjust layout
                plt.tight_layout()
                plt.grid()
                plt.legend()
               # plt.figure()
                plt.show()



    return np.mean(test_loss), np.mean(test_loss_deriv)

def main():

    window_size = 4
    start_time = 0
    stop_time = 90
    timesteps = 600
    num_of_inits = 2
    option_odestep = True
    losses=[]

    input_data, test_data, time, initial_values, input_data_w_time = get_data(x0 = np.pi/4, y0 = 0.1, use_fixed_init = False, t0=start_time, t1=stop_time, 
                                                                                   time_steps=timesteps, num_of_inits=num_of_inits,
                                                                                     normalize=False, add_noise=False, u_option="tanh",  set_seed=False)


    train_size = 1
    test_size = len(input_data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(input_data, [train_size, test_size])

    model = LSTMmodel(input_size=3,hidden_size=5, out_size=2, layers=1).to(device)
    #model = LSTMmodel(input_size=3,hidden_size=6, out_size=2, layers=2).to(device)

    model.load_state_dict(torch.load("lstm_wsize_4_smaller_net.pth", map_location=torch.device('cpu')))
    #this works good:  window_size = 4 / small model
    #model.load_state_dict(torch.load("lstm_wsize_4_smaller_net.pth", map_location=torch.device('cpu')))

    ic(test(test_dataset, time, model, plot_opt=True, ws=window_size, steps=timesteps, odestep=option_odestep, plot_derivative=False, error_plot=True)) 

    return None


if __name__ == "__main__":
    main()

#These settings worked:
    #
    #input_data, test_data, time, initial_values, input_data_w_time = get_data(x0 = np.pi/4, y0 = 0.1, use_fixed_init = False, t0=start_time, t1=stop_time, 
    #                                                                              time_steps=timesteps, num_of_inits=num_of_inits, normalize=False, add_noise=False, u_option="sin")
    #     window_size = 2
    # start_time = 0
    # stop_time = 30
    # timesteps = 200
    # num_of_inits = 100
    # option_odestep = True