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
 


def test(test_data, time, model, plot_opt=False, ws=1, steps=200, odestep=False, use_autograd=False):

    model.eval() 

    loss_fn = nn.MSELoss()

    test_loss = 0 
    test_loss_deriv = 0

    for x in test_data: 
        with torch.inference_mode():
            #pred = torch.ones_like(x)


            pred = torch.zeros((steps,3))
            pred_next_step = torch.zeros((steps,3))
            #ic(pred)
            #ic(x)
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
                #out = model(pred[i:i+ws,:])
               
                if odestep:
                    
                          # x[k+1] = x[k]              + NN(x[k])
                    pred[i+ws, 1:] = pred[i+ws-1 , 1:] + out[-1, :]

                    pred_next_step[i+ws, 1:] = x[i+ws-1 , 1:] + out[-1, :]

                else:
                 #print("error you shouldnt have come here") 
                 pred[i+ws, 1:] = out[-1]
                 outt , _  = model(x[i:i+ws,:])
                 pred_next_step[i+ws, 1:] = outt[-1]

                if use_autograd:
                  print("not implemented yet")


            test_loss += loss_fn(pred[:,1], x[:,1]).detach().numpy()
            test_loss_deriv += loss_fn(pred[:,2], x[:,2]).detach().numpy()

            if plot_opt:
                #plt.plot(pred.detach().numpy()[:,0])
                plt.plot(time, pred.detach().numpy()[:,1], color="red", label="pred")
                #plt.plot(time, pred_next_step.detach().numpy()[:,1], color="green", label="next step from data")
                plt.plot(time, x.detach().numpy()[:,1], color="blue", label="true", linestyle="dashed")
                #plt.plot(pred.detach().numpy()[:,2])
                
            plt.grid()
            plt.legend()
            plt.show()

    return np.mean(test_loss), np.mean(test_loss_deriv)

def main():

    window_size = 1
    start_time = 0
    stop_time = 90
    timesteps = 600
    num_of_inits = 10
    option_odestep = True
    losses=[]

    input_data, test_data, time, initial_values, input_data_w_time = get_data(x0 = np.pi/4, y0 = 0.1, use_fixed_init = False, t0=start_time, t1=stop_time, 
                                                                                   time_steps=timesteps, num_of_inits=num_of_inits,
                                                                                     normalize=False, add_noise=False, u_option="random_walk",  set_seed=False)

    
    train_size = 1
    test_size = len(input_data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(input_data, [train_size, test_size])


    model = LSTMmodel(input_size=3,hidden_size=5, out_size=2, layers=1).to(device)
    model.load_state_dict(torch.load("lstm_derivative_with_input3.pth"))

    
    # model2 = LSTMmodel(input_size=3,hidden_size=5, out_size=2, layers=1).to(device)
    # model2.load_state_dict(torch.load("lstm_derivative_with_input.pth"))

    ic(test(test_dataset, time, model, plot_opt=True, ws=window_size, steps=timesteps, odestep=option_odestep)) 
     # ,test(test_dataset, time, model2, plot_opt=True, ws=window_size, steps=timesteps, odestep=option_odestep) )

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