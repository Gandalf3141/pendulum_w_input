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

class NeuralNetwork(nn.Module): 

    def __init__(self, input_dim=1, output_dim=1, num_neurons=6, layers=5):
        super().__init__()
        
        layers = [nn.Linear(input_dim, num_neurons), nn.Tanh()] + list(chain.from_iterable(zip([nn.Linear(num_neurons, num_neurons, bias=True) for i in range(layers)], 
                                              [nn.Tanh() for i in range(layers)]))) + [nn.Linear(num_neurons, output_dim)]
        self.linear_relu_stack = nn.Sequential(*layers)
    
    def forward(self, x):
        # x = self.flatten(x)
        
        return  self.linear_relu_stack(x)  
    
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

def slice_batch(batch, window_size=1):
    l = []
    for i in range(len(batch)- window_size):

     l.append((batch[i:i+window_size,:], batch[i+1:i+window_size+1, 1: ]))  # hier ist u(t) nicht beim label dabei
     #l.append((batch[i:i+window_size,:], batch[i+window_size:i+window_size+1, 1: ]))  # hier ist u(t) nicht beim label dabei

    return l    

def train(input_data, model, ws=1, odestep=False, use_autograd=False):

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    model.train()

    total_loss = []

    for batch in input_data:

        input = slice_batch(batch, ws)
        
        batch_loss = 0

        for inp, label in input: # inp = (u, x) label = x
            
            #inp.to(device)
            #label.to(device)

            output, _ = model(inp)

            out = output[-1]
            out = out.view(1,out.size(dim=0))#out.view(1, out.size(dim=0))
           
            if odestep:
             #out = inp[-1, 1:] + out # u wird nicht geändert!
             out = inp[:, 1:] + output
            
            
            if use_autograd:
                print("not implemented yet")
               
            optimizer.zero_grad()
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()

            batch_loss += loss.detach().numpy()

        total_loss.append(batch_loss) #loss over all initial conditions

    return np.mean(total_loss)

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
                plt.plot(time, pred_next_step.detach().numpy()[:,1], color="green", label="next step from data")
                plt.plot(time, x.detach().numpy()[:,1], color="blue", label="true", linestyle="dashed")
                #plt.plot(pred.detach().numpy()[:,2])
                
            plt.grid()
            plt.legend()
            plt.show()

    return np.mean(test_loss), np.mean(test_loss_deriv)

def main():

    window_size = 6 #4 besser als 2?
    start_time = 0
    stop_time = 30
    timesteps = 200
    num_of_inits = 400
    option_odestep = True
    losses=[]

    input_data, test_data, time, initial_values, input_data_w_time = get_data(x0 = np.pi/4, y0 = 0.1, use_fixed_init = False, t0=start_time, t1=stop_time, 
                                                                                   time_steps=timesteps, num_of_inits=num_of_inits,
                                                                                     normalize=False, add_noise=False, u_option="random_walk",  set_seed=True)

    
    train_size = int(0.9 * len(input_data))
    test_size = len(input_data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(input_data, [train_size, test_size])
    #train_dataset = input_data

    slice_of_data = 30
    train_dataset = train_dataset[:][:,0:slice_of_data, :]

    model = LSTMmodel(input_size=3,hidden_size=5, out_size=2, layers=1).to(device)

    epochs = 100

    ic(epochs, window_size, start_time, stop_time, timesteps, num_of_inits, option_odestep)
 
    for e in range(epochs): 

        loss_epoch = train(train_dataset, model, ws=window_size, odestep=option_odestep, use_autograd=False)
        losses.append(loss_epoch)
        if e%2 == 0:
          ic(loss_epoch, e)

    plt.plot(losses[1:])
    plt.show()
    torch.save(model.state_dict(), "lstm_derivative_with_input3.pth")
    ic(test(test_dataset, time, model, plot_opt=True, ws=window_size, steps=timesteps, odestep=option_odestep))

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