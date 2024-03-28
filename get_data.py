import torch
import numpy as np
import scipy



def input(t, u_option):
  
  #np.random.seed(seed=np.random.randint(1))

  if u_option=="noise":
   return np.random.normal(0, 0.1, len(t))
  
  if u_option=="sin":
   return 0.1 * np.tanh(t/len(t))
  
  if u_option=="rand_sin":
   
   a = np.random.uniform(0.05, 0.2, 1)
   b = np.random.uniform(1, 5, 1)
   c = np.random.uniform(0, 5, 1)

   return a * np.sin( b * t + c)

  if u_option=="random_walk":
   
   steps = np.random.uniform(-0.05, 0.05, len(t))
   steps[0] = np.random.uniform(-0.2, 0.2, 1)
   walk = np.cumsum(steps) 
   walk = 0.3*(walk / np.max(abs(walk)))
   return walk
  
  if u_option=="cos":
   return 0.1 * np.cos(t) 
  
  if u_option=="test":
   return 0.4 * np.cos(t**2) + np.sin(2*t)
  
  if u_option=="none":
   return 0*np.cos(t) 

  else: 
   return 0  

def func(y,t, u):

  f = 1
  
  return np.array([y[1], 1/f*(-np.sin(y[0])- 1/7*y[1] + u )])

def get_data(x0 = np.pi/4, y0 = 0.1, use_fixed_init = False, t0=0, t1=30, time_steps=1000, num_of_inits=1, normalize=True, add_noise=False, u_option="noise", set_seed=True):

    if set_seed:
     np.random.seed(seed=42)

    
    y0 = [x0, y0]
    
    t = np.linspace(t0, t1,num=time_steps)
    y0_list_w = np.random.uniform(-np.pi/2, np.pi/2, size=(num_of_inits, 1)) 
    y0_list_o = np.random.uniform(-0.2, 0.2, size=(num_of_inits, 1)) 
    y0_list = np.concatenate([y0_list_w, y0_list_o], axis=1)
    
    if use_fixed_init:
      y0_list = [y0]
      
    trajectory_list = []
    input_data_list = []
    input_data_w_time_list = []

    for i, y0 in enumerate(y0_list):

      #np.random.seed(seed=i)
      
      out = np.zeros((len(t), 2))
     
      out[0] = y0_list[i]
      
      u = input(t, u_option)
     
      for i in range(1, len(t)):

        t_span = [t[i-1], t[i]]
        z = scipy.integrate.odeint(func, out[i-1], t_span, args = (u[i],))

        out[i] = z[1]

      out = torch.tensor(out)
      u = torch.tensor(u).view(1,len(t))
      u = u.transpose(0,1)
      time = torch.tensor(t)
      time = time.view(1,len(t))
      time = time.transpose(0,1)

      trajectory_list.append(out)
      input_data_list.append(torch.cat((u,out), dim=-1))
      input_data_w_time_list.append(torch.cat((time,u,out), dim=-1))

    test_data = torch.stack(trajectory_list)
    input_data = torch.stack(input_data_list)

    input_data_w_time = torch.stack(input_data_w_time_list)

    if normalize:
      #nicht jede traj. einzeln
     test_data = torch.nn.functional.normalize(test_data) 
     input_data = torch.nn.functional.normalize(input_data) 
     input_data_w_time = torch.nn.functional.normalize(input_data_w_time) 
    #  test_data = test_data/torch.max(abs(test_data))
    #  input_data = input_data/torch.max(abs(input_data))
    #  input_data_w_time = input_data_w_time/torch.max(abs(input_data_w_time))

     
    return input_data, test_data, time, torch.tensor(y0_list), input_data_w_time
