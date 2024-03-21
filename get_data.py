import torch
import numpy as np
import scipy



def input(t, u_option):
  
  if u_option=="noise":
   return np.random.normal(0, 0.5, len(t))
  
  if u_option=="sin":
   return np.sin(t)
  
  if u_option=="cos":
   return np.cos(t)
  

  else: 
   return 0
  

def func(y,t, args):

  f = 1

  u = args
  
  return np.array([y[1], 1/f*(-np.sin(y[0])- 1/7*y[1] + u )])


def generate_data(x0 = np.pi/4, y0 = 0.1, use_fixed_init = False, t0=0, t1=30, time_steps=1000, num_of_inits=1, normalize=True, add_noise=False, u_option="noise"):

    np.random.seed(seed=42)

    
    y0 = [x0, y0]
    
    t = np.linspace(t0, t1,num=time_steps)
    y0_list_w = np.random.uniform(-np.pi/3, np.pi/3, size=(num_of_inits, 1)) 
    y0_list_o = np.random.uniform(-0.1, 0.1, size=(num_of_inits, 1)) 
    y0_list = np.concatenate([y0_list_w, y0_list_o], axis=1)
    
    if use_fixed_init:
      y0_list = [y0]
      
    trajectory_list = []
    for i, y0 in enumerate(y0_list):
     if add_noise:
      dydt = scipy.integrate.odeint(func, y0, t) + np.random.normal(0, 0.005, (len(t), 2))

     else:
      
      out = np.zeros((len(t), 2))
     
      out[0] = y0_list[i]
      print(out[0])

      u = input(t, u_option)

      for i in range(1, len(t)):

        t_span = [t[i-1], t[i]]
        z = scipy.integrate.odeint(func, out[i-1], t_span, args = (u[i],))

        out[i] = z[1]

     trajectory_list.append(torch.tensor(out))
    data = torch.stack(trajectory_list)

    if normalize:
     data = data/torch.max(abs(data))
     
    return data, torch.tensor(t), torch.tensor(y0_list)


def get_batch(data, time_series, t_end_batch , batch_size): 

    device="cpu"
    s = torch.from_numpy(np.random.choice(np.arange(data.size(dim=0) - t_end_batch, dtype=np.int64), batch_size, replace=False))
    batch_y0 = data[s]  # (M, D)
    batch_t = time_series[:t_end_batch]  # (T)
    batch_y = torch.stack([data[s + i] for i in range(t_end_batch)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

def get_batch_RNN(data, time_series, t_end_batch , batch_size):
    
    

    device="cpu"
    #s = torch.from_numpy(np.random.choice(np.arange(data.size(dim=0) - t_end_batch, dtype=np.int64), batch_size, replace=False))
    s = np.array([0 + t_end_batch*x for x in range(batch_size) if t_end_batch*x < len(time_series)])
    
    batch_y0 = data[s]  # (M, D)
    batch_t = time_series[:t_end_batch]  # (T)
    batch_y = torch.stack([data[s + i] for i in range(t_end_batch)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


