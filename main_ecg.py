import numpy as np
import torch
from data.synthetic_dataset import create_synthetic_dataset, SyntheticDataset
from models.seq2seq import EncoderRNN, DecoderRNN, Net_GRU
from loss.dilate_loss import dilate_loss
from torch.utils.data import DataLoader
import random
import pandas as pd
import os
from tslearn.metrics import dtw, dtw_path
import matplotlib.pyplot as plt
import warnings
import warnings; warnings.simplefilter('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
random.seed(0)

# parameters
batch_size = 100
N = 500
N_input = 20
N_output = 20  
sigma = 0.01
gamma = 0.01

# Load synthetic dataset
X_train_input,X_train_target,X_test_input,X_test_target,train_bkp,test_bkp = create_synthetic_dataset(N,N_input,N_output,sigma)
dataset_train = SyntheticDataset(X_train_input,X_train_target, train_bkp)
dataset_test  = SyntheticDataset(X_test_input,X_test_target, test_bkp)
trainloader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, num_workers=1)
testloader  = DataLoader(dataset_test, batch_size=batch_size,shuffle=False, num_workers=1)


def train_model(net,loss_type, learning_rate, epochs=1000, gamma = 0.001,
                print_every=50,eval_every=50, verbose=1, Lambda=1, alpha=0.5):
    
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
    criterion = torch.nn.MSELoss()

    train_losses = []
    train_losses_shape = []
    train_losses_temp = []

    test_mse = []
    test_dtw = []
    test_tdi = []
    
    for epoch in range(epochs): 
        for i, data in enumerate(trainloader, 0):
            inputs, target, _ = data
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            target = torch.tensor(target, dtype=torch.float32).to(device)
            batch_size, N_output = target.shape[0:2]                     

            # forward + backward + optimize
            outputs = net(inputs)
            loss_mse,loss_shape,loss_temporal = torch.tensor(0),torch.tensor(0),torch.tensor(0)
            
            if (loss_type=='mse'):
                loss_mse = criterion(target,outputs)
                loss = loss_mse
                loss_dilate, loss_shape, loss_temporal = dilate_loss(target,outputs,alpha, gamma, device) 
                train_losses.append(loss_dilate.item())
                train_losses_shape.append(loss_shape.item())
                train_losses_temp.append(loss_temporal.item())                   
 
            if (loss_type=='dilate'):    
                loss, loss_shape, loss_temporal = dilate_loss(target,outputs,alpha, gamma, device) 
                train_losses.append(loss.item())
                train_losses_shape.append(loss_shape.item())
                train_losses_temp.append(loss_temporal.item())           
                  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()          
        
        if(verbose):
            if (epoch % print_every == 0):
                print('epoch ', epoch, ' loss ',loss.item(),' loss shape ',loss_shape.item(),' loss temporal ',loss_temporal.item())
                mse_, dtw_, tdi_ = eval_model(net,testloader, gamma,verbose=1)
                test_mse.append(mse_)
                test_dtw.append(dtw_)
                test_tdi.append(tdi_)
    return train_losses, train_losses_shape, train_losses_temp, test_mse, test_dtw, test_tdi
  

def eval_model(net,loader, gamma,verbose=1):   
    criterion = torch.nn.MSELoss()
    losses_mse = []
    losses_dtw = []
    losses_tdi = []   

    for i, data in enumerate(loader, 0):
        loss_mse, loss_dtw, loss_tdi = torch.tensor(0),torch.tensor(0),torch.tensor(0)
        # get the inputs
        inputs, target, breakpoints = data
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        target = torch.tensor(target, dtype=torch.float32).to(device)
        batch_size, N_output = target.shape[0:2]
        outputs = net(inputs)
         
        # MSE    
        loss_mse = criterion(target,outputs)    
        loss_dtw, loss_tdi = 0,0
        # DTW and TDI
        for k in range(batch_size):         
            target_k_cpu = target[k,:,0:1].view(-1).detach().cpu().numpy()
            output_k_cpu = outputs[k,:,0:1].view(-1).detach().cpu().numpy()

            path, sim = dtw_path(target_k_cpu, output_k_cpu)   
            loss_dtw += sim
                       
            Dist = 0
            for i,j in path:
                    Dist += (i-j)*(i-j)
            loss_tdi += Dist / (N_output*N_output)            
                        
        loss_dtw = loss_dtw /batch_size
        loss_tdi = loss_tdi / batch_size

        # print statistics
        losses_mse.append( loss_mse.item() )
        losses_dtw.append( loss_dtw )
        losses_tdi.append( loss_tdi )

    print( ' Eval mse= ', np.array(losses_mse).mean() ,' dtw= ',np.array(losses_dtw).mean() ,' tdi= ', np.array(losses_tdi).mean()) 
    
    return(np.array(losses_mse).mean(), np.array(losses_dtw).mean(), np.array(losses_tdi).mean())

encoder = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
decoder = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=1).to(device)
net_gru_dilate = Net_GRU(encoder,decoder, N_output, device).to(device)
train_losses_dilate, train_losses_shape_dilate, train_losses_temp_dilate, test_mse_dilate, test_dtw_dilate, test_tdi_dilate = train_model(net_gru_dilate,loss_type='dilate',learning_rate=0.001, epochs=50, gamma=gamma, print_every=5, eval_every=5,verbose=1)
final_mse, final_dtw, final_tdi = eval_model(net_gru_dilate, testloader, gamma, verbose=0)


encoder = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
decoder = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=1).to(device)
net_gru_mse = Net_GRU(encoder,decoder, N_output, device).to(device)
train_losses_mse, train_losses_shape_mse, train_losses_temp_mse, test_mse_mse, test_dtw_mse, test_tdi_mse = train_model(net_gru_mse,loss_type='mse',learning_rate=0.001, epochs=50, gamma=gamma, print_every=5, eval_every=5,verbose=1)
final_mse_2, final_dtw_2, final_tdi_2 = eval_model(net_gru_dilate, testloader, gamma, verbose=0)

metrics_df = pd.DataFrame({
    'Loss' : ['DILATE', 'MSE'],
    'MSE': [final_mse, final_mse_2],
    'DTW': [final_dtw, final_dtw_2],
    'TDI': [final_tdi, final_tdi_2]
})

# Visualize results

# Create a directory 'plots' if it doesn't exist
if not os.path.exists('plots/ecg'):
    os.makedirs('plots/ecg')

metrics_df.to_csv('plots/tab_metrics_ecg.csv', index=False)

gen_test = iter(testloader)
test_inputs, test_targets, breaks = next(gen_test)

test_inputs  = torch.tensor(test_inputs, dtype=torch.float32).to(device)
test_targets = torch.tensor(test_targets, dtype=torch.float32).to(device)
criterion = torch.nn.MSELoss()

nets = [net_gru_mse,net_gru_dilate]

for ind in range(1,51):
    plt.figure()
    plt.rcParams['figure.figsize'] = (17.0,5.0)  
    k = 1
    for net in nets:
        pred = net(test_inputs).to(device)

        input = test_inputs.detach().cpu().numpy()[ind,:,:]
        target = test_targets.detach().cpu().numpy()[ind,:,:]
        preds = pred.detach().cpu().numpy()[ind,:,:]

        plt.subplot(1,3,k)
        plt.plot(range(0,N_input) ,input,label='input',linewidth=3)
        plt.plot(range(N_input-1,N_input+N_output), np.concatenate([ input[N_input-1:N_input], target ]) ,label='target',linewidth=3)   
        plt.plot(range(N_input-1,N_input+N_output),  np.concatenate([ input[N_input-1:N_input], preds ])  ,label='prediction',linewidth=3)       
        plt.xticks(range(0,40,2))
        plt.legend()
        k = k+1

    plt.savefig(f'plots/ecg/plot_ecg_{ind}.png')  
    plt.close()

# Plots des évolution des loss au cours des epochs

plt.figure(figsize=(12, 8))

# Premier subplot
plt.subplot(3, 1, 1)
plt.plot(range(1, len(train_losses_dilate) + 1), train_losses_dilate, label='loss = dilate', color='blue')
plt.plot(range(1, len(train_losses_dilate) + 1), train_losses_mse, label='loss = mse', color='red')
plt.title('Evolution de L_dilate en fonction de la loss choisie pour le modèle')
plt.legend()

# Deuxième subplot
plt.subplot(3, 1, 2)
plt.plot(range(1, len(train_losses_shape_dilate) + 1), train_losses_shape_dilate, label='loss = dilate', color='blue')
plt.plot(range(1, len(train_losses_shape_dilate) + 1), train_losses_shape_mse, label='loss = mse', color='red')
plt.title('Evolution de L_shape en fonction de la loss choisie pour le modèle')
plt.legend()

# Troisième subplot
plt.subplot(3, 1, 3)
plt.plot(range(1, len(train_losses_temp_dilate) + 1), train_losses_temp_dilate, label='loss = dilate', color='blue')
plt.plot(range(1, len(train_losses_temp_dilate) + 1), train_losses_temp_mse, label='loss = mse', color='red')
plt.title('Evolution de L_temp en fonction de la loss choisie pour le modèle')
plt.legend()

plt.tight_layout()
plt.savefig(f'plots/ecg/plots_losses.png')  
plt.close()
