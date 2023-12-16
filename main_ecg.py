import numpy as np
import torch
from data.synthetic_dataset import create_synthetic_dataset, SyntheticDataset
from models.seq2seq import Seq2Seq
from loss.dilate_loss import dilate_loss
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import random
from copy import deepcopy
import os
import pandas as pd
from tslearn.metrics import dtw, dtw_path
import matplotlib.pyplot as plt
import warnings
import warnings; warnings.simplefilter('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

### SEED POUR LA REPRODUCTIBILITE 

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# parameters
batch_size = 35
N = 500
N_input = 84
N_output = 56  
sigma = 0.01
gamma = 0.01

### IMPORTATION DU DATASET ECG

DATA_PATH = "./data/"

ecg_train = np.array(pd.read_table(DATA_PATH + "ECG5000/ECG5000_TRAIN.tsv"))[:, :, np.newaxis]
ecg_test = np.array(pd.read_table(DATA_PATH + "ECG5000/ECG5000_TEST.tsv"))[:, :, np.newaxis]

ecg_train_flat = ecg_train.reshape(-1, ecg_train.shape[1])
ecg_test_flat = ecg_test.reshape(-1, ecg_test.shape[1])

# Normalisation
scaler = StandardScaler()
ecg_train_flat = scaler.fit_transform(ecg_train_flat)
ecg_test_flat = scaler.transform(ecg_test_flat)

ecg_train = ecg_train_flat.reshape(ecg_train.shape[0], ecg_train.shape[1], 1)
ecg_test = ecg_test_flat.reshape(ecg_test.shape[0], ecg_test.shape[1], 1)

# Tronquer pour s'assurer que la taille est un multiple de batch_size
num_train_batches = ecg_train.shape[0] // batch_size
ecg_train = ecg_train[:num_train_batches * batch_size]
num_test_batches = ecg_test.shape[0] // batch_size
ecg_test = ecg_test[:num_test_batches * batch_size]

print(ecg_train.shape, ecg_test.shape)

class ECG5000Dataset(Dataset):

    def __init__(self, data, output_length=56):
        self.data = torch.from_numpy(data).to(dtype=torch.float32)
        self.output_length = output_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index, :-self.output_length], self.data[index, -self.output_length:]
    
ecg_train_dataset = ECG5000Dataset(ecg_train)
ecg_test_dataset = ECG5000Dataset(ecg_test)
trainloader = DataLoader(ecg_train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(ecg_test_dataset, batch_size=batch_size, shuffle=False)


### FONCTION ENTRAINEMENT ET EVALUATION

def train_model(net,loss_type, learning_rate, epochs=1000, gamma = 0.001,
                print_every=50, alpha=0.5):
    
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
    ### learning rate adaptatif qui diminue au cours des epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    criterion = torch.nn.MSELoss()
    
    for epoch in range(epochs): 
        for i, data in enumerate(trainloader, 0):
            inputs, target = data
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            target = torch.tensor(target, dtype=torch.float32).to(device)

            # forward + backward + optimize
            outputs = net(inputs)
            loss_mse,loss_shape,loss_temporal = torch.tensor(0),torch.tensor(0),torch.tensor(0)
            
            if (loss_type=='mse'):
                loss_mse = criterion(target,outputs)
                loss = loss_mse                   
 
            if (loss_type=='dilate'):    
                loss, loss_shape, loss_temporal = dilate_loss(target,outputs,alpha, gamma, device)             
                  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

        ### on ajoute un pas au scheduler pour mettre le learning rate à jour
        scheduler.step()
        
        if (epoch % print_every == 0):
            print('epoch ', epoch, ' loss ',loss.item(),' loss shape ',loss_shape.item(),' loss temporal ',loss_temporal.item())
            m, d, t = eval_model(net,testloader, gamma,verbose=1)

  

def eval_model(net,loader, gamma,verbose=1):   
    criterion = torch.nn.MSELoss()
    losses_mse = []
    losses_dtw = []
    losses_tdi = []   

    for i, data in enumerate(loader, 0):
        loss_mse, loss_dtw, loss_tdi = torch.tensor(0),torch.tensor(0),torch.tensor(0)
        # get the inputs
        inputs, target = data
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        target = torch.tensor(target, dtype=torch.float32).to(device)
        batch_size, N_output = target.shape[0:2]
        outputs = net(inputs)
         
        # MSE    
        loss_mse = criterion(target,outputs)    

        # DTW and TDI
        loss_dtw, loss_tdi = 0,0
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


### CREATION DU MODELE GRU (Seq2Seq) ET ENTRAINEMENT

net_gru_dilate = Seq2Seq(input_size=1, hidden_size=128, num_layers=1, fc_units=16, output_size=1, target_length=N_output, device=device).to(device)
train_model(net_gru_dilate,loss_type='dilate',learning_rate=0.005, epochs=500, gamma=gamma, alpha=0.7, print_every=5)
final_mse, final_dtw, final_tdi = eval_model(net_gru_dilate, testloader, gamma)

net_gru_mse = Seq2Seq(input_size=1, hidden_size=128, num_layers=1, fc_units=16, output_size=1, target_length=N_output, device=device).to(device)
train_model(net_gru_mse,loss_type='mse',learning_rate=0.005, epochs=150, gamma=gamma, print_every=5)
final_mse_2, final_dtw_2, final_tdi_2 = eval_model(net_gru_mse, testloader, gamma)

net_gru_soft_dtw = Seq2Seq(input_size=1, hidden_size=128, num_layers=1, fc_units=16, output_size=1, target_length=N_output, device=device).to(device)
train_model(net_gru_soft_dtw,loss_type='dilate',learning_rate=0.005, epochs=500, gamma=gamma, alpha =1, print_every=5)
final_mse_3, final_dtw_3, final_tdi_3 = eval_model(net_gru_soft_dtw, testloader, gamma)


# VISUALISATION DES RESULTATS

# Create a directory 'plots' if it doesn't exist
if not os.path.exists('plots/ecg'):
    os.makedirs('plots/ecg')

### TABLEAU RECAPITULATIF DES METRICS

metrics_df = pd.DataFrame({
    'Loss' : ['DILATE', 'MSE', 'SOFT DTW'],
    'MSE': [final_mse, final_mse_2, final_mse_3],
    'DTW': [final_dtw, final_dtw_2, final_dtw_3],
    'TDI': [final_tdi, final_tdi_2, final_tdi_3]
})


metrics_df.to_csv('plots/ecg/tab_metrics_ecg.csv', index=False)


### PREDICTION DE QUELQUES ECG

gen_test = iter(testloader)
test_inputs, test_targets = next(gen_test)

test_inputs  = torch.tensor(test_inputs, dtype=torch.float32).to(device)
test_targets = torch.tensor(test_targets, dtype=torch.float32).to(device)
criterion = torch.nn.MSELoss()

nets = [net_gru_mse,net_gru_dilate,net_gru_soft_dtw]

for ind in range(1,20):
    plt.figure()
    plt.rcParams['figure.figsize'] = (17.0,5.0)  
    k = 1
    for net in nets:
        pred = net(test_inputs).to(device)

        input = test_inputs.detach().cpu().numpy()[ind,:,:]
        target = test_targets.detach().cpu().numpy()[ind,:,:]
        preds = pred.detach().cpu().numpy()[ind,:,:]

        plt.subplot(1,3,k)
        plt.plot(range(0, len(input)), input.flatten(), label='input', linewidth=3)
        plt.plot(range(len(input)-1,len(input)+len(preds)), np.concatenate([ input[len(input)-1:len(input)].flatten(), target.flatten() ]) ,label='target',linewidth=3)   
        plt.plot(range(len(input)-1,len(input)+len(preds)),  np.concatenate([ input[len(input)-1:len(input)].flatten(), preds.flatten() ])  ,label='prediction',linewidth=3)       
        plt.xticks(range(0,140,10))
        plt.legend()
        k = k+1

    plt.savefig(f'plots/ecg/plot_ecg_{ind}.png')  # Save figure
    plt.close()