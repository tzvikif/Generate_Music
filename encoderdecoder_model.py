# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 08:12:18 2018

@author: dkang
"""
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import random

#########################################################MODELS###########################################################

#Input is 100x1x219
#100 is the batch size
#219 is the length of the vector for each note
CURRENT_PATH = '/Users/tzvikif/Documents/Msc/Deep Learning/Project/MID/Bach/'
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers = 2, dropout = .5)
        
    def forward(self, input):
        output, (hidden, cell) = self.gru(input)
        return hidden, cell
    
    def initHidden(self):
        return torch.zeros(2,102, self.hidden_size, device = device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(output_size, hidden_size, num_layers = 2, dropout = .5)
        self.out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input, hidden, cell):
        output = F.relu(input)
        output, (hidden, cell) = self.gru(output, torch.stack((hidden, cell), dim = 0))
        output = self.sigmoid(self.out(output))
        return output, hidden, cell
    
    def initHidden(self):
        return torch.zeros(2,102, self.hidden_size, device = device)

class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hidden_size == decoder.hidden_size, "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #initialize output vector
        outputs = torch.zeros((102,219)).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src.view(102,1,219))
        
        input = trg[0,:] #SOS token
            
        for t in range(0, 102):
            output, hidden, cell = self.decoder(input.view(1,1,219), hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            input = (trg[t] if teacher_force else output)
        
        return outputs

####################################################SETTING UP DATA#########################################################
if __name__ == "__main__":
    
    print('Newest version 12/7/18 7:00 PM')
    
    with open(CURRENT_PATH + 'classical_notes.pkl', 'rb') as handle:
        data = pickle.load(handle)
        
    #If GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Turning songs into just a whole bunch of notes
    one_big_file = []
    for song in data:
        for note in song:
            one_big_file.append(note)
    
    #Trim off a bit of the data, so it's a nice number
    one_big_file = one_big_file[0:406800]
    
    
    #Organize data into one giant vector
    dataset = torch.stack(one_big_file, dim = 0).to(device)
    dataset.unsqueeze(0)
    
    #Organize data into batches of 100
    dataset = dataset.view(-1, 100, 219)
    
    #Add start of phrase vector [a vector of 2's] and an end of phrase vector [a vector of 3's]
    new_dataset = torch.zeros((dataset.shape[0],dataset.shape[1]+2, dataset.shape[2])).to(device)
    for idx, batch in enumerate(dataset):
        batch = torch.cat((((torch.zeros(batch.shape[1])).unsqueeze(0)).to(device),batch))
        batch = torch.cat((batch, ((torch.zeros(batch.shape[1])+1).unsqueeze(0)).to(device)))
        new_dataset[idx] = batch
        
    
    ###########Split data into train/test##############
    train_x = new_dataset[:-1] #Remove last batch
    train_y = new_dataset[1:] #Offset data by 1
    
    #Get random indices to split data 80-10-10. 
    total_idx = random.sample(range(0, len(dataset)-1), int(np.round(len(train_x)*.2,0)) )
    rand_idx = total_idx[0:int(len(total_idx)/2)]
    rand_idx2 = total_idx[int(len(total_idx)/2):int(len(total_idx))]
    test_x = train_x.clone()[rand_idx]
    test_y = train_y.clone()[rand_idx]
    eval_x = train_x.clone()[rand_idx2]
    eval_y = train_y.clone()[rand_idx2]
    
    #Deleting the random elements
    mask = np.ones(len(dataset)-1)
    mask[total_idx] = 0
    keep_idx = np.where(mask == 1)[0]
    train_x = train_x[keep_idx]
    train_y = train_y[keep_idx]
    
    #Setting to GPU the data
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    eval_x = eval_x.to(device)
    eval_y = eval_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
        
    print("Finished Loading Data")
    
    #########################################################TRAINING###########################################################
    
    encoder = EncoderRNN(219, 512)
    decoder = DecoderRNN(512, 219)
    
    encoder.to(device)
    decoder.to(device)
    
    s2s = seq2seq(encoder, decoder, device).to(device)
    
    optimizer = optim.Adam(s2s.parameters(), lr = 1e-5)
    
    criterion = nn.BCELoss()
    
    num_epochs = 200
    
    tr_loss_list = []
    ev_loss_list = []
    tt_loss_list = []
    
    for epoch in range(num_epochs):
        
        #TRAINING
        s2s.train()
        epoch_tr_loss = 0
        for i in range(len(train_x)): 
            optimizer.zero_grad()
            output = s2s(train_x[i], train_y[i])
            loss = criterion(output[1:101], train_y[i][1:101])
            
            loss.backward()
            clip_grad_norm_(s2s.parameters(), .25)
            
            optimizer.step()
            
            epoch_tr_loss += loss.item()
            
            if i % 100 == 0:
                print("Training iteration ", i, " out of ", len(train_x))
            
            
        #Eval
        s2s.eval()
        epoch_ev_loss = 0
        for i in range(len(eval_x)): 
            output = s2s(eval_x[i], eval_y[i], 0)
            
            loss = criterion(output[1:101], eval_y[i][1:101])
            
            epoch_ev_loss += loss.item()
            
            if i % 100 == 0:
                print("Evaluation iteration ", i, " out of ", len(eval_x))
            
            
        tr_loss_list.append(epoch_tr_loss/len(train_x))
        ev_loss_list.append(epoch_ev_loss/len(eval_x))
    
        print('We are on epoch ', epoch)
        print('The current training loss is ', epoch_tr_loss, " ", epoch_tr_loss/len(train_x))
        print('The current test loss is ', epoch_ev_loss, " ", epoch_ev_loss/len(eval_x))
        print()
        
        #Testing
        if epoch % 50 == 0:
            s2s.eval()
            epoch_tt_loss = 0
            for i in range(len(test_x)):
                output = s2s(test_x[i], test_y[i], 0)
            
                loss = criterion(output[1:101], test_y[i][1:101])
                
                epoch_tt_loss += loss.item()
                
                if i % 100 == 0:
                    print("Evaluation iteration ", i, " out of ", len(test_x))
                    
            tt_loss_list.append(epoch_tt_loss/len(test_x))
            
            with open('test_loss.pkl', 'wb') as handle:
                pickle.dump(tt_loss_list, handle, protocol= pickle.HIGHEST_PROTOCOL )
        
        if epoch % 15 == 0:
            
            state = {
                    'epoch': epoch,
                    'enc_state_dict': encoder.state_dict(),
                    'dec_state_dict': decoder.state_dict(),
                    's2s_state_dict': s2s.state_dict(),
                    'optimizer': optimizer.state_dict(),
            }
            
            
            torch.save(state, str(epoch)+'modelstate.pth')
            
            with open('train_loss.pkl', 'wb') as handle:
                pickle.dump(tr_loss_list, handle, protocol= pickle.HIGHEST_PROTOCOL )
    
            with open('eval_loss.pkl', 'wb') as handle:
                pickle.dump(ev_loss_list, handle, protocol= pickle.HIGHEST_PROTOCOL )
    
            #state = torch.load(filepath)
            #s2s.load_state_dict(state['state_dict'])
            #optimizer.load_state_dict(state['optimizer'])
