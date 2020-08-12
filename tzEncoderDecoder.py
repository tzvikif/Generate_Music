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
import glob
from music21 import *
from torch import from_numpy

CURRENT_PATH = '/Users/tzvikif/Documents/Msc/Deep Learning/Project/MID/Bach/'
LOAD_WEIGHTS = False
MODEL1_NAME = 'model1.pth'
SEQ_LEN = 64
HIDDEN = 100
OUTPUT = HIDDEN
FEATURES = HIDDEN


#########################################################MODELS###########################################################

#Input is 100x1x219
#100 is the batch size
#219 is the length of the vector for each note
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers = 2, dropout = .5)
        self.embedding= nn.Embedding(vocab_size,hidden_size)
    def forward(self, input):
        input = self.embedding(input)
        input = input.transpose(0,1)
        output, (hidden, cell) = self.gru(input)
        return hidden, cell
    
    def initHidden(self):
        return torch.zeros(2,SEQ_LEN, self.hidden_size, device = device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(output_size, hidden_size, num_layers = 2, dropout = .5)
        self.out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input, hidden, cell):
        #input = input.squeeze(dim=0)
        output = F.relu(input)
        #output = output.transpose(0,1)
        #(1,1,100)
        output, (hidden, cell) = self.gru(output.view(1,1,-1), torch.stack((hidden, cell), dim = 0))
        output = self.sigmoid(self.out(output))
        return output, hidden, cell
    
    def initHidden(self):
        return torch.zeros(2,SEQ_LEN, self.hidden_size, device = device)

class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.embedding= nn.Embedding(vocab_size,100)
        
        assert encoder.hidden_size == decoder.hidden_size, "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #initialize output vector
        outputs = torch.zeros((SEQ_LEN,FEATURES)).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        #src.view(102,1,219)
        trg = trg.transpose(0,1)
        input = self.embedding(trg[0,:]) #SOS token
        for t in range(0, SEQ_LEN-1):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            input = (self.embedding(trg[t]) if teacher_force else output)
        
        return outputs

####################################################READING DATA#########################################################

# prepare music data
def process_data(songs):
    whole_data = []
    for song in songs:
        midi_data = converter.parse(song).flat
        song_data = []
        prev_offset = -1
        for element in midi_data:
            if isinstance(element, note.Note):
                if element.offset != prev_offset:
                    song_data.append([element.pitch.nameWithOctave, 
                                      element.quarterLength])
                else:
                    if len(song_data[-1]) < 4:
                        song_data[-1].append(element.pitch.nameWithOctave)   
                        song_data[-1].append(element.quarterLength)       
                prev_offset = element.offset
            elif isinstance(element, chord.Chord):
                pitch_names = '.'.join(n.nameWithOctave for n in element.pitches)
                if element.offset != prev_offset:
                    song_data.append([pitch_names, element.quarterLength])
                else:
                    if len(song_data[-1]) < 4:
                        song_data[-1].append(pitch_names)   
                        song_data[-1].append(element.quarterLength)      
                prev_offset = element.offset
        for item in song_data:
            if len(item) < 4:
                item.append(None)
                item.append(None)
        whole_data.append(song_data)
    return whole_data
# transform data to tuple instead of list and pad songs that are shorter in
# length with (None, None, None, None)
def transform_data(songs):
    max_len = 0
    for song in songs:
        max_len = max(max_len, len(song))
    for song in songs:
        for i in range(max_len - len(song)):
            song.append([None, None, None, None])
    transform_data = []
    for song in songs:
        t_song_data = []
        for item in song:
            t_song_data.append(tuple(item))
        transform_data.append(t_song_data)
    return transform_data

# get a dictionary of the unique notes
def get_dictionary(songs):
    possible_combs = set(item for song in songs for item in song)
    data_to_int = dict((v, i) for i, v in enumerate(possible_combs))
    int_to_data = dict((i, v) for i, v in enumerate(possible_combs))
    return data_to_int, int_to_data    
def create_data():
    songs = glob.glob(CURRENT_PATH +  '/*.mid')
    songs_data = process_data(songs)
    songs_data = transform_data(songs_data)
    return songs_data
def get_batches(songs, data_int):
    train_dataset = []
    batch_size = min(int(len(songs[0])/SEQ_LEN),16) #max batch size is 16
    batch_size = 1
    song_len = len(songs[0])    #all songs have the same length
    for song in songs:
        #batched_song = np.zeros([batch_size,SEQ_LEN,vocab_size],dtype=np.double)
        batched_song = np.zeros([batch_size,SEQ_LEN])
        batch_idx = 0
        for _,seq_idx in enumerate(range(0,song_len,SEQ_LEN)):
            if batch_idx == batch_size:
                train_dataset.append(torch.LongTensor(batched_song))
                batched_song = np.zeros([batch_size,SEQ_LEN])
                batch_idx = 0
            start = seq_idx
            end = start + SEQ_LEN
            if end+1 >= len(song):
                break
            sequence = song[start:end]
            #batched_song[batch_idx,:] = single_sequence
            #if(len(batch_data) != batch_size):
            #    break
            note_list = []
            #one_hot_note = np.zeros([batch_size, len(data_int)])
            for i,note in enumerate(sequence):
                idx = data_to_int[note]
                #batched_song[batch_idx,i,idx] = 1
                batched_song[batch_idx,i] = idx
            batch_idx+=1
        
    return train_dataset,batch_size
def prepareData(data,seq_len=SEQ_LEN):
    X = []
    Y = []
    
    for i,batch in enumerate(data):

        x_temp = np.copy(batch[:,:-1])
        y_temp = np.copy(batch[:,1:])
        X.append(x_temp)
        Y.append(y_temp)
    return X,Y
def splitData():
    train_set,batch_size = get_batches(songs_data, data_to_int)
    X,Y = prepareData(train_set)
    total_idx = np.random.permutation( len(X) )
    rand_idx = total_idx[0:int(len(total_idx)*0.75)]
    rand_idx2 = total_idx[int(len(total_idx)*0.75):int(len(total_idx))]
    train_x = [from_numpy(X[i]) for i in rand_idx]
    train_y = [from_numpy(Y[i]) for i in rand_idx]
    eval_x = [from_numpy(X[i]) for i in rand_idx2]
    eval_y = [from_numpy(Y[i]) for i in rand_idx2]
    #dont need X,Y anymore
    X = None
    Y = None

    #Setting to GPU the data
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    eval_x = eval_x.to(device)
    eval_y = eval_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
        
    print("Finished Loading Data")

####################################################SETTING UP DATA#########################################################
if __name__ == "__main__":
    
    songs_data = create_data()
    data_to_int, int_to_data = get_dictionary(songs_data)    
    vocab_size = len(data_to_int)
    #If GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set,batch_size = get_batches(songs_data, data_to_int)
    X,Y = prepareData(train_set)
    total_idx = np.random.permutation( len(X) )
    rand_idx = total_idx[0:int(len(total_idx)*0.75)]
    rand_idx2 = total_idx[int(len(total_idx)*0.75):int(len(total_idx))]
    train_x = [from_numpy(X[i]) for i in rand_idx]
    train_y = [from_numpy(Y[i]) for i in rand_idx]
    eval_x = [from_numpy(X[i]) for i in rand_idx2]
    eval_y = [from_numpy(Y[i]) for i in rand_idx2]
    #dont need X,Y anymore
    X = None
    Y = None

    #Setting to GPU the data
    '''
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    eval_x = eval_x.to(device)
    eval_y = eval_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    '''
    print("Finished Loading Data")

    #########################################################TRAINING###########################################################
    
    encoder = EncoderRNN(FEATURES, HIDDEN)
    decoder = DecoderRNN(HIDDEN, FEATURES)
    
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
            t = encoder.embedding(train_y[i])
            loss = criterion(output[0:SEQ_LEN-1], t.view(SEQ_LEN-1,-1).detach())
            
            loss.backward()
            clip_grad_norm_(s2s.parameters(), .25)
            
            optimizer.step()
            
            epoch_tr_loss += loss.item()
            
            if i % 20 == 0:
                print("Training iteration ", i, " out of ", len(train_x))
            
            
        #Eval
        s2s.eval()
        epoch_ev_loss = 0
        for i in range(len(eval_x)): 
            output = s2s(eval_x[i], eval_y[i], 0)
            t = encoder.embedding(eval_y[i])
            loss = criterion(output[0:SEQ_LEN-1], t.view(SEQ_LEN-1,-1).detach())
            
            epoch_ev_loss += loss.item()
            
            if i % 100 == 0:
                print("Evaluation iteration ", i, " out of ", len(eval_x))
            
            
        tr_loss_list.append(epoch_tr_loss/len(train_x))
        ev_loss_list.append(epoch_ev_loss/len(eval_x))
    
        print('We are on epoch ', epoch)
        print('The current training loss is ', epoch_tr_loss, " ", epoch_tr_loss/len(train_x))
        print('The current test loss is ', epoch_ev_loss, " ", epoch_ev_loss/len(eval_x))
        print()
        '''
        #Testing
        if epoch % 50 == 0:
            s2s.eval()
            epoch_tt_loss = 0
            for i in range(len(test_x)):
                output = s2s(test_x[i], test_y[i], 0)
            
                t = encoder.embedding(train_y[i])
                loss = criterion(output[0:SEQ_LEN-1], t.view(SEQ_LEN-1,-1).detach())
                
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
            '''
