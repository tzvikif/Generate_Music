import numpy as np
import pickle
import torch.nn as nn
import torch
from torch import optim
from torch import from_numpy
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import random
from music21 import *
import glob

CURRENT_PATH = '/Users/tzvikif/Documents/Msc/Deep Learning/Project/MID/Bach/'
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

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = 2, dropout = .5)
        
    def forward(self, input):
        output, (hidden, cell) = self.lstm(input)
        return hidden, cell
    
    def initHidden(self):
        return torch.zeros(2,102, self.hidden_size, device = device)
class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers = 2, dropout = .5)
        self.out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hidden_size == decoder.hidden_size, "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg):
        
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

class Model_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,batch_size=4,num_layers=2):
        super(Model_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        #self.lstm = nn.LSTM(self.input_dim, self.hidden_dim,num_layers=self.num_layers,dropout=0.5,batch_first=True)
        self.lstm = nn.LSTM(input_size=self.input_dim,hidden_size=self.hidden_dim,num_layers=self.num_layers,batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = nn.Softmax(dim=1)
        #self.h0 = np.random.normal(mean,std,(input_units+hidden_units,hidden_units))
    def forward(self, x):
         # Initialize hidden state with zeros
         #self.layer_dim, batch_dim, self.hidden_dim
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim,dtype=torch.double).requires_grad_()
        # Initialize cell state
        #self.layer_dim, x.size(0), self.hidden_dim
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim,dtype=torch.double).requires_grad_()
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        #out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out, (hn, cn) = self.lstm(x, (h0, c0) )
        batch_first = out.transpose(0,1)
        batch_size = x.shape[0]
        linear_input = batch_first.view(-1,self.hidden_dim)
        out = self.fc(linear_input)
        out_softmax = self.softmax(out)
        return out_softmax, (hn, cn)


songs_data = create_data()
# # of songs (samples) x # of timestamps x tuple of 4 (combination of notes
# and rhythms of both hands)
print("Whole music data size:", np.array(songs_data).shape)
data_to_int, int_to_data = get_dictionary(songs_data)
print("Number of unique notes:", len(data_to_int))

INPUT = 100
HIDDEN = 256
vocab_size = len(data_to_int)
OUTPUT = vocab_size
BATCH_SIZE = 2
SEQ_LEN = 200
#LEARNING_RATE = 0.005
#BETA1 = 0.9
#BETA2 = 0.999
#BATCH_SIZE = 50

# get batched dataset
def get_batches(songs, data_int):
    train_dataset = []
    batch_size = min(int(len(songs[0])/SEQ_LEN),16) #max batch size is 16
    song_len = len(songs[0])    #all songs have the same length
    for song in songs:
        batched_song = np.zeros([batch_size,SEQ_LEN,vocab_size],dtype=np.double)
        for batch_idx,seq_idx in enumerate(range(0,song_len,SEQ_LEN)):
            if batch_idx == batch_size:
                train_dataset.append(batched_song)
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
                batched_song[batch_idx,i,idx] = 1
        
    return train_dataset,batch_size
def prepareData(data,seq_len=SEQ_LEN):
    X = []
    Y = []
    
    for i,batch in enumerate(data):
        x_temp = np.copy(batch[:,:-1,:])
        y_temp = np.copy(batch[:,1:,:])
        X.append(x_temp)
        Y.append(y_temp)
    return X,Y
def main():
    train_set = get_batches(songs_data, data_to_int)
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
    
    model = Model_LSTM(vocab_size,hidden_dim=HIDDEN,output_dim=OUTPUT)
    model.double()
    #s2s = seq2seq(encoder, decoder, device).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr = 1e-5)
    
    criterion = nn.BCELoss()
    
    num_epochs = 200
    
    tr_loss_list = []
    ev_loss_list = []
    tt_loss_list = []
    
    for epoch in range(num_epochs):
        
        #TRAINING
        model.train()
        epoch_tr_loss = 0
        accuracy = 0
        for i in range(len(train_x)): 
            optimizer.zero_grad()
            output, (hn, cn) = model(train_x[i])
            output = output.view(4,SEQ_LEN-1,-1)    #batch_size=4
            loss = criterion(output[:], train_y[i][:])
            o = np.argmax(output.detach(),axis=2)
            loss.backward()
            #clip_grad_norm_(model.parameters(), .25)
            
            optimizer.step()
            
            epoch_tr_loss += loss.item()
            
            if i % 1 == 0:
                print("Training iteration ", i, " out of ", len(train_x))
            
            
        #Eval
        model.eval()
        epoch_ev_loss = 0
        for i in range(len(eval_x)): 
            output, (hn, cn) = model(eval_x[i])
            output = output.view(4,SEQ_LEN-1,-1)    #batch_size=4
            loss = criterion(output[:], eval_y[i][:])
            
            epoch_ev_loss += loss.item()
            
            if i % 100 == 0:
                print("Evaluation iteration ", i, " out of ", len(eval_x))
            
            
        tr_loss_list.append(epoch_tr_loss/len(train_x))
        ev_loss_list.append(epoch_ev_loss/len(eval_x))
    
        print('We are on epoch ', epoch)
        print('The current training loss is ', epoch_tr_loss, " ", epoch_tr_loss/len(train_x))
        print('The current test loss is ', epoch_ev_loss, " ", epoch_ev_loss/len(eval_x))
        print()
main()


