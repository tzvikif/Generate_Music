import numpy as np
import pickle
import torch.nn as nn
from torch import optim
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
#LEARNING_RATE = 0.005
#BETA1 = 0.9
#BETA2 = 0.999
#BATCH_SIZE = 50

# get batched dataset
def get_batches(songs, data_int):
    train_dataset = []
    for i in range(len(songs) - BATCH_SIZE + 1):
        start = i * BATCH_SIZE
        end = start + BATCH_SIZE
        batch_data = songs[start:end]
        if(len(batch_data) != BATCH_SIZE):
            break
        note_list = []
        for j in range(len(batch_data[0])):
            batch_dataset = np.zeros([BATCH_SIZE, len(data_int)])
            for k in range(BATCH_SIZE):
                note = batch_data[k][j]
                idx = data_to_int[note]
                batch_dataset[k, idx] = 1
            note_list.append(batch_dataset)
        train_dataset.append(note_list)
    return train_dataset
def main():
    train_set = get_batches(songs_data, data_to_int)

    encoder = EncoderLSTM(219, 512)
    decoder = DecoderLSTM(512, 219)
    
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
            
            if i % 1 == 0:
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
    



main()


