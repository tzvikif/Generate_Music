# -*- coding: utf-8 -*-
"""
LSTM training and prediction on midi dataset to generate a new music
Closely followed the tutorial at https://www.kaggle.com/navjindervirdee/
lstm-neural-network-from-scratch/notebook
 
@author: Jay
"""

from music21 import *
import matplotlib.pyplot as plt
import numpy as np
import glob
CURRENT_PATH = '/Users/tzvikif/Documents/Msc/Deep Learning/Project/MID/'
# PART.1 #####################################################################
# Process data and get it ready for LSTM run

# visualizations of midi data
#midi_data = converter.parse(CURRENT_PATH +  'bach_846_format0.mid')
#midi_data.write('text')
#midi_data.show()
#midi_data.plot('histogram', 'pitchclass')
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

songs = glob.glob(CURRENT_PATH +  '/*.mid')
songs_data = process_data(songs)
songs_data = transform_data(songs_data)
# # of songs (samples) x # of timestamps x tuple of 4 (combination of notes
# and rhythms of both hands)
print("Whole music data size:", np.array(songs_data).shape)
data_to_int, int_to_data = get_dictionary(songs_data)
print("Number of unique notes:", len(data_to_int))


# PART.2 #####################################################################
# train data which is divided into specified number of batches in LSTM

# hyperparameters ############################################################
INPUT = 100
HIDDEN = 256
vocab_size = len(data_to_int)
OUTPUT = vocab_size
LEARNING_RATE = 0.005
BETA1 = 0.9
BETA2 = 0.999
#BATCH_SIZE = 50
BATCH_SIZE = 2

# activation functions #######################################################
def sigmoid(X):
    return 1. / (1 + np.exp(-X))

def softmax(X):
    exp_X = np.exp(X)
    return exp_X / np.sum(exp_X, axis=1).reshape(-1, 1)

def tanh(X):
    return np.tanh(X)

def dtanh(X):
    return 1 - X**2

# LSTM #######################################################################
def initialize_parameters():
    parameters = {}
    parameters['fgw'] = np.random.normal(0,0.01,(INPUT+HIDDEN,HIDDEN))
    parameters['igw'] = np.random.normal(0,0.01,(INPUT+HIDDEN,HIDDEN))
    parameters['ogw'] = np.random.normal(0,0.01,(INPUT+HIDDEN,HIDDEN))
    parameters['ggw'] = np.random.normal(0,0.01,(INPUT+HIDDEN,HIDDEN))
    parameters['how'] = np.random.normal(0,0.01,(HIDDEN,OUTPUT)) 
    return parameters


def initialize_V(parameters):
    V = {}
    V['vfgw'] = np.zeros(parameters['fgw'].shape)
    V['vigw'] = np.zeros(parameters['igw'].shape)
    V['vogw'] = np.zeros(parameters['ogw'].shape)
    V['vggw'] = np.zeros(parameters['ggw'].shape)
    V['vhow'] = np.zeros(parameters['how'].shape)
    return V


def initialize_S(parameters):
    S = {}
    S['sfgw'] = np.zeros(parameters['fgw'].shape)
    S['sigw'] = np.zeros(parameters['igw'].shape)
    S['sogw'] = np.zeros(parameters['ogw'].shape)
    S['sggw'] = np.zeros(parameters['ggw'].shape)
    S['show'] = np.zeros(parameters['how'].shape)
    return S


def get_embeddings(batch_dataset, embeddings):
    embedding_dataset = np.matmul(batch_dataset, embeddings)
    return embedding_dataset


def lstm_cell(batch_dataset, prev_activation_matrix, prev_cell_matrix, parameters):
    concat_dataset = np.concatenate((batch_dataset, prev_activation_matrix), axis=1)
    fa = sigmoid(np.matmul(concat_dataset, parameters['fgw']))
    ia = sigmoid(np.matmul(concat_dataset, parameters['igw']))
    oa = sigmoid(np.matmul(concat_dataset, parameters['ogw']))
    ga = tanh(np.matmul(concat_dataset, parameters['ggw']))
    cell_memory_matrix = np.multiply(fa, prev_cell_matrix) + np.multiply(ia, ga)
    activation_matrix = np.multiply(oa, tanh(cell_memory_matrix))
    lstm_activations = {}
    lstm_activations['fa'] = fa
    lstm_activations['ia'] = ia
    lstm_activations['oa'] = oa
    lstm_activations['ga'] = ga
    return lstm_activations,cell_memory_matrix,activation_matrix


def output_cell(activation_matrix, parameters):
    output_matrix = softmax(np.matmul(activation_matrix, parameters['how'])) 
    return output_matrix


def cal_loss_accuracy(batch_labels, output_cache):
    loss, accuracy, prob = 0, 0, 1
    batch_size = batch_labels[0].shape[0]
    for i in range(1, len(output_cache)+1):
        labels = batch_labels[i]
        pred = output_cache['o' + str(i)]
        prob = np.multiply(prob, np.sum(np.multiply(labels, pred), axis=1).reshape(-1, 1))
        loss += np.sum((np.multiply(labels, np.log(pred)) + np.multiply(1-labels, np.log(1-pred))), axis=1).reshape(-1, 1)
        accuracy += np.array(np.argmax(labels, 1) == np.argmax(pred, 1), dtype=np.float32).reshape(-1, 1)
    #perplexity = np.sum((1 / prob)**(1 / len(output_cache))) / batch_size
    perplexity = 0
    if prob.all() > 0:
        perplexity = np.sum((1 / prob)**(1 / len(output_cache))) / batch_size
    loss = np.sum(loss) * (-1 / batch_size)
    accuracy = (np.sum(accuracy) / (batch_size)) / len(output_cache)
    
    return perplexity, loss, accuracy
    

def forward_propagation(batches, parameters, embeddings):
    batch_size = batches[0].shape[0]
    lstm_cache, activation_cache, cell_cache = {}, {}, {}
    output_cache, embedding_cache = {}, {}
    a0 = np.zeros([batch_size, HIDDEN], dtype=np.float32)
    c0 = np.zeros([batch_size, HIDDEN], dtype=np.float32)
    activation_cache['a0'] = a0
    cell_cache['c0'] = c0
    for i in range(len(batches) - 1):
        batch_dataset = batches[i]
        batch_dataset = get_embeddings(batch_dataset, embeddings)
        embedding_cache['emb' + str(i)] = batch_dataset
        lstm_activations, ct, at = lstm_cell(batch_dataset, a0, c0, parameters)
        ot = output_cell(at, parameters)
        lstm_cache['lstm' + str(i+1)]  = lstm_activations
        activation_cache['a'+str(i+1)] = at
        cell_cache['c' + str(i+1)] = ct
        output_cache['o'+str(i+1)] = ot
        a0 = at
        c0 = ct  
    return embedding_cache, lstm_cache, activation_cache, cell_cache, output_cache


def calculate_output_cell_error(batch_labels, output_cache, parameters):
    output_error_cache, activation_error_cache = {}, {}
    for i in range(1, len(output_cache)+1):
        error_output = output_cache['o' + str(i)] - batch_labels[i]
        error_activation = np.matmul(error_output, parameters['how'].T)
        output_error_cache['eo'+str(i)] = error_output
        activation_error_cache['ea'+str(i)] = error_activation
    return output_error_cache, activation_error_cache


def calculate_single_lstm_cell_error(activation_output_error, next_activation_error,
                                     next_cell_error, parameters, lstm_activation,
                                     cell_activation, prev_cell_activation):
    activation_error = activation_output_error + next_activation_error
    oa = lstm_activation['oa']
    ia = lstm_activation['ia']
    ga = lstm_activation['ga']
    fa = lstm_activation['fa']
    eo = np.multiply(np.multiply(np.multiply(activation_error, tanh(cell_activation)), oa), 1-oa)
    cell_error = np.multiply(np.multiply(activation_error, oa), dtanh(tanh(cell_activation)))
    cell_error += next_cell_error
    ei = np.multiply(np.multiply(np.multiply(cell_error, ga), ia), 1-ia)
    eg = np.multiply(np.multiply(cell_error, ia), dtanh(ga))
    ef = np.multiply(np.multiply(np.multiply(cell_error, prev_cell_activation), fa), 1-fa)
    prev_cell_error = np.multiply(cell_error, fa)
    embed_activation_error = np.matmul(ef, parameters['fgw'].T)
    embed_activation_error += np.matmul(ei, parameters['igw'].T)
    embed_activation_error += np.matmul(eo, parameters['ggw'].T)
    embed_activation_error += np.matmul(eg, parameters['ogw'].T)
    input_units = parameters['fgw'].shape[0] - parameters['fgw'].shape[1]
    prev_activation_error = embed_activation_error[:, input_units:]
    embed_error = embed_activation_error[:, :input_units]
    lstm_error = {}
    lstm_error['ef'] = ef
    lstm_error['ei'] = ei
    lstm_error['eo'] = eo
    lstm_error['eg'] = eg
    return prev_activation_error, prev_cell_error, embed_error, lstm_error


def backward_propagation(batch_labels, embedding_cache, lstm_cache,
                         activation_cache, cell_cache, output_cache, parameters):
    output_error_cache, activation_error_cache = calculate_output_cell_error(batch_labels, output_cache, parameters)
    lstm_error_cache, embedding_error_cache = {}, {}
    eat = np.zeros(activation_error_cache['ea1'].shape)
    ect = np.zeros(activation_error_cache['ea1'].shape)
    for i in range(len(lstm_cache), 0, -1):
        pae, pce, ee, le = calculate_single_lstm_cell_error(activation_error_cache['ea'+str(i)], eat, ect, parameters, lstm_cache['lstm'+str(i)], cell_cache['c'+str(i)], cell_cache['c'+str(i-1)])
        lstm_error_cache['elstm'+str(i)] = le
        embedding_error_cache['eemb'+str(i-1)] = ee
        eat = pae
        ect = pce
    derivatives = {}
    derivatives['dhow'] = calculate_output_cell_derivatives(output_error_cache, activation_cache, parameters)
    lstm_derivatives = {}
    for i in range(1, len(lstm_error_cache)+1):
        lstm_derivatives['dlstm'+str(i)] = calculate_single_lstm_cell_derivatives(lstm_error_cache['elstm'+str(i)], embedding_cache['emb'+str(i-1)], activation_cache['a'+str(i-1)])
    derivatives['dfgw'] = np.zeros(parameters['fgw'].shape)
    derivatives['digw'] = np.zeros(parameters['igw'].shape)
    derivatives['dogw'] = np.zeros(parameters['ogw'].shape)
    derivatives['dggw'] = np.zeros(parameters['ggw'].shape)
    for i in range(1, len(lstm_error_cache)+1):
        derivatives['dfgw'] += lstm_derivatives['dlstm'+str(i)]['dfgw']
        derivatives['digw'] += lstm_derivatives['dlstm'+str(i)]['digw']
        derivatives['dogw'] += lstm_derivatives['dlstm'+str(i)]['dogw']
        derivatives['dggw'] += lstm_derivatives['dlstm'+str(i)]['dggw']
    return derivatives, embedding_error_cache


def calculate_output_cell_derivatives(output_error_cache, activation_cache, parameters):
    dhow = np.zeros(parameters['how'].shape)
    batch_size = activation_cache['a1'].shape[0]
    for i in range(1, len(output_error_cache)+1):
        output_error = output_error_cache['eo' + str(i)]
        activation = activation_cache['a'+str(i)]
        dhow += np.matmul(activation.T,output_error)/batch_size
    return dhow


def calculate_single_lstm_cell_derivatives(lstm_error, embedding_matrix, activation_matrix):
    concat_matrix = np.concatenate((embedding_matrix, activation_matrix), axis=1) 
    batch_size = embedding_matrix.shape[0]
    derivatives = {}
    derivatives['dfgw'] = np.matmul(concat_matrix.T, lstm_error['ef']) / batch_size
    derivatives['digw'] = np.matmul(concat_matrix.T, lstm_error['ei']) / batch_size
    derivatives['dogw'] = np.matmul(concat_matrix.T, lstm_error['eo']) / batch_size
    derivatives['dggw'] = np.matmul(concat_matrix.T, lstm_error['eg']) / batch_size
    return derivatives


def update_parameters(parameters, derivatives, V, S):
    vfgw = BETA1 * V['vfgw'] + (1 - BETA1) * derivatives['dfgw']
    vigw = BETA1 * V['vigw'] + (1 - BETA1) * derivatives['digw']
    vogw = BETA1 * V['vogw'] + (1 - BETA1) * derivatives['dogw']
    vggw = BETA1 * V['vggw'] + (1 - BETA1) * derivatives['dggw']
    vhow = BETA1 * V['vhow'] + (1 - BETA1) * derivatives['dhow']
    sfgw = BETA2 * S['sfgw'] + (1 - BETA2) * derivatives['dfgw']**2
    sigw = BETA2 *S['sigw'] + (1 - BETA2) * derivatives['digw']**2
    sogw = BETA2 *S['sogw'] + (1 - BETA2) * derivatives['dogw']**2
    sggw = BETA2 * S['sggw'] + (1 - BETA2) * derivatives['dggw']**2
    show = BETA2 * S['show'] + (1 - BETA2) * derivatives['dhow']**2
    parameters['fgw'] -= LEARNING_RATE * (vfgw / (np.sqrt(sfgw) + 1e-6))
    parameters['igw'] -= LEARNING_RATE * (vigw / (np.sqrt(sigw) + 1e-6))
    parameters['ogw'] -= LEARNING_RATE * (vogw / (np.sqrt(sogw) + 1e-6))
    parameters['ggw'] -= LEARNING_RATE * (vggw / (np.sqrt(sggw) + 1e-6))
    parameters['how'] -= LEARNING_RATE * (vhow / (np.sqrt(show) + 1e-6))
    V['vfgw'], V['vigw'], V['vogw'], V['vggw'], V['vhow'] = vfgw, vigw, vogw, vggw, vhow
    S['sfgw'], S['sigw'], S['sogw'], S['sggw'], S['show'] = sfgw, sigw, sogw, sggw, show
    return parameters, V, S


def update_embeddings(embeddings, embedding_error_cache, batch_labels):
    embedding_derivatives = np.zeros(embeddings.shape)
    batch_size = batch_labels[0].shape[0]
    for i in range(len(embedding_error_cache)):
        embedding_derivatives += np.matmul(batch_labels[i].T, embedding_error_cache['eemb'+str(i)]) / batch_size
    embeddings = embeddings - LEARNING_RATE * embedding_derivatives
    return embeddings


# train ######################################################################
    
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

def train(train_dataset, iters):
    # initalize the parameters
    parameters = initialize_parameters()
    # initialize the parameters for Adam optimizer
    V = initialize_V(parameters)
    S = initialize_S(parameters)
    # generate the random embeddings
    embeddings = np.random.normal(0, 0.01, (vocab_size, INPUT))
    # Loss, Perplexity and Accuracy for each batch
    L, P, A = [], [], []
    for step in range(iters):
        # get batch dataset
        index = step % len(train_dataset)
        batches = train_dataset[index]
        # forward propagation
        embedding_cache, lstm_cache, activation_cache, cell_cache, output_cache = forward_propagation(batches, parameters, embeddings)
        # calculate the loss, perplexity and accuracy
        perplexity, loss, acc = cal_loss_accuracy(batches, output_cache)
        # backward propagation
        derivatives, embedding_error_cache = backward_propagation(batches, embedding_cache, lstm_cache, activation_cache, cell_cache, output_cache, parameters) 
        # update the parameters
        parameters, V, S = update_parameters(parameters, derivatives, V, S)
        # update the embeddings
        embeddings = update_embeddings(embeddings, embedding_error_cache, batches)
        # print error measures every 100 epochs
        L.append(loss)
        P.append(perplexity)
        A.append(acc)
        if(step % 10 == 0):
            print("For Single Batch :")
            print('Step       = {}'.format(step))
            print('Loss       = {}'.format(round(loss,2)))
            print('Perplexity = {}'.format(round(perplexity,2)))
            print('Accuracy   = {}'.format(round(acc*100,2)))
            print()
    return embeddings, parameters, L, P, A

train_set = get_batches(songs_data, data_to_int)
embeddings, parameters, L, P, A = train(train_set, 50)

# plot
avg_loss, avg_acc, avg_perp = [], [], []
i = 0
while(i < len(L)):
    avg_loss.append(np.mean(L[i:i+50]))
    avg_acc.append(np.mean(A[i:i+50]))
    avg_perp.append(np.mean(P[i:i+50]))
    i += 50

plt.plot(list(range(len(avg_loss))), avg_loss)
plt.xlabel("Iteration of 50 batches")
plt.ylabel("Average Loss")
plt.title("Average Loss of Each 50 Batches")
plt.show()

plt.plot(list(range(len(avg_perp))), avg_perp)
plt.xlabel("Iteration of 50 batches")
plt.ylabel("Average Perplexity")
plt.title("Average Perplexity of Each 50 Batches")
plt.show()

plt.plot(list(range(len(avg_acc))), avg_acc)
plt.xlabel("Iteration of 50 batches")
plt.ylabel("Average Accuracy")
plt.title("Average Accuracy of Each 50 Batches")
plt.show()

# PART.3 #####################################################################
# predict data from random initial value using the trained LSTM

def predict(parameters, embeddings, idx2note, vocab_size):
    out_notes = []    
    # produce 10 pieces of music
    for i in range(10):
        a0 = np.zeros([1, HIDDEN], dtype=np.float32)
        c0 = np.zeros([1, HIDDEN], dtype=np.float32)
        notes = []
        batch_dataset = np.zeros([1, vocab_size])
        # get random start note
        index = np.random.randint(0, vocab_size, 1)[0]     
        batch_dataset[0, index] = 1.0
        # add first note to the generating piece
        notes.append(idx2note[index])
        # get actual note from idx2note dict
        note = idx2note[index]
        while(note != (None, None, None, None)):
            # get embeddings
            batch_dataset = get_embeddings(batch_dataset, embeddings)
            # lstm cell
            lstm_activations, ct, at = lstm_cell(batch_dataset, a0, c0, parameters)
            # output cell
            ot = output_cell(at, parameters)
            # select random.choice with output distribution!
            # this helps eliminating repetition and gives more diversity
            # better result than pred = np.argmax(ot)
            pred = np.random.choice(vocab_size, 1, p=ot[0])[0]         
            # add note to song
            notes.append(idx2note[pred])
            note = idx2note[pred]
            # change the batch_dataset to this new predicted note
            batch_dataset = np.zeros([1, vocab_size])
            batch_dataset[0, pred] = 1.0
            # update new 'at' and 'ct' for next lstm cell
            a0 = at
            c0 = ct
        out_notes.append(notes)
    return out_notes

pred_notes = predict(parameters, embeddings, int_to_data, vocab_size)

# write the generated songs as midi files
count = 0
for piece in pred_notes:
    p1 = stream.Part()
    p1.insert(0, instrument.Piano())
    p2 = stream.Part()
    p2.insert(0, instrument.Piano())
    for item in piece:
        if item != (None, None, None, None):
            if item[0] != None and item[1] != None:
                if '.' in item[0]:
                    chord_pitches = item[0].split('.')
                    p1.append(chord.Chord(chord_pitches, quarterLength = item[1]))
                else:
                    p1.append(note.Note(item[0], quarterLength = item[1]))
            if item[2] != None and item[3] != None:
                if '.' in item[2]:
                    chord_pitches = item[2].split('.')
                    p2.append(chord.Chord(chord_pitches, quarterLength = item[3]))
                else:
                    p2.append(note.Note(item[2], quarterLength = item[3]))
    s = stream.Stream([p1, p2])
    mid = midi.translate.streamToMidiFile(s)
    mid.open('out'+str(count)+'.mid', 'wb')
    mid.write()
    mid.close()
    print("saved generated song! check your directory.")
    count += 1