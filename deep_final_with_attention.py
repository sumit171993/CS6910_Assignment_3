

import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import array as arr
import wandb
from torch.utils.data import Dataset
import random
import csv 
import re

GPU_fr_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#For Indexing Each "Akshars" in Hindi language to each Indexes
padding = 'padding'
Akshar_hindi = [chr(akshar) for akshar in range(2304, 2432)]
Akshar_hindi_size = len(Akshar_hindi)

Akshar2pointer = {padding: 0}
for i, hindi_akshar in enumerate(Akshar_hindi):
    Akshar2pointer[hindi_akshar] = i+1

print(Akshar2pointer)

alphabets = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

English2pointer = {padding: 0}
for i, English_char in enumerate(alphabets):
    English2pointer[English_char] = i+1

print(English2pointer)

regex_english_letters = re.compile('[^a-zA-Z ]')

#In Order to form the clean English Vocabulary. 
def Plain_English_Vocabulary(input_line):
    print("Clean English Vocabulary for cleaning the English Vocab")
    modified_line = input_line.replace('-', ' ').replace(',', ' ').upper()
    print("Cleaned line formation started")
    cleaned_line = regex_english_letters.sub('', modified_line)
    return cleaned_line.split()

def Shudh_hindi_shabd_bhandar(input_line):
    print("Shudh Hindi Vocabulary for cleaning the Hindi Vocab")
    modified_line = input_line.replace('-', ' ').replace(',', ' ')
    cleaned_line = ''
    for char in modified_line:
        if char in Akshar2pointer or char == ' ':
            cleaned_line += char
    return cleaned_line.split()

#Class defined for Transliteration Purpose
class loadcustomdata(Dataset):
    def __init__(self, name_file):
        print("Starts reading the csv file")
        self.alphabets_english, self.akshars_hindi = self.csv_data(name_file, Shudh_hindi_shabd_bhandar)
        self.pointers = list(range(len(self.alphabets_english)))
        random.shuffle(self.pointers)
        self.pointer_start = 0
        
    def __len__(self):
        return len(self.alphabets_english)
    
    def __getitem__(self, d):
        return self.alphabets_english[d], self.akshars_hindi[d]
    
    def csv_data(self,name_file, cleaner_vocab):
        print("Accessing function for reading CSV data")
        English_words = []
        Hindi_words = []
        with open(name_file, 'r',encoding="utf8") as csvfile:
            csvreader = csv.reader(csvfile)
            for line in csvreader:
                print("Reading word list 1 i.e. English Words")
                wordlist1 = Plain_English_Vocabulary(line[0])
                print("Reading word list 2 i.e. Hindi Words")
                wordlist2 = cleaner_vocab(line[1])

                if len(wordlist1) != len(wordlist2):
                    print('Skipping: ', line[0], ' - ', line[1])
                    continue

                English_words.extend(wordlist1)
                Hindi_words.extend(wordlist2)

        return English_words, Hindi_words
    
    def get_random_sample(self):
        return self.__getitem__(np.random.randint(len(self.alphabets_english)))
    
    def batch_formation(self, batch_size, array):
        end = self.pointer_start + batch_size
        batch = []
        if end >= len(self.alphabets_english):
            batch = [array[i] for i in self.pointers[0:end % len(self.alphabets_english)]]
            end = len(self.alphabets_english)
        return batch + [array[i] for i in self.pointers[self.pointer_start:end]]
    
    def get_batch(self, batch_size, postprocess=True):
        print("Function for getting batches")
        print("Batches formation for English")
        batch_formation_english = self.batch_formation(batch_size, self.alphabets_english)
        print("Batches formation for Hindi")
        batch_formation_hindi = self.batch_formation(batch_size, self.akshars_hindi)
        self.pointer_start = self.pointer_start + batch_size + 1
        
        if self.pointer_start >= len(self.alphabets_english):
            random.shuffle(self.pointers)
            self.pointer_start = 0
            
        return batch_formation_english, batch_formation_hindi

data_training = loadcustomdata('hin_valid.csv')
data_testing = loadcustomdata('hin_test.csv')

print("Size for Training Set:\t", len(data_training))
print("Size for Testing Set:\t", len(data_testing))

print('\nSome "English_Words" and "Hindi_Shabd" from Training Data-set:')
for k in range(10):
    Words, Shabd = data_training.get_random_sample()
    print(Words + ' , ' + Shabd)

def generate_word_representation(word, char_to_index, device='cpu'):
    print("Generating the Representation")
    Representation = torch.zeros(len(word) + 1, 1, len(char_to_index)).to(device)
    for pointers_fr_letter, chara in enumerate(word):
        if chara in char_to_index:
            pos = char_to_index[chara]
            Representation[pointers_fr_letter][0][pos] = 1
    pad_pos = char_to_index.get(padding, 0)
    Representation[pointers_fr_letter + 1][0][pad_pos] = 1
    return Representation

def generate_ground_truth_representation(word, char_to_index, device='cpu'):
    print("Printing the Ground Truth Representation")
    ground_truth_representation = torch.zeros([len(word) + 1, 1], dtype=torch.long).to(device)
    for pointers_fr_letter, chara in enumerate(word):
        if chara in char_to_index:
            pos = char_to_index[chara]
            ground_truth_representation[pointers_fr_letter][0] = pos
    pad_pos = char_to_index.get(padding, 0)
    ground_truth_representation[pointers_fr_letter + 1][0] = pad_pos
    return ground_truth_representation

#For Demonstarting Each English-Hindi Pairs in Training data, by selecting them randomly and demonstrating,
#its corresponding tensor representation 
ENGLISH, HINDI = data_training.get_random_sample()#For getting Random Samples From Training Data
Representation_in_English = generate_word_representation(ENGLISH, English2pointer)
print(ENGLISH, Representation_in_English)

#For Demonstarting Each English-Hindi Pairs in Training data, by selecting them randomly and demonstrating,
#its corresponding tensor representation 
Generate_Representation_fr_Hindi = generate_word_representation(HINDI, Akshar2pointer)
print(HINDI, Generate_Representation_fr_Hindi)

class CustomEncoderDecoder(nn.Module):
    
    def __init__(self, size_input, size_hidden, size_output,type_cell,num_layers,dropout):
        
        super(CustomEncoderDecoder, self).__init__()
        self.type_cell = type_cell 
        self.size_hidden = size_hidden
        self.size_output = size_output
        self.num_layers = num_layers
        self.dropout = dropout
        if type_cell == 'gru':
            print("Cell type is GRU")
            self.cell_encoder = nn.GRU(size_input, size_hidden,num_layers = num_layers,dropout = dropout)
            self.cell_decoder = nn.GRU(size_output, size_hidden,num_layers = num_layers,dropout = dropout)
        elif type_cell == 'lstm':
            print("Cell type is LSTM")
            self.cell_encoder = nn.LSTM(size_input, size_hidden,num_layers = num_layers,dropout = dropout)
            self.cell_decoder = nn.LSTM(size_output, size_hidden,num_layers = num_layers,dropout = dropout)
        else:
            print("Cell type is RNN")
            self.cell_encoder = nn.RNN(size_input, size_hidden,num_layers = num_layers,dropout = dropout)
            self.cell_decoder = nn.RNN(size_output, size_hidden,num_layers = num_layers,dropout = dropout)
        self.hidden_conn_output = nn.Linear(size_hidden, size_output)
        self.softmax = nn.LogSoftmax(dim=2)
        
    def forward(self, pt, chars_maximum_out = 30, device = 'cpu', ground_truth = None):
        print("Forward Function Part for Encoder")
        if self.type_cell == 'lstm':
            O_e, (hi,cl) = self.cell_encoder(pt)
        else:
            O_e, hi = self.cell_encoder(pt)
  
        print('Input for Encoder is', pt.shape)
        print('Output from Encoder is', O_e.shape)
        print('Shape of Encoder Hidden layer is', hi.shape)
        
        print("Forward Function Part for decoder")
        fr_decoder_state = hi
        if self.type_cell == 'lstm':
            cell_decoder = cl
        
        fr_decoder_input = torch.zeros(1, 1, self.size_output).to(device)
        ots = []
        print('State of Decoder is', fr_decoder_state.shape)
        print('Input for Decoder is', fr_decoder_input.shape)
        for k in range(chars_maximum_out):
            if self.type_cell == 'lstm':
                ot, (fr_decoder_state,cell_decoder) = self.cell_decoder(fr_decoder_input, (fr_decoder_state,cell_decoder))
            else:
                ot, fr_decoder_state = self.cell_decoder(fr_decoder_input, fr_decoder_state)
            ot = self.hidden_conn_output(fr_decoder_state)
            ot = self.softmax(ot)
            ots.append(ot.view(1, -1))
            
            pointr_max = torch.argmax(ot, 2, keepdim=True)
            if not ground_truth is None:
                pointr_max = ground_truth[k].reshape(1, 1, 1)
            hot_one = torch.FloatTensor(ot.shape).to(device)
            hot_one.zero_()
            hot_one.scatter_(2, pointr_max, 1)
            decoder_input = hot_one.detach()
        return ots

def conjecture(model, input_word,out_characters_limit, device='cpu'):
    print("Evaluation of the model")
    model.eval().to(device)
    print("English Representation")
    Representation_English = generate_word_representation(input_word, English2pointer)
    fo = model(Representation_English, out_characters_limit)
    return fo

class CustomEncoderDecoder_Attention(nn.Module):
    def __init__(self, size_input, size_hidden, size_output,type_cell):
        super(CustomEncoderDecoder_Attention, self).__init__()
        self.type_cell = type_cell 
        self.size_output = size_output
        self.size_hidden = size_hidden
        if type_cell == 'lstm':
            self.cell_encoder = nn.LSTM(size_input, size_hidden)
            self.cell_decoder = nn.LSTM(size_hidden*2, size_hidden)
        elif type_cell == 'gru':
            self.cell_encoder = nn.GRU(size_input, size_hidden)
            self.cell_decoder = nn.GRU(size_hidden*2, size_hidden)
        else:
            self.cell_encoder = nn.RNN(size_input, size_hidden)
            self.cell_decoder = nn.RNN(size_hidden*2, size_hidden)

        print("Encoder-Decoder Class Definition")
        self.hiddenconnoutput = nn.Linear(size_hidden, size_output)
        self.softmax = nn.LogSoftmax(dim=2)
        self.dw = nn.Linear(self.size_hidden, self.size_hidden)
        self.yu = nn.Linear(self.size_hidden, self.size_hidden)
        self.Attention = nn.Linear(self.size_hidden, 1)
        self.Outputconnhidden = nn.Linear(self.size_output, self.size_hidden)   
        
    def forward(self, pt, chars_otpt_max = 30, device = 'cpu', ground_truth = None):
        print("Encoder Part")
        if self.type_cell == 'lstm':
            O_e, (hi,cl) = self.cell_encoder(pt)
        else:
            O_e, hi = self.cell_encoder(pt)
        O_e = O_e.view(-1, self.size_hidden)
        print('Output From the Encoder with Attention into Consideration', O_e.shape)
        # Decoder_Part
        fr_state_decoder = hi
        fr_in_decoder = torch.zeros(1, 1, self.size_output).to(device)

        if self.type_cell == 'lstm':
            cell_decoder = cl
        otpts = []
        yu = self.yu(O_e)
        
        print('Intermediate decoder input', fr_in_decoder.shape)
        print('Encoder output* U', yu.shape)
        print('State of Decoder is', fr_state_decoder.shape)
        for k in range(chars_otpt_max):
            dw = self.dw(fr_state_decoder.view(1, -1).repeat(O_e.shape[0], 1))
            vi = self.Attention(torch.tanh(yu + dw))
            Weights_fr_attn  = F.softmax(vi.view(1, -1), dim = 1) 
            
            embedding = self.Outputconnhidden(fr_in_decoder)
            Applied_attn = torch.bmm(Weights_fr_attn.unsqueeze(0),O_e.unsqueeze(0))
            fr_input_decoder = torch.cat((embedding[0], Applied_attn[0]), 1).unsqueeze(0)
            
            if self.type_cell == 'lstm':
                ot, (fr_state_decoder,cell_decoder) = self.cell_decoder(fr_input_decoder, (fr_state_decoder,cell_decoder))
            else:
                ot, fr_state_decoder = self.cell_decoder(fr_input_decoder, fr_state_decoder)
            
            print("The Output from the decoder is")
            ot = self.hiddenconnoutput(fr_state_decoder)
            ot = self.softmax(ot)
            otpts.append(ot.view(1, -1))
            print("Definition of a one hot vector")
            hot_one = torch.FloatTensor(ot.shape).to(device)
            print("Maximum_Index_Assignment")
            pointr_max = torch.argmax(ot, 2, keepdim=True)
            if not ground_truth is None:
               pointr_max = ground_truth[k].reshape(1, 1, 1)
            hot_one.zero_()
            hot_one.scatter_(2, pointr_max, 1) 
            decoder_input = hot_one.detach()
    
    
        return otpts

def t_batch(netwrk,method_opti, rule, size_batch, device = 'cpu'):
    print("Function for training Batches start here")
    method_opti.zero_grad()
    netwrk.train().to(device)
    print("Spliting the Data into English and Hindi Batches")
    batch_english, batch_hindi = data_training.get_batch(size_batch)
    total_privation = 0
    for j in range(size_batch):
        print("Ground Truth Generation")
        Ground_truth = generate_ground_truth_representation(batch_hindi[j], Akshar2pointer, device)
        print("Prediction Generation")
        pt = generate_word_representation(batch_english[j], English2pointer, device)
        yields = netwrk(pt, Ground_truth.shape[0], device)
        
        for pointr, opt in enumerate(yields):
            privation = rule(opt, Ground_truth[pointr]) /size_batch
            privation.backward(retain_graph = True)
            total_privation = total_privation + privation
    method_opti.step()
    return total_privation/size_batch

def fr_training(netwrk, learning_rate , batches_number , size_batch , device = 'cpu'):
    print("Here is the train function definition starts")
    netwrk = netwrk.to(device)
    print("Defining the criteria i.e. Negative Log-liklihood")
    rule = nn.NLLLoss()
    method_opti = optim.Adam(netwrk.parameters(), lr=learning_rate)
    fr_loss_arr = np.zeros(batches_number + 1)
    for k in range(batches_number):
        print("The loss matrix calculation starts as follows")
        fr_loss_arr[k+1] = (fr_loss_arr[k]*k + t_batch(netwrk,method_opti, rule, size_batch, device = device))/(k + 1)
        wandb.log({'training_loss':fr_loss_arr[k]})        
        print('ith iteration', k, 'Loss is', fr_loss_arr[k])
    torch.save(netwrk, 'netwrk.pt')
    return fr_loss_arr

def accuracy_calc(netwrk, device = 'cpu'):
    print("Accuracy Calculations starts here")
    estimations = []
    Pre = 0
    netwrk = netwrk.eval().to(device)
    for j in range(len(data_testing)):
        true = 0
        English, Hindi = data_testing[j]
        Ground_truth = generate_ground_truth_representation(Hindi, Akshar2pointer, device)
        yields = conjecture(netwrk, English, Ground_truth.shape[0], device)
        print("Prediction Calculation started")
        for pointr, i in enumerate(yields):
            la, pointers = i.topk(1)
            pos_hindi = pointers.tolist()[0]
            if pos_hindi[0] == Ground_truth[pointr][0]:
                true += 1
        Pre += Pre/Ground_truth.shape[0]
    Pre/= len(data_testing)
    wandb.log({'Testing_accuracy with Attention':Pre*100})
    return Pre

sweep_config = {
    'method': 'bayes', 
    'metric': {
      'name': 'validation_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {

        'dropout': {
            'values': [0.0, 0.1, 0.2]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4]
        },
        'size_batch': {
            'values': [64, 128]
        },
        
        'number_of_Encoder_decoder_layers': {
            'values': [1, 2, 3]
        },
        'size_hidden':{
            'values': [128, 256,512]
        },
        'type_cell': {
            'values': ['rnn', 'gru', 'lstm']
        }
    }
}
#API KEY   e65a3753f55c5eca34b4c1ae489d00abea918792

# Initialize a new sweep
sweep_id = wandb.sweep(sweep_config, entity="dl_sumit", project="Deep_Learning_Assignment_3")

def function_sweep():
    wandb.init(config = sweep_config)
  
    config = wandb.config
    wandb.run.name = str(config.type_cell)+ '_' + '_bs_'+str(config.size_batch)
    #model = CustomEncoderDecoder(len(English2pointer),wandb.config.size_hidden, len(Akshar2pointer),wandb.config.type_cell,wandb.config.number_of_Encoder_decoder_layers,wandb.config.dropout)
    #model.to(GPU_fr_device)
    #iterations = 1000
    #Accuracy_wo_attn = accuracy_calc(model) * 100
    #wandb.log({'Accuracy w/o attention':Accuracy_wo_attn})
    #Loss = setup_train(model, learning_rate = wandb.config.learning_rate,batches_number = 2000, size_batch = wandb.config.size_batch)
    
    
    
    Attention_Network = CustomEncoderDecoder_Attention(len(English2pointer),wandb.config.size_hidden, len(Akshar2pointer),wandb.config.type_cell)
    Attention_Network.to(GPU_fr_device)
    iterations = 1000
    Attention_loss = fr_training(Attention_Network, learning_rate = wandb.config.learning_rate, batches_number=2000, size_batch = wandb.config.size_batch, device = GPU_fr_device)
    Accuracy_w_attn = accuracy_calc(Attention_Network) * 100
    wandb.log({'Accuracy with attention':Accuracy_w_attn})
    #Accuracy_wo_attn = accuracy_calc(model) * 100

os.environ['WANDB_NOTEBOOK_NAME'] = 'Deep_final_with_Attention.ipynb'
wandb.login()
sweep_id = wandb.sweep(sweep_config, entity="dl_sumit", project="Deep_Learning_Assignment_3")
wandb.agent(sweep_id, lambda:function_sweep())