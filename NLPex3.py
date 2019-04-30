__authors__ = ['Ahmad Chaar','Abhishek Singh','Rebecca Erbanni']
__emails__  = ['b00739600@essec.edu','b00748269@essec.edu','B00746038@essec.edu']

import argparse
import pickle as pkl
import numpy as np
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import re
import os
from io import open
import itertools
from torch.nn import init
import torch
import torch.autograd as autograd
import torch.nn as nn
from nltk.corpus import brown 
import gensim
import nltk
nltk.download('brown')
#setup the device
USE_CUDA = torch.cuda.is_available()
global device
device = torch.device("cuda" if USE_CUDA else "cpu")

####################PICKBEST#############################################
####################PICKBEST#############################################
####################PICKBEST#############################################
####################PICKBEST#############################################
####################PICKBEST#############################################
####################PICKBEST#############################################


def loadData_part_1(path):
    """
        :param path: containing dialogue data of ConvAI (eg:  train_both_original.txt, valid_both_original.txt)
        :return: for each dialogue, yields (description_of_you, description_of_partner, dialogue) where a dialogue
            is a sequence of (utterance, answer, options)
    """
    """
        :param path: containing dialogue data of ConvAI (eg:  train_both_original.txt, valid_both_original.txt)
        :return: for each dialogue, yields (description_of_you, description_of_partner, dialogue) where a dialogue
            is a sequence of (utterance, answer, options)
    """
    with open(path) as f:
        descYou, descPartner = [], []
        dialogue = []
        your_description=""
        partner_description=""
        
        reset_persona_bol=False
        for l in f:
            l=l.strip()
            lxx = l.split()
            lxx_persona = l.split("persona:")
            
            
            idx = int(lxx[0])
            if idx == 1:
                if len(dialogue) != 0:
                    yield descYou,  descPartner, dialogue
                # reinit data structures
                descYou, descPartner = [], []
                dialogue = []

            if lxx[2] == 'persona:':
                if reset_persona_bol==True:
                    your_description=""
                    partner_description=""
                    reset_persona_bol=False
                    # description of people involved
                if lxx[1] == 'your':
                    description = descYou
                    your_description=your_description+lxx_persona[1]
                    bol_persona=0
                elif lxx[1] == "partner's":
                    description = descPartner
                    partner_description=partner_description+lxx_persona[1]
                else:
                    assert 'Error, cannot recognize that persona ({}): {}'.format(lxx[1],l)
                description.append(lxx[3:])

            else:
                reset_persona_bol=True
                if bol_persona==0:                
                    # the dialogue
                    lxx = l.split('\t')
                    utterance = ' '.join(lxx[0].split()[1:])
                    answer = lxx[1]
                    options_full = [o for o in lxx[-1].split('|')]
                    options = [o for o in lxx[-1].split('|')]
                    options.remove(answer)  
                    option_1,option_2,option_3,option_4,option_5 = random.sample(options,5)                            
                    dialogue.append([idx, str(partner_description + " " + your_description + " " + utterance), answer, options,option_1,option_2,option_3,option_4,option_5,options_full])
                    bol_persona=1
                    

                else: 
                    # the dialogue
                    lxx = l.split('\t')
                    utterance = ' '.join(lxx[0].split()[1:])
                    answer = lxx[1]
                    options_full = [o for o in lxx[-1].split('|')]
                    options = [o for o in lxx[-1].split('|')]
                    options.remove(answer)
                    option_1,option_2,option_3,option_4,option_5 = random.sample(options,5)                            
                    dialogue.append([idx, str(partner_description + " " + your_description + " " + utterance), answer, options,option_1,option_2,option_3,option_4,option_5,options_full])
                    bol_persona=0


def create_id_to_vec(word_to_id: dict, new_model) -> dict:
    """
    Extracts the embedding weights for each word in the vocabulary and maps each word ids to its weight in a dictionary.
    
    Parameters
    ----------
    path_to_glove_weights : str
        Path to the file containing the embedding weights.
    word_to_id : Dict[str, int]
        Vocabulary mapping each word to an unique id.

    Returns
    -------
    id_to_vec : Dict[int, np.ndarray]
        Map of each word id to its embedding form.
    """
    
    id_to_vec = {}
    vector = None
    
    for key, value in new_model.wv.vocab.items():
        word = key
    
        vector = np.array(new_model[word], dtype='float32')
    
        if word in word_to_id:
            id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(vector))
            
    for word, id in word_to_id.items(): 
        if word_to_id[word] not in id_to_vec:
            v = np.zeros(*vector.shape, dtype='float32')
            v[:] = np.random.randn(*v.shape)*0.01
            id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(v))
                
    return id_to_vec



def normalizeString(s):
    try:
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        s = s.replace("'s", ' is')
        s = s.replace("n't", ' not')
        s = s.replace("'re", ' are')
        s = s.replace("'m", ' am')
        s = s.replace("'ve", ' have')
        s = s.replace("'ll", ' will')
        s = s.replace("'d", ' would')
        s = s.lower()
    except:
        pass
    return s


def clean_up_sentences_part_1(text_train):
    for conv_counter in range(len(text_train)):
        text_train[conv_counter][1]=normalizeString(text_train[conv_counter][1])
        text_train[conv_counter][2]=normalizeString(text_train[conv_counter][2])
        text_train[conv_counter][4]=normalizeString(text_train[conv_counter][4])
        text_train[conv_counter][5]=normalizeString(text_train[conv_counter][5])
        text_train[conv_counter][6]=normalizeString(text_train[conv_counter][6])
        text_train[conv_counter][7]=normalizeString(text_train[conv_counter][7])        
    return text_train


class Voc_part_1:
#PAD_token = 0  # Used for padding short sentences
#SOS_token = 1  # Start-of-sentence token
#EOS_token = 2  # End-of-sentence token
    def __init__(self):
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):        
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)

#Below function loops on sentences in the pairs and 
#populates the voc class, the voc class
#will be repopulated again after trimming for rare words

def prepare_voc_part_1(text_train,voc):

    for conv_counter in range(len(text_train)):
        
        try:
            #In case it does not contain prime, meaning its the second
            #iteration of the same conversation

            float(text_train[conv_counter][0])
            voc.addSentence(text_train[conv_counter][1]) 
        
        except:
            pass
        
        if conv_counter>0 and text_train[conv_counter-1][2]!=text_train[conv_counter][1]:
            voc.addSentence(text_train[conv_counter-1][2]) 

    return voc

#Below function removes rare words
def trimRareWords_part_1(voc, text_train, MIN_COUNT=3):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    list_for_removal=list()
    # Filter out pairs with trimmed words
    for dialogue_counter in range(len(text_train)):  
        input_sentence = text_train[dialogue_counter][1]
        output_sentence = text_train[dialogue_counter][2]
        option_1 = text_train[dialogue_counter][4]
        option_2 = text_train[dialogue_counter][5]
        option_3 = text_train[dialogue_counter][6]
        option_4 = text_train[dialogue_counter][7]
        option_5 = text_train[dialogue_counter][8]
        
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break        

        # Check Options sentence
        for word in option_1.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break        

        # Check Options sentence
        for word in option_2.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break        

        # Check Options sentence
        for word in option_3.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break        

        # Check Options sentence
        for word in option_4.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break        

        # Check Options sentence
        for word in option_5.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break        
        
        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        
        if not (keep_input and keep_output):
            list_for_removal.append(dialogue_counter)
 
    for index in sorted(list_for_removal, reverse=True):
        del text_train[index]

    #populate dictionnaries again    
    voc=prepare_voc_part_1(text_train,voc)
            
    return voc, text_train


def remove_rare_words_test(sentence,voc):
    new_sentence=list()
    sentence = normalizeString(sentence)
    temp_sentence=sentence.split(" ")
    for word in temp_sentence:
        if word in voc.word2index:
            new_sentence.append(word)
    
    new_sentence=" ".join(new_sentence)    

    return new_sentence
        
#The below function replaces words in a sentence with the index (obtained from voc class)
def indexesFromSentence_part_1(voc, sentence,EOS_TOKEN=2):
    return [voc.word2index.get(word,100000) for word in sentence.split(' ')] + [EOS_TOKEN]

#The below function zero pads words after the EOS token (standarize sentence length )
#in the input, the padtoken is set to zero

def zeroPadding_part_1(l, PAD_TOKEN=0):
    return list(itertools.zip_longest(*l, fillvalue=PAD_TOKEN))

#The function below, creates a binary matrix (1 if the word is real, 0 if it's a pad token)
#the mask will be used later 

def binaryMatrix_part_1(l, PAD_TOKEN=0):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == 0:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar_part_1(l, voc):
    indexes_batch = [indexesFromSentence_part_1(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding_part_1(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar_part_1(lst, voc):
    indexes_batch = [indexesFromSentence_part_1(voc, sentence) for sentence in lst]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding_part_1(indexes_batch)
    mask = binaryMatrix_part_1(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


def testVar(sentence, voc):
    indexes_batch = indexesFromSentence_part_1(voc, sentence)
    padVar = torch.LongTensor(indexes_batch)
    return padVar


# Returns input and target tensors for a given pair of sentences, using the functions above
def batch2TrainData_part_1(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch, output_batch_neg_1,output_batch_neg_2,output_batch_neg_3,output_batch_neg_4,output_batch_neg_5 = [], [], [],[],[],[],[]
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
        options.append(pair[2])
        output_batch_neg_1.append(pair[3]) 
        output_batch_neg_2.append(pair[4])
        output_batch_neg_3.append(pair[5]) 
        output_batch_neg_4.append(pair[6]) 
        output_batch_neg_5.append(pair[7])
    
    
    inp, _ = inputVar_part_1(input_batch, voc)
    output, _, _ = outputVar_part_1(output_batch, voc)
    output_neg_1, _, _ = outputVar_part_1(output_batch_neg_1, voc)
    output_neg_2, _, _ = outputVar_part_1(output_batch_neg_2, voc)
    output_neg_3, _, _ = outputVar_part_1(output_batch_neg_3, voc)
    output_neg_4, _, _ = outputVar_part_1(output_batch_neg_4, voc)
    output_neg_5, _, _ = outputVar_part_1(output_batch_neg_5, voc)
        
    
    return inp, output , output_neg_1,output_neg_2,output_neg_3,output_neg_4,output_neg_5


def batch2TestData(voc, sentence):
    sentence = testVar(sentence, voc)    
    return sentence


class Encoder_part_1(nn.Module):
    """LSTM encoder"""

    def __init__(self, emb_size, hidden_size, p_dropout, id_to_vec): 
    
            super(Encoder_part_1, self).__init__()
             
            self.emb_size = emb_size
            self.hidden_size = hidden_size
            self.vocab_size = len(id_to_vec)
            self.p_dropout = p_dropout
       
            self.embedding = nn.Embedding(self.vocab_size+100, self.emb_size)
            self.lstm = nn.LSTM(self.emb_size, self.hidden_size)
            self.dropout_layer = nn.Dropout(self.p_dropout) 

            self.init_weights(id_to_vec)
             
    def init_weights(self, id_to_vec):
        init.uniform_(self.lstm.weight_ih_l0, a = -0.01, b = 0.01)
        init.orthogonal_(self.lstm.weight_hh_l0)
        self.lstm.weight_ih_l0.requires_grad = True
        self.lstm.weight_hh_l0.requires_grad = True
        
        embedding_weights = torch.FloatTensor(self.vocab_size+100, self.emb_size)
            
        for idx, vec in id_to_vec.items():
            embedding_weights[idx] = vec                                
        self.embedding.weight = nn.Parameter(embedding_weights, requires_grad = True)
            
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        _, (last_hidden, _) = self.lstm(embeddings) #dimensions: (num_layers * num_directions x batch_size x hidden_size)
        last_hidden = self.dropout_layer(last_hidden[-1])#access last lstm layer, dimensions: (batch_size x hidden_size)

        return last_hidden


class DualEncoder(nn.Module):
    """Dual LSTM encoder"""
     
    def __init__(self, encoder):
        super(DualEncoder, self).__init__()
        self.encoder = encoder
        self.hidden_size = self.encoder.hidden_size
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)     
        init.xavier_normal_(M)
        self.M = nn.Parameter(M, requires_grad = True)

    def forward(self, context_tensor, response_tensor):
        
        context_last_hidden = self.encoder(context_tensor.transpose(0,1)) #dimensions: (batch_size x hidden_size)
        response_last_hidden = self.encoder(response_tensor.transpose(0,1)) #dimensions: (batch_size x hidden_size)
        
        context = context_last_hidden.mm(self.M).to(device)
        #context = context_last_hidden.mm(self.M) #dimensions: (batch_size x hidden_size)
        context = context.view(-1, 1, self.hidden_size) #dimensions: (batch_size x 1 x hidden_size)
        
        response = response_last_hidden.view(-1,self.hidden_size,1) #dimensions: (batch_size x hidden_size x 1)
        
        score = torch.bmm(context, response).view(-1, 1).to(device)
        #score = torch.bmm(context, response).view(-1, 1) #dimensions: (batch_size x 1 x 1) and lastly --> (batch_size x 1)

        return score

def train_model_part_1(pairs,voc,learning_rate, l2_penalty, nb_epochs, dual_encoder,optimizer,batch_size=256,negative_samples=5,refresh_rate=20):

    loss_func = torch.nn.BCEWithLogitsLoss()
    sum_loss_training=0
    dual_encoder.train()
          
    for epoch in range(nb_epochs):
        
        if epoch%refresh_rate==0:
            training_batches = [batch2TrainData_part_1(voc, [random.choice(pairs)[1:] for _ in range(batch_size)]) for _ in range(refresh_rate)]
            train_iter=0

        input_variable, target_variable,  neg_example_1,neg_example_2,neg_example_3,neg_example_4,neg_example_5 = training_batches[train_iter]
        
        #Postive Exampe
        context = autograd.Variable(torch.LongTensor(input_variable).view(-1,len(input_variable)), requires_grad = False).to(device)
        response = autograd.Variable(torch.LongTensor(target_variable).view(-1, len(target_variable)), requires_grad = False).to(device)   
        label = autograd.Variable(torch.FloatTensor(np.ones(batch_size).reshape(batch_size,1)), requires_grad = False).to(device)

        # Predict
        score = dual_encoder(context, response)
        loss = loss_func(score, label)

        # Train
        sum_loss_training += loss.data.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        

        #Negative Exampe 1 - Batch
        context = autograd.Variable(torch.LongTensor(input_variable).view(-1,len(input_variable)), requires_grad = False).to(device)
        response = autograd.Variable(torch.LongTensor(neg_example_1).view(-1, len(neg_example_1)), requires_grad = False).to(device)   
        label = autograd.Variable(torch.FloatTensor(np.zeros(batch_size).reshape(batch_size,1)), requires_grad = False).to(device)

        # Predict
        score = dual_encoder(context, response)
        loss = loss_func(score, label)

        # Train
        sum_loss_training += loss.data.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if negative_samples > 1:
            #Negative Exampe 2 - Batch
            context = autograd.Variable(torch.LongTensor(input_variable).view(-1,len(input_variable)), requires_grad = False).to(device)
            response = autograd.Variable(torch.LongTensor(neg_example_2).view(-1, len(neg_example_2)), requires_grad = False).to(device)   
            label = autograd.Variable(torch.FloatTensor(np.zeros(batch_size).reshape(batch_size,1)), requires_grad = False).to(device)
    
            # Predict
            score = dual_encoder(context, response)
            loss = loss_func(score, label)
    
            # Train
            sum_loss_training += loss.data.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if negative_samples > 2:
            #Negative Exampe 3 - Batch
            context = autograd.Variable(torch.LongTensor(input_variable).view(-1,len(input_variable)), requires_grad = False).to(device)
            response = autograd.Variable(torch.LongTensor(neg_example_3).view(-1, len(neg_example_3)), requires_grad = False).to(device)   
            label = autograd.Variable(torch.FloatTensor(np.zeros(batch_size).reshape(batch_size,1)), requires_grad = False).to(device)
    
            # Predict
            score = dual_encoder(context, response)
            loss = loss_func(score, label)
    
            # Train
            sum_loss_training += loss.data.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if negative_samples > 3:
            #Negative Exampe 4 - Batch
            context = autograd.Variable(torch.LongTensor(input_variable).view(-1,len(input_variable)), requires_grad = False).to(device)
            response = autograd.Variable(torch.LongTensor(neg_example_4).view(-1, len(neg_example_4)), requires_grad = False).to(device)   
            label = autograd.Variable(torch.FloatTensor(np.zeros(batch_size).reshape(batch_size,1)), requires_grad = False).to(device)
    
            # Predict
            score = dual_encoder(context, response)
            loss = loss_func(score, label)
    
            # Train
            sum_loss_training += loss.data.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if negative_samples > 4:
            #Negative Exampe 5 - Batch
            context = autograd.Variable(torch.LongTensor(input_variable).view(-1,len(input_variable)), requires_grad = False).to(device)
            response = autograd.Variable(torch.LongTensor(neg_example_5).view(-1, len(neg_example_5)), requires_grad = False).to(device)   
            label = autograd.Variable(torch.FloatTensor(np.zeros(batch_size).reshape(batch_size,1)), requires_grad = False).to(device)
    
            # Predict
            score = dual_encoder(context, response)
            loss = loss_func(score, label)
    
            # Train
            sum_loss_training += loss.data.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


#        if epoch%refresh_rate==0:
#            print("finished epoch ",epoch, " with a loss ", loss.item() )

    return dual_encoder

def test_model(pairs,voc, dual_encoder):
#     print("====================Data and Hyperparameter Overview====================\n")
#     print("Number of training examples: %d, Number of validation examples: %d" %(len(training_dataframe), len(validation_dataframe)))
#     print("Learning rate: %.5f, Embedding Dimension: %d, Hidden Size: %d, Dropout: %.2f, L2:%.10f\n" %(learning_rate, emb_dim, encoder.hidden_size, encoder.p_dropout, l2_penalty))
#     print("================================Results...==============================\n")

    dual_encoder.eval()
    option_selected=list()
    correct =0
    n_iter=0
    
    for pair in pairs:        

        sentence=remove_rare_words_test(pair[1],voc)
        options_full=pair[len(pair)-1]
        encoded_sentence = batch2TestData(voc, sentence)        
        option_score_temp=list()
        for x in range(len(options_full)):
        #Postive Exampe
            option = remove_rare_words_test(options_full[x],voc)
            option = batch2TestData(voc, option)
            context = autograd.Variable(torch.LongTensor(encoded_sentence).view(-1,len(encoded_sentence)), requires_grad = False).to(device)
            response = autograd.Variable(torch.LongTensor(option).view(-1, len(option)), requires_grad = False).to(device)   

            # Predict
            score = dual_encoder(context, response)
            option_score_temp.append(score.item())
                
                
        selected_option_arg=np.argmax(np.array(option_score_temp))
        
        option_selected.append(selected_option_arg)
        
        if options_full.index(pair[2]) == selected_option_arg:
            correct = correct + 1
                    
        print(str(pair[0])+ " " + options_full[selected_option_arg])
        
#        n_iter=n_iter+1    
#        if n_iter%20==0:
#            print("Accuracy at example",n_iter," is ",round(correct/len(pairs),2)*100)
                    
    return option_selected    


def creating_model(emb_size, hidden_size, p_dropout, id_to_vec):
    encoder = Encoder_part_1(emb_size, hidden_size, p_dropout, id_to_vec)
    dual_encoder = DualEncoder(encoder)    
    return dual_encoder.to(device)




####################GENERATOR#############################################
####################GENERATOR#############################################
####################GENERATOR#############################################
####################GENERATOR#############################################
####################GENERATOR#############################################
####################GENERATOR#############################################

def loadData(path):
    """
        :param path: containing dialogue data of ConvAI (eg:  train_both_original.txt, valid_both_original.txt)
        :return: for each dialogue, yields (description_of_you, description_of_partner, dialogue) where a dialogue
            is a sequence of (utterance, answer, options)
    """
    with open(path) as f:
        descYou, descPartner = [], []
        dialogue = []
        for l in f:
            l=l.strip()
            lxx = l.split()
            idx = int(lxx[0])
            if idx == 1:
                if len(dialogue) != 0:
                    yield descYou,  descPartner, dialogue
                # reinit data structures
                descYou, descPartner = [], []
                dialogue = []

            if lxx[2] == 'persona:':
                # description of people involved
                if lxx[1] == 'your':
                    description = descYou
                elif lxx[1] == "partner's":
                    description = descPartner
                else:
                    assert 'Error, cannot recognize that persona ({}): {}'.format(lxx[1],l)
                description.append(lxx[3:])

            else:
                # the dialogue
                lxx = l.split('\t')
                utterance = ' '.join(lxx[0].split()[1:])
                answer = lxx[1]
                options = [o for o in lxx[-1].split('|')]
                dialogue.append( (idx, utterance, answer, options))


def normalizeString(s):
    try:
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        s = s.replace("'s", ' is')
        s = s.replace("n't", ' not')
        s = s.replace("'re", ' are')
        s = s.replace("'m", ' am')
        s = s.replace("'ve", ' have')
        s = s.replace("'ll", ' will')
        s = s.replace("'d", ' would')
        s = s.lower()
    except:
        pass
    return s



def clean_up_sentences(text_gen):
    for conv_counter in range(len(text_gen)):
        text_gen[conv_counter][1]=normalizeString(text_gen[conv_counter][1])
        text_gen[conv_counter][2]=normalizeString(text_gen[conv_counter][2])            
    return text_gen


class Voc:
#PAD_token = 0  # Used for padding short sentences
#SOS_token = 1  # Start-of-sentence token
#EOS_token = 2  # End-of-sentence token
    def __init__(self):
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):        
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)


#Below function loops on sentences in the pairs and 
#populates the voc class, the voc class
#will be repopulated again after trimming for rare words

def prepare_voc(text_gen,voc):

    for conv_counter in range(len(text_gen)):
        
        try:
            #In case it does not contain prime, meaning its the second
            #iteration of the same conversation

            float(text_gen[conv_counter][0])
            voc.addSentence(text_gen[conv_counter][1]) 
        
        except:
            pass
        
        if conv_counter>0 and text_gen[conv_counter-1][2]!=text_gen[conv_counter][1]:
            voc.addSentence(text_gen[conv_counter-1][2]) 

    return voc


#Below function removes rare words
def trimRareWords(voc, text_gen, MIN_COUNT=3):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    list_for_removal=list()
    # Filter out pairs with trimmed words
    for dialogue_counter in range(len(text_gen)):  
        input_sentence = text_gen[dialogue_counter][1]
        output_sentence = text_gen[dialogue_counter][2]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break
        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        
        if not (keep_input and keep_output):
            list_for_removal.append(dialogue_counter)
 
    for index in sorted(list_for_removal, reverse=True):
        del text_gen[index]

    #populate dictionnaries again    
    voc=prepare_voc(text_gen,voc)
            
    return voc, text_gen

#The below function replaces words in a sentence with the index (obtained from voc class)
def indexesFromSentence(voc, sentence,EOS_TOKEN=2):
    pass
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_TOKEN]

#The below function zero pads words after the EOS token (standarize sentence length )
#in the input, the padtoken is set to zero

def zeroPadding(l, PAD_TOKEN=0):
    return list(itertools.zip_longest(*l, fillvalue=PAD_TOKEN))

#The function below, creates a binary matrix (1 if the word is real, 0 if it's a pad token)
#the mask will be used later 

def binaryMatrix(l, PAD_TOKEN=0):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == 0:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns input and target tensors for a given pair of sentences, using the functions above
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


#Encoder Part of the RNN:
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)



class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden

#Since we are dealing with batches of padded sequences, we cannot simply consider all elements 
#of the tensor when calculating loss. We define maskNLLLoss to calculate our loss based on our
# decoder’s output tensor, the target tensor, and a binary mask tensor describing the padding 
#of the target tensor. This loss function calculates the average negative log likelihood of 
#the elements that correspond to a 1 in the mask tensor.
        
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()



#The ``train`` function contains the algorithm for a single training
#iteration (a single batch of inputs).
#
#We will use a couple of clever tricks to aid in convergence:
#
#-  The first trick is using **teacher forcing**. This means that at some
#   probability, set by ``teacher_forcing_ratio``, we use the current
#   target word as the decoder’s next input rather than using the
#   decoder’s current guess. This technique acts as training wheels for
#   the decoder, aiding in more efficient training. However, teacher
#   forcing can lead to model instability during inference, as the
#   decoder may not have a sufficient chance to truly craft its own
#   output sequences during training. Thus, we must be mindful of how we
#   are setting the ``teacher_forcing_ratio``, and not be fooled by fast
#   convergence.
#
#-  The second trick that we implement is **gradient clipping**. This is
#   a commonly used technique for countering the “exploding gradient”
#   problem. In essence, by clipping or thresholding gradients to a
#   maximum value, we prevent the gradients from growing exponentially
#   and either overflow (NaN), or overshoot steep cliffs in the cost
#   function.


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip,SOS_token=1,teacher_forcing_ratio=1):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()



def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, n_iteration, batch_size,  clip):
    global loss

    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs)[1:] for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    # Initializations
    start_iteration = 1

    # Training loop
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)




class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.SOS_token=1

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * self.SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores
    
    
def evaluate(encoder, decoder, searcher, voc, sentence,MAX_LENGTH=20):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths,MAX_LENGTH)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(idx, utterance,encoder, decoder, searcher, voc):
    input_sentence = utterance
    try:
        # Normalize sentence
        input_sentence = normalizeString(input_sentence)
        # Evaluate sentence
        output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
        # Format and print response sentence
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        print(idx, ' '.join(output_words))

    except KeyError:

        print(idx, "Matthias, Julien, there are some words that I don't understand, could you please rephrase.")    


class Model_Paramaters:
    def __init__(self):
        self.model_name = 'cb_model'
        #attn_model = 'dot'
#        self.attn_model = 'general'
        self.attn_model = 'concat'
        self.hidden_size = 500
        self.encoder_n_layers = 2
        self.decoder_n_layers = 2
        self.dropout = 0.1
        self.batch_size = 64

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path to model file (for saving/loading)', required=True)    
    parser.add_argument('--text', help='path to text file (for training/testing)', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--test', action='store_true')
    parser.add_argument('--gen', help='enters generative mode',action='store_true',default=False)
    opts = parser.parse_args()

        
    if opts.train:
        #########################################TRAIN PICK BEST##############
        #########################################TRAIN PICK BEST##############
        #########################################TRAIN PICK BEST##############
        #########################################TRAIN PICK BEST##############
        #########################################TRAIN PICK BEST##############
        
        
        EMBEDDING_DIM = 50
        
        model = gensim.models.Word2Vec(brown.sents(),size=50)
        directory = os.path.join(opts.model, "brown.embedding")
        model.save(directory)
        new_model = gensim.models.Word2Vec.load(directory)
        
        text_train=[]
                
        for your_persona,partner_persona, dialogue in loadData_part_1(opts.text):
            for idx, utterance, answer,options,option_1,option_2,option_3,option_4,option_5,options_full in dialogue:
                text_train.append([idx, utterance, answer,options,option_1,option_2,option_3,option_4,option_5,options_full])
            
        #Cleans Up Sentences (lower case and replaces inconsistencies)
        text_train=clean_up_sentences_part_1(text_train)
        
        #Creates Voc class which contains 
        #indicies for each word dictionnaries (word2index and index2word)
        #occurence count for each word        
        
        voc_part_1=Voc_part_1()
        voc_part_1=prepare_voc_part_1(text_train,voc_part_1)
        
        ##removes rare words from the corpus, repopulates voc dictionnaries
        voc_part_1,text_train=trimRareWords_part_1(voc_part_1, text_train,3)
        
        
        # Training paramaters
        learning_rate = 1e-4
        l2_penalty = 1e-4
        nb_epochs = 200000
        
        id_to_vec = create_id_to_vec(voc_part_1.word2index, new_model)
        USE_CUDA = torch.cuda.is_available()
        device = torch.device("cuda" if USE_CUDA else "cpu")
        encoder = creating_model(EMBEDDING_DIM, 50, 0.1, id_to_vec)        
        optimizer = torch.optim.Adam(encoder.parameters(), lr = learning_rate, weight_decay = l2_penalty)        
        # launch training
        dual_encoder = train_model_part_1(text_train,voc_part_1,learning_rate, l2_penalty, nb_epochs, encoder,optimizer)                
        torch.save({'dual_encoder': dual_encoder,'voc_1': voc_part_1}, os.path.join(opts.model, 'part_1.tar'))

                        
        ########################################TRAIN GEN#####################
        ########################################TRAIN GEN#####################
        ########################################TRAIN GEN#####################
        ########################################TRAIN GEN#####################
        ########################################TRAIN GEN#####################
        #Training for Text Generation
        text_gen=[]
        previous_answer="!not#an#answer!"

        for _,_, dialogue in loadData(opts.text):
            new_dialogue=True
            for idx, utterance, answer,options in dialogue:
                if new_dialogue==False and previous_answer !="!not#an#answer!":                
                    text_gen.append([str(idx-1)+"-prime", previous_answer, utterance])    
                previous_answer=answer
                text_gen.append([idx, utterance, answer])
                new_dialogue=False                
            
        #Cleans Up Sentences (lower case and replaces inconsistencies)
        text_gen=clean_up_sentences(text_gen)
        
        #Creates Voc class which contains 
        #indicies for each word dictionnaries (word2index and index2word)
        #occurence count for each word        
        
        voc=Voc()
        voc=prepare_voc(text_gen,voc)

        #removes rare words from the corpus, repopulates voc dictionnaries
        voc,text_gen=trimRareWords(voc, text_gen,3)

            
        # Configure model paramaters
        #attn_model = 'dot'
        #attn_model = 'concat'
        model_paramaters = Model_Paramaters()
        model_name = model_paramaters.model_name
        attn_model = model_paramaters.attn_model
        hidden_size = model_paramaters.hidden_size
        encoder_n_layers = model_paramaters.encoder_n_layers
        decoder_n_layers = model_paramaters.decoder_n_layers
        dropout = model_paramaters.dropout
        batch_size= model_paramaters.batch_size                
        embedding = nn.Embedding(voc.num_words, hidden_size)
        
        # Initialize encoder & decoder models
        encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
        decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
        
        # Use appropriate device
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        
        # Configure training/optimization paramaters
        clip = 50.0
        teacher_forcing_ratio = 1.0
        learning_rate = 0.0001
        decoder_learning_ratio = 5.0
        n_iteration = 100000
        
        # Ensure dropout layers are in train mode
        encoder.train()
        decoder.train()
        
        # Initialize optimizers
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
        
        # Run training iterations
        trainIters(model_name, voc, text_gen, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, n_iteration, batch_size,clip)
        
        #saves the model for testing
        directory = os.path.join(opts.model, "RNN")
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'iteration': n_iteration,
            'en': encoder.state_dict(),
            'de': decoder.state_dict(),
            'en_opt': encoder_optimizer.state_dict(),
            'de_opt': decoder_optimizer.state_dict(),
            'loss': loss,
            'voc_dict': voc,
            'embedding': embedding.state_dict()            
        }, os.path.join(directory, 'RNN.tar'))
                


    elif opts.gen==False:
        #testing dataset
        USE_CUDA = torch.cuda.is_available()
        device = torch.device("cuda" if USE_CUDA else "cpu")
                    
        # load models
        loadFilename = os.path.join(opts.model, "part_1.tar")
        try:
            model = torch.load(loadFilename)
        except:
            model = torch.load(loadFilename, map_location='cpu')


        dual_encoder = model["dual_encoder"]            
        voc_part_1=model["voc_1"]
        dual_encoder = dual_encoder.to(device)
        
        text_dev=[]
        _=""
        
        for your_persona,partner_persona, dialogue in loadData_part_1(opts.text):
            for idx, utterance, answer,options,option_1,option_2,option_3,option_4,option_5,options_full in dialogue:
                text_dev.append([idx, utterance, answer,options,option_1,option_2,option_3,option_4,option_5,options_full])
        
#        launch testing
        option_score = test_model(text_dev,voc_part_1,dual_encoder)                                


                                    
    else:
            
        USE_CUDA = torch.cuda.is_available()
        device = torch.device("cuda" if USE_CUDA else "cpu")
        
        #Load model paramaters            
        model_paramaters=Model_Paramaters()
        model_name = model_paramaters.model_name
        attn_model = model_paramaters.attn_model
        hidden_size = model_paramaters.hidden_size
        encoder_n_layers = model_paramaters.encoder_n_layers
        decoder_n_layers = model_paramaters.decoder_n_layers
        dropout = model_paramaters.dropout
        batch_size= model_paramaters.batch_size

        # Load model encoder, decoder, voc and others
        loadFilename = os.path.join(opts.model, "RNN","RNN.tar")
        try:
            model = torch.load(loadFilename)
        except:
            model = torch.load(loadFilename, map_location='cpu')

        encoder_sd = model['en']
        decoder_sd = model['de']
        encoder_optimizer_sd = model['en_opt']
        decoder_optimizer_sd = model['de_opt']
        embedding_sd = model['embedding']
        voc = model['voc_dict']
        

        # Initialize word embeddings
        embedding = nn.Embedding(voc.num_words, hidden_size)
        embedding.load_state_dict(embedding_sd)
        # Initialize encoder & decoder models
        encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
        decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
        # Use appropriate device
        encoder = encoder.to(device)
        decoder = decoder.to(device)

        
        encoder.eval()
        decoder.eval()            
    
        # Initialize search module
        searcher = GreedySearchDecoder(encoder, decoder)
        
        for _,_, dialogue in loadData(opts.text):
            for idx, utterance, answer, options in dialogue:
                evaluateInput(idx, utterance,encoder, decoder, searcher, voc)                                    