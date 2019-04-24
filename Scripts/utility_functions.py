#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:21:41 2019
@author: Hesham El Abd
@Description: 
"""
from music21 import converter, note, chord
import glob
import numpy as np
import tensorflow as tf
## defining function
def Midi2List(midi_dir,mus_sep,start_token,end_toekn,print_progress=False):
    """
    The function reads all the midi files available at a data dir, extract the
    events pitches and duration, combine them as text and returns it as a list
    representation of the midi file. Incase, more than one midi file are used
    start and end taoken are used to mark the start and the end of the token
    respectively 
    """
    musicAsList=[]
    counter=0
    for track_name in glob.glob(midi_dir):
        midi_file=converter.parse(track_name).flat.notes
        musicAsList.append(start_token)
        for element in midi_file:
            if isinstance(element, note.Note):
                musicAsList.append(str(element.pitch)+"$"+
                                    element.duration.fullName)
            elif isinstance(element, chord.Chord):
                musicAsList.append('.'.join(str(n) for n in element.normalOrder)
                +"$"+element.duration.fullName)
        musicAsList.append(end_toekn)
        if print_progress:
            counter+=1
            print("Number of songs processed is {}".format(counter))
    return musicAsList

def UniqueElements(text):
    """
    The functions takes as an input a text corpa and returns as an output the 
    unique elemets in the corpa
    """
    return sorted(set(text))

def GenChar2IdxMap(uniqElements_text):
    """
    The functions takes as an input a unique element text corpa and returns as
    an output a dictionary that maps each character into an int.
    """
    return {char:idx for idx , char in enumerate(uniqElements_text)}
    
def GenIdx2Char(uniqElements_text):
    """
    The functions takes as an input a unique element text and returns as an 
    output a map for each index to the corresponding character. 
    """
    return np.array(uniqElements_text)

def EncodeText(text, encoding_scheme):
    """
    The function takes as an input a string and a mapping dictionary 
    and returns numpy array the represents a numerical encoding of the 
    input text string. 
    """
    return np.array([encoding_scheme[char] for char in text])

def DecodeText(encoded_text, decoding_scheme):
    """
    The function takes as an input a numpy array which represents a 
    numerically encoded text and decoding scheme and return a string as an 
    output. 
    """
    return "".join([decoding_scheme[idx] for idx in encoded_text])

def PrepareTrainingTensors(seq_fragment): 
    """
    The function takes as an input a seq_fragment that has been generated from 
    a BatchDataset that yeild batches of length equal to seq_length+1
    where the sequence length is the contional sequence length and 1 is the 
    target character.
    """
    inputs=seq_fragment[:-1]
    targets=seq_fragment[1:]
    return inputs, targets

def loss(labels,preds):
    return tf.keras.losses.sparse_categorical_crossentropy(
            y_true=labels, y_pred=preds, from_logits=True)
    
    
def CreateModels(vocab_size, emb_dim, num_lstm_units, 
                 recurrent_dropout, input_dropout,batch_size):
    """
    The function creates a LSTM based model to be used for creating the text
    generation, the function uses keras functional API. 
    
    ### inputs:
    # vocab_size=is the vocabulary size 
    
    # emb_dim= is the embedding dim to be used with the embedding layers
    
    # num_lstm_units= the number of units of LSTM layer
    
    # recurent_dropout= is the dropout between different time steps inside the 
    # LSTM units 
    
    # input_dropout= is the dropout of the LSTM unit input.
    
    # batch_size= batch size
    
    """
    inputs=tf.keras.layers.Input(shape=(None,),
                                 batch_size=batch_size, 
                                 name="Input_layer")
    
    embedding=tf.keras.layers.Embedding(input_dim=vocab_size,
                                        output_dim=emb_dim,
                                        name="Embeeding_layer")(inputs)
    
    lstm=tf.keras.layers.LSTM(units=num_lstm_units,
                              return_sequences=True,
                              recurrent_dropout=recurrent_dropout,
                              dropout=input_dropout,
                              stateful=True,
                              name="LSTM_Layer")(embedding)
    
    output_neurons=tf.keras.layers.Dense(units=vocab_size,
                                         name="OutPut_units")(lstm)
    
    model=tf.keras.models.Model(inputs=inputs,outputs=output_neurons)
    return model

def CreateStackedModels(vocab_size, emb_dim, num_lstm_units, 
                 recurrent_dropout, input_dropout,batch_size,
                 num_stacked_LSTM, num_units_stacks,return_sequence):
    """
    The function creates a LSTM based model to be used for creating the text
    generation, the function uses keras functional API. 
    
    ### inputs:
    # vocab_size=is the vocabulary size 
    
    # emb_dim= is the embedding dim to be used with the embedding layers
    
    # num_lstm_units= the number of units of LSTM layer
    
    # recurent_dropout= A list of dropouts to be applied between different
    time steps, the len of this list must equal num_stacked_LSTM.
    
    # input_dropout= is the dropout of the LSTM unit input.
    
    # batch_size= batch size
    
    #num_stacked_LSTM: int, the number of layers to stack on top of each other.
    
    #num_stacked_units: a list that contian the number of units for each LSTM
    layer in the stack, the len of this list must equal num_stacked_LSTM. 
    
    # return_Sequences: a bool, if true the last LSTM will return sequences,
    i.e. a 3D Tensor, otherwise, it returns the last time step output, 
    i.e. a 2D tensor.
    """
    inputs=tf.keras.layers.Input(shape=(None,),
                                 batch_size=batch_size, 
                                 name="Input_layer")
    
    embedding=tf.keras.layers.Embedding(input_dim=vocab_size,
                                        output_dim=emb_dim,
                                        name="Embeeding_layer")(inputs)
    
    lstmStack=CreateLSTMSack(num_stacked_LSTM,
                             num_units_stacks,
                             recurrent_dropout,
                             return_sequence)(embedding)
    
    output_neurons=tf.keras.layers.Dense(units=vocab_size,
                                         name="OutPut_units")(lstmStack)
    
    model=tf.keras.models.Model(inputs=inputs,outputs=output_neurons)
    return model

def CreateLSTMSack(num_stacked_LSTM, 
                   num_units_stacks,recurrent_dropout,return_sequence):
    """
    The function create a stacke of LSTMs using Keras Sequential model.
    
    ## inputs: 
    
    #num_stacked_LSTM: an int donates the number of stacks.
    
    # num_units_stakcs: a list that contian the number of units for each LSTM in
    the stack, the length of num_units_stacks must equal num_stacked_LSTM. 
    
    # recurrent_dropout: is a list that contains the recurrent dropout at each 
    layers.
    
    # return_Sequences: a bool, if true the last LSTM will return sequences,
    i.e. a 3D Tensor, otherwise, it returns the last time step output, 
    i.e. a 2D tensor
    """
    model=tf.keras.models.Sequential()
    for idx in range(num_stacked_LSTM):
        if (idx+1)==num_stacked_LSTM:
            model.add(tf.keras.layers.LSTM(
                    units=num_units_stacks[idx],
                    return_sequence=return_sequence,
                    recurrent_dropout=recurrent_dropout[idx]))
        else:
            model.add(tf.keras.layers.LSTM(
                    units=num_units_stacks[idx],
                    recurrent_dropout=recurrent_dropout[idx],
                    return_sequence=True))
    return model

def LoadModel(path,cust_obj=None):
    """
    the function takes the path to an hf5 model, loads the model and return 
    it.The Load function also takes a dictionary that has the name and def.
    of any impelemented custom layers or functions.
    """
    return tf.keras.models.load_model(path,cust_obj)


def GenerateText(model, TextLength, cond_string, sampling_temp,
                 from_char_to_int_map, from_int_to_char_map): 
    """
    The Function takes a trained language model, feeds it a cond_string and then
    gets distrbution of the output from the model, next it rescles the output
    distrbution using the sampling temp and sample the next character, this 
    allow the model to generate more creative words and phrases. This process
    is repeated untill the generated text is equal to TextLength. 
    
    ## inputs: 
   
    # model: a trained language model
    
    # TextLength: is the length of the output text in the number of characters
    
    #cond_string: the conditional string to feed it into the model inorder 
    to make prediction
    
    #sampling_temp: the sampling temp used to scale the model output distrbtions

    # from_char_to_int_map: a dictionary that can be used for mapping from
    characters to ints
    
    # from_int_to_char_map: is the numpy array that can be used to map from 
    ints to characters
    """
    # encoding the input string 
    Encoded_input_string=EncodeText(cond_string,from_char_to_int_map).reshape(
            1,len(cond_string))
    
    # a place holder for the generated text 
    generated_text=[]
    
    # Resetting the model state:
    model.reset_states()
    # the generation loop
    for i in range(TextLength):
        preds=model(Encoded_input_string) 
        preds=tf.squeeze(preds,axis=0) # remove the batch axis as we are 
        # generating only one samples at a time 
        preds=preds/sampling_temp # rescale the disrbution
        idx_pre_char=tf.random.categorical(preds,num_samples=1)[-1,0].numpy()
        # note the indexing: -1 is used to access the last elmenet,because this
        # is the character we are interested in all the other elements are from
        # the input of prev. time steps. 
        Encoded_input_string=tf.expand_dims([idx_pre_char],0)
        generated_text.append(idx_pre_char)
        
    generated_text=DecodeText(generated_text,from_int_to_char_map)
    return generated_text
    
    
    
    