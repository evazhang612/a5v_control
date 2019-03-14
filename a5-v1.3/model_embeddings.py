#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.pad_token_idx = vocab.char2id['<pad>'] 
        self.e_char = self.char_embed_size = 50 
        self.e_word = self.embed_size = embed_size #e_word
        self.embeddings = nn.Embedding(len(vocab.char2id), self.char_embed_size, padding_idx=self.pad_token_idx)
        self.vocab = vocab #vocab object 
        
        self.dropoutp = 0.3 
        self.dropout = nn.Dropout(self.dropoutp)
        self.cnnlay = CNN(self.e_char, f = self.e_word)
        self.highwaylay = Highway(self.e_word) 


        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        x_emb = self.embeddings(input) #x_emb = charEmbedding(xpadded )
        (sentence_length, batch, max_word_length, char_embed_size) = x_emb.size()

        x_reshaped = x_emb.transpose(2,3)

        x_reshaped = x_reshaped.view(batch*sentence_length, self.e_char, max_word_length)

        # print(x_reshaped.size())

        x_convout = self.cnnlay(x_reshaped) 

        # print(x_convout.size())
        squeeze_convout = x_convout.squeeze(dim = 2)

        x_word_emb = self.highwaylay(squeeze_convout)
        x_word_emb = self.dropout(x_word_emb)

        output = x_word_emb.view(sentence_length, batch, self.e_word)
        return output

        ### END YOUR CODE

