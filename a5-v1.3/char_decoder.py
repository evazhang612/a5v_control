#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        super(CharDecoder, self).__init__()
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        self.charDecoder = nn.LSTM(input_size = char_embedding_size, hidden_size = hidden_size, bias = True, bidirectional = False) 
        self.char_output_projection = nn.Linear(in_features = hidden_size, out_features = len(target_vocab.char2id), bias = True)
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id), char_embedding_size, padding_idx = target_vocab.char2id['<pad>'])
        self.target_vocab = target_vocab
        self.padding_idx = target_vocab.char2id['<pad>']

        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        char_embeddings = self.decoderCharEmb(input)
        # print(char_embeddings.permute(2,0,1).size())
        state, dec_hidden = self.charDecoder(char_embeddings, dec_hidden) 
        scores = self.char_output_projection(state) #s_t = W[dec] ht + b_dec #is this part correct
        
        return scores, dec_hidden  
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

        length, batch = char_sequence.size()


        calc_loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx, reduction='sum')
        scores, dec_hidden = self.forward(char_sequence, dec_hidden=dec_hidden)
        truncate_srcsequence = char_sequence[1:] #.contiguous().view(-1)
        probs_tgtsequence = scores[:-1] #.view((length-1)*batch, vocab_size)

        resized_src = truncate_srcsequence.contiguous().view(-1)
        resized_probs = probs_tgtsequence.view((length-1)*batch, scores.size()[2])

        CEloss = calc_loss(resized_probs,resized_src)
        
        return CEloss

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        start_idx, end_idx = self.target_vocab.start_of_word, self.target_vocab.end_of_word
        start_char, end_char = self.target_vocab.id2char[start_idx], self.target_vocab.id2char[end_idx]
        _, batch, hidden_size = (initialStates[0].size())

        out_words = [] 
        decodedWords = []

        #initialize starting character 
        current_char_batch = torch.LongTensor([start_idx] * batch)
        #print(current_char_batch.size())
        current_char_batch = current_char_batch.unsqueeze(dim = 0)
        last_state = initialStates

        for t in range(max_length):
            scores_last, dec_hidden = self.forward(current_char_batch, dec_hidden = last_state)
            probs = nn.functional.softmax(scores_last, dim =2 ) #dim = 2 over vocab

            likely_char_batch = probs.argmax(dim = 2)
            batch_likely = likely_char_batch.squeeze(0)
            out_words.append([self.target_vocab.id2char[j] for j in batch_likely.tolist()])
            last_state = dec_hidden
            current_char_batch = likely_char_batch


        transposedwords =  [list(i) for i in zip(*out_words)]
        joinedwords = list(map(''.join, transposedwords)) 

        for word in joinedwords: 
            addtolist = word
            if (len(word)> max_length):
                addtolist = word[:max_length]
            addtolist = addtolist.split(end_char, 1)[0]
            decodedWords.append(addtolist)

        print(decodedWords)

        return decodedWords 
        ### END YOUR CODE

