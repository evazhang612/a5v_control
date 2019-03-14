#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
import torch.nn.utils

### YOUR CODE HERE for part 1i
class CNN(nn.Module):

	def __init__(self, char_embed_size , f , k = 5, max_word_length = 21, bias = True):
		"""
		@param: embed_size, e_word 
		@param f = filter size, # of output channels 
		@param k = kernel size, window size
		@param max_word_length = m_word in the pdf 
	
		"""
		super(CNN,self).__init__()

		self.e_char = char_embed_size
		self.k = k 
		self.filter_size = f #output channels 
		self.relu = nn.ReLU()
		self.conv = nn.Conv1d(in_channels = self.e_char, out_channels = f,kernel_size = k, bias = bias)
		osize = (max_word_length - k + 1)
		self.maxpool = nn.MaxPool1d(kernel_size = osize) #stride = None by default , 0 padding 

		#kernel size = 5 

	def forward(self, reshaped_embedding): 
		""""
		@param: reshaped_embedding, has shape (e_char * max_word_length)

		@returns: x_conv_out returns (e_word)
		operate on batches of words 
		"""
		x = self.relu(self.conv(reshaped_embedding))
#        print(x.size())

		conv_out = self.maxpool(x)

#        print("cnn forward output shape")
#        print(conv_out.size())

		return conv_out

### END YOUR CODE

