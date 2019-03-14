#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
import torch.nn.utils

### YOUR CODE HERE for part 1h
class Highway(nn.Module):

	def __init__(self, embed_size, dropoutp = 0.3 , bias = True):
		""" Init Highway Model. 
		@param embed_size (int) Embedding size (dim)
		@param f_size (int) Filter size (dim)
		@param bias 

		"""
		super(Highway, self).__init__()
		#2 nn layers relu
		self.e_word = embed_size #e_word 
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.proj = nn.Linear(embed_size, embed_size, bias = bias)
		self.gate = nn.Linear(embed_size, embed_size, bias = bias)
		# self.dropout = nn.Dropout(p = dropoutp)


	def forward(self, conv_out) -> torch.Tensor:
		"""
		@param embedding: tensor with shape (e_word) 

		@returns x_word_embed 
		"""

		#Wproj*x_conv_out + b_proj in R^e_word 
		#Wgate*x_conv_out + b_gate #\in R e_word 

		#xproj = ReLu(Wproj*xconv_out + bproj)
		#xgate = sigmoid(Wgate*xconv_out + bgate)

		# print("Highwayinput size check")
		# assert(conv_out.size() == (self.e_word))

		xproj = self.relu(self.proj(conv_out))
		# print("Highway Proj relu shape check")
		# assert(xproj.size() == (self.e_word))
		xgate = self.sigmoid(self.gate(conv_out))

		# print("Highway gate sigoid shape check")
		# assert(xgate.size() == (self.e_word))
		xhighway = torch.mul(xproj, xgate)+torch.mul((1.0-xgate),conv_out)
		# xword_emb = self.dropout(xhighway)

		# print("Highway output shape check")
		# assert(xword_emb.size() == (self.e_word))

		return xhighway


