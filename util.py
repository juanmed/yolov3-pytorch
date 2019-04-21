from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np 
import cv2 

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
	#print("Input dimension: {}".format(inp_dim))
	batch_size = prediction.size(0)
	stride = inp_dim // prediction.size(2)				# stride de la red
	#print("Stride: {}".format(stride))
	#print(" Output size: {}".format(prediction.size()))
	grid_size = inp_dim // stride						# el grid size es distinto debio al multi-dimension predictions
	#print(" Grid Size: {}".format(grid_size))
	bbox_attrs = 5 + num_classes						# (x,y,w,h, pc) + num_classes
	num_anchors = len(anchors)

	# El metodo 'view' hace lo mismo que np.reshape
	prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
	prediction = prediction.transpose(1,2).contiguous()
	prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

	# Re-escalar las dimensiones de anchors por un factor de '1/stride'
	# por que la red redujo en un factor de 'stride'
	anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

	# Aplicar las transformaciones a cada elemento correspondiente 
	prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])	# x centroide
	prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])	# y centroide
	prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])	# objectness score


	# ************************************ #
	# 	 	Recuperar posicion del centroide
	# ************************************ #
	# Sumar a cada centroide los indices del bounding box de donde proviene
	# para recuperar su posicion absoluta x, y

	grid = np.arange(grid_size)
	a, b= np.meshgrid(grid, grid)

	# crear listas de los indices 
	x_offset = torch.FloatTensor(a).view(-1,1)
	y_offset = torch.FloatTensor(b).view(-1,1)

	# mover al GPU
	if CUDA:
		x_offset = y_offset.cuda()
		y_offset = y_offset.cuda()

	# crea un lista de indices que se pueden sumar directamente
	# a los centroides
	x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
	prediction[:,:,:2] = prediction[:,:,:2] + x_y_offset

	# ************************************ #
	# 	 	Recuperar ancho y alto
	# ************************************ #
	anchors = torch.FloatTensor(anchors)

	if CUDA:
		anchors = anchors.cuda()

	anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
	prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors


	# aplicar funcion sigmoide a las confianzas por clase
	prediction[:,:,5: 5 + num_classes] = torch.sigmoid(prediction[:,:, 5: 5 + num_classes])

	# Re escalar las dimensiones extraidas por un factor de 'stride'
	prediction[:,:,:4] = prediction[:,:,:4]*stride

	return prediction






	

