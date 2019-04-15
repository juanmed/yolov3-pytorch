from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np 


class EmptyLayer(nn.Module):
    """
    Esta capa representa una capa generica que sera utilizada para 
    representar las capas tipo 'route' y 'shortcut'. Se hara overrida
    del metodo 'forward' para este proposito.
    """

    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    """
    Esta capa representa la capa final de YOLO. Contiene las anchor boxes,
    etc parametros que definen las predicciones finales
    """
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors



def parse_cfg(cfgfile):
    """
    Takes a darknet configuration file and output a list of blocks.
    
    Each block is a dictionary that contains the description of 
    each block in the neural network to be built using the configuration
    file.
    """

    # Abrir y preparar archivo de configuracion
    conf_file = open(cfgfile, 'r')
    lines = conf_file.read().split('\n')            # leer todas las lines y separarlas
    lines = [x for x in lines if len(x) > 0]        # eliminar lineas vacias
    lines = [x for x in lines if x[0] != '#']       # eliminar comentarios
    lines = [x.rstrip().lstrip() for x in lines]    # eliminar espacios en blanco

    # crear lista de bloques
    blocks = []                                     # devolver una lista
    block = {}                                      # cada bloque es un diccionario

    # leer todas las lines, formar bloques y 
    # anadir cada bloque a la lista de bloques
    for line in lines:
        if line[0] == "[":                          # Identificar si es inicio de bloque
            if len(block) != 0:                     # ¿Antiguo bloque tiene datos?
                blocks.append(block)                # Anadir el antiguo bloque
                block = {}                          # y crear uno nuevo 
            block['type'] = line[1:-1].rstrip()     # Almacenar tipo de bloque
        else:
            key, value = line.split('=')            # 
            block[key.rstrip()] = value.lstrip()    # 
    blocks.append(block)

    return blocks

def create_modules(blocks):
    """
    Recibe una lista de bloques y crear una lista de capas (layers) de la 
    red neural basado en la misma.
    """

    net_parameters = blocks[0]                      # guardar parametros para la red neural
    module_list = nn.ModuleList()                   # estructura para almacenar una lista de nn.Modules
    prev_filters = 3                                # inicializar en 3 por que la image tiene 3 canales: RGB
    output_filters = []                             # 

    for index, block in enumerate(blocks[1:]):      
        module = nn.Sequential()

        # Extraer parametros de cada capa (layer), 
        # crear capa, y anadirla a la lista de capas

        if( block['type'] == 'convolutional'):
            
            activation = block['activation']        # extraer tipo de activation
            
            try:                                    # aplicar normalizacion por grupos
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(block['filters'])
            padding = int(block['pad'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])

            if padding:                             
                pad = (kernel_size - 1) // 2        # calcular pixeles a anadir al contorno
            else:                                   # de imagen para aplicar el kernel
                pad = 0

            # crear y anadir capa convolucional
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{}".format(index), conv)

            # crear y anadir capa de normalizacion
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{}".format(index), bn)

            # crear y anadir capa de activacion
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{}".format(index), activn)

        elif(block['type'] == 'upsample'):
            stride = int(block['stride'])
            upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear')
            module.add_module("upsample_{}".format(index), upsample)
        elif(block['type'] == 'route'):
            layers = block['layers'].split(',')     # obtener lista de layers a conectar

            start = int(layers[0])

            try:
                end = int(layers[1])
            except:
                end = 0

            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()                    # 
            module.add_module("route_{}".format(index), route)

            # Llevar cuenta del numero de filtros de salida de esta capa route
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif(block['type'] == 'shortcut'):
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        elif(block['type'] == 'yolo'):
            mask = block['mask'].split(",")
            mask = [int(x) for x in mask]           # obtener indices de anchor boxes

            anchors = block['anchors'].split(",")   # obtener y castear a 'int' las 
            anchors = [int(a) for a in anchors]     # dimensions de las anchor boxes
            # hacer una lista de parejas (ancho, alto) de anchor boxes
            anchors = [(anchors[i], anchors[i+1]) for i in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in mask]    # elejir solo las primeras n indicadas
                                                    # en mask

            detection = DetectionLayer(anchors)
            module.add_module('Detection_{}'.format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_parameters, module_list) 





if __name__ == '__main__':
    import sys
    cfgfile_dir = sys.argv[1]                       # leer archivo de configuracion
    bloques = parse_cfg(cfgfile_dir)                # parsear archivo de conf
    net_info, module_list = create_modules(bloques) # crear lista de capas
    print("\n Network Information: \n{}".format(net_info))
    print("\" Capas: \n{}".format(module_list))
    #print(bloques,len(bloques))

