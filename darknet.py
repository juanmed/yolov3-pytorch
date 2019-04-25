from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np 

from util import *


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

class Darknet(nn.Module):
    """
    Esta clase contiene la red neural profunda conformada por una lista de 
    capas recibida en module_list y parametros en net_params
    """

    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)            # leer archivo de conf
        self.net_params, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        """
        @x imagen de entrada
        @CUDA Si es TRUE, usar GPU

        Recibir imagen de entrada, pasarla por cada capa en self.module_list,
        almacenar la salida de cada capa en un diccionario, y por ultimo
        retorna la salida de la ultima capa
        """

        modules = self.blocks[1:]
        outputs = {}

        write = 0
        for i, module in enumerate(modules):
            
            module_type = module['type']            
            
            if(module_type == 'convolutional' or module_type == 'upsample'):
            
                x = self.module_list[i](x)          # simplement pasar imagen por la capa

            elif(module_type == 'route'):

                layers = module['layers']           # obtener indice de capas a rutear
                #print("Route Layer {} layers: {}".format(i,layers))
                layers = [int(layer) for layer in layers]   # y convertir a int

                if (layers[0] > 0):                  # en teoria, esto no debe suceder
                    layers[0] = layers[0] - i
                    print("Capa {} Tipo {} tiene layers[0] > 0!".format(i, module_type))

                if len(layers) == 1:
                    x = outputs[i + layers[0]]      # solo rutea 1 capa asi que
                                                    # obtener salida de la capa (previa) i + layers[0]
                else:
                    if (layers[1] > 0):             # rutear varias capas
                        layers[1] = layers[1] - i 
                        #print("Capa {} Tipo {} tiene layers[1] > 0!".format(i, module_type))

                    # obtener la salida de las capas a rutear
                    # *** Esto esta hard-coded, se puede mejorar
                    feature_map1 = outputs[i + layers[0]]   
                    feature_map2 = outputs[i + layers[1]]

                    x = torch.cat((feature_map1, feature_map2), 1)  # concatenar a lo profundo

            elif( module_type == 'shortcut'):

                from_ = int(module['from'])
                x = outputs[i-1] + outputs[i + from_]

            elif( module_type == 'yolo'):

                anchors = self.module_list[i][0].anchors

                inp_dim = int(self.net_params['height'])

                num_classes = int(module['classes'])

                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections

    def load_weights(self, weightfile_dir):
        """
        @description lee un archivo de pesos de la red y los carga a las
        capas apropiadas de la red
        @weightfile_dir directory de los pesos de la red
        @return 
        """

        fp = open(weightfile_dir, "rb")

        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        conv_layers = 0
        #num_weights_network = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]['type']

            # cargar pesos solo si la red es convolucional
            if (module_type == 'convolutional'):
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]['batch_normalize'])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    # cargar capa
                    bn = model[1]

                    # obtener numero de pesos de la capa
                    num_bn_biases = bn.bias.numel()

                    # leer los pesos desde archivo
                    # primero los biases
                    bn_biases = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  = ptr + num_bn_biases          # actualizar puntero

                    # ahora los pesos
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr = ptr + num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr = ptr + num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr = ptr + num_bn_biases

                    # dimensionar los pesos de acuerdo a la forma de la capa
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # cargar los pesos hacia la capa
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:

                    # Numero de biases
                    num_biases = conv.bias.numel()

                    # leer los pesos desde archivo
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases              # actualizar puntero

                    # dimensionar los pesos de acuerdo a la forma de la capa
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # cargar los pesos hacia la capa
                    conv.bias.data.copy_(conv_biases)
                
                # cargar pesos de capa convolucional
                num_weights = conv.weight.numel()

                # leer los pesos desde el archivo
                conv_weights = torch.from_numpy(weights[ptr : ptr + num_weights])
                ptr = ptr + num_weights

                # dimensionar los pesos de acuerdo a la forma de la capa
                conv_weights = conv_weights.view_as(conv.weight.data)

                # cargar los pesos hacia la capa
                conv.weight.data.copy_(conv_weights)

                conv_layers = conv_layers + 1

        print(" Se cargo {} pesos para {} capas convolucionales.".format(ptr,conv_layers))

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
            if len(block) != 0:                     # Antiguo bloque tiene datos?
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
            block['layers'] = block['layers'].split(',')     # obtener lista de layers a conectar
            
            start = int(block['layers'][0])

            try:
                end = int(block['layers'][1])
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

        module_list.append(module)                  # anadir la capa creada a la lista de capas
        prev_filters = filters                      # actualizar info, sera utilizada por siguiente capa
        output_filters.append(filters)              # 
        #print("Output Filters \n {}".format(output_filters))

    return (net_parameters, module_list) 

def get_test_input(img_dir, dims):
    img = cv2.imread(img_dir)
    img = cv2.resize(img, dims)
    img_ = img[:,:,::-1].transpose((2,0,1)) #cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # BGR to RGB
    #img_ = img_.transpose((2,0,1))                    # h x w x c -> c x h x w
    img_ = img_[np.newaxis,:,:,:]/255.0              # anadir canal (para batch)
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_


# example command
# python darknet.py cfg/yolov3.cfg backup/yolov3.weights dog-cycle-car.png

if __name__ == '__main__':
    import sys
    cfgfile_dir = sys.argv[1]                       # leer archivo de configuracion
    tstimg_dir = sys.argv[3]                        # leer imagen a probar
    wgtfile_dir = sys.argv[2]                        # leer pesos

    bloques = parse_cfg(cfgfile_dir)                # parsear archivo de conf
    net_info, module_list = create_modules(bloques) # crear lista de capas

    img_dims = (int(net_info['height']), int(net_info['width']))
    #print("\n Network Information: \n{}".format(net_info))
    #print("\" Capas: \n{}".format(module_list))

    model = Darknet(cfgfile_dir)                    # crear red
    model.load_weights(wgtfile_dir)                 # cargar pesos
    inp = get_test_input(tstimg_dir, img_dims)                
    pred = model(inp, False)
    print("Predictions: {}, {}\n {}".format(pred.size(),type(pred),pred[:,:,:5]))
    # post processing
    detections = write_results(pred, 0.1, 80)
    print (detections)



