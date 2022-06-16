from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np 
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet 
import pickle as pkl 
import pandas as pd 
import random


def arg_parse():
    """
    Parse command line arguments to the detector
    """

    parser = argparse.ArgumentParser(description = " Red Neuronal Yolo V3")
    parser.add_argument('--images', 
                        dest = 'images', 
                        help = 'Directorio donde se encuentran las imagenes a procesar',
                        default = 'imgs',
                        type = str)
    parser.add_argument('--dest', 
                        dest = 'dest',
                        help = 'Directorio para almacenar detecciones',
                        default = 'dest',
                        type = str )
    parser.add_argument('--bs', 
                        dest = 'bs',
                        help = 'Batch Size',
                        default = 1)
    parser.add_argument('--conf',
                        dest = 'confidence',
                        help = 'Umbral de confianza para filtrar detectiones',
                        default = 0.5)
    parser.add_argument('--nms_thresh',
                        dest = 'nms_thresh',
                        help = 'Umbral para Non Maximum Supression',
                        default = 0.4)
    parser.add_argument('--cfg',
                        dest = 'cfgfile',
                        help = 'Archivo de configuracion .cfg de Yolo',
                        default = 'cfg/yolov3.cfg',
                        type = str)
    parser.add_argument('--weights',
                        dest = 'weightsfile',
                        help = 'Archivo .weights de pesos de la red Yolo',
                        default = 'backup/yolov3.weights',
                        type = str)
    parser.add_argument('--res',
                        dest = 'res',
                        help = 'Resolution the entrada para la red. Mayor resolucion es igual a mejor precision pero menor rapidez',
                        default = '416',
                        type = str)
    return parser.parse_args()

def draw_bb(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

# parse command line arguments
args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()                    # check if CUDA is present

# Cargar los nombres en COCO dataset
classes = load_classes('data/coco.names')
num_classes = len(classes)

# Configurar la red neuronal
print(">> Cargando Red...")
try:
    model = Darknet(args.cfgfile)
    print(">> Red inicializada, cargando pesos...")
except:
    print(">> No se puede inicializar red. Abortando...")

try:
    params = model.load_weights(args.weightsfile)
    print(">> Se cargo {} parameteros en {} capas convolucionales exitosamente!".format(params[0],params[1]))
except:
    print(">> No se pudo cargar pesos. Abortando...")

model.net_params['height'] = args.res               # 
inp_dim = int(model.net_params['height'])
assert inp_dim % 32 == 0
assert inp_dim > 32

if CUDA:
    model.cuda()                                    # si CUDA esta presente, mover el modelo

# Configurar el model en modo evaluacion
model.eval()

# Cargar direcciones de imagenes
read_dir = time.time()
try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print('No file or directory with name {}'.format(images))
    exit()

# Cargar o crear directorio destino
if not os.path.exists(args.dest):
    os.makedirs(args.dest)

# Cargar imagenes
load_batch = time.time()
loaded_imgs = [cv2.imread(x) for x in imlist]

# Crear lista de todas las imagenes pero convertidas a Variable
im_batches = list(map(prep_image, loaded_imgs, [inp_dim]*len(loaded_imgs)))
#print(im_batches[0],im_batches[0].size())


# Lista de dimensiones originales
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_imgs]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
#print(im_dim_list,len(im_dim_list))

if CUDA:
    im_dim_list = im_dim_list.cuda()

# verificar si, con el batch_size actual, quedarian imagenes sobrantes
leftover = 0
if (len(im_dim_list) % batch_size):
   leftover = 1

# crear batches
if batch_size != 1:
    num_batches = (len(imlist) // batch_size) + leftover
    im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size, len(im_batches))]))  for i in range(num_batches)] 
#print(im_batches[0], im_batches[0].size())

# Hacer el forward pass de todos los batches
write = 0
start_det_loop = time.time()
for i, batch in enumerate(im_batches):

    start = time.time()
    if CUDA:
        batch = batch.cuda()

    # forward pass el primer batch
    with torch.no_grad():
        prediction= model( Variable(batch), CUDA)

    # Filtrar por confianza, hacer Non Max Supression
    prediction = write_results(prediction, confidence, num_classes, nms_thresh = nms_thresh)
    end = time.time()
    print(prediction/int(args.res))

    if type(prediction) == int:         # No hay detecciones
        
        # Desplegar mensaje (indicando 0 detecciones)
        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("-----------------------------------------------------")

        continue

    # cambiar indice en batch por indice en la lista
    prediction[:,0] += i*batch_size

    if not write:                       # Si aun no se ha inicializado
        output = prediction             # inicializar salida
        write = 1
    else:
        output = torch.cat((output,prediction))     # anadir predicciones a la salida

    for im_num, image in enumerate(imlist[i*batch_size: min((i+1)*batch_size, len(imlist))]):

        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0])==im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")     

    if CUDA:
        torch.cuda.synchronize() 

# Verificar si hay detecciones
try:
    output
except NameError:
    print ("No detections were made")
    exit()

# Reconstruir las dimensiones del bounding box para las imagenes con padding
im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
scaling_factor = torch.min(inp_dim/im_dim_list, 1)[0].view(-1,1)
output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2

# Escalar las bounding boxes al tamano original
output[:,1:5] /= scaling_factor

# Limitar las coordenadas de bounding boxes para que esten dentro de la 
# imagen
for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])

output_recast = time.time()

# ------------------------- #
#   Draw Bounding Boxes     #
# ------------------------- #

class_load = time.time()
colors = pkl.load(open("pallete", "rb"))

draw = time.time()

list(map(lambda x: draw_bb(x, loaded_imgs), output))

# -------------------------------- #
#   Save images with detections    #
# -------------------------------- #

#dest_names = list(map(lambda x: "{}/det_{}".format(args.dest,x),imlist))
dest_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.dest,x.split("/")[-1]))
print(dest_names)
list(map(cv2.imwrite, dest_names, loaded_imgs))
end = time.time()


# -----------------  #
#   Print Summary    #
# ------------------ #
print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()