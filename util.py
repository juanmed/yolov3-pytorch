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
    stride = inp_dim // prediction.size(2)              # stride de la red
    #print("Stride: {}".format(stride))
    #print(" Output size: {}".format(prediction.size()))
    grid_size = inp_dim // stride                       # el grid size es distinto debio al multi-dimension predictions
    #print(" Grid Size: {}".format(grid_size))
    bbox_attrs = 5 + num_classes                        # (x,y,w,h, pc) + num_classes
    num_anchors = len(anchors)

    # El metodo 'view' hace lo mismo que np.reshape
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # Re-escalar las dimensiones de anchors por un factor de '1/stride'
    # por que la red redujo en un factor de 'stride'
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # ************************************ #
    #       Recuperar posicion del centroide
    # ************************************ #

    # Aplicar las transformaciones a cada elemento correspondiente 
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])    # x centroide
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])    # y centroide
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])    # objectness score

    # Sumar a cada centroide los indices del bounding box de donde proviene
    # para recuperar su posicion absoluta x, y

    grid = np.arange(grid_size)
    a, b= np.meshgrid(grid, grid)

    # crear listas de los indices 
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    # mover al GPU
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    # crea un lista de indices que se pueden sumar directamente
    # a los centroides
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    prediction[:,:,:2] = prediction[:,:,:2] + x_y_offset

    # ************************************ #
    #       Recuperar ancho y alto
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

def write_results(prediction, confidence, num_classes, nms_thresh = 0.4):
    """
    @prediction salida de la red neural
    @confidence objectness 
    @nms_conf non maximum supression confidence
    @description En base a la confianza de la prediccion y las clases se devuelve
    la prediccion final de la red, luego de post-procesar utilizando non-maximum
    supression para obtener la prediccion mas precisa
    """

    # considerar solo aquellas bounding box con confianza mayor al limite
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask               # si objecteness<confidence, objectess == 0

    # obtener coordenadas x,y de esquinas del bounding box
    # Ejemplo
    # esquina superior izquierda, y = centro,y - alto/2
    # esquina superior izquierda, x = centro,x - ancho/2
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - (prediction[:,:,2]/2)) # top-left x
    box_corner[:,:,1] = (prediction[:,:,1] - (prediction[:,:,3]/2)) # top-left y
    box_corner[:,:,2] = (prediction[:,:,0] + (prediction[:,:,2]/2)) # top-right x
    box_corner[:,:,3] = (prediction[:,:,1] + (prediction[:,:,3]/2)) # top-right y

    # sustituir centro x,y,ancho,alto  por  esquina izquierda x,y, esquina derecha x,y
    prediction[:,:,:4] = box_corner[:,:,:4]

    
    batch_size = prediction.size(0)                         # leer numero de imagenes
    #print("Batch size is: {}".format(batch_size))
    write = False

    # hacer Non Maximum Suppresion  image por imagen (no por batch)
    for ind in range(batch_size):
        image_pred = prediction[ind]                    # leer imagen 'i'

        # ****************************
        # @IMPROVEMENTE: creo que esto se puede hacer antes, no en este punto
        # y ganar eficiencia
        # ********************************
        # recuperar solo la clase con mayor confianza
        max_conf, max_conf_index = torch.max(image_pred[:,5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_index = max_conf_index.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_index)      # almacenar info en un tuple
        image_pred = torch.cat(seq, 1)                          # concatenar todos los valores en 1 solo tensor

        # eliminar las bounding box con objecteness < confidence
        non_zero_ind = (torch.nonzero(image_pred[:,4]))
        #print("Objectess > Confidence para {} elementos".format(non_zero_ind.size(0)))
        try:
            # seleccionar solo las ubicaciones donde objectness > confidence
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue

        #print(">> image_pred all: {}".format(image_pred.size()))
        #print(">> image_pred_ conf< {}: {}\n{}".format(confidence, image_pred_.size(), image_pred_))

        # si no hay detectiones con objecteness > confidence, continuar a 
        # siguiente imagen
        if image_pred_.shape[0]==0:
            continue

        img_classes = unique(image_pred_[:,-1])                 # obtener las classes detectadas
        #print(">> img_classes: {}".format(img_classes))
        # ************************************ #
        #    NON MAXIMUM SUPRESSION
        # See for reference: https://www.youtube.com/watch?v=VAo84c1hQX8
        # ************************************ #
        for cls in img_classes:

            # obtener detecciones para la actual clase 'cls'
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            #print(">> Elementos de la clase {}:\n {}".format(cls, cls_mask))
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)
            # sort by objecteness in descending order 
            conf_sort_index = torch.sort(image_pred_class[:,4], descending =True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            #print(">> Elementos de la clase {} ordenados descendente: \n {}".format(cls, image_pred_class))

            idx = image_pred_class.size(0)      
            #print(">> IDX: {}".format(idx))
            # PERFORM Non Maximum Supression
            for i in range(idx):

                # obtener el IOU entre el bounding box con max conf, y el resto
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    #print("Value error en elemento {}".format(i))
                    break
                except IndexError:
                    #print("IndexError en elemento {}".format(i))
                    break
                #print("IOUS para NMS: \n{}".format(ious))

                # identificar bounding boxes cuyo IOU > threshold
                iou_mask = (ious < nms_thresh).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask
                
                # eliminar bounding boxes cuyo IOU > threshold
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)


            # 
            batch_ind = image_pred_class.new(image_pred_class.size(0),1).fill_(ind)
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output,out))

    try:
        return output
    except: 
        # Si no hubo detecciones
        return 0

def bbox_iou(box1, box2):
    """
    Calcular el IOU entre box1 y box2
    """

    # Obtener coordenadas de las esquinas de cada bounding box
    #print(">> Boxes\n Box1 \n{} \nBox2 \n{}".format(box1,box2))
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # calcular coordenadas del rectangulo interseccion
    int_rect_x1 = torch.max(b1_x1, b2_x1) 
    int_rect_y1 = torch.max(b1_y1, b2_y1)
    int_rect_x2 = torch.max(b1_x2, b2_x2)
    int_rect_y2 = torch.max(b1_y2, b2_y2)

    # area de interseccion = ancho * alto
    int_area = torch.clamp(int_rect_x2 - int_rect_x1 +1, min=0)* torch.clamp(int_rect_y2 - int_rect_y1 + 1, min=0)

    # area de union: area1 + area 2 - inter_area
    box1_area = (b1_x2 - b1_x1 + 1 ) * (b1_y2 - b1_y1 + 1)
    box2_area = (b2_x2 - b2_x1 + 1 ) * (b2_y2 - b2_y1 + 1)
    union_area = box2_area + box1_area - int_area

    # IOU = int_area / (un_area)
    iou = int_area/union_area

    return iou

def unique(tensor):
    """
    Devuelve un tensor cuyos elementos son las clases detectadas en el tensor
    enviado como parametro
    """
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def load_classes(namesfile):
    """
    Cargar los nombres de todas las classes
    """
    fp = open(namesfile, 'r')
    names = fp.read().split('\n')[:-1]              
    return names

def resize_pad_image(img, inp_dim):
    """
    @img image to resize (without changing aspect ratio) and pad 
    @inp_dim dimensions to which resize the image 
    @description resize an image without changing aspect ratio using padding
    @ return resized and padded image 
    """
    o_width, o_height = img.shape[1], img.shape[0]              # original dimensions
    w, h = inp_dim

    # factor de redimension
    f = min(w/o_width, h/o_height)                                              
    new_w = int(o_width * f)
    new_h = int(o_height * f)

    # redimensionar imagen
    resized_image = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)            

    # crear un canvas con 'fondo' gris (128,128,128)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    # ubicar imagen original en el centro del canvas
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image

    return canvas

def prep_image(img, inp_dim):
    """
    @img image to prepare
    @inp_dim 
    @description redimensionar, convertir de BGR a RGB, dividing por 255,
                 reorganizar en orden Canales, Alto, Ancho, y transformar
                 en Variable la imagen de entrada 
    """
    img = resize_pad_image(img, (inp_dim, inp_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

