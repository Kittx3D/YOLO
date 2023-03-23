# **Iniciando - Dowload do Darknet (Rede Neural)**
!git clone https://github.com/AlexeyAB/darknet
ls 
cd darknet
ls
# **2° Passo: Compilando Biblioteca**
!make
# **3º Passo: Baixando os pesos do modelo pré-treinado**
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
# **4º Passo: Testando o Detector**
!./darknet detect cfg/yolov4.cfg yolov4.weights data/person.jpg
# **5° Passo: Vizualizando Resultado**
import cv2
import matplotlib.pyplot as plt

def mostrar(caminho):
  imagem = cv2.imread(caminho) #leitura da imagem
  fig = plt.gcf()
  fig.set_size_inches(18,10)#define o tamanho da imagem
  plt.axis('off') #desativa os eixos
  plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)) #faz a conversão para RGB
  plt.show()
mostrar('predictions.jpg')
# **Detecntando Objetos com YOLO v4 - Com suporte a GPU/CUDA**
import tensorflow as tf
device_name = tf.test.gpu_device_name()

print(device_name)
rm -rf ./obj/image_opencv.o ./obj/http_stream.o ./obj/gemm.o ./obj/utils.o ./obj/dark_cuda.o ./obj/convolutional_layer.o ./obj/list.o ./obj/image.o ./obj/activations.o ./obj/im2col.o ./obj/col2im.o ./obj/blas.o ./obj/crop_layer.o ./obj/dropout_layer.o ./obj/maxpool_layer.o ./obj/softmax_layer.o ./obj/data.o ./obj/matrix.o ./obj/network.o ./obj/connected_layer.o ./obj/cost_layer.o ./obj/parser.o ./obj/option_list.o ./obj/darknet.o ./obj/detection_layer.o ./obj/captcha.o ./obj/route_layer.o ./obj/writing.o ./obj/box.o ./obj/nightmare.o ./obj/normalization_layer.o ./obj/avgpool_layer.o ./obj/coco.o ./obj/dice.o ./obj/yolo.o ./obj/detector.o ./obj/layer.o ./obj/compare.o ./obj/classifier.o ./obj/local_layer.o ./obj/swag.o ./obj/shortcut_layer.o ./obj/activation_layer.o ./obj/rnn_layer.o ./obj/gru_layer.o ./obj/rnn.o ./obj/rnn_vid.o ./obj/crnn_layer.o ./obj/demo.o ./obj/tag.o ./obj/cifar.o ./obj/go.o ./obj/batchnorm_layer.o ./obj/art.o ./obj/region_layer.o ./obj/reorg_layer.o ./obj/reorg_old_layer.o ./obj/super.o ./obj/voxel.o ./obj/tree.o ./obj/yolo_layer.o ./obj/gaussian_yolo_layer.o ./obj/upsample_layer.o ./obj/lstm_layer.o ./obj/conv_lstm_layer.o ./obj/scale_channels_layer.o ./obj/sam_layer.o darknet  
# **Modificando o arquivo Makefile para poder usar o GPU**
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
# **Compilando novamente a biblioteca**
!make
# **Testando o Detector**
!./darknet detect cfg/yolov4.cfg yolov4.weights data/person.jpg
mostrar('predictions.jpg')
!./darknet detect cfg/yolov4.cfg yolov4.weights data/giraffe.jpg
mostrar('predictions.jpg')
#Mostrando dados da GPU
!nvidia-smi
!/usr/local/cuda/bin/nvcc --version # mostra os dados da GPU
# **Criando Função para detecção de imagens**
import os
def detectar (imagem):
  os.system("cd /content/darknet && ./darknet detect cfg/yolov4.cfg yolov4.weights {}".format(imagem))
  mostrar('predictions.jpg')
detectar('data/person.jpg')
imagens = ['data/horses.jpg', 'data/eagle.jpg', 'data/dog.jpg']
for img in imagens:
  detectar(img)


# **Detecção de fotos personalizadas**
from google.colab import drive
drive.mount('/content/gdrive')
!ls /content/gdrive/My\ Drive/YOLO-\TESTE/
!cp /content/gdrive/My\ Drive/YOLO-\TESTE/dogcar.JPG data/
detectar('data/dogcar.JPG')

# **Salvando o resultado da predição**
!cp predictions.jpg /content/gdrive/My\ Drive/YOLO-\TESTE/dogcar.JPG
