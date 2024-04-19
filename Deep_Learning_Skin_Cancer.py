# Imports

# Manipulação de dados e imagens
import os                             # Manipulacao Sistema
import cv2                            # Open CV  
import itertools                      # Interacao com todas imagens no disco
import matplotlib.pyplot as plt       # Diagrama/Grafico
import numpy as np                
import pandas as pd
import nvidia_smi
from tqdm import tqdm                 # Permite criar uma barra de progressao
from glob import glob                 # Manipulacao imagens
from PIL import Image                 # ||
import warnings
warnings.filterwarnings('ignore')     # ignorar qualquer aviso

# Pytorch
import torch
from torch import nn, optim                            # Neural Network , Otimizador
from torch.autograd import Variable                    # Gradientes
from torch.utils.data import DataLoader, Dataset       # DataLoader
from torchvision import models, transforms             # Transformacao

# Scikit-learn - Machine Learning for Python
from sklearn.model_selection import train_test_split   # Dividir em treino e teste
from sklearn.metrics import confusion_matrix           # Avaliacao
from sklearn.metrics import classification_report      # Avaliacao

# Pacotes para o relatório de hardware
import gc                              
import types
import pkg_resources
import pytorch_lightning as pl

# Seed para reproduzir os mesmos resultados
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)

# Versões dos pacotes usados neste jupyter notebook
#%reload_ext watermark
#%watermark -a "Weber Souza" --iversions

# Relatório completo

# Verificando o dispositivo
processing_device = "cuda" if torch.cuda.is_available() else "cpu"

# Verificando se GPU pode ser usada (isso depende da plataforma CUDA estar instalada)
torch_aval = torch.cuda.is_available()

# Labels para o relatório de verificação
lable_1 = 'Visão Geral do Ambiente'
lable_2 = 'Se NVIDIA-SMI não for encontrado, então CUDA não está disponível'
lable_3 = 'Fim da Checagem'

# Função para verificar o que está importado nesta sessão
def get_imports():

    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):            
            name = val.__module__.split(".")[0]

        poorly_named_packages = {"PIL": "Pillow", "sklearn": "scikit-learn"}

        if name in poorly_named_packages.keys():
            name = poorly_named_packages[name]

        yield name

# Imports nesta sessão
imports = list(set(get_imports()))

# Loop para verificar os requerimentos
requirements = []
for m in pkg_resources.working_set:
    if m.project_name in imports and m.project_name!="pip":
        requirements.append((m.project_name, m.version))
        
# Pasta com os dados (quando necessário)
pasta_dados = r'dados'

print(f'{lable_1:-^100}')
print()
print(f"Device:", processing_device)
print(f"Pasta de Dados: ", pasta_dados)
print(f"Versões dos Pacotes Requeridos: ", requirements)
print(f"Dispositivo Que Será Usado Para Treinar o Modelo: ", processing_device)
print(f"CUDA Está Disponível? ", torch_aval)
print("Versão do PyTorch: ", torch.__version__)
print("Versão do Lightning: ", pl.__version__)
print()
print(f'{lable_2:-^100}\n')
nvidia-smi
gc.collect()
print()
print(f"Limpando a Memória da GPU (se disponível): ", torch.cuda.empty_cache())
print("\nModelo da GPU:")
# Modelo da GPU usada
!nvidia-smi --query-gpu=name --format=csv,noheader
print(f'\n{lable_3:-^100}')