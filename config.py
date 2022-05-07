import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
SOS_token = 2
EOS_token = 3

MAX_LENGTH = 128  #句子最大长度

R_PATH = '..' if torch.cuda.is_available() else '.'
DATASET_PATH=f'{R_PATH}/dataset/'

MODEL_PATH = './model/v1.0.1'  # model解压位置

LANG1 = 'zh'
LANG2 = 'en'

TRAIN_BATCH_SIZE = 64
TEACHER_FORCING_RATIO = 0.5

TOTAL_TRAIN_SIZE = 39315
