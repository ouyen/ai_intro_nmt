import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
SOS_token = 0
EOS_token = 1

MAX_LENGTH = 128  #句子最大长度

MODEL_PATH = './model/v1.0.1'  # model解压位置

LANG1 = 'zh'
LANG2 = 'en'

TEACHER_FORCING_RATIO = 0.5