import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 10 #句子最大长度

R_PATH='.' #相对路径
# data_path='..' # 在训练平台时

LANG1='zh'
LANG2='en'

# TEACHER_FORCING_RATIO = 0.5