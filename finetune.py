import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, D_in, D_out):
        super().__init__()
        self.linear = nn.Linear(D_in, D_out)
    def forward(self,x):
        x = self.linear(x)
        x = F.relu(x)
        return x


class SCORE(nn.Module):
    def __init__(self, D_in, D_mid, D_out):
        super().__init__()
        self.linear1 = nn.Linear(D_in, D_mid)
        self.linear2 = nn.Linear(D_mid, D_out)
    def forward(self,x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def sst_trainer():
    #1. 그냥 버전
    #2. fairfilter 추가버전
    #3. 우리 개선사항 추가버전 일케 3종류 finetuning

    #fairfil.py랑 거의 똑같이 하면 될듯?
    #bert for sequence classification 참고
    # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
    return