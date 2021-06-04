import argparse
import collections
import logging
import json
import re
from tqdm import tqdm
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

from utils import accuracy
from utils import save_checkpoint


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


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


def contrastive_loss(features, score_function, args, device):

    features = features.reshape(-1,2,768).permute(1,0,2)
    features = torch.cat((features[0],features[1]),dim=0)
    labels = torch.cat([torch.arange(features.shape[0]//2) for i in range(2)], dim=0)
    # labels = torch.arange(features.shape[0]//2).repeat_interleave(2)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    #score function 기준으로 바꾸기
    similarity_matrix = torch.matmul(features, features.T)
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0]).to(device) > 0
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    # logits = logits / self.args.temperature
    return logits, labels


def club():
    return 0


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputFeature`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    for k,v in input_file.items():
        for i in range(len(v['f'])):
            senf = v['f'][i]
            senm = v['m'][i]
            text_f = senf
            text_m = senm
            text_b = None
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_f, text_b=text_b))
            examples.append(
                InputExample(unique_id=unique_id+1, text_a=text_m, text_b=text_b))
        unique_id += 2
    return examples


def fairfil_trainer(input_file, args):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

    layer_indexes = [int(x) for x in args.layers.split(",")]

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    examples = read_examples(input_file)

    features = convert_examples_to_features(
        examples=examples, seq_length=args.max_seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model = BertModel.from_pretrained(args.bert_model)
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
    if args.local_rank == -1:
        train_sampler = SequentialSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    #구체적인 값 명시 안되어서 적당히 설정
    filter = MLP(768,768).to(device)
    score_function = SCORE(768*2,100,1).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    params = list(filter.parameters())+list(score_function.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)

    n_iter = 0
    for epoch in tqdm(range(args.epochs)):
        for input_ids, input_mask, example_indices in (train_dataloader):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
            all_encoder_layers = all_encoder_layers
            sent_emb = all_encoder_layers[-1].permute(1,0,2)[0]
            
            fair_filter = filter(sent_emb)
            
            nce_logits, nce_labels = contrastive_loss(fair_filter, score_function, args, device)
            nce_loss = criterion(nce_logits, nce_labels)

            #club loss 추가해주면 될듯
            #짝수번 Index가 original. sensitive word index를 알아서 그거랑 loss 걸어야되는데 그부분을 앞에서 안뽑았다ㅠ 일단 랜덤 인덱스로 구현해도 될듯.
            club_loss = club()

            loss = nce_loss + club_loss

            #pretrain graph 아직 안끊음
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if n_iter % args.log_step ==0:
                top1, top5 = accuracy(nce_logits, nce_labels, topk=(1,5))
                print(f"Iter: {n_iter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}\tTop5 accuracy: {top5[0]}")

            n_iter+=1

    print("Fairfil training has finished")
    
    filter_ckpt_name = f'filter_ckpt_{args.epochs}.pth'
    score_ckpt_name = f'score_ckpt_{args.epochs}.pth'
    save_checkpoint({'state_dict':filter.state_dict(),'optimizer':optimizer.state_dict()},filename=os.path.join(args.log_dir,filter_ckpt_name))
    save_checkpoint({'state_dict':score_function.state_dict(),'optimizer':optimizer.state_dict()},filename=os.path.join(args.log_dir,score_ckpt_name))

    print("Checkpoint has been saved")