import augmentation as aug
from fairfil import fairfil_trainer
from finetune import sst_trainer

import argparse
import collections
import logging
import json
import re
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_file", default='feature.json', type=str)
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    ## Other parameters
    parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for predictions.")
    parser.add_argument("--epochs", default=10, type=int, help="num of epochs.")
    parser.add_argument("--log_step", default=10, type=int, help="")
    parser.add_argument("--temperature", default=0.07, type=float, help="temperature scaling")
    parser.add_argument("--log_dir", default="./log", type=str, help="log directory")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help = "local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    args = parser.parse_args()

    aug_data = aug.get_data()
    f = open('aug_data.json','w')
    json.dump(aug_data, f)

    fairfil_trainer(input_file=aug_data, args=args)
    #finetuning
    sst_trainer()
    #seat 계산하기


if __name__ == "__main__":
    main()