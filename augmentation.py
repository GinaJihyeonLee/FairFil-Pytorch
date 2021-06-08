# standard library
from itertools import combinations
import numpy as np
import os, sys
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset
import pickle
import json

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

np.random.seed(42)

words2 = [["woman", "man"], ["girl", "boy"], ["she", "he"], ["mother", "father"], ["daughter", "son"], ["gal", "guy"], ["female", "male"], ["her", "his"], ["herself", "himself"], ["Mary", "John"]]
words_more = [["actress", "actor"], ["girlfriend", "boyfriend"], ["lady", "gentleman"], ["ladies", "gentlemen"], ["heroin", "hero"], 
        ["queen", "king"], ["princess", "prince"], ["female", "male"], ["woman", "man"], ["women", "men"], ["aunt", "uncle"], ["granddauter", "grandson"], 
        ["stepmother", "stepfather"], ["husband", "wife"], ["Mrs.", "Mr."], ["spokeswoman", "spokesman"], ["sister", "brother"]]

words2 =  words2 + words_more 

DIRECTORY = '.'

GENDER = 0

def match(a,L):
	for b in L:
		if a == b:
			return True
	return False

def replace(a,new,L):
    word = ""
    Lnew = []
    for i, b in enumerate(L):
        if a == b:
            Lnew.append(new)
            word = b
        else:
            Lnew.append(b)
    return ' '.join(Lnew), word

def template2(words, sent, sent_list, all_pairs):
    for i, (female, male) in enumerate(words):
        if match(female, sent_list):
            sent_f = sent
            sent_m, word = replace(female,male,sent_list)
            all_pairs[i]['f'].append([sent_f, word])
            all_pairs[i]['m'].append([sent_m, ""])
            # import pdb
            # pdb.set_trace()
        if match(male, sent_list):
            sent_f, word = replace(male,female,sent_list)
            sent_m = sent
            all_pairs[i]['f'].append([sent_f,""])
            all_pairs[i]['m'].append([sent_m, word])
            # import pdb
            # pdb.set_trace()
    return all_pairs

def get_sst():
    sst = load_dataset("sst","default")
    sentence = sst['train']['sentence']
    all_pairs2 = defaultdict(lambda: defaultdict(list))
    total = 0
    num = 0
    for sent in sentence:
        try:
            sent = sent
        except:
            pass
        sent = sent.lower().strip()
        sent_list = sent.split(' ')
        total += len(sent_list)
        num += 1
        all_pairs2 = template2(words2, sent, sent_list, all_pairs2)
    return all_pairs2

def check_bucket_size(D):
	n = 0
	for i in D:
		for key in D[i]:
			n += len(D[i][key])
			break
	return n

def get_data():
    gender = get_sst()
    bucket_size = check_bucket_size(gender)
    print("sst has {} pairs of templates".format(bucket_size))
    print('')
    return gender

# if __name__ == "__main__":
#     data = get_data()
#     import pdb
#     pdb.set_trace()
