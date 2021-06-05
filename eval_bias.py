from __future__ import absolute_import, division, print_function
import numpy as np
import json
import os
import logging
import argparse
from scipy import spatial
import torch
import itertools
from itertools import combinations
import collections
import pickle
from tqdm import tqdm

# word embeddings
import gensim
import gensim.downloader as api
from gensim.utils import tokenize
# from eval_utils import isInSet
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)


# first party
import weat
# from run_classifier import get_encodings, compute_gender_dir, get_tokenizer_encoder
# from run_classifier import get_def_examples
from def_sent_utils import get_all, get_all_domains, get_def_pairs
# from my_debiaswe import my_we
# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertModel, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule


logger = logging.getLogger(__name__)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt = '%m/%d/%Y %H:%M:%S',
					level = logging.INFO)

DATA_DIR = "./data/sent-bias/"
MAX_SEQ_LENGTH = 128
DEVICE = torch.device("cuda") if torch.cuda.is_available() else None


def load_json(sent_file):
    ''' Load from json. We expect a certain format later, so do some post processing '''
    logger.info("Loading %s..." % sent_file)
    all_data = json.load(open(sent_file, 'r'))
    data = {}
    for k, v in all_data.items():
        examples = v["examples"]
        data[k] = examples
    return all_data  # data


def parse_args():
	'''Parse command line arguments.'''
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_path",
						type=str,
						default="bert-base-uncased",
						help="Path of the model to be evaluated")
	parser.add_argument("--debias",
						action='store_true',
						help="Whether to debias.")
	parser.add_argument("--equalize",
						action='store_true',
						help="Whether to equalize.")
	parser.add_argument("--def_pairs_name", default="all", type=str,
						help="Name of definitional sentence pairs.")
	parser.add_argument("--model", "-m", type=str, default="dummy")
	parser.add_argument("--output_name", type=str)
	parser.add_argument("--results_dir", type=str,
						help="directory for storing results")
	parser.add_argument("--encode_only", action='store_true')
	parser.add_argument("--num_dimension", "-k", type=int, default=1,
						help="dimensionality of bias subspace")
	args = parser.parse_args()
	if (args.output_name == None):
		args.output_name = args.def_pairs_name if args.debias else "biased"
	print("outputname: {}".format(args.output_name))
	if (args.results_dir == None):
		args.results_dir = os.path.join("results", args.model)
	args.do_lower_case = True
	args.cache_dir = None
	args.local_rank = -1
	args.max_seq_length = 128
	args.eval_batch_size = 8
	args.n_samples = 100000
	args.parametric = True
	args.tune_bert = False
	args.normalize = True

	# word embeddings
	args.word_model = 'fasttext-wiki-news-subwords-300'
	wedata_path = 'my_debiaswe/data'
	args.wedata_path = wedata_path
	args.definitional_filename = os.path.join(wedata_path, 'definitional_pairs.json')
	args.equalize_filename = os.path.join(wedata_path, 'equalize_pairs.json')
	args.gendered_words_filename = os.path.join(wedata_path, 'gender_specific_complete.json')

	return args


def binary_weat(targets, attributes):
	targetOne = []
	targetTwo = []
	for x in targets[0]:
		targetOne.append(_binary_s(x, attributes))
	for y in targets[1]:
		targetTwo.append(_binary_s(y, attributes))

	weat_score = np.absolute(sum(targetOne) - sum(targetTwo))

	wtmp = [_binary_s(t, attributes) for t in targets[0] + targets[1]]
	effect_std = np.std(wtmp)
	num = np.absolute((sum(targetOne)/float(len(targetOne)) - sum(targetTwo)/float(len(targetTwo))))
	effect_size = (num/effect_std)
	return weat_score, effect_size


def _binary_s(target, attributes):
	groupOne = []
	groupTwo = []
	for ai in attributes[0]:
		groupOne.append(spatial.distance.cosine(target, ai))
	for aj in attributes[1]:
		groupTwo.append(spatial.distance.cosine(target, aj))
	return sum(groupOne)/float(len(groupOne)) - sum(groupTwo)/float(len(groupTwo))
	

def save_dict_to_json(D, output_eval_file):
	with open(output_eval_file, 'w') as f:
		json.dump(D, f)


def run_binary_weat_test(encs):
	targ1 = list(encs['targ1']['encs'].values())
	targ2 = list(encs['targ2']['encs'].values())
	attr1 = list(encs['attr1']['encs'].values())
	attr2 = list(encs['attr2']['encs'].values())
	targets = [targ1, targ2]
	attributes = [attr1, attr2]
	weat_score, effect_size = binary_weat(targets, attributes)
	return weat_score, effect_size


# def evaluate(args, def_pairs, word_level=False):
# 	'''Evaluate bias level with given definitional sentence pairs.'''
# 	results_path = os.path.join(args.results_dir, args.output_name)

# 	if (not args.encode_only):
# 		if (os.path.exists(results_path)): 
# 			print("Results already evaluated in {}".format(results_path))
# 			return
# 		if (not os.path.exists(args.results_dir)): os.makedirs(args.results_dir)

# 	results = []
# 	all_tests_dict = dict()

# 	tokenizer, bert_encoder = get_tokenizer_encoder(args, DEVICE)
# 	print("tokenizer: {}".format(tokenizer==None))
# 	gender_subspace = None
# 	if (args.debias):
# 		gender_subspace = compute_gender_dir(DEVICE, tokenizer, bert_encoder, def_pairs, 
# 			args.max_seq_length, k=args.num_dimension, load=True, task=args.model, word_level=word_level, keepdims=True)
# 		logger.info("Computed (gender) bias direction")

# 	with open(args.gendered_words_filename, "r") as f:
# 		gender_specific_words = json.load(f)
# 	specific_set = set(gender_specific_words)

# 	abs_esizes = []
# 	for test_id in ['6', '6b', '7', '7b', '8', '8b']:
# 		filename = "sent-weat{}.jsonl".format(test_id)
# 		sent_file = os.path.join(DATA_DIR, filename)
# 		data = load_json(sent_file)

# 		encs = get_encodings(args, data, tokenizer, bert_encoder, gender_subspace, 
# 			DEVICE, word_level=word_level, specific_set=specific_set)
# 		if (args.encode_only):
# 			if (args.debias):
# 				outfile_name = 'debiased_encs{}.pkl'.format(test_id)
# 			else:
# 				outfile_name = 'biased_encs{}.pkl'.format(test_id)

# 			with open(os.path.join(args.results_dir, outfile_name), 'wb') as outfile:
# 				pickle.dump(encs, outfile)
# 			continue
# 		'''
# 		encs: targ1, targ2, attr1, attr2
# 		         -> category
# 		         -> encs
# 		         	-> (id1, sent1_emb), (id2, sent2_emb), ...
# 		'''

# 		esize, pval = weat.run_test(encs, n_samples=args.n_samples, parametric=args.parametric)
# 		abs_esizes.append(abs(esize))

# 		result = "{}: esize={} pval={}".format(filename, esize, pval)
# 		print(filename, result)
# 		results.append(result)
# 		test_results = {"esize": esize, "pval": pval}
		
# 		all_tests_dict[filename] = test_results
# 	avg_absesize = np.mean(np.array(abs_esizes))
# 	print("Averge of Absolute esize: {}".format(avg_absesize))
# 	all_tests_dict['avg_absesize'] = avg_absesize

# 	if (args.encode_only): return
	
# 	# print and save results
# 	for result in results: logger.info(result)
# 	save_dict_to_json(all_tests_dict, results_path)

# 	return


def eval_sent_debias():
    import pdb
    pdb.set_trace()
    args = parse_args()
    def_pairs_name = args.def_pairs_name
    size_prefix = "allsize"
    accdomain_prefix = "accdomain"
    domain_prefix = "moredomain"
    if (def_pairs_name.startswith(size_prefix)):
        # evaluate model 
        bucket_list = get_single_domain_in_buckets()
        indices = np.arange(len(bucket_list))

        size = int(def_pairs_name[len(size_prefix):])
        
        choices_list = list(combinations(indices, size))
        logger.info(choices_list)
        for choices in choices_list:
            logger.info(choices)
            chosen_buckets = [bucket_list[i] for i in choices]
            def_pairs = []
            for bucket in chosen_buckets:
                def_pairs += bucket
            evaluate(args, def_pairs)
    elif (def_pairs_name.startswith(accdomain_prefix)):
        domain_list = get_all_domains(1000)
        for domain in domain_list: print("domain size={}".format(len(domain)))
        indices = np.arange(len(domain_list))

        size = int(def_pairs_name[len(accdomain_prefix):])
        choices_list = list(combinations(indices, size))
        logger.info(choices_list)
        for choices in choices_list:
            logger.info(choices)
            chosen_buckets = [domain_list[i] for i in choices]
            def_pairs = []
            for bucket in chosen_buckets:
                def_pairs += bucket
            evaluate(args, def_pairs)
    elif (def_pairs_name.startswith(domain_prefix)):
        indices = np.arange(4) # 4 domains
        size = int(def_pairs_name[len(domain_prefix):])
        choices_list = list(combinations(indices, size))

        fixed_size = 1080
        domain_size = int(fixed_size / size)
        logger.info("{} samples per domain; domain: {}".format(domain_size, choices_list))
        domain_list = get_all_domains(domain_size)
        for choices in choices_list: 
            logger.info(choices)
            chosen_buckets = [domain_list[i] for i in choices]
            def_pairs = []
            for bucket in chosen_buckets: def_pairs += bucket
            evaluate(args, def_pairs)
    else:
        def_pairs = get_def_pairs(def_pairs_name)
        evaluate(args, def_pairs)


# class WordEvaluator(object):
# 	"""Evaluator for fastText"""
# 	def __init__(self, args):
# 		super(WordEvaluator, self).__init__()
# 		self.args = args

# 		# define files for evaluation
# 		self.filenames = []
# 		for i in [6, 7, 8]:
# 			self.filenames.append("sent-weat{}.jsonl".format(i))
# 			self.filenames.append("sent-weat{}b.jsonl".format(i))
# 		self.word_filenames = []
# 		for i in [6, 7, 8]:
# 			self.word_filenames.append("weat{}.jsonl".format(i))
# 			self.word_filenames.append("weat{}b.jsonl".format(i))

# 		self.vocab = self.init_vocab() # 190 words
# 		self.expand_specific_vocab()
		
# 		self.E = my_we.WordEmbedding(args.word_model, self.vocab)
# 		if (args.debias): self.debias()

# 	def init_vocab(self):
# 		print("Initializing vocab for evaluation...")
# 		vocab = set()
# 		for filename in self.filenames:
# 			sent_file = os.path.join(DATA_DIR, filename)
# 			data = load_json(sent_file)
# 			for key in ['targ1', 'targ2', 'attr1', 'attr2']:
# 				texts = data[key]['examples']
# 				for text in texts:
# 					words = set(tokenize(text))
# 					vocab = vocab.union(words)

# 		args = self.args
# 		with open(args.definitional_filename, "r") as f:
# 			definitional = json.load(f)

# 		with open(args.equalize_filename, "r") as f:
# 			equalize = json.load(f)

# 		with open(args.gendered_words_filename, "r") as f:
# 			gender_specific_words = json.load(f)
# 		print("gender specific", len(gender_specific_words), gender_specific_words[:10])

# 		for pair in definitional:
# 			vocab = vocab.union(set(pair))

# 		for pair in equalize:
# 			if (pair[0] in vocab): vocab.add(pair[1])
# 			if (pair[1] in vocab): vocab.add(pair[0])

# 		print("Vocabulary size {}.".format(len(vocab)))
# 		assert('gal' in vocab)

# 		self.definitional = definitional
# 		self.equalize = equalize
# 		self.gender_specific_words = gender_specific_words
		
# 		return vocab

# 	# expanding gender_specific_full to gender_specific_complete
# 	# with gender specific words from tests.
# 	def expand_specific_vocab(self):
# 		# expand gender specific words 
# 		gender_specific_words = set(self.gender_specific_words)
# 		for word_filename in self.word_filenames:
# 			word_file = os.path.join(DATA_DIR, word_filename)
# 			data = load_json(word_file)
# 			for key in ['targ1', 'targ2', "attr1", "attr2"]:
# 				category = data[key]["category"]
# 				print("category={}".format(category))
# 				if (not "male" in category.lower()): continue
# 				words = data[key]["examples"]
# 				print(words)
# 				gender_specific_words = gender_specific_words.union(set(words))
# 		self.gender_specific_words = list(gender_specific_words)

# 	def debias(self):
# 		print("debiasing...")
# 		definitional = self.definitional
# 		equalize = self.equalize
# 		gender_specific_words = self.gender_specific_words

# 		gender_subspace = my_we.doPCA(definitional, self.E).components_[:args.num_dimension]
# 		print("gender subspace shape: {}".format(gender_subspace.shape))
# 		specific_set = set(gender_specific_words)
# 		for i, w in enumerate(self.vocab):
# 			if (not isInSet(w, specific_set)):
# 				self.E.vecs[i] = my_we.dropspace(self.E.vecs[i], gender_subspace)
# 		self.E.normalize()

# 		# Equalize
# 		equalize_subset = []
# 		for pair in equalize:
# 			if (pair[0] in self.vocab):
# 				equalize_subset.append(pair)
# 		candidates = {x for e1, e2 in equalize_subset for x in [(e1.lower(), e2.lower()),
# 														 (e1.title(), e2.title()),
# 														 (e1.upper(), e2.upper())]}

# 		for (a, b) in candidates:
# 			if (a in self.E.index and b in self.E.index):
# 				y = my_we.drop((self.E.v(a) + self.E.v(b)) / 2, gender_direction)
# 				z = np.sqrt(1 - np.linalg.norm(y)**2)
# 				if (self.E.v(a) - self.E.v(b)).dot(gender_direction) < 0:
# 					z = -z
# 				self.E.vecs[self.E.index[a]] = z * gender_direction + y
# 				self.E.vecs[self.E.index[b]] = -z * gender_direction + y
# 		self.E.normalize()
# 		print("finished debiasing")


# 	def get_sent_embedding(self, sent):
# 		words = tokenize(sent)
# 		word_embeddings = np.array([self.E.v(w) for w in words]) # T x W(300)
# 		sent_embeddings = np.mean(word_embeddings, axis=0)
# 		return sent_embeddings

# 	def get_encodings(self, data):
# 		results = collections.defaultdict(dict)
# 		for key in ['targ1', 'targ2', 'attr1', 'attr2']:
# 			texts = data[key]['examples']
# 			category = data[key]['category'].lower()
# 			logger.info("category={}".format(category))

# 			results[key]['category'] = category
# 			encs = dict()
# 			for i, text in enumerate(texts):
# 				encs[text] = self.get_sent_embedding(text)
# 			results[key]['encs'] = encs

# 		return results

# 	def evaluate(self):
# 		args = self.args
# 		if (not os.path.exists(args.results_dir)): os.makedirs(args.results_dir)
# 		results_path = os.path.join(args.results_dir, args.output_name)
# 		results = []
# 		all_tests_dict = dict()

# 		for filename in self.filenames:
# 			sent_file = os.path.join(DATA_DIR, filename)
# 			data = load_json(sent_file)
# 			encs = self.get_encodings(data)
# 			esize, pval = weat.run_test(encs, n_samples=args.n_samples, parametric=args.parametric)

# 			result = "{}: esize={} pval={}".format(filename, esize, pval)
# 			print(filename, result)
# 			results.append(result)
# 			test_results = {"esize": esize, "pval": pval}
			
# 			all_tests_dict[filename] = test_results

# 		# print and save results
# 		for result in results: logger.info(result)
# 		save_dict_to_json(all_tests_dict, results_path)


def test_fastText():
	args = parse_args()
	evaluator = WordEvaluator(args)
	evaluator.evaluate()


def test_bertword():
	args = parse_args()
	def_pairs = json.load(open(args.definitional_filename, "r"))
	evaluate(args, def_pairs, word_level=True)


class BertEncoder(object):
	def __init__(self, model, device):
		self.device = device
		self.bert = model

	def encode(self, input_ids, token_type_ids=None, attention_mask=None, word_level=False):
		self.bert.eval()
		embeddings = self.bert(input_ids, token_type_ids=token_type_ids, 
			attention_mask=attention_mask, word_level=word_level, 
			remove_bias=False, bias_dir=None, encode_only=True)
		return embeddings


def get_tokenizer_encoder(args, device=None):
    '''Return BERT tokenizer and encoder based on args. Used in eval_bias.py.'''
    print("get tokenizer from {}".format(args.model_path))
    tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case=args.do_lower_case)
    # cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    # model_weights_path = args.model_path
    model = BertModel.from_pretrained(args.model_path)
    model.to(device)
    # model = BertForSequenceClassification.from_pretrained(model_weights_path,
    # 		  cache_dir=cache_dir,
    # 		  num_labels=2,
    # 		  normalize=args.normalize,
    # 		  tune_bert=args.tune_bert)
    # if (device != None): model.to(device)
    # bert_encoder = BertEncoder(model, device)

    return tokenizer, model
    

class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, guid, text_a, text_b=None, label=None):
		"""Constructs a InputExample.

		Args:
			guid: Unique id for the example.
			text_a: string. The untokenized text of the first sequence. For single
			sequence tasks, only this sequence must be specified.
			text_b: (Optional) string. The untokenized text of the second sequence.
			Only must be specified for sequence pair tasks.
			label: (Optional) string. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.text_a = text_a
		self.text_b = text_b
		self.label = label


class DualInputFeatures(object):
	"""A single set of dual features of data."""

	def __init__(self, input_ids_a, input_ids_b, mask_a, mask_b, segments_a, segments_b):
		self.input_ids_a = input_ids_a
		self.input_ids_b = input_ids_b
		self.mask_a = mask_a
		self.mask_b = mask_b
		self.segments_a = segments_a
		self.segments_b = segments_b


def convert_examples_to_dualfeatures(examples, label_list, max_seq_length, tokenizer, output_mode):
	"""Loads a data file into a list of dual input features."""
	'''
	output_mode: classification or regression
	'''	
	features = []
	for (ex_index, example) in enumerate(examples):
		if ex_index % 10000 == 0:
			logger.info("Writing example %d of %d" % (ex_index, len(examples)))

		tokens_a = tokenizer.tokenize(example.text_a)
		# truncate length
		if len(tokens_a) > max_seq_length - 2:
			tokens_a = tokens_a[:(max_seq_length - 2)]

		tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]
		segments_a = [0] * len(tokens_a)
		input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
		mask_a = [1] * len(input_ids_a)
		padding_a = [0] * (max_seq_length - len(input_ids_a))
		input_ids_a += padding_a
		mask_a += padding_a
		segments_a += padding_a
		assert(len(input_ids_a) == max_seq_length)
		assert(len(mask_a) == max_seq_length)
		assert(len(segments_a) == max_seq_length)

		if example.text_b:
			tokens_b = tokenizer.tokenize(example.text_b)
			if len(tokens_b) > max_seq_length - 2:
				tokens_b = tokens_b[:(max_seq_length - 2)]

			tokens_b = ["[CLS]"] + tokens_b + ["[SEP]"]
			segments_b = [0] * len(tokens_b)
			input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)
			mask_b = [1] * len(input_ids_b)
			padding_b = [0] * (max_seq_length - len(input_ids_b))
			input_ids_b += padding_b
			mask_b += padding_b
			segments_b += padding_b
			assert(len(input_ids_b) == max_seq_length)
			assert(len(mask_b) == max_seq_length)
			assert(len(segments_b) == max_seq_length)
		else:
			input_ids_b = None
			mask_b = None
			segments_b = None

		features.append(
				DualInputFeatures(input_ids_a=input_ids_a,
						     	  input_ids_b=input_ids_b,
								  mask_a=mask_a,
								  mask_b=mask_b,
								  segments_a=segments_a,
								  segments_b=segments_b))
	return features


def extract_embeddings(bert_encoder, tokenizer, examples, max_seq_length, device, 
		label_list, output_mode, norm, word_level=False):
    '''Encode examples into BERT embeddings in batches.'''
    features = convert_examples_to_dualfeatures(
        examples, label_list, max_seq_length, tokenizer, output_mode)
    all_inputs_a = torch.tensor([f.input_ids_a for f in features], dtype=torch.long)
    all_mask_a = torch.tensor([f.mask_a for f in features], dtype=torch.long)
    all_segments_a = torch.tensor([f.segments_a for f in features], dtype=torch.long)

    data = TensorDataset(all_inputs_a, all_mask_a, all_segments_a)
    dataloader = DataLoader(data, batch_size=8, shuffle=False)
    all_embeddings = []
    for step, batch in enumerate(tqdm(dataloader)):
        inputs_a, mask_a, segments_a = batch
        if (device != None):
            inputs_a = inputs_a.to(device)
            mask_a = mask_a.to(device)
            segments_a = segments_a.to(device)
        bert_encoder.eval()
        embeddings, _ = bert_encoder(input_ids=inputs_a, token_type_ids=segments_a, attention_mask=mask_a)
        # import pdb
        # pdb.set_trace()
        embeddings = embeddings[-1][0].cpu().detach().numpy()
        all_embeddings.append(embeddings)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings


def get_encodings(args, encs, tokenizer, bert_encoder, gender_space, device, 
        word_level=False, specific_set=None):
    '''Extract BERT embeddings from encodings dictionary.
        Perform the debiasing step if debias is specified in args.
    '''
    # if (word_level): assert(specific_set != None)

    logger.info("Get encodings")

    examples_dict = dict()
    for key in ['targ1', 'targ2', 'attr1', 'attr2']:
        texts = encs[key]['examples']
        category = encs[key]['category'].lower()
        examples = []
        encs[key]['text_ids'] = dict()
        for i, text in enumerate(texts):
            examples.append(InputExample(guid='{}'.format(i), text_a=text, text_b=None, label=None))
            encs[key]['text_ids'][i] = text
        examples_dict[key] = examples
        all_embeddings = extract_embeddings(bert_encoder, tokenizer, examples, args.max_seq_length, device, 
                    label_list=None, output_mode=None, norm=False, word_level=False)
        # logger.info("Debias category {}".format(category))

        emb_dict = {}
        for index, emb in enumerate(all_embeddings):
            emb /= np.linalg.norm(emb)
            # if (args.debias and not category in {'male','female'}): # don't debias gender definitional sentences
            # 	emb = my_we.dropspace(emb, gender_space)
            emb /= np.linalg.norm(emb) # Normalization actually doesn't affect e_size
            emb_dict[index] = emb

        encs[key]['encs'] = emb_dict
    return encs

def eval_seat(args):
    '''Evaluate bias level with given definitional sentence pairs.'''
    results_path = os.path.join(args.results_dir, args.output_name)

    # if (not args.encode_only):
    # 	if (os.path.exists(results_path)): 
    # 		print("Results already evaluated in {}".format(results_path))
    # 		return
    # 	if (not os.path.exists(args.results_dir)): os.makedirs(args.results_dir)

    results = []
    all_tests_dict = dict()

    tokenizer, bert_encoder = get_tokenizer_encoder(args, DEVICE)
    print("tokenizer: {}".format(tokenizer==None))
    gender_subspace = None

    # if (args.debias):
    # 	gender_subspace = compute_gender_dir(DEVICE, tokenizer, bert_encoder, def_pairs, 
    # 		args.max_seq_length, k=args.num_dimension, load=True, task=args.model, word_level=word_level, keepdims=True)
    # 	logger.info("Computed (gender) bias direction")

    # with open(args.gendered_words_filename, "r") as f:
    # 	gender_specific_words = json.load(f)
    # specific_set = set(gender_specific_words)
    abs_esizes = []
    for test_id in ['6', '6b', '7', '7b', '8', '8b']:
        filename = "sent-weat{}.jsonl".format(test_id)
        sent_file = os.path.join(DATA_DIR, filename)
        data = load_json(sent_file)

        encs = get_encodings(args, data, tokenizer, bert_encoder, gender_subspace, 
            DEVICE)
        if (args.encode_only):
            if (args.debias):
                outfile_name = 'debiased_encs{}.pkl'.format(test_id)
            else:
                outfile_name = 'biased_encs{}.pkl'.format(test_id)

            with open(os.path.join(args.results_dir, outfile_name), 'wb') as outfile:
                pickle.dump(encs, outfile)
            continue
        '''
        encs: targ1, targ2, attr1, attr2
                    -> category
                    -> encs
                    -> (id1, sent1_emb), (id2, sent2_emb), ...
        '''

        esize, pval = weat.run_test(encs, n_samples=args.n_samples, parametric=args.parametric)
        abs_esizes.append(abs(esize))

        result = "{}: esize={} pval={}".format(filename, esize, pval)
        print(filename, result)
        results.append(result)
        test_results = {"esize": esize, "pval": pval}
        
        all_tests_dict[filename] = test_results

    avg_absesize = np.mean(np.array(abs_esizes))
    print("Averge of Absolute esize: {}".format(avg_absesize))
    all_tests_dict['avg_absesize'] = avg_absesize

    if (args.encode_only): return

    # print and save results
    for result in results: logger.info(result)
    save_dict_to_json(all_tests_dict, results_path)

    return



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        type=str,
                        default="bert-base-uncased",
                        help="Path of the model to be evaluated")
    parser.add_argument("--debias",
                        action='store_true',
                        help="Whether to debias.")
    parser.add_argument("--equalize",
                        action='store_true',
                        help="Whether to equalize.")
    parser.add_argument("--def_pairs_name", default="all", type=str,
                        help="Name of definitional sentence pairs.")
    parser.add_argument("--model", "-m", type=str, default="dummy")
    parser.add_argument("--output_name", default="seat.txt", type=str)
    parser.add_argument("--results_dir", default="./results/", type=str,
                        help="directory for storing results")
    parser.add_argument("--encode_only", action='store_true')
    parser.add_argument("--num_dimension", "-k", type=int, default=1,
                        help="dimensionality of bias subspace")
    args = parser.parse_args()
    if (args.output_name == None):
        args.output_name = args.def_pairs_name if args.debias else "biased"
    print("outputname: {}".format(args.output_name))
    if (args.results_dir == None):
        args.results_dir = os.path.join("results", args.model)
    args.do_lower_case = True
    args.cache_dir = None
    args.local_rank = -1
    args.max_seq_length = 128
    args.eval_batch_size = 8
    args.n_samples = 100000
    args.parametric = True
    args.tune_bert = False
    args.normalize = True

    # word embeddings
    args.word_model = 'fasttext-wiki-news-subwords-300'
    wedata_path = 'my_debiaswe/data'
    args.wedata_path = wedata_path
    args.definitional_filename = os.path.join(wedata_path, 'definitional_pairs.json')
    args.equalize_filename = os.path.join(wedata_path, 'equalize_pairs.json')
    args.gendered_words_filename = os.path.join(wedata_path, 'gender_specific_complete.json')

    eval_seat(args)





