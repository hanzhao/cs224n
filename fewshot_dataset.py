# lm-bff/.../dataset.py

import os
import copy
import logging
import torch
import numpy as np
import time
from filelock import FileLock
import json
import itertools
import random
import transformers
from transformers.data.processors.utils import InputFeatures
from transformers import DataProcessor, InputExample
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer, util
from copy import deepcopy
import pandas as pd

# from src.processors import
# processors_mapping: this should always just be default QAProcessor, prepare_eval_data and prepare_train_data
# num_labels_mapping: # of possible start/end locations. len(context), get_labels should return range(len(context))
# output_modes_mapping: classification
# compute_metrics_mapping: F1 and EM

# For regression task only:
# This is not used for QA. Values are assigned for correctness.
# median_mapping = {
#     "sts-b": 2.5
# }

bound_mapping = {
    "sts-b": (0, 5)
}

logger = logging.getLogger(__name__)

# c_handler = logging.StreamHandler()
# c_handler.setLevel(logging.INFO)
# c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
# c_handler.setFormatter(c_format)
# logger.addHandler(c_handler)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

@dataclass(frozen=True)
class OurInputFeatures(InputFeatures):
    """
    Inherit from Transformers' InputFeatuers.
    """

    # INFO: the following two features are shared btw QA and in-context
    # TODO(chen): make sure they fit. especially with the new loss functions.
    # id: List[int]
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    # start/end positions are inherited from QA
    # They should only be present in "train" and NOT "eval"
    # FewShotDataset should check for that during tokenization
    # TODO(chen): they should be assigned with in-context labeling process but
    # with QA labels.
    start_positions: Optional[List[int]] = None
    end_positions: Optional[List[int]] = None
    # TODO(chen, shan): in-context features, see how they can become useful.
    token_type_ids: Optional[List[int]] = None
    # label: Optional[Union[int, float]] = None
    mask_pos: Optional[List[int]] = None # Position of the mask token
    label_word_list: Optional[List[int]] = None # Label word mapping (dynamic)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

# This dataclass is necessary due to the column to row example transformation
# needed for demonstration rearrangement.
@dataclass
class OurInputExample(InputExample):
    """
    Inherit from Transformers' InputExample.
    """

    offset_mapping: List[int] = None
    sequence_ids: List[int] = None
    overflow_to_sample_mapping: Optional[List[int]] = None

def input_example_to_string(example, sep_token): 
    if example.text_b is None:
        return example.text_a
    else:
        # Warning: very simple hack here
        return example.text_a + ' ' + sep_token + ' ' + example.text_b

def input_example_to_tuple(example): 
    if example.text_b is None:
        if pd.isna(example.text_a) or example.text_a is None:
            return ['']
            logger.warn("Empty input")
        else:
            return [example.text_a]
    else:
        return [example.text_a, example.text_b]

# def input_example_to_boundary(example):
#     if example.start_pos is None or example.end_pos is None:
#         return None
#     else:
#         # TODO(chen): might need to use mapping for non trivial loss fn.
#         return [example.start_pos, example.end_pos]

def tokenize_multipart_input(
    example,
    input_text_list,
    max_length,
    tokenizer,
    task_name=None, 
    prompt=False, 
    template=None,
    label_word_list=None, 
    first_sent_limit=None,
    other_sent_limit=None,
    gpt3=False,
    truncate_head=False,
    support_labels=None,
):
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    input_ids = []
    attention_mask = []
    token_type_ids = [] # Only for BERT
    mask_pos = None # Position of the mask token
    input_q_boundary = None

    if prompt:
        """
        Concatenate all sentences and prompts based on the provided template.
        Template example: '*cls*It was*mask*.*sent_0**<sep>*label_0:*sent_1**<sep>**label_1*:*sent_2**<sep>*'
        *xx* represent variables:
            *cls*: cls_token
            *mask*: mask_token
            *sep*: sep_token
            *sep+*: sep_token, also means +1 for segment id
            *sent_i*: sentence i (input_text_list[i])
            *sent-_i*: same as above, but delete the last token
            *sentl_i*: same as above, but use lower case for the first word
            *sentl-_i*: same as above, but use lower case for the first word and delete the last token
            *+sent_i*: same as above, but add a space before the sentence
            *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
            *label_i*: label_word_list[i]
            *label_x*: label depends on the example id (support_labels needed). this is only used in GPT-3's in-context learning
        Use "_" to replace space.
        PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.
        """
        assert template is not None

        special_token_mapping = {
            'cls': tokenizer.cls_token_id, 'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id, 'sep+': tokenizer.sep_token_id, 
        }
        template_list = template.split('*') # Get variable list in the template
        # TODO(chen): what does segment_id, token_type_ids and sep+ do for BERT?
        segment_id = 0 # Current segment id. Segment id +1 if encountering sep+.
        # INFO reference for prompt offset.
        original_sep = 0

        # INFO: enc no longer necessary for tokenized encodings from input_text_list.
        # but still necessary for added spaces, prompts, etc.
        # Special characters are handled via map.
        for part_id, part in enumerate(template_list):
            new_tokens = []
            segment_plus_1_flag = False

            if part in special_token_mapping:
                if part == 'cls' and 'T5' in type(tokenizer).__name__:
                    # T5 does not have cls token
                    continue
                new_tokens.append(special_token_mapping[part])
                if part == 'sep+':
                    segment_plus_1_flag = True
            elif part[:6] == 'label_':
                # Note that label_word_list already has extra space, so do not add more space ahead of it.
                label_id = int(part.split('_')[1])
                label_word = label_word_list[label_id]
                new_tokens.append(label_word)
            elif part[:7] == 'labelx_':
                instance_id = int(part.split('_')[1])
                label_id = support_labels[instance_id]
                label_word = label_word_list[label_id]
                new_tokens.append(label_word)
            elif part[:5] == 'sent_':
                sent_id = int(part.split('_')[1])
                # new_tokens += enc(input_text_list[sent_id]) 
                new_tokens += input_text_list[sent_id]
                # INFO: record the the original <sep> index as reference for offset.
                if sent_id == 0:
                    original_sep = len(new_tokens) + len(input_ids)
            elif part[:6] == '+sent_':
                # Add space
                sent_id = int(part.split('_')[1])
                # new_tokens += enc(' ' + input_text_list[sent_id])
                new_tokens += enc(' ') + input_text_list[sent_id]
            elif part[:6] == 'sent-_':
                # Delete the last token
                sent_id = int(part.split('_')[1])
                # new_tokens += enc(input_text_list[sent_id][:-1])
                new_tokens += input_text_list[sent_id][:-1]
            # INFO: avoid dealing with upper() and lower() of embeddings
            # QA task uses uncased DistilBert anyway.
                '''
                elif part[:6] == 'sentl_':
                    # Lower case the first token
                    sent_id = int(part.split('_')[1])
                    text = input_text_list[sent_id]
                    text = text[:1].lower() + text[1:]
                    new_tokens += text
                elif part[:7] == '+sentl_':
                    # Lower case the first token and add space 
                    sent_id = int(part.split('_')[1])
                    text = input_text_list[sent_id]
                    text = text[:1].lower() + text[1:]
                    # new_tokens += enc(' ' + text)
                    new_tokens += enc(' ' + text)
                elif part[:7] == 'sentl-_':
                    # Lower case the first token and discard the last token
                    sent_id = int(part.split('_')[1])
                    text = input_text_list[sent_id]
                    text = text[:1].lower() + text[1:]
                    new_tokens += enc(text[:-1])
                elif part[:6] == 'sentu_':
                    # Upper case the first token
                    sent_id = int(part.split('_')[1])
                    text = input_text_list[sent_id]
                    text = text[:1].upper() + text[1:]
                    new_tokens += enc(text)
                elif part[:7] == '+sentu_':
                    # Upper case the first token and add space
                    sent_id = int(part.split('_')[1])
                    text = input_text_list[sent_id]
                    text = text[:1].upper() + text[1:]
                    new_tokens += enc(' ' + text)
                '''
            else:
                # Just natural language prompt
                part = part.replace('_', ' ') 
                # handle special case when T5 tokenizer might add an extra space
                if len(part) == 1:
                    # new_tokens.append(tokenizer._convert_token_to_id(part))
                    new_tokens.append(tokenizer.convert_tokens_to_ids(part)[0])                    
                else:
                    new_tokens += enc(part)
            
            # TODO(chen): this truncation might affect QA pair quality.
            # Not using sent_limit for now!
            if part[:4] == 'sent' or part[1:5] == 'sent':
                # If this part is the sentence, limit the sentence length
                sent_id = int(part.split('_')[1])
                if sent_id == 0:
                    if first_sent_limit is not None:
                        new_tokens = new_tokens[:first_sent_limit]
                else:
                    if other_sent_limit is not None:
                        new_tokens = new_tokens[:other_sent_limit]

            input_ids += new_tokens
            attention_mask += [1 for i in range(len(new_tokens))]
            token_type_ids += [segment_id for i in range(len(new_tokens))]

            if segment_plus_1_flag:
                segment_id += 1

        # INFO: calculate prompt_offset
        new_sep_idx = input_ids.index(tokenizer.sep_token_id)
        prompt_offset = new_sep_idx - original_sep
        assert prompt_offset > 0

        # INFO: Use this offset to also pad offset_mapping and sequence_ids
        offset_mapping = example.offset_mapping[:original_sep] + [(0, 0)] * prompt_offset + example.offset_mapping[original_sep:]
        # TODO(chen): test accuracy of this offset by making sure boundaries are right: 1 starts at context
        # TODO(chen): test if it is appropriate for prompt to receive None seq_ids.
        sequence_ids = example.sequence_ids[:original_sep] + [None] * prompt_offset + example.sequence_ids[original_sep:]

        # TODO(chen): make sure <cls> is the only condition where boundary shouldn't move
        if example.label is not None:
            input_q_boundary=example.label
            # INFO: assuming <cls> is at [0]
            assert input_ids.index(tokenizer.cls_token_id) == 0
            if input_q_boundary is not None and input_q_boundary[0] != 0 and input_q_boundary[1] != 0:
                # logger.info("input_q_boundary: %s", input_q_boundary)
                input_q_boundary[0] = input_q_boundary[0] + prompt_offset
                input_q_boundary[1] = input_q_boundary[1] + prompt_offset
    else:
        input_ids = [tokenizer.cls_token_id]
        attention_mask = [1]
        token_type_ids = [0]

        for sent_id, input_text in enumerate(input_text_list):
            if input_text is None:
                # Do not have text_b
                continue
            if pd.isna(input_text) or input_text is None:
                # Empty input
                input_text = ''
            input_tokens = enc(input_text) + [tokenizer.sep_token_id]
            input_ids += input_tokens
            attention_mask += [1 for i in range(len(input_tokens))]
            token_type_ids += [sent_id for i in range(len(input_tokens))]

        if 'T5' in type(tokenizer).__name__: # T5 does not have CLS token
            input_ids = input_ids[1:]
            attention_mask = attention_mask[1:]
            token_type_ids = token_type_ids[1:]

    # Padding
    if first_sent_limit is not None and len(input_ids) > max_length:
        # If using sentence limit, the total length still exceeds the maximum limit, report a warning
        logger.warn("Input exceeds max_length limit: {}".format(tokenizer.decode(input_ids)))

    while len(input_ids) < max_length:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)
        token_type_ids.append(0)

    # Truncate
    if len(input_ids) > max_length:
        if truncate_head:
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]
            token_type_ids = token_type_ids[-max_length:]
        else:
            # Default is to truncate the tail
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]

    # Find mask token
    if prompt:
        mask_pos = [i for i, t in enumerate(input_ids) if t == tokenizer.mask_token_id]
        # mask_pos = [input_ids.index(tokenizer.mask_token_id)]
        # Make sure that the masked position is inside the max_length
        assert mask_pos[0] < max_length
        assert len(mask_pos) == 2

    result = {'id': example.guid, 'input_ids': input_ids, 'attention_mask': attention_mask, \
              'offset_mapping': offset_mapping, 'sequence_ids': sequence_ids}
    if 'BERT' in type(tokenizer).__name__:
        # Only provide token type ids for BERT
        result['token_type_ids'] = token_type_ids

    if prompt:
        result['mask_pos'] = mask_pos

    # INFO: return the offsetted q_boundary
    if input_q_boundary is not None:
        result['start_positions'] = input_q_boundary[0]
        result['end_positions'] = input_q_boundary[1]
        result['overflow_to_sample_mapping'] = example.overflow_to_sample_mapping

    return result

class FewShotDataset(torch.utils.data.Dataset):
    """Few-shot dataset."""

    def __init__(self, args, tokenizer, qa_data_encodings_features, qa_data_encodings_features_train_for_test = None, data_dict = None, cache_dir=None, mode="train", use_demo=False):
        self.args = args
        # INFO: only one QA Task
        # self.task_name = args.task_name
        # self.processor = processors_mapping[args.task_name]
        self.encodings = qa_data_encodings_features
        self.encodings['sequence_ids'] = []
        self.encodings['mask_pos'] = []
        self.encodings['token_type_ids'] = []
        # self.qa_data_enc_f = qa_data_encodings_features
        self.data_dict = data_dict

        self.tokenizer = tokenizer
        self.mode = mode

        # If not using demonstrations, use use_demo=True
        self.use_demo = use_demo
        if self.use_demo:
            logger.info("Use demonstrations")
        
        print(mode)
        assert mode in ["train", "dev", "test"]

        self.keys = ['input_ids', 'attention_mask']
        if args.prompt:
            self.keys += ['mask_pos']
        if mode == 'train':
            self.keys += ['start_positions', 'end_positions']

        # INFO
        # Get label list and (for prompt) label word list
        # self.label_list = self.processor.get_labels()
        # self.max_length should be 512 for GPT just in case
        if self.args.double_demo:
            self.max_length = self.args.max_seq_length * 2
        else:
            self.max_length = self.args.max_seq_length
        # if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
        #     # When using GPT-3's in-context learning, take the maximum tokenization length of the model (512)
        #     self.max_length = 512
        # TODO(chen): figure out relative index vs absolute
        self.label_list = range(self.max_length)
        self.label_list = [str(i) for i in self.label_list]
        self.num_labels = len(self.label_list)

        if args.prompt:
            # TODO(chen): Remember to use a trivial (args.mapping) for index when passing in args
            assert args.mapping is not None
            self.label_to_word = eval(args.mapping)

            for key in self.label_to_word:
                # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
                if self.label_to_word[key][0] not in ['<', '[', '.', ',']:
                    # Make sure space+word is in the vocabulary
                    # INFO: This is needed for correctness during template processing.
                    # INFO: missing some tokens, and use convert_tokens_to_ids instead
                    if len(tokenizer.tokenize(' ' + self.label_to_word[key])) != 1:
                        num_added_toks = tokenizer.add_tokens(' ' + self.label_to_word[key])
                        assert len(tokenizer.tokenize(' ' + self.label_to_word[key])) == 1
                    # assert len(tokenizer.tokenize(' ' + self.label_to_word[key])) == 1
                    # self.label_to_word[key] = tokenizer._convert_token_to_id(tokenizer.tokenize(' ' + self.label_to_word[key])[0])
                    self.label_to_word[key] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' ' + self.label_to_word[key]))[0]
                else:
                    # self.label_to_word[key] = tokenizer._convert_token_to_id(self.label_to_word[key])
                    self.label_to_word[key] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' ' + self.label_to_word[key]))[0]
                # logger.info("Label {} to word {} ({})".format(key, tokenizer._convert_id_to_token(self.label_to_word[key]), self.label_to_word[key]))
            
            if len(self.label_list) > 1:
                self.label_word_list = [self.label_to_word[label] for label in self.label_list]
            else:
                # Regression task
                # '0' represents low polarity and '1' represents high polarity.
                self.label_word_list = [self.label_to_word[label] for label in ['0', '1']]
        else:
            self.label_to_word = None
            self.label_word_list = None

        # Multiple sampling: when using demonstrations, we sample different combinations of demonstrations during 
        # inference and aggregate the results by averaging the logits. The number of different samples is num_sample.
        if (mode == "train") or not self.use_demo:
            # We do not do multiple sampling when not using demonstrations or when it's the training mode 
            self.num_sample = 1
        else:
            self.num_sample = args.num_sample

        # If we use multiple templates, we also need to do multiple sampling during inference.
        if args.prompt and args.template_list is not None:
            logger.info("There are %d templates. Multiply num_sample by %d" % (len(args.template_list), len(args.template_list)))
            self.num_sample *= len(args.template_list)
                
        logger.info("Total num_sample for mode %s: %d" % (mode, self.num_sample))

        '''
        # TODO(chen): Properly configure cache, or just test without it.
        # Load cache
        # Cache name distinguishes mode, task name, tokenizer, and length. So if you change anything beyond these elements, make sure to clear your cache.
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.task_name,
            ),
        )

        logger.info(f"Creating/loading examples from dataset file at {args.data_dir}")

        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.support_examples, self.query_examples = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                # TODO(chen): read from tokenized QA task instead of disk.

                # The support examples are sourced from the training set.
                # INFO: Intuitively, support_examples should ALL come from labelled examples from indomain_train and oodomain_train,
                # since eval and test datasets' "golden answers" should not be exposed to the model
                # in the form of demonstration.
                # TODO(chen): confirm the following order:
                # train: indomain + oodomain together in an interleaving manner
                # eval: indomain then outdomain.
                # TODO(chen): read the different ways in which eval and train datasets are
                # prepared and make sure that QA and in-context are compatible.
                # TODO(chen): why does feature set contains "start" and "end" in dataset anyway?
                # Aren't we have a data_dict available at the side anyway? Look at QA.
                # self.support_examples = self.processor.get_train_examples(args.data_dir)

                # if mode == "dev":
                #     self.query_examples = self.processor.get_dev_examples(args.data_dir)
                # elif mode == "test":
                #     self.query_examples = self.processor.get_test_examples(args.data_dir)
                # else:
                #     self.query_examples = self.support_examples

                # data_encodings contains tokenized encoding.
                # For training: self.keys = ['input_ids', 'attention_mask'] + ['start_positions', 'end_positions']
                # For eval: self.keys = ['input_ids', 'attention_mask']
                # Also contains QA specific features from BertTokenizer: ["overflow_to_sample_mapping"] + ["offset_mapping"]
                self.support_examples = []
                self.query_examples = []

                start = time.time()
                torch.save([self.support_examples, self.query_examples], cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )
        '''
        # # INFO: creating row examples (only inputs) from QA Tokenized features.
        # assert 'input_ids' in self.qa_data_enc_f
        # assert 'id' in self.qa_data_enc_f
        # # Attention mask is NOT passed. In-context tokenization generates its own attention_masks.
        # # assert 'attention_mask' in self.qa_data_enc_f
        # if mode == 'train':
        #     assert 'start_positions' in self.qa_data_enc_f
        #     assert 'end_positions' in self.qa_data_enc_f
        # else:
        #     assert 'input_ids' in qa_data_encodings_features_train_for_test
        #     assert 'id' in qa_data_encodings_features_train_for_test
        #     assert 'start_positions' in qa_data_encodings_features_train_for_test
        #     assert 'end_positions' in qa_data_encodings_features_train_for_test

        # Unlike original impl, query_examples for all 3 cases ['train', 'dev', 'test'] comes from the same source.
        self.query_examples = []
        # self._populate_examples(self.query_examples, self.qa_data_enc_f, mode)
        self._populate_examples(self.query_examples, self.encodings, mode)
        logger.info("self.query_examples has: %s examples.", len(self.query_examples))

        # INFO: For demonstration to work in test/dev, use supporting examples from train.
        # This is true for both indomain and oodomain.
        self.support_examples = []
        if self.use_demo:
            if mode == 'train':
                self.support_examples = self.query_examples
            else:
                # Note that the decoding mode must be train to retrieve demonstration with golden labels for demo.
                self._populate_examples(self.support_examples, qa_data_encodings_features_train_for_test, 'train')
        logger.info("self.support_examples has: %s examples.", len(self.support_examples))
        # TODO(chen): Not using filtering for first ver. As stretch goal.
        '''
                # For filtering in using demonstrations, load pre-calculated embeddings
                if self.use_demo and args.demo_filter:
                    split_name = ''
                    if mode == 'train':
                        split_name = 'train'
                    elif mode == 'dev':
                        if args.task_name == 'mnli':
                            split_name = 'dev_matched'
                        elif args.task_name == 'mnli-mm':
                            split_name = 'dev_mismatched'
                        else:
                            split_name = 'dev'
                    elif mode == 'test':
                        if args.task_name == 'mnli':
                            split_name = 'test_matched'
                        elif args.task_name == 'mnli-mm':
                            split_name = 'test_mismatched'
                        else:
                            split_name = 'test'
                    else:
                        raise NotImplementedError

                    self.support_emb = np.load(os.path.join(args.data_dir, "train_{}.npy".format(args.demo_filter_model)))
                    self.query_emb = np.load(os.path.join(args.data_dir, "{}_{}.npy".format(split_name, args.demo_filter_model)))
                    logger.info("Load embeddings (for demonstration filtering) from {}".format(os.path.join(args.data_dir, "{}_{}.npy".format(split_name, args.demo_filter_model))))

                    assert len(self.support_emb) == len(self.support_examples)
                    assert len(self.query_emb) == len(self.query_examples)
        '''
        # Size is expanded by num_sample
        self.size = len(self.query_examples) * self.num_sample
        
        # Prepare examples (especially for using demonstrations)
        support_indices = list(range(len(self.support_examples)))
        self.example_idx = []
        for sample_idx in range(self.num_sample):
            for query_idx in range(len(self.query_examples)):
                # If training, exclude the current example. Else keep all.
                if self.use_demo and args.demo_filter:
                    # TODO(chen): Not using filtering for first ver. As stretch goal.
                    raise ValueError("Should not using filtering!!")
                    '''
                    # Demonstration filtering
                    candidate = [support_idx for support_idx in support_indices
                                   if support_idx != query_idx or mode != "train"]
                    sim_score = []
                    for support_idx in candidate:
                        sim_score.append((support_idx, util.pytorch_cos_sim(self.support_emb[support_idx], self.query_emb[query_idx])))
                    sim_score.sort(key=lambda x: x[1], reverse=True)
                    if self.num_labels == 1:
                        # Regression task
                        limit_each_label = int(len(sim_score) // 2 * args.demo_filter_rate)
                        count_each_label = {'0': 0, '1': 0}
                        context_indices = []

                        if args.debug_mode:
                            print("Query %s: %s" % (self.query_examples[query_idx].label, self.query_examples[query_idx].text_a)) # debug
                        for support_idx, score in sim_score:
                            if count_each_label['0' if float(self.support_examples[support_idx].label) <= median_mapping[args.task_name] else '1'] < limit_each_label:
                                count_each_label['0' if float(self.support_examples[support_idx].label) <= median_mapping[args.task_name] else '1'] += 1
                                context_indices.append(support_idx)
                                if args.debug_mode:
                                    print("    %.4f %s | %s" % (score, self.support_examples[support_idx].label, self.support_examples[support_idx].text_a)) # debug
                    else:
                        limit_each_label = int(len(sim_score) // self.num_labels * args.demo_filter_rate)
                        count_each_label = {label: 0 for label in self.label_list}
                        context_indices = []

                        if args.debug_mode:
                            print("Query %s: %s" % (self.query_examples[query_idx].label, self.query_examples[query_idx].text_a)) # debug
                        for support_idx, score in sim_score:
                            if count_each_label[self.support_examples[support_idx].label] < limit_each_label:
                                count_each_label[self.support_examples[support_idx].label] += 1
                                context_indices.append(support_idx)
                                if args.debug_mode:
                                    print("    %.4f %s | %s" % (score, self.support_examples[support_idx].label, self.support_examples[support_idx].text_a)) # debug
                    '''
                else:
                    # Using demonstrations without filtering
                    # INFO: it is ok to have overlapping examples across samples. This will largely go away with sampling.
                    # For non-train we can just use training set as support so ALL
                    # can be used as support.
                    context_indices = [support_idx for support_idx in support_indices
                               if support_idx != query_idx or mode != "train"]

                # We'll subsample context_indices further later.
                self.example_idx.append((query_idx, context_indices, sample_idx))

        # If it is not training, we pre-process the data; otherwise, we process the data online.
        # INFO: Don't process data online for correctness.
        # if mode != "train":
        if True:
            # self.features = []
            
            _ = 0
            for query_idx, context_indices, bootstrap_idx in self.example_idx:
                # The input (query) example
                example = self.query_examples[query_idx]
                # The demonstrations
                supports = self.select_context([self.support_examples[i] for i in context_indices])

                if args.template_list is not None:
                    template = args.template_list[sample_idx % len(args.template_list)] # Use template in order
                else:
                    template = args.template

                # self.features.append(self.convert_fn(
                #     example=example,
                #     supports=supports,
                #     use_demo=self.use_demo,
                #     label_list=self.label_list,
                #     prompt=args.prompt,
                #     template=template,
                #     label_word_list=self.label_word_list,
                #     verbose=True if _ == 0 else False,
                # ))

                features=self.convert_fn(
                    example=example,
                    supports=supports,
                    use_demo=self.use_demo,
                    label_list=self.label_list,
                    prompt=args.prompt,
                    template=template,
                    label_word_list=self.label_word_list,
                    verbose=True if _ == 0 else False,
                )
                # for key, val in features.items():
                #     self.encodings[key].append(val)
                # assert(all(key in self.encodings for key in self.keys))

                # INFO: actually updates the encoding, crucial for evaluation
                for key, val in features.items():
                    if key in ['sequence_ids', 'mask_pos', 'token_type_ids']:
                        self.encodings[key].append(val)
                    self.encodings[key][query_idx] = val

                _ += 1
        '''
        else:
            # self.features = None
            self.encodings = None
        '''

    def select_context(self, context_examples):
        """
        Select demonstrations from provided examples.
        """
        # INFO deals with 0 context_example for prompting with out demo.
        if len(context_examples) == 0:
            return []
        max_demo_per_label = 1
        counts = {k: 0 for k in self.label_list}
        if len(self.label_list) == 1:
            # Regression
            counts = {'0': 0, '1': 0}
        selection = []
        
        if False:
            '''
            if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
                # For GPT-3's in-context learning, we sample gpt3_in_context_num demonstrations randomly. 
                order = np.random.permutation(len(context_examples))
                for i in range(min(self.args.gpt3_in_context_num, len(order))):
                    selection.append(context_examples[order[i]])
            '''
        else:
            # Our sampling strategy
            order = np.random.permutation(len(context_examples))

            for i in order:
                label = context_examples[i].label
                # if len(self.label_list) == 1:
                    # Regression
                    # label = '0' if float(label) <= median_mapping[self.args.task_name] else '1'
                if counts[label] < max_demo_per_label:
                    selection.append(context_examples[i])
                    counts[label] += 1
                if sum(counts.values()) == len(counts) * max_demo_per_label:
                    break
        
            assert len(selection) > 0
        
        return selection

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        # INFO: don't calculate online to mirror QADataset's behavior just in case.
        # if self.features is None:
        if False:
            '''
        if self.encodings is None:
            query_idx, context_indices, bootstrap_idx = self.example_idx[i]
            # The input (query) example
            example = self.query_examples[query_idx]
            # The demonstrations
            supports = self.select_context([self.support_examples[i] for i in context_indices])

            if self.args.template_list is not None:
                template = self.args.template_list[sample_idx % len(self.args.template_list)]
            else:
                template = self.args.template

            # logger.info("raw example from QATokenizer: %s", example)

            all_features = self.convert_fn(
                example=example,
                supports=supports,
                use_demo=self.use_demo,
                label_list=self.label_list,
                prompt=self.args.prompt,
                template=template,
                label_word_list=self.label_word_list,
                # verbose=False,
                # INFO for testing only. Remember to switch it back for training.
                verbose=True,
            )
            features={key : torch.tensor(all_features[key]) for key in self.keys}
            # INFO: should have all 4 keys
            # logger.info("features for train: %s", features)
            '''
        else:
            # features = self.features[i]
            # INFO: should only contains ids and mask
            features={key : torch.tensor(self.encodings[key][i]) for key in self.keys}
            # logger.info("features for eval: %s", features)
        # TODO(chen)
        # if train: return more keys, if others, don't...
        # ['start'] and 'end' should be returned with an offset???? Since context has actually moved further right!!!!!!!
        # Read prepare_train_data to make sure that start and finish should actually be offset in both cases...
        # e.g. Sometimes the start and end both point to <cls>
        return features

    def _populate_examples(self, examples, enc_f, enc_mode):
        # Creating row examples from QA Tokenized features.
        # 1-to-1 index mapping btw self.query_examples and self.encoding.
        # When it comes the time to update, Please directy updates the self.encoding
        # for fidality which will be crucial for eval step later.
        assert 'input_ids' in enc_f
        assert 'id' in enc_f
        assert 'offset_mapping' in enc_f
        assert enc_f.sequence_ids() is not None
        for i, (enc, qid, offset_mapping) in enumerate(zip(enc_f['input_ids'], enc_f['id'], enc_f["offset_mapping"])):
            # 'attention_mask' is NOT passed. In-context tokenization generates its own attention_masks.
            seq_ids = enc_f.sequence_ids(i)
            # logger.info("original seq_ids: %s" % [(j, seq_id) for j, seq_id in enumerate(seq_ids)])
            # logger.info("original offset_mapping: %s" % offset_mapping)

            # text_a is question sentence, and text_b is context sentence.
            q_start = enc.index(self.tokenizer.cls_token_id) + 1
            q_end = enc.index(self.tokenizer.sep_token_id)
            assert seq_ids[q_end] == None
            assert seq_ids[q_end+1] == 1
            text_a = enc[q_start:q_end]

            c_start = q_end + 1
            c_end = enc.index(self.tokenizer.sep_token_id, c_start)
            text_b = enc[c_start:c_end]
            # logger.info("c_end: %s", c_end)
            assert seq_ids[c_end] == None
            assert seq_ids[c_end - 1] == 1
            # enc_mode == mode when retrieving query_examples.
            # enc_mode == 'train' when retrieving support_examples for test/dev query_examples
            if enc_mode=='train':
                assert 'start_positions' in enc_f
                assert 'end_positions' in enc_f
                assert 'overflow_to_sample_mapping' in enc_f
                start_pos = enc_f['start_positions'][i]
                end_pos = enc_f['end_positions'][i]
                sample_mapping = enc_f['overflow_to_sample_mapping'][i]
                examples.append(OurInputExample(guid=qid, text_a=text_a, text_b=text_b, \
                                             offset_mapping=offset_mapping, sequence_ids=seq_ids, \
                                             label=[start_pos, end_pos], overflow_to_sample_mapping=sample_mapping))
            else:
                examples.append(OurInputExample(guid=qid, text_a=text_a, text_b=text_b, \
                                                offset_mapping=offset_mapping, sequence_ids=seq_ids))

    def get_labels(self):
        return self.label_list

    def convert_fn(
        self,
        example,
        supports,
        use_demo=False,
        label_list=None,
        prompt=False,
        template=None,
        label_word_list=None,
        verbose=False
    ):
        """
        Returns a list of processed "InputFeatures".
        """
        # args.max_seq_length should be the same length as QATokenizer at 384.
        # use --double_demo for demonstration.
        max_length = self.args.max_seq_length

        # Prepare labels
        label_map = {label: i for i, label in enumerate(label_list)} # Mapping the label names to label ids
        if len(label_list) == 1:
            # Regression
            label_map = {'0': 0, '1': 1}

        # INFO: example_label is NOT in use for QA which has 2 labels.
        '''
        # Get example's label id (for training/inference)
        if example.label is None:
            example_label = None
        elif len(label_list) == 1:
            # Regerssion
            example_label = float(example.label)
        else:
            example_label = label_map[example.label]
        '''

        # Prepare other features
        if not use_demo:
            # No using demonstrations
            inputs = tokenize_multipart_input(
                example=example,
                input_text_list=input_example_to_tuple(example),
                max_length=max_length,
                tokenizer=self.tokenizer,
                # task_name=self.args.task_name,
                prompt=prompt,
                template=template,
                label_word_list=label_word_list,
                # first_sent_limit=self.args.first_sent_limit,
                # other_sent_limit=self.args.other_sent_limit,
            )

            # features = OurInputFeatures(**inputs, label=example_label)
            # features = OurInputFeatures(**inputs)
            features = inputs
        else:
            raise ValueError("Should not using demo!!")
            # TODO(chen): figure out how to perform demo with augmented examples with the right q_boundary offset
            # basically you should perform the same offset to support_labels, in/outside of tokenize.
            # Using demonstrations

            # Max length
            if self.args.double_demo:
                # When using demonstrations, double the maximum length
                # Note that in this case, args.max_seq_length is the maximum length for a single sentence
                max_length = max_length * 2
            # if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
            #     # When using GPT-3's in-context learning, take the maximum tokenization length of the model (512)
            #     max_length = 512

            # All input sentences, including the query and the demonstrations, are put into augmented_examples, 
            # and are numbered based on the order (starting from 0). For single sentence tasks, the input (query)
            # is the sentence 0; for sentence-pair tasks, the input (query) is the sentence 0 and 1. Note that for GPT-3's 
            # in-context learning, the input (query) might be at the end instead of the beginning (gpt3_in_context_head)
            augmented_example = []
            query_text = input_example_to_tuple(example) # Input sentence list for query
            support_by_label = [[] for i in range(len(label_map))]

            if False:
               '''
                if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
                    support_labels = []
                    augmented_example = query_text
                    for support_example in supports:
                        augmented_example += input_example_to_tuple(support_example)
                        current_label = support_example.label
                        if len(label_list) == 1:
                            # current_label = '0' if float(current_label) <= median_mapping[self.args.task_name] else '1' # Regression
                        support_labels.append(label_map[current_label])
                '''
            else:
                # Group support examples by label
                for label_name, label_id in label_map.items():
                    if False:
                        '''
                        if len(label_list) == 1:
                            Regression
                            for support_example in filter(lambda s: ('0' if float(s.label) <= median_mapping[self.args.task_name] else '1') == label_name, supports):
                                support_by_label[label_id] += input_example_to_tuple(support_example)
                        '''
                    else:
                        for support_example in filter(lambda s: s.label == label_name, supports):
                            support_by_label[label_id] += input_example_to_tuple(support_example)

                augmented_example = query_text
                for label_id in range(len(label_map)):
                    augmented_example += support_by_label[label_id]

            # Tokenization (based on the template)
            # TODO(chen): if want to use demo, must adjust label offsets.
            inputs = tokenize_multipart_input(
                example=example,
                input_text_list=augmented_example,
                max_length=max_length,
                tokenizer=self.tokenizer,
                # task_name=self.args.task_name,
                prompt=prompt,
                template=template,
                label_word_list=label_word_list,
                # first_sent_limit=self.args.first_sent_limit,
                # other_sent_limit=self.args.other_sent_limit,
                # truncate_head=self.args.truncate_head,
                # gpt3=self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail,
                # support_labels=None if not (self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail) else support_labels
            )
            # features = OurInputFeatures(**inputs, label=example_label)
            # features = OurInputFeatures(**inputs)
            features = inputs

        if verbose:
            logger.info("*** Example ***")
            logger.info("id has length %s with content: %s", len(features['id']), features['id'])
            # logger.info("features: %s" % features)
            # logger.info("text: %s" % self.tokenizer.decode(features.input_ids))
            # logger.info("input_ids text: %s" % self.tokenizer.decode(features['input_ids']))
            logger.info("index to word has length %s and content: %s", len(features['input_ids']), [[i, self.tokenizer.decode(features['input_ids'][i])] for i in range(len(features['input_ids']))])
            logger.info("attention_mask has length %s and content: %s", len(features['attention_mask']), features['attention_mask'])
            logger.info("offset_mapping has length %s and content: %s", len(features['offset_mapping']), features['offset_mapping'])
            logger.info("sequence_ids has length %s and content: %s", len(features['sequence_ids']), features['sequence_ids'])
            if prompt:
                logger.info("mask_pos has length %s and content: %s", len(features['mask_pos']), features['mask_pos'])
            if 'start_positions' in features:
                logger.info("prompt_offset_start_position: %s" % features['start_positions'])
                logger.info("prompt_offset_end_position: %s" % features['end_positions'])
                logger.info("answer text: %s" % self.tokenizer.decode(features['input_ids'][(features['start_positions'] - 1):(features['end_positions'])]))
                logger.info("overflow_to_sample_mapping: %s" % features['overflow_to_sample_mapping'])
                # TODO(chen): test accuracy here with the offset versions.

        return features
