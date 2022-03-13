import argparse
import json
import os
from collections import OrderedDict
import torch
import csv
import util
import re
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW
from tensorboardX import SummaryWriter
from fewshot_dataset import FewShotDataset
from model import MLMModel

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args

from tqdm import tqdm

import nlpaug.augmenter.word as naw

def prepare_eval_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    # INFO: overflow_to_sample_mapping is popped because it is not needed for eval.
    # For eval we use qid to refer back to original QA.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["id"] = []
    for i in tqdm(range(len(tokenized_examples["input_ids"]))):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["id"].append(dataset_dict["id"][sample_index])
        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        # INFO: this modification to offset_mapping is a way to completely rule out
        # answering attempts at non-context segments.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples



def prepare_train_data(dataset_dict, tokenizer):
    print(f"preparing {len(dataset_dict['context'])} examples")
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    offset_mapping = tokenized_examples["offset_mapping"]

    # Let's label those examples!
    # TODO(shan): be aware of the shifting indices due to segmentation
    # TODO(chen): use the accuracy check after prompt/demo.
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples['id'] = []
    inaccurate = 0
    for i, offsets in enumerate(tqdm(offset_mapping)):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answer = dataset_dict['answer'][sample_index]
        # Start/end character index of the answer in the text.
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        tokenized_examples['id'].append(dataset_dict['id'][sample_index])
        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
            # assertion to check if this checks out
            context = dataset_dict['context'][sample_index]
            offset_st = offsets[tokenized_examples['start_positions'][-1]][0]
            offset_en = offsets[tokenized_examples['end_positions'][-1]][1]
            if context[offset_st : offset_en] != answer['text'][0]:
                inaccurate += 1

    total = len(tokenized_examples['id'])
    print(f"Preprocessing not completely accurate for {inaccurate}/{total} instances")
    return tokenized_examples

def read_and_process(args, tokenizer, dataset_dict, dir_name, dataset_name, split):
    #TODO: cache this if possible
    cache_path = f'{dir_name}/{dataset_name}_encodings.pt'
    tokenized_examples_train_for_test = None
    if os.path.exists(cache_path) and not args.recompute_features:
        tokenized_examples = util.load_pickle(cache_path)
    else:
        if split=='train':
            tokenized_examples = prepare_train_data(dataset_dict, tokenizer)
        else:
            tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
            if args.in_context and args.use_demo:
                tokenized_examples_train_for_test = prepare_train_data(dataset_dict, tokenizer)
        util.save_pickle(tokenized_examples, cache_path)
    return tokenized_examples, tokenized_examples_train_for_test

class Tokenizer:
    @staticmethod
    def tokenizer(text):
        return text.split(' ')

    @staticmethod
    def reverse_tokenizer(tokens):
        return ' '.join(tokens)

#TODO: use a logger, use tensorboard
class Trainer():
    def __init__(self, args, log):
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_every = args.eval_every
        self.eval_after_epoch = args.eval_after_epoch
        self.path = os.path.join(args.save_dir, 'checkpoint')
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.visualize_predictions = args.visualize_predictions
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, model):
        model.save_pretrained(self.path)

    def evaluate(self, model, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        model.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), \
                tqdm(total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_size = len(input_ids)
                outputs = model(input_ids, attention_mask=attention_mask, return_dict = False)
                # Forward
                if (len(outputs) > 2):
                    start_logits, end_logits = outputs[1], outputs[2]
                else:
                    start_logits, end_logits = outputs[0], outputs[1] # when no loss can be computed
                # TODO: compute loss
                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = util.postprocess_qa_predictions(data_dict,
                                                 data_loader.dataset.encodings,
                                                 (start_logits, end_logits))
        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    def train(self, model, train_dataloader, eval_dataloader, val_dict, model_type):
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    optim.zero_grad()
                    model.train()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    start_positions = batch['start_positions'].to(device)
                    end_positions = batch['end_positions'].to(device)
                    if model_type == "mlm":
                        outputs = model(input_ids, attention_mask=attention_mask,
                                        start_positions=start_positions,
                                        end_positions=end_positions, decay_alpha=True, mask_inputs=True)
                    else:
                        outputs = model(input_ids, attention_mask=attention_mask,
                                        start_positions=start_positions,
                                        end_positions=end_positions)
                    loss = outputs[0]
                    loss.backward()
                    optim.step()
                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                    tbx.add_scalar('train/NLL', loss.item(), global_idx)
                    if ((global_idx % self.eval_every) == 0 and not self.eval_after_epoch) or global_idx == 0:
                        self.log.info(f'Evaluating at step {global_idx}...')
                        preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                        self.log.info('Visualizing in TensorBoard...')
                        for k, v in curr_score.items():
                            tbx.add_scalar(f'val/{k}', v, global_idx)
                        self.log.info(f'Eval {results_str}')
                        if self.visualize_predictions:
                            util.visualize(tbx,
                                           pred_dict=preds,
                                           gold_dict=val_dict,
                                           step=global_idx,
                                           split='val',
                                           num_visuals=self.num_visuals)
                        if curr_score['F1'] >= best_scores['F1']:
                            best_scores = curr_score
                            self.save(model)
                    global_idx += 1
                if self.eval_after_epoch:
                    self.log.info(f'Evaluating at epoch {epoch_num}...')
                    preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                    self.log.info('Visualizing in TensorBoard...')
                    for k, v in curr_score.items():
                        tbx.add_scalar(f'val/{k}', v, global_idx)
                    self.log.info(f'Eval {results_str}')
                    if self.visualize_predictions:
                        util.visualize(tbx,
                                        pred_dict=preds,
                                        gold_dict=val_dict,
                                        step=global_idx,
                                        split='val',
                                        num_visuals=self.num_visuals)
                    if curr_score['F1'] >= best_scores['F1']:
                        best_scores = curr_score
                        self.save(model)
        return best_scores

def get_dataset(args, datasets, data_dir, tokenizer, split_name, augment_datasets = {}, augmenters = []):
    datasets = datasets.split(',')
    dataset_dict = None
    dataset_name=''
    for dataset in datasets:
        dataset_name += f'_{dataset.replace("/", "_")}'
        dataset_dict_curr = util.read_squad(f'{data_dir}/{dataset}', augmenters if split_name == 'train' and dataset in augment_datasets else [])
        print(f"dataset {dataset} has {len(dataset_dict_curr['question'])} examples")
        dataset_dict = util.merge(dataset_dict, dataset_dict_curr)
    # INFO: data_encodings['input_ids']: pair of sequences: `[CLS] A [SEP] B [SEP]`
    data_encodings, qa_data_encodings_features_train_for_test = read_and_process(args, tokenizer, dataset_dict, data_dir, dataset_name, split_name)
    # TODO(chen): perform query sampling, demonstration, etc
    # process the data_encodings to retrofit into a template. No need to encode again for sentences since they are already encoded by a basic DistilBertTokenizer.
    if args.in_context:
        mode = split_name
        if split_name == 'validation' or split_name == 'val':
          mode = 'dev'
        return FewShotDataset(args, tokenizer, data_encodings, qa_data_encodings_features_train_for_test, dataset_dict, cache_dir=None, mode=mode, use_demo=False), dataset_dict
    else:
        return util.QADataset(data_encodings, train=(split_name=='train')), dataset_dict

# generate range of alphas
def get_alphas(alpha_start, alpha_end, n_steps, scheme):
    if scheme == "linear":
        return torch.arange(alpha_start, alpha_end, (alpha_end - alpha_start) / n_steps)
    else:
        return [0.0] * n_steps # no MLM loss

def get_augmenter(name):
    # 1.06s
    if name == 'synonym_wordnet':
        return naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=30, aug_p=0.3, tokenizer=Tokenizer.tokenizer, reverse_tokenizer=Tokenizer.reverse_tokenizer)
    # 0.0014s
    if name == 'random_swap':
        return naw.RandomWordAug(action='swap', tokenizer=Tokenizer.tokenizer, reverse_tokenizer=Tokenizer.reverse_tokenizer)
    if name == 'random_delete':
        return naw.RandomWordAug(action='delelte', tokenizer=Tokenizer.tokenizer, reverse_tokenizer=Tokenizer.reverse_tokenizer)
    # 10.44s
    if name == 'wordembs_word2vec':
        # Slow
        return naw.WordEmbsAug(action='substitute', model_type='word2vec', model_path='./GoogleNews-vectors-negative300.bin', tokenizer=Tokenizer.tokenizer, reverse_tokenizer=Tokenizer.reverse_tokenizer, top_k=10)
    # 0.0024s
    if name == 'wordembs_word2vec_insert':
        return naw.WordEmbsAug(action='insert', model_type='word2vec', model_path='./GoogleNews-vectors-negative300.bin', tokenizer=Tokenizer.tokenizer, reverse_tokenizer=Tokenizer.reverse_tokenizer, top_k=10)
    # 0.47s
    if name == 'contextembs_distilbert':
        return naw.ContextualWordEmbsAug(action='substitute', model_path='distilbert-base-uncased', top_k=10, device='cuda')
    if name == 'contextembs_distilbert_insert':
        return naw.ContextualWordEmbsAug(action='insert', model_path='distilbert-base-uncased', top_k=10, device='cuda')
    # 4.129s
    if name == 'back_translation_ru':
        # EN->RU->EN
        return naw.BackTranslationAug(from_model_name='facebook/wmt19-en-ru',  to_model_name='facebook/wmt19-ru-en', device='cuda')
    # 4.97s
    if name == 'back_translation_de':
        # EN->DE->EN
        return naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de',  to_model_name='facebook/wmt19-de-en', device='cuda')

def main():
    # define parser and arguments
    args = get_train_test_args()

    util.set_seed(args.seed)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    vocab_size = len(tokenizer.get_vocab().keys())

    if args.model == 'bert':
        if args.checkpoint_path:
            model = DistilBertForQuestionAnswering.from_pretrained(args.checkpoint_path)
        else:
            model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    elif args.model == 'mlm':
        if args.checkpoint_path:
            model = MLMModel.from_pretrained(args.checkpoint_path)
        else:
            model = MLMModel.from_pretrained('distilbert-base-uncased')
        model.set_mask_token(tokenizer.mask_token_id)
        model.add_vocab_size(vocab_size)
    else:
        raise ValueError('--model parameter must be one of the following:{"bert", "mlm"}')

    if args.do_train:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        trainer = Trainer(args, log)
        augmenters = [get_augmenter(aug_name) for aug_name in args.augment_methods.split(',') if aug_name]
        train_dataset, _ = get_dataset(
            args, args.train_datasets, args.data_dir, tokenizer, 'train', args.augment_datasets, augmenters)
        log.info("Preparing Validation Data...")
        val_dataset, val_dict = get_dataset(args, args.eval_datasets, args.data_dir, tokenizer, 'val')
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))
        # INFO: Always resize model after get_dataset for new tokens has been added.
        model.resize_token_embeddings(len(tokenizer))
        best_scores = trainer.train(model, train_loader, val_loader, val_dict)
        if args.model == 'mlm':
            alpha_start = 2.0
            alpha_end   = 0.5
            n_steps = args.num_epochs * len(train_loader)
            alphas = get_alphas(alpha_start, alpha_end, n_steps, "linear")
            model.set_alphas(alphas)

        best_scores = trainer.train(model, train_loader, val_loader, val_dict, args.model)
    if args.do_eval:
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        split_name = 'test' if 'test' in args.eval_datasets else 'validation'
        log = util.get_logger(args.save_dir, f'log_{split_name}')
        trainer = Trainer(args, log)
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
        if args.model == 'mlm':
            model = MLMModel.from_pretrained(checkpoint_path)
        elif args.model == 'bert':
            model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)
        eval_dataset, eval_dict = get_dataset(args, args.eval_datasets, args.data_dir, tokenizer, split_name)
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 sampler=SequentialSampler(eval_dataset))
        # INFO: Always resize model after get_dataset for new tokens has been added.
        model.resize_token_embeddings(len(tokenizer))
        eval_preds, eval_scores = trainer.evaluate(model, eval_loader,
                                                   eval_dict, return_preds=True,
                                                   split=split_name)
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
        log.info(f'Eval {results_str}')
        # Write submission file
        sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
        log.info(f'Writing submission file to {sub_path}...')
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(eval_preds):
                csv_writer.writerow([uuid, eval_preds[uuid]])


if __name__ == '__main__':
    main()
