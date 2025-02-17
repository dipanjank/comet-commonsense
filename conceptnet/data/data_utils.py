import json
import logging
import os
import pickle
import re

import ftfy
import spacy
import torch
from tqdm import tqdm

from data_loader.data import config as cfg
from data_loader.utils import make_name

start_token = "<START>"
end_token = "<END>"
blank_token = "<blank>"


def save_checkpoint(state, filename):
    logging.info("Saving model to {}".format(filename))
    torch.save(state, filename)


def save_step(model, vocab, optimizer, opt, length, lrs):
    if cfg.test_save:
        name = "{}.pickle".format(make_name(
            opt, prefix="garbage/models/", is_dir=False, eval_=True))
    else:
        name = "{}.pickle".format(make_name(
            opt, prefix="models/", is_dir=False, eval_=True))
    save_checkpoint({
        "epoch": length, "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(), "opt": opt,
        "vocab": vocab, "epoch_learning_rates": lrs},
        name)


def save_eval_file(opt, stats, eval_type="losses", split="dev", ext="pickle"):
    if cfg.test_save:
        name = "{}/{}.{}".format(make_name(
            opt, prefix="garbage/{}/".format(eval_type),
            is_dir=True, eval_=True), split, ext)
    else:
        name = "{}/{}.{}".format(make_name(
            opt, prefix="results/{}/".format(eval_type),
            is_dir=True, eval_=True), split, ext)
    logging.info("Saving {} {} to {}".format(split, eval_type, name))

    if ext == "pickle":
        with open(name, "wb") as f:
            pickle.dump(stats, f)
    elif ext == "txt":
        with open(name, "w") as f:
            f.write(stats)
    elif ext == "json":
        with open(name, "w") as f:
            json.dump(stats, f)
    else:
        raise


def load_checkpoint(filename):
    if os.path.exists(filename):
        checkpoint = torch.load(
            filename, map_location=lambda storage, loc: storage)
        return checkpoint
    else:
        logging.info("No model found at {}".format(filename))


def set_max_sizes(data_loader, force_split=None):
    data_loader.total_size = {}
    if force_split is not None:
        data_loader.total_size[force_split] = \
            data_loader.sequences[force_split]["total"].size(0)
        return
    for split in data_loader.sequences:
        data_loader.total_size[split] = \
            data_loader.sequences[split]["total"].size(0)


def load_existing_data_loader(data_loader, path):
    old_data_loader = torch.load(path)
    for attr in data_loader.__dict__.keys():
        if attr not in old_data_loader.__dict__.keys():
            continue
        setattr(data_loader, attr, getattr(old_data_loader, attr))


################################################################################
#
# Code Below taken from HuggingFace pytorch-openai-lm repository
#
################################################################################


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub(r'\s*\n\s*', ' \n ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()


class TextEncoder(object):
    """
    mostly a wrapper for a public python bpe tokenizer
    """

    def __init__(self, encoder_path, bpe_path):
        self.nlp = spacy.load(
            'en_core_web_sm', disable=['parser', 'tagger', 'ner', 'textcat', 'lemmatizer'])
        self.encoder = json.load(open(encoder_path))
        self.decoder = {v: k for k, v in self.encoder.items()}
        merges = open(bpe_path, encoding='utf-8').read().split('\n')[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(
                pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if (word[i] == first and i < len(word) - 1 and
                        word[i+1] == second):
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[token] = word
        return word

    def encode(self, texts, verbose=True):
        texts_tokens = []
        if verbose:
            for text in tqdm(texts, ncols=80, leave=False):
                text = self.nlp(text_standardize(ftfy.fix_text(text)))
                text_tokens = []
                for token in text:
                    text_tokens.extend(
                        [self.encoder.get(t, 0) for t in
                         self.bpe(token.text.lower()).split(' ')])
                texts_tokens.append(text_tokens)
        else:
            for text in texts:
                text = self.nlp(text_standardize(ftfy.fix_text(text)))
                text_tokens = []
                for token in text:
                    text_tokens.extend(
                        [self.encoder.get(t, 0) for t in
                         self.bpe(token.text.lower()).split(' ')])
                texts_tokens.append(text_tokens)
        return texts_tokens
