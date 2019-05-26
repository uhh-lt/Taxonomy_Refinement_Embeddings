#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import gensim
import csv
import io
import sys
import numpy as np
import gzip
import os
import argparse
import logging
from sklearn.manifold import TSNE
from gensim.models.poincare import PoincareModel, PoincareRelations
from gensim.test.utils import datapath
from gensim.models.word2vec import LineSentence
from data_loader import read_all_data, read_trial_data, compound_operator, preprocess_com, preprocess_wordnet
import pickle
import itertools
from nltk.corpus import wordnet as wn
import pandas


def main():
    parser = argparse.ArgumentParser(description="Embeddings for Taxonomy")
    parser.add_argument('mode', type=str, default='preload', choices=["train_poincare_wordnet", "train_poincare_custom", "train_word2vec"], help="Mode of the system.")
    parser.add_argument('language', type=str, default='preload', choices=["EN", "FR", "NL", "IT"], help="Mode of the system.")
    args = parser.parse_args()
    print("Mode: ", args.mode)
    run(args.mode, args.language)


def run(mode, language):
    if mode == "train_poincare_custom":
        if language == 'EN':
            gold_s,_ = read_all_data(domain = "science")
            gold_e,_ = read_all_data(domain = "environment")
            gold_f,_ = read_all_data(domain = "food")
            vocabulary = set([relation[0].lower() for relation in gold_s] + [relation[1].lower() for relation in gold_s])
            vocabulary = vocabulary | set([relation[0].lower() for relation in gold_f] + [relation[1].lower() for relation in gold_f])
            vocabulary = vocabulary | set([relation[0].lower() for relation in gold_e] + [relation[1].lower() for relation in gold_e])
            relations ="data/poincare_common_domains02L.tsv"
            poincare_rel = PoincareRelations(relations)
            dim = 50
            model = PoincareModel(poincare_rel, size = dim)
            print("Starting Training...")
            model.train(epochs=400)
            model.save("embeddings/poincare_common_domains_5_3_EN" + "_" + str(dim))
            break


    if mode == 'train_poincare_wordnet':
        gold_s,_ = read_all_data(domain = "science")
        gold_e,_ = read_all_data(domain = "environment")
        gold_f,_ = read_all_data(domain = "food")
        vocabulary = set([relation[0].lower() for relation in gold_s] + [relation[1].lower() for relation in gold_s])
        vocabulary = vocabulary | set([relation[0].lower() for relation in gold_f] + [relation[1].lower() for relation in gold_f])
        vocabulary = vocabulary | set([relation[0].lower() for relation in gold_e] + [relation[1].lower() for relation in gold_e])

        preprocess_wordnet('data/noun_closure.tsv', vocabulary)
        poincare_rel = PoincareRelations('data/noun_closure_filtered.tsv')
        dim = 50
        model = PoincareModel(poincare_rel, size = dim)
        print("Starting Training...")
        model.train(epochs=400)
        model.save("embeddings/wordnet_filtered" + "_" + str(dim))

    if mode == "train_word2vec":
        gold_s,relations_s = read_all_data("science")
        gold_e,relations_e = read_all_data("environment")
        gold_f,relations_f = read_all_data("food")
        vocabulary = set([relation[0].lower() for relation in gold_s] + [relation[1].lower() for relation in gold_s])
        vocabulary = vocabulary | set([relation[0].lower() for relation in gold_f] + [relation[1].lower() for relation in gold_f])
        vocabulary = vocabulary | set([relation[0].lower() for relation in gold_e] + [relation[1].lower() for relation in gold_e])
        documents =  []

        documents = list(read_input("/srv/data/5aly/data_text/wikipedia_utf8_filtered_20pageviews.csv",vocabulary))
        model = gensim.models.Word2Vec(documents, size= 300, window = 10, min_count = 2, workers = 10)
        model.train(documents, total_examples=len(documents), epochs=30)
        print("Finished building word2vec model")
        model.save("embeddings/own_embeddings_w2v")


if __name__ == '__main__':
    main()
