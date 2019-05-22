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
from data_loader import read_all_data, read_trial_data, read_input, compound_operator, preprocess_com, preprocess_wordnet
import pickle
import itertools
from nltk.corpus import wordnet as wn
import pandas


def main():
    parser = argparse.ArgumentParser(description="Embeddings for Taxonomy")
    parser.add_argument('mode', type=str, default='preload', choices=["train_poincare", "analysis", "visualize_embedding_w2v", "visualize_embedding_poincare", "train_word2vec"], help="Mode of the system.")
    args = parser.parse_args()
    print("Mode: ", args.mode)
    run(args.mode)


def run(mode):
    if embedding == "train_poincare":
            #model = PoincareModel.load('embeddings/poincare_common_domains02_5_3_50')
            rel = set([])
            relations = 'data/noun_closure.tsv'
            notin = set([])
            with open(relations, 'r') as f:
                reader = csv.reader(f, delimiter = '\t')
                f.readline()
                for i, line in enumerate(reader):
                    rel.add(line[0])
                    rel.add(line[1])
            with open('not.tsv', 'r') as f:
                reader = csv.reader(f, delimiter = '\t')
                f.readline()
                for i, line in enumerate(reader):
                    notin.add(line[0])
            print(len(rel))
            model = PoincareModel.load('embeddings/embeddings_poincare_wordnet')
            print(len(model.kv.vocab))
            for element in notin:
                print(element)
                print( wn.synsets(element) , wn.synsets(element)[0].name() in model.kv.vocab)
            words = ["computer_science", "biology", "physics", "science", "virology", "life_science", "chemistry", "earth_science", "algebra", "economics", "optics" "immunology"]
            for word in words:
                print("Current word: ", word)

                if word in model.kv.vocab:
                    try:
                        print("Closest Parent: ", model.kv.closest_parent(word))
                        print("Closest Child ", model.kv.closest_child(word))
                        print("Descendants: ", model.kv.descendants(word))
                        print("Ancestors: ", model.kv.ancestors(word))
                        print("Hierarchy diff to Science: ", model.kv.difference_in_hierarchy(word, "science"))
                        print('\n')
                    except:
                        continue
                else:
                    print("Word not in Vocab")


    if mode == 'train_poincare':
        # gold,relations = read_all_data()
        # freq_science = [3,5]
        # for entry_science in freq_science:
        #     relations = './data/' + domain +'_crawl_' + str(entry_science) +'.tsv'
        #     #relations = './data/science_crawl_merge_10_3_02.tsv'
        #     poincare_rel = PoincareRelations(relations)
        #     dim = 50
        #     model = PoincareModel(poincare_rel, size = dim)
        #     print("Starting Training...")
        #     model.train(epochs=400)
        #     model.save("embeddings/embeddings_" + domain + "_crawl_poincare_" + str(entry_science) + "_" + str(dim))
        #     #model.save("embeddings/embeddings_science_crawl_merge_poincare_10_3_50_02")
        #     break
        gold_s,relations_s = read_all_data("science")
        gold_e,relations_e = read_all_data("environment")
        gold_f,relations_f = read_all_data("food")
        vocabulary = set([relation[0].lower() for relation in gold_s] + [relation[1].lower() for relation in gold_s])
        vocabulary = vocabulary | set([relation[0].lower() for relation in gold_f] + [relation[1].lower() for relation in gold_f])
        vocabulary = vocabulary | set([relation[0].lower() for relation in gold_e] + [relation[1].lower() for relation in gold_e])

        #relations = './data/poincare_common_domains.tsv'
        preprocess_wordnet('data/noun_closure.tsv', vocabulary)
        print("Finished filtering.")
        #relations ="data/poincare_common_domains02L.tsv"
        #relations = './data/science_crawl_merge_10_3_02.tsv'
        poincare_rel = PoincareRelations('data/noun_closure_filtered.tsv')
        dim = 50
        print(poincare_rel)
        model = PoincareModel(poincare_rel, size = dim)
        print("Starting Training...")
        model.train(epochs=400)
        #model.save("embeddings/poincare_common_domains_5_3" + "_" + str(dim))
        model.save("embeddings/wordnet_filtered" + "_" + str(dim))

    if mode == "train_word2vec":
        gold_s,relations_s = read_all_data("science")
        gold_e,relations_e = read_all_data("environment")
        gold_f,relations_f = read_all_data("food")
        vocabulary = set([relation[0].lower() for relation in gold_s] + [relation[1].lower() for relation in gold_s])
        vocabulary = vocabulary | set([relation[0].lower() for relation in gold_f] + [relation[1].lower() for relation in gold_f])
        vocabulary = vocabulary | set([relation[0].lower() for relation in gold_e] + [relation[1].lower() for relation in gold_e])
        documents =  []
        preprocess = False

        if preprocess:
            preprocess_com('/srv/data/5aly/data_text/en_news-2014.txt', vocabulary)
            preprocess_com('/srv/data/5aly/data_text/ukwac-noxml.txt', vocabulary)
            preprocess_com('/srv/data/5aly/data_text/wikipedia.txt', vocabulary)
            preprocess_com('/srv/data/5aly/data_text/gigaword.txt', vocabulary)

        #documents = list(read_input("/srv/data/5aly/data_text/wikipedia_utf8_filtered_20pageviews.csv",vocabulary))
        # model.build_vocab(documents)
        # #model.train(documents, total_examples = len(documents), epochs=10)
        # model.train(documents, total_examples=model.corpus_count, epochs=30)
        print("Loading okwac and en_news...")
        sentences = itertools.chain(LineSentence('/srv/data/5aly/data_text/ukwac-noxml.txt_rep'), LineSentence('/srv/data/5aly/data_text/en_news-2014.txt_rep'))
        print("Training...")
        # model.build_vocab(sentences)
        # model.train(documents, total_examples = len(documents), epochs=10)
        print("Loading Wikipedia...")
        sentences = itertools.chain(sentences, LineSentence('/srv/data/5aly/data_text/wikipedia.txt_rep'))
        print("Loading gigaword")
        sentences = itertools.chain(sentences, LineSentence('/srv/data/5aly/data_text/gigaword.txt_rep'))
        print("Finished loading")
        model = gensim.models.Word2Vec(sentences, size= 300, window = 5, min_count = 1000, workers = 20, iter = 10)
        print("Finished building word2vec model")
        # model.build_vocab(sentences)
        # print("Finished building vocabulary")
        # #model.train(documents, total_examples = len(documents), epochs=10)
        # model.train(sentences, total_examples=model.corpus_count, epochs=30)
        model.save("embeddings/own_embeddings_w2v_taxi_corpora")


if __name__ == '__main__':
    main()
