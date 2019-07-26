#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
import pandas
import logging
import gzip
import sys
import string
import gensim
from gensim.test.utils import datapath
punctuations = string.punctuation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
import pickle
import re
import argparse

compound_operator = "_"

parser = None

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9()\'\`äöüß ]", " ", string)
    return string.strip().lower()


def preprocess_com(input_file, vocabulary):
    vocabulary = set(vocabulary)
    vocabulary_compound = {}
    vocabulary_dash_com = {}
    voc_not_same = set([])
    cleared_lines = []
    for word in vocabulary:
        vocabulary_compound[word] = word.replace(' ', compound_operator)
    for word in vocabulary:
        vocabulary_dash_com[word] = word.replace(' ', "-")
    for word in vocabulary:
        if word != word.replace(' ', compound_operator):
            voc_not_same.add(word)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logging.info("reading file {0}...this may take a while".format(input_file))
    with open(input_file, "r") as f:
        text = f.readlines()
    output = open(input_file + '_rep', 'w')
    print("Num lines", len(text))
    print(text[:3])
    freq = {}
    print("Number of Reviews: " + str(len(text)))
    for i in range(len(text)):
        line = text[i]
        if (i%100000==0):
            logging.info ("read {0} reviews".format (i))
            print(line)

        line = line.lower()
        for word_voc in vocabulary:
            if word_voc in line:
                if word_voc in voc_not_same:
                    compound =  vocabulary_compound[word_voc]
                    line = line.replace(word_voc,  compound)
                comp_dash = vocabulary_dash_com[word_voc]
                if comp_dash in line:
                    compound =  vocabulary_compound[word_voc]
                    line = line.replace(comp_dash, compound)

        cleared_line = clean_str(line)
        yield cleared_line


def preprocess_wordnet(filename, vocabulary):
    vocabulary = set(vocabulary)
    vocabulary_com = set([])
    for word in vocabulary:
        vocabulary_com.add(word.replace(" ", compound_operator))
    file_out = open('data/noun_closure_filtered.tsv', "w")
    relations = []
    with open(filename, "r") as f:
        text = f.readlines()
    for line in text:
        elements = line.strip().split('\t')
        if elements[0].split('.',1)[0] in vocabulary_com and elements[1].split('.', 1)[0] in vocabulary_com:
            file_out.write(elements[0] + '\t' + elements[1] + '\n')
    file_out.close()


def replace_str_index(text,index=0,replacement=''):
    return '%s%s%s'%(text[:index],replacement,text[index+1:])

def spacy_tokenizer(sentence):
    tokens = parser(sentence, disable=['parser', 'tagger', 'ner', 'textcat'])
    tokens = [tok.lemma_ for tok in tokens]
    #print(tokens)
    tokens = [tok for tok in tokens]
    sentence_norm = " ".join(tokens)
    # print(sentence_norm)
    return sentence_norm

def adjust_input(target_word, vocabulary):
    target_original = target_word
    if target_word in vocabulary:
        return target_word #MAKE TO LOWER IF IT DOESNT WORK BETTER
    target_word = spacy_tokenizer(target_word)
    if target_word in vocabulary:
        return target_word
    else:
        return target_original

def create_relation_files(relations_all, output_file_name, min_freq):
    f_out = open("data/" + output_file_name, 'w')
    output_freqs = []
    output_rels_all = []
    for output in relations_all:
        output_relations = output[0]
        output_freq = output[1]
        output_rels_all.append(output_relations)
        output_freqs.append(output_freq)

    for i, output_rels in enumerate(output_rels_all):
        if i== len(output_rels) - 2:
            break
        for j, other_out in enumerate(output_rels_all):
            if i <= j:
                continue
            for k,entry1  in enumerate(output_rels):
                for l, entry2 in enumerate(other_out):
                    #print(entry1[1], entry1[0])
                    if (entry1[1], entry1[0]) == entry2:
                        print("Found contradicting entry: ", entry2)
                        if j == len(output_freqs) - 1:
                            other_out.remove(entry2)
                            print("Removed entry from commoncrawl")
                        else:
                            diff_freq = output_freqs[i][entry1] - output_freqs[j][entry2]
                            if diff_freq >= min_freq:
                                print("Freq_diff:", diff_freq, "therefore remove from other rel")
                                other_out.remove(entry2)
                            elif abs(diff_freq) >= min_freq:
                                print("freq_diff:", diff_freq, "therefore remove from current rel")
                                output_rels.remove(entry1)
                            else:
                                print("freq_diff:", diff_freq, "therefore remove both entries")
                                output_rels.remove(entry1)
                                other_out.remove(entry2)

    for relations in output_rels_all:
        for relation in relations:
            f_out.write(relation[0].replace(' ', compound_operator) + '\t' + relation[1].replace(' ', compound_operator) + '\n')
    f_out.close()

def process_rel_file(min_freq, input_file, vocabulary):
    relations =  []
    relations_with_freq = {}
    filename_in = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/"  + input_file)
    with open(filename_in, 'r',  newline='\n') as f:
        reader = csv.reader(f, delimiter = '\t', quoting=csv.QUOTE_NONE)
        next(reader)
        for i, line in enumerate(reader):
            if len(line) != 3:
                continue
            freq = int(line[2])
            #remove reflexiv and noise relations
            hyponym = adjust_input(line[0], vocabulary)
            hypernym = adjust_input(line[1], vocabulary)
            valid = int(freq) >= min_freq  and line[0] != line[1] and len(line[0]) > 3 and len(line[1]) > 3 and (line[0] in vocabulary and line[1] in vocabulary)
            if valid:
                vocabulary.add(hyponym)
                vocabulary.add(hypernym)
                #remove symmetric relations
                if (hypernym, hyponym) in relations:
                    freq_sym = relations_with_freq[(hypernym, hyponym)]
                    if freq > freq_sym:
                        relations.remove((hypernym, hyponym))
                        if freq - freq_sym > min_freq:
                            relations.append((hyponym, hypernym))
                            relations_with_freq[(hyponym,hypernym)] =  freq
                    else:
                        continue
                else:
                    relations.append((hyponym, hypernym))
                    relations_with_freq[(hyponym,hypernym)] =  freq
    print("For input file: ", input_file, "extracted: " + str(len(relations)) + " relations")
    return relations, relations_with_freq


def read_all_data(filename_in = None, system = "taxi", domain = 'science', language = 'EN'):
    #EN, FR, IT, NL
    if domain in ["environment", "environnement", "ambiente", "milieu"]:
        domain_l = "environment"
    elif domain in ["science", "scienze", "wetenschap"]:
        domain_l = "science"
    elif domain in ["food", "alimentation", "alimenti", "voedsel"]:
        domain_l = "food"
    global compound_operator
    filename_gold = "eval/" + language + "/gold_" + domain_l + ".taxo"

    relations = []
    if filename_in != None:
        with open(filename_in, 'r', newline='\n') as f:
            reader = csv.reader(f, delimiter = '\t')
            for i, line in enumerate(reader):
                relations.append((line[1], line[2]))

    gold= []
    with open(filename_gold, 'r', newline='\n') as f:
        reader = csv.reader(f, delimiter = '\t')
        for i, line in enumerate(reader):
            gold.append((line[1], line[2]))
    return [gold, relations]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create data for poincaré embeddings")
    parser.add_argument('-d', '--lang', type=str, default='EN', choices=["EN", "FR", "NL", "IT"], help="Choose language to generate data for, EN, FR, IT, or NL")
    args = parser.parse_args()
    import spacy
    spacy_lower = language.lower()
    if language == 'EN':
        parser = spacy.load('en_core_web_sm')
    else:
        parser = spacy.load(spacy_lower +'_core_news_sm')
    freq_common = 5
    freq_domain = 3
    all_vocabulary = []
    output_domains = []
    domains = ['science', 'food', 'environment']
    for domain in domains:
        gold, _ = read_all_data(domain = domain, language = language)
        gold = set([relation[0] for relation in gold] + [relation[1] for relation in gold])
        all_vocabulary += gold
        output_domains.append(process_rel_file(freq_domain,language + "/" + spacy_lower + "_" + domain + ".csv" ,gold))
    output_domains.append(process_rel_file(freq_common, language + "/" + spacy_lower  + ".csv", set(all_vocabulary))) #en_ps59g -> en.csv
    create_relation_files(output_domains,language + "/poincare_common_and_domains_" + language + ".tsv",freq_common)
