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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gensim.models.poincare import PoincareModel, PoincareRelations, PoincareKeyedVectors
from gensim.viz.poincare import poincare_2d_visualization
from gensim.test.utils import datapath
from data_loader import read_all_data, read_trial_data, read_input, compound_operator
import plotly.plotly as py
from nltk.corpus import wordnet as wn
#py.sign_in('RamiA', 'lAA8oTL51miiC79o3Hrz')

from collections import Counter


from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas


def compare_to_gold(gold, taxonomy, model,  model_poincare = None, outliers = [], threshold_add = 0.4, new_nodes = [], log = "", write_file = ""):
    taxonomy_c = taxonomy.copy()
    global compound_operator
    removed_outliers = []
    for element in taxonomy_c:
        if (element[0].replace(' ', compound_operator), element[1].replace(' ', compound_operator)) in outliers:
            continue
        removed_outliers.append((element[0], element[1]))

    if new_nodes:
        for element in new_nodes:
            removed_outliers.append((element[0].replace(compound_operator, " "), element[1].replace(compound_operator, " ")))

    removed_outliers = list(set(removed_outliers))

    correct = 0
    for element in removed_outliers:
        for ele_g in gold:
            if element[0] == ele_g[0] and element[1] == ele_g[1]:
                correct+=1
                break
    precision = correct / float(len(removed_outliers))
    recall = correct / float(len(gold))
    print(str(recall).replace(".", ',') +'\t' + str(precision).replace(".", ',') + '\t' + str(2*precision *recall / (precision + recall)).replace(".", ',') + '\t' + str(len(new_nodes)) + '\t' + str(len(outliers)))
    if log != None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log)
        with open(path + ".txt", 'w') as f:
            for element in outliers:
                f.write(element[0] + '\t' + element[1] + '\n')
            f.write("Elements Taxonomy:" + str(float(len(removed_outliers))))
            f.write(str((float(len(gold)))) + '\n')
            f.write("Correct: " + str(correct) + '\n')
            f.write("Precision: " + str(precision) + '\n')
            f.write("Recall: " + str(recall) + '\n')
            f.write("F1: " + str(2*precision *recall / (precision + recall)) + '\n')
            f.close()
    if write_file != None:
        path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), write_file + ".csv")
        with open(path, 'w') as f:
            for i, element in enumerate(removed_outliers):
                f.write(str(i) + '\t' + str(element[0]) + '\t' + str(element[1])  + '\n')
        f.close()

    return removed_outliers

def get_parent(relations,child):
    for relation in relations:
        if child == relation[0]:
            return relation[1]
    return None

def connected_to_root(element, list_data, root):
    parent = element
    parent_last = None
    while parent != parent_last:
        parent_last = parent
        for relation in list_data:
            if parent == relation[0]:
                parent = relation[1]
                #print parent
                #break
    #print '\n'
    return [parent == root, parent]

def get_rank(entity1, entity2, model, threshhold):
    rank_inv = None
    similarities_rev = model.wv.similar_by_word(entity1, threshhold)
    similarities_rev = [entry[0] for entry in similarities_rev]
    for j in range(len(similarities_rev)):
        temp_rev = similarities_rev[j]
        if entity2 == temp_rev:
            rank_inv = j
    return rank_inv


def connect_to_root(gold, taxonomy, domain):
    new_nodes = set([])
    new_relationships = []
    gold_nodes = [relation[0] for relation in gold] + [relation[1] for relation in gold]
    taxonomy_nodes = (set([relation[0] for relation in taxonomy] + [relation[1] for relation in taxonomy]))
    for element in gold_nodes:
        if element not in taxonomy_nodes:
            new_nodes.add((element, domain))
    return new_nodes

#do not need to check if words in vocab since outliers must be in vocab
#TODO could happen that outlier would connect to new outlier, but is not regarded, so currently adding all but outlier, so order of replacing outliers is not irrelevant
def connect_new_nodes(gold, taxonomy, model, model_poincare, threshold, no_parents, no_co, wordnet = False, exclude_sub = False, outliers = None, domain = None):
    structure = {}
    new_nodes = set([])
    new_relationships = []
    gold_nodes = [relation[0] for relation in gold] + [relation[1] for relation in gold]
    taxonomy_nodes = (set([relation[0] for relation in taxonomy] + [relation[1] for relation in taxonomy]))
    results_parents = []
    results_substring = []
    pairs_parents = []
    results_co = []
    pairs_co = []
    for element in gold_nodes:
        if element not in taxonomy_nodes:
            new_nodes.add(element)

    relations = taxonomy.copy()
    for i in range(len(relations)):
        relations[i] = (relations[i][0].replace(" ", compound_operator), relations[i][1].replace(" ", compound_operator))

    for parent in [relation[1] for relation in relations]:
        structure[parent] = [relation[0] for relation in relations if relation[1] == parent]

    for node in new_nodes:
        node = node.replace(" ", compound_operator)
        result_co_min = 10000000
        pair_co_min  = 0
        result_parent_min = 10000000
        pair_parent_min = 0
        for key in structure:
            #print(key)
            if structure[key] == []:
                print("no children: " + key)
                continue
            cleaned_co_hyponyms = []
            if key == node:
                continue
            # if len(structure[key]) < 1:
            #     continue
            result_parent, pair_parent, result_co, pair_co  = get_rank(node, key, structure[key], model, model_poincare, no_parents, no_co, compound = True, wordnet = wordnet)
            if result_parent < result_parent_min and result_parent != 0:
                result_parent_min = result_parent
                pair_parent_min = pair_parent
            if result_co < result_co_min and result_co != 0:
                result_co_min = result_co
                pair_co_min = pair_co
        if result_parent_min != 10000000 or result_co_min != 10000000:
            if result_parent_min != 10000000:
                results_parents.append(result_parent_min)
                pairs_parents.append(pair_parent_min)
            if result_co_min != 10000000:
                results_co.append(result_co_min)
                pairs_co.append(pair_co_min)
        elif node.split('_')[0] in structure and not exclude_sub:
            results_substring.append((node, node.split('_')[0]))
        elif node.split('_')[-1] in structure and not exclude_sub:
            results_substring.append((node, node.split('_')[-1]))
    #print(len(results_co))

    results_substring = set(results_substring)
    results_normalized1 = []
    results_normalized2 = []
    if not no_parents:
        #results_normalized1= list(preprocessing.scale(results_parents))
        outliers_parents = find_outliers(results_parents, pairs_parents, threshold, mode = 'min')
        new_relationships = list(set(outliers_parents)|results_substring)

    if not no_co:
        #results_normalized2= list(preprocessing.scale(results_co))
        outliers_co = find_outliers(results_co, pairs_co, threshold, mode = 'min')
        new_relationships = list(set(outliers_co)|results_substring)

    count_o = 0
    outlier_n = [outlier[0] for outlier in outliers if not outlier[0] in taxonomy_nodes]
    for entry in new_relationships:
        if entry[0] in outlier_n or entry[1] in outlier_n:
            count_o+=1
    print("disconnected from Step 3.2", count_o)
    return new_relationships

def get_rank(current_child, parent, children, model, model_poincare, no_parents, no_co, compound  = True, wordnet = False):
    result_co = 0
    pair_co  = 0
    result_parent = 0
    pair_parent = 0
    current_child2  = current_child.replace(compound_operator, " ")
    parent2 = parent.replace(compound_operator, " ")
    if not no_co:
        try:
            children = [chi for chi in children if chi != current_child]
            if children:
                most_similar_child = model.wv.most_similar_to_given(current_child, children)
                index_child = model.wv.distance(current_child, most_similar_child)

                result_co = index_child
                pair_co = (current_child,parent)
            # else:
            #     index_child = 0
        except (KeyError,ZeroDivisionError) as e:
            index_child = 0
    if not no_parents:
        try:
            if wordnet:
                if compound:
                    current_child2 = current_child
                    parent2 = parent
                index_parent = 1000000
                parents = [ele for ele in model_poincare.kv.vocab if parent2 == ele.split(".")[0]]
                children = [ele for ele in model_poincare.kv.vocab if current_child2 == ele.split(".")[0]]
                for parento in parents:
                    for child in children:
                        index_parent_c = model_poincare.kv.rank(child, parento)
                        if index_parent_c < index_parent:
                            index_parent = index_parent_c
                if index_parent == 1000000:
                    index_parent = 0
            else:
                if compound:
                    index_parent = model_poincare.kv.rank(current_child, parent)
                else:
                    index_parent = model_poincare.kv.rank(current_child2,parent2)

            result_parent = index_parent
            pair_parent = (current_child,parent)
        except KeyError as e:
            index_parent = 0
    return [result_parent, pair_parent, result_co, pair_co]


#create dictionary mit den begirffen wegen bindestrich
def calculate_outliers(relations_o, model, model_poincare = None, threshold = None, no_parents = False, no_co = True, compound = False, wordnet = False, exclude_sub = False):
    outliers = []
    structure = {}
    results_parents = []
    pairs_parents = []
    results_co = []
    pairs_co = []
    relations = relations_o.copy()
    for i in range(len(relations)):
        relations[i] = (relations[i][0].replace(" ", compound_operator), relations[i][1].replace(" ", compound_operator))

    #Dictionary with each parent and its children in the taxonomy
    for parent in [relation[1] for relation in relations]:
        structure[parent] = [relation[0] for relation in relations if relation[1] == parent]

    for key in structure:
        #print(key)
        if structure[key] == []:
            print("no children: " + key)
            continue
        elif not key in model.wv:
            continue
        cleaned_co_hyponyms = []
        for word in structure[key]:
            if word in model.wv:
                cleaned_co_hyponyms.append(word)
        if len(cleaned_co_hyponyms) < 1:
            continue

        cleaned_co_hyponyms_copy = cleaned_co_hyponyms.copy()
        for child in cleaned_co_hyponyms_copy:
            result_parent, pair_parent, result_co, pair_co = get_rank(child, key, cleaned_co_hyponyms, model, model_poincare, no_parents, no_co, compound, wordnet)
            if result_parent != 0:
                if not exclude_sub:
                     if child.split("_")[0] != key and child.split("_")[-1] != key:
                        results_parents.append(result_parent)
                        pairs_parents.append(pair_parent)
                else:
                    results_parents.append(result_parent)
                    pairs_parents.append(pair_parent)
            if result_co != 0:
                if not exclude_sub:
                     if child.split("_")[0] != key and child.split("_")[-1] != key:
                        results_co.append(result_co)
                        pairs_co.append(pair_co)
                else:
                    results_co.append(result_co)
                    pairs_co.append(pair_co)


    if not no_parents:
        #results_normalized1= list(preprocessing.scale(results_parents))
        outliers_parents = find_outliers(results_parents, pairs_parents, threshold)
        outliers = list(outliers_parents)

    if not no_co:
        #results_normalized2= list(preprocessing.scale(results_co))
        outliers_co = find_outliers(results_co, pairs_co, threshold)
        outliers = list(outliers_co)

    if not no_co and not no_parents:
        outliers = list(outliers_parents.intersection(outliers_co))

    return outliers


def replace_outliers(taxonomy, outliers, domain, model, model_poincare, no_parents, no_co, wordnet = False, exclude_sub = False):
    orphans =set([])

    structure = {}
    new_relationships = []
    taxonomy_nodes = (set([relation[0] for relation in taxonomy] + [relation[1] for relation in taxonomy]))
    results_parents = []
    results_substring = []
    pairs_parents = []
    results_co = []
    pairs_co = []
    for element in [outlier[0] for outlier in outliers] + [outlier[1] for outlier in outliers]:
        if element not in taxonomy_nodes:
            continue
        ele_parent = connected_to_root(element, taxonomy, domain)
        if not ele_parent[0] and ele_parent[1] == element:
            orphans.add(element)
    #print(orphans)

    relations = taxonomy.copy()
    for i in range(len(relations)):
        relations[i] = (relations[i][0].replace(" ", compound_operator), relations[i][1].replace(" ", compound_operator))

    for parent in [relation[1] for relation in relations]:
        structure[parent] = [relation[0] for relation in relations if relation[1] == parent]

    for node in orphans:
        node = node.replace(" ", compound_operator)
        result_co_min = 10000000
        pair_co_min  = 0
        result_parent_min = 10000000
        pair_parent_min = 0
        for key in structure.keys():
            #print(key)
            if structure[key] == []:
                print("no children: " + key)
                continue
            cleaned_co_hyponyms = []
            if key == node:
                continue
            # if len(structure[key]) < 1:
            #     continue
            result_parent, pair_parent, result_co, pair_co  = get_rank(node, key, structure[key], model, model_poincare, no_parents, no_co, compound = True, wordnet = wordnet)
            if result_parent < result_parent_min and result_parent != 0:
                result_parent_min = result_parent
                pair_parent_min = pair_parent
            if result_co < result_co_min and result_co != 0:
                result_co_min = result_co
                pair_co_min = pair_co
        if result_parent_min != 10000000 or result_co_min != 10000000:
            if result_parent_min != 10000000:
                results_parents.append(result_parent_min)
                pairs_parents.append(pair_parent_min)
            if result_co_min != 10000000:
                results_co.append(result_co_min)
                pairs_co.append(pair_co_min)
        elif node.split('_')[0] in structure and not exclude_sub:
            results_substring.append((node, node.split('_')[0]))
        elif node.split('_')[-1] in structure and not exclude_sub:
            results_substring.append((node, node.split('_')[-1]))
    #print(len(results_co))

    results_substring = set(results_substring)

    pairs_parents = set(pairs_parents)
    pairs_co = set(pairs_co)
    if not no_parents:
        new_relationships = list(set(pairs_parents|results_substring))
    if not no_co:
        new_relationships = list(set(pairs_co)|results_substring)
    print(new_relationships)
    return new_relationships


def find_outliers(results, pairs, threshold, mode = "max"):
    print("num results", len(results))
    outliers = set([])
    num_clusters = threshold#15 wordnet #6 own_poincare
    results_all_s = np.asarray(results).reshape(-1,1)
    best_cluster = threshold
    if len(results_all_s) == 0:
        return set([])
    results_pairs = []
    for i in range(len(results)):
        results_pairs.append((i, results[i]))
    results_sorted = sorted(results_pairs, key=lambda x: x[1])
    average = 0
    for element in results:
        average+=element
    average /= len(results)
    if mode == 'min':
        for value in results_sorted:
            if value[1] <= average:
                outliers.add(pairs[value[0]])
    elif mode == 'max':
        for value in results_sorted:
            if value[1] > average: #wn -0.3
                outliers.add(pairs[value[0]])
    return outliers

def calculate_Nemar():
    systems = ["TAXI", "JUNLP", "USAAR"]
    domains = ["environment", "science", "food"]
    for i in range(3):
        for j in range(3):
            domain = domains[j]
            system = systems[i]
            file_poincare = "out/distributed_semantics_" + domain + "_" + system + "_True.csv"
            file_poincare_WN = "out/distributed_semantics_" + domain + "_" + system + "_WN.csv"
            file_w2v = "out/distributed_semantics_"+domain+"_" + system + "_False.csv"
            file_base = "../out/" + system +"_" + domain + ".taxo-pruned.csv-cleaned.csv"
            file_root = "out/distributed_semantics_"+domain+"_" + system + "_root.csv"
            filename_gold = "data/gold_"+domain+".taxo"
            poincare_f = open(file_poincare, 'r').readlines()
            poincare_wv_f = open(file_poincare_WN, 'r').readlines()
            baseline_f = open(file_base, 'r').readlines()
            w2v_f = open(file_w2v, 'r').readlines()
            gold_f = open(filename_gold, 'r').readlines()
            root_f = open(file_root, 'r').readlines()
            poincare = []
            poincare_wn = []
            baseline = []
            w2v = []
            gold = []
            root = []
            for line in baseline_f:
                content = line.split('\t')
                baseline.append((content[1], content[2]))
            for line in poincare_wv_f:
                content = line.split('\t')
                poincare_wn.append((content[1], content[2]))
            for line in w2v_f:
                content = line.split('\t')
                w2v.append((content[1], content[2]))
            for line in poincare_f:
                content = line.split('\t')
                poincare.append((content[1], content[2]))
            for line in gold_f:
                content = line.split('\t')
                gold.append((content[1], content[2]))
            for line in root_f:
                content = line.split('\t')
                root.append((content[1], content[2]))
            yes_no =  0
            no_yes = 0
            for entry in gold:
                # if entry in w2v and entry not in baseline:
                #     yes_no+=1
                # if entry in baseline and entry not in w2v:
                #     no_yes+=1

                # if entry in poincare and entry not in baseline:
                #     yes_no+=1
                # if entry in baseline and entry not in poincare:
                #     no_yes+=1

                # if entry in poincare_wn and entry not in baseline:
                #     yes_no+=1
                # if entry in baseline and entry not in poincare_wn:
                #     no_yes+=1

                if entry in root and entry not in baseline:
                    yes_no+=1
                if entry in baseline and entry not in root:
                    no_yes+=1
            if yes_no + no_yes == 0:
                nemar = 0
            else:
                nemar = (yes_no - no_yes)**2 /(yes_no + no_yes)
            print(yes_no, no_yes)
            print(system, domain, nemar)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Embeddings for Taxonomy")
    parser.add_argument('-m', '--mode', type=str, default='preload', choices=["root", "combined_embeddings_removal_and_new", "combined_embeddings_new_nodes", "combined_embeddings_removal", "nemar"], help="Mode of the system.")
    parser.add_argument('-d', '--domain', type=str, default='science', choices=["science", "food", "environment"], help="Domain")
    parser.add_argument('-e', '--embedding', type=str, nargs='?', default=None, choices=["own_and_poincare", "poincare", "poincare_all", "fasttext", "wiki2M", "wiki1M_subword", "own_w2v", "quick", "none"], help="Embedding to use")
    parser.add_argument('-ep', '--exparent', action='store_true', help='Exclude "parent" relations')
    parser.add_argument('-sys','--system', type=str, choices =["TAXI", "USAAR", "QASSIT", "JUNLP", "NUIG-UNLP"])
    parser.add_argument('-ico', '--inco', action='store_true', help='Include "co-hypernym relations')
    parser.add_argument('-com', '--compound', action='store_true', help='Includes compound word in outlier removal')
    parser.add_argument('-wn', '--wordnet', action ='store_true', help= 'Use Wordnet instead of own embeddings')
    parser.add_argument('-es', '--exclude_sub', action='store_true', help="Uses substring method")
    args = parser.parse_args()
    print("Mode: ", args.mode)
    run(args.mode, args.domain, args.embedding, args.exparent, args.inco, args.compound, args.wordnet, args.exclude_sub, args.system)


def run(mode, domain, embedding, exclude_parent = False, include_co = False, compound = False, wordnet = False, exclude_sub = False, system = "TAXI"):
    if embedding == "fasttext":
        #model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M-subword.vec', binary=False)
        model = gensim.models.FastText.load_fasttext_format('wiki.en.bin')
        #model = gensim.models.FastText.load_fasttext_format('crawl-300d-2M.vec')
    elif embedding == "wiki2M":
        #model = gensim.models.FastText.load_fasttext_format('crawl-300d-2M.vec','vec')
        model = gensim.models.KeyedVectors.load_word2vec_format('embeddings/crawl-300d-2M.vec', binary=False)
        #model.save("crawl-300d-2M.bin")
    elif embedding == "wiki1M_subword":
        model = gensim.models.KeyedVectors.load_word2vec_format('embeddings/wiki-news-300d-1M-subword.vec', binary=False)

    elif embedding == "own_w2v":
        model = gensim.models.KeyedVectors.load('embeddings/own_embeddings_w2v')

    elif embedding == "quick":
        model = gensim.models.KeyedVectors.load_word2vec_format('embeddings/crawl-300d-2M.vec', binary=False, limit = 50000)

    elif embedding == 'own_and_poincare':
        print("init")
        model = gensim.models.KeyedVectors.load('embeddings/own_embeddings_w2v_all') #n2 #all
        print("Word2vec vocab size", len(model.wv.vocab))
        #model_poincare = PoincareModel.load('embeddings/embeddings_' + domain +'_crawl_poincare_3_50')
        #model_poincare = PoincareModel.load('embeddings/embeddings_science_crawl_merge_poincare_10_3_50_02')

        if wordnet:
            #model_poincare = PoincareModel.load('embeddings/embeddings_poincare_wordnet')
            model_poincare = PoincareModel.load('embeddings/wordnet_filtered_50')

        else:
            model_poincare = PoincareModel.load('embeddings/poincare_common_domains02_5_3_50')

            # wordlist = ["volcanic_eruption", "whipped_cream", 'ordinary_differential_equations']
            # for word in wordlist:
            #     print(word)
            #     distances = list(model_poincare.kv.distances(word))
            #     pairs = list(zip(distances, list(model_poincare.kv.vocab)))
            #     pairs = sorted(pairs)
            #     closest = [element[1] for element in pairs[:5]]
            #     print(closest, '\n')


        print("Poincare vocab size", len(model_poincare.kv.vocab))


    gold = []
    relations = []
    taxonomy = []
    outliers = []
    exclude_co = not include_co

    if mode =='combined_embeddings_removal':
        thresholds = range(2,50,2)#poincare and co-hyper testrun
        #thresholds = [6]
        for value in thresholds:
            gold, relations = read_all_data(system, domain)
            outliers = calculate_outliers(relations, model, threshold = value, model_poincare = model_poincare, compound = compound, no_parents = exclude_parent, no_co = exclude_co, wordnet = wordnet, exclude_sub = exclude_sub)
            relations2 = compare_to_gold(gold = gold, taxonomy = relations, model = model, model_poincare = model_poincare, outliers = outliers)
    elif mode == 'combined_embeddings_new_nodes':
        thresholds = [2]
        #thresholds = [2,4,6,8,10,12,14] #poincare testrun
        #thresholds = [12,14,18,20] #co-hyper testrun
        for value in thresholds:
            gold, relations = read_all_data(system, domain)
            new_nodes = connect_new_nodes(taxonomy = relations, gold = gold, model = model, model_poincare = model_poincare, threshold = value,  no_parents = exclude_parent, no_co = exclude_co, wordnet = wordnet, exclude_sub = exclude_sub, domain = domain)
            compare_to_gold(gold = gold, taxonomy = relations, model = model, model_poincare = model_poincare, new_nodes =  new_nodes)

#result not always same as random selection because random init of kmeans, outliers are not the same
#Remove outliers of taxonomy, then use the outlier children and the new nodes to attach to taxonomy

    elif mode == 'combined_embeddings_removal_and_new':

        gold, relations = read_all_data(system, domain)
        voc = set([rel[0] for rel in relations] + [rel[1] for rel in relations])
        g_voc = set([rel[0] for rel in gold] + [rel[1] for rel in gold])
        diff = len(g_voc) - len(voc)
        print(len(voc))
        print("Orphans at start", diff)
        outlier_thresh = 2 #6#20
        orphan_thresh = 2
        compare_to_gold(gold = gold, taxonomy = relations, model = model, model_poincare = model_poincare)
        # new_nodes = connect_new_nodes(taxonomy = relations, gold = gold, model = model, model_poincare = model_poincare, threshold = orphan_thresh,  no_parents = exclude_parent, no_co = exclude_co, wordnet = wordnet, exclude_sub = exclude_sub, domain = domain)
        outliers = calculate_outliers(relations, model, threshold = outlier_thresh, model_poincare = model_poincare, compound = compound, no_parents = exclude_parent,
         no_co = exclude_co, wordnet = wordnet, exclude_sub = exclude_sub)
        # relations1 = compare_to_gold(gold = gold, taxonomy = relations, model = model, model_poincare = model_poincare, new_nodes =  new_nodes)
        relations2 = compare_to_gold(gold = gold, taxonomy = relations, model = model, model_poincare = model_poincare, outliers = outliers)

        replaced_outliers = replace_outliers(taxonomy = relations2, outliers = outliers, domain = domain, model = model, model_poincare = model_poincare, no_parents = exclude_parent, no_co = exclude_co, wordnet = wordnet, exclude_sub = exclude_sub)
        relations3 = compare_to_gold(gold = gold, taxonomy = relations2, model = model, model_poincare = model_poincare, new_nodes =  replaced_outliers)
        new_nodes = connect_new_nodes(taxonomy = relations3, gold = gold,  model = model, model_poincare = model_poincare, threshold = orphan_thresh,
          no_parents = exclude_parent, no_co = exclude_co, wordnet = wordnet, exclude_sub = exclude_sub, outliers = outliers, domain = domain)
        # new_nodes2 = connect_new_nodes(taxonomy = relations, gold = gold,  model = model, model_poincare = model_poincare, threshold = orphan_thresh,
        #     no_parents = exclude_parent, no_co = exclude_co, wordnet = wordnet, exclude_sub = exclude_sub, outliers = outliers, domain = domain)
        # compare_to_gold(gold = gold, taxonomy = relations, model = model, model_poincare = model_poincare, new_nodes =  new_nodes2)
        if wordnet:
            outfile = "out/distributed_semantics_" + domain + "_" + system + "_" + 'WN'
        else:
            outfile = "out/distributed_semantics_" + domain + "_" + system + "_" + str(exclude_co)

        compare_to_gold(gold = gold, taxonomy = relations3, new_nodes = new_nodes, model = model, model_poincare = model_poincare,  write_file = outfile)

    elif mode == 'nemar':
        calculate_Nemar()

    elif mode == 'root':
        gold, relations = read_all_data(system, domain)
        root = connect_to_root(taxonomy = relations, gold =  gold, domain = domain)
        compare_to_gold(gold = gold, taxonomy = relations, new_nodes = root, model = model, model_poincare = model_poincare, write_file = "refinement_out/distributed_semantics_" +domain + "_" + system + "_root")



if __name__ == '__main__':
    main()
