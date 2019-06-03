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
from data_loader import read_all_data, compound_operator
import plotly.plotly as py
from nltk.corpus import wordnet as wn
from collections import Counter
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas


MAX_INDEX = 10000000

def compare_to_gold(gold, taxonomy, model = None,  model_poincare = None, outliers = [], threshold_add = 0.4, new_nodes = [], write_file = ""):
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
    print(str(precision).replace(".", ',') +'\t' + str(recall).replace(".", ',') + '\t' + \
     str(2*precision *recall / (precision + recall)).replace(".", ',') + '\t' + str(len(new_nodes)) + '\t' + str(len(outliers)))

    if write_file != None:
        path = '/'.join(write_file.split('/')[:-1]) + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        path =  os.path.join(os.path.dirname(os.path.abspath(__file__)), write_file + "_refined_taxonomy.csv")
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

def connect_new_nodes(gold, taxonomy, model, model_poincare, threshold, no_parents, no_co, wordnet = False, exclude_sub = False, outliers = None, domain = None):
    structure = {}
    new_nodes = set([])
    new_relationships = []
    gold_nodes = [relation[0] for relation in gold] + [relation[1] for relation in gold]
    taxonomy_nodes = (set([relation[0] for relation in taxonomy] + [relation[1] for relation in taxonomy]))
    results_parents, results_substring, pairs_parents, results_co, pairs_co = [], [], [], [], []

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
        result_co_min, result_parent_min = MAX_INDEX, MAX_INDEX
        pair_co_min, pair_parents_min  = 0, 0
        for key in structure:
            if key == node:
                continue
            parents = structure[key]
            if not no_co:
                parents = [word for word in structure[key].copy() if word in model.wv]
            if not no_parents:
                parents = [word for word in structure[key].copy() if word in model_poincare.kv]
            result_parent, pair_parent, result_co, pair_co  = get_rank(node, key, parents, model, model_poincare, no_parents, no_co, compound = True, wordnet = wordnet)
            if result_parent < result_parent_min and result_parent != 0 and pair_parent[0] != domain:
                result_parent_min = result_parent
                pair_parent_min = pair_parent
            if result_co < result_co_min and result_co != 0 and pair_co[0] != domain:
                result_co_min = result_co
                pair_co_min = pair_co
        if result_parent_min != MAX_INDEX or result_co_min != MAX_INDEX:
            if result_parent_min != MAX_INDEX:
                results_parents.append(result_parent_min)
                pairs_parents.append(pair_parent_min)
            if result_co_min != MAX_INDEX:
                results_co.append(result_co_min)
                pairs_co.append(pair_co_min)
        elif node.split('_')[0] in structure and not exclude_sub:
            results_substring.append((node, node.split('_')[0]))
        elif node.split('_')[-1] in structure and not exclude_sub:
            results_substring.append((node, node.split('_')[-1]))

    if not no_parents:
        outliers_parents = find_outliers(results_parents, pairs_parents, threshold, mode = 'attach')
        new_relationships = list(set(outliers_parents)|set(results_substring))

    if not no_co:
        outliers_co = find_outliers(results_co, pairs_co, threshold, mode = 'attach')
        new_relationships = list(set(outliers_co)|set(results_substring))

    return new_relationships

def get_rank(current_child, parent, children, model, model_poincare, no_parents, no_co, compound  = True, wordnet = False):
    result_co, pair_co, result_parent, pair_parent = 0, 0, 0, 0
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
        except (KeyError,ZeroDivisionError) as e:
            result_co = 0
    if not no_parents:
        try:
            if wordnet:
                if compound:
                    current_child2 = current_child
                    parent2 = parent
                index_parent = MAX_INDEX
                parents = [ele for ele in model_poincare.kv.vocab if parent2 == ele.split(".")[0]]
                children = [ele for ele in model_poincare.kv.vocab if current_child2 == ele.split(".")[0]]
                for parento in parents:
                    for child in children:
                        index_parent_c = model_poincare.kv.rank(child, parento)
                        if index_parent_c < index_parent:
                            index_parent = index_parent_c
                if index_parent == MAX_INDEX:
                    index_parent = 0
            else:
                if compound:
                    index_parent = model_poincare.kv.rank(current_child, parent)
                else:
                    index_parent = model_poincare.kv.rank(current_child2,parent2)

            result_parent = index_parent
            pair_parent = (current_child,parent)
        except KeyError as e:
            result_parent = 0
    return [result_parent, pair_parent, result_co, pair_co]


#create dictionary mit den begirffen wegen bindestrich
def calculate_outliers(relations_o, model, model_poincare = None, threshold = None, no_parents = False, no_co = True, compound = False, wordnet = False, exclude_sub = False):
    outliers, results_parents, pairs_parents, results_co, pairs_co = [], [], [] ,[], []
    structure = {}
    relations = relations_o.copy()
    for i in range(len(relations)):
        relations[i] = (relations[i][0].replace(" ", compound_operator), relations[i][1].replace(" ", compound_operator))

    #Dictionary with each parent and its children in the taxonomy
    for parent in [relation[1] for relation in relations]:
        structure[parent] = [relation[0] for relation in relations if relation[1] == parent]
    for key in structure:
        parents = structure[key]
        if not no_co:
            parents = [word for word in structure[key].copy() if word in model.wv]
        if not no_parents:
            parents = [word for word in structure[key].copy() if word in model_poincare.kv]
        for child in parents:
            try:
                result_parent, pair_parent, result_co, pair_co = get_rank(child, key, parents, model, model_poincare, no_parents, no_co, compound, wordnet)
            except:
                result_parent, result_co = 0, 0
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
        outliers_parents = find_outliers(results_parents, pairs_parents, threshold)
        outliers = list(outliers_parents)

    if not no_co:
        outliers_co = find_outliers(results_co, pairs_co, threshold)
        outliers = list(outliers_co)

    return outliers


def replace_outliers(taxonomy, outliers, domain, model, model_poincare, no_parents, no_co, wordnet = False, exclude_sub = False):
    orphans =set([])
    structure = {}
    taxonomy_nodes = (set([relation[0] for relation in taxonomy] + [relation[1] for relation in taxonomy]))
    results_parents, results_substring, pairs_parents, results_co, pairs_co, new_relationships = [], [] ,[] ,[] ,[], []
    for element in [outlier[0] for outlier in outliers] + [outlier[1] for outlier in outliers]:
        if element not in taxonomy_nodes:
            continue
        ele_parent = connected_to_root(element, taxonomy, domain)
        if not ele_parent[0] and ele_parent[1] == element:
            orphans.add(element)

    relations = taxonomy.copy()
    for i in range(len(relations)):
        relations[i] = (relations[i][0].replace(" ", compound_operator), relations[i][1].replace(" ", compound_operator))

    for parent in [relation[1] for relation in relations]:
        structure[parent] = [relation[0] for relation in relations if relation[1] == parent]

    for node in orphans:
        node = node.replace(" ", compound_operator)
        result_co_min, result_parent_min = MAX_INDEX, MAX_INDEX
        pair_co_min, pair_parents_min  = 0, 0
        for key in structure.keys():
            if key == node:
                continue
            parents = structure[key]
            if not no_co:
                parents = [word for word in structure[key].copy() if word in model.wv]
            if not no_parents:
                parents = [word for word in structure[key].copy() if word in model_poincare.kv]
            result_parent, pair_parent, result_co, pair_co  = get_rank(node, key, structure[key], model, model_poincare, no_parents, no_co, compound = True, wordnet = wordnet)
            if result_parent < result_parent_min and result_parent != 0:
                result_parent_min = result_parent
                pair_parent_min = pair_parent
            if result_co < result_co_min and result_co != 0:
                result_co_min = result_co
                pair_co_min = pair_co
        if result_parent_min != MAX_INDEX or result_co_min != MAX_INDEX:
            if result_parent_min != MAX_INDEX:
                results_parents.append(result_parent_min)
                pairs_parents.append(pair_parent_min)
            if result_co_min != MAX_INDEX:
                results_co.append(result_co_min)
                pairs_co.append(pair_co_min)
        elif node.split('_')[0] in structure and not exclude_sub:
            results_substring.append((node, node.split('_')[0]))
        elif node.split('_')[-1] in structure and not exclude_sub:
            results_substring.append((node, node.split('_')[-1]))

    results_substring = set(results_substring)
    pairs_parents = set(pairs_parents)
    pairs_co = set(pairs_co)
    if not no_parents:
        new_relationships = list(set(pairs_parents|results_substring))
    if not no_co:
        new_relationships = list(set(pairs_co)|results_substring)
    return new_relationships


def find_outliers(results, pairs, threshold, mode = "removal"):
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
    if mode == 'attach':
        for value in results_sorted:
            if value[1] <= average:
                outliers.add(pairs[value[0]])
    elif mode == 'removal':
        for value in results_sorted:
            if value[1] > average: #wn -0.3
                outliers.add(pairs[value[0]])
    return outliers

def load_embeddings(include_co, exclude_parent, wordnet, domain, language = 'EN'):
    model = None
    model_poincare = None
    if include_co:
        if language == 'EN':
            model = gensim.models.KeyedVectors.load('embeddings/own_embeddings_w2v_all')
            print("Word2vec vocab size", len(model.wv.vocab))
        else:
            print("There is no wordnet poincaré model for a non-english language\nAbort...")
            sys.exit()
    if not exclude_parent:
        if wordnet:
            if language == 'EN':
                model_poincare = PoincareModel.load('embeddings/wordnet_filtered_50')
            else:
                print("There is no wordnet poincaré model for a non-english language\nAbort...")
                sys.exit()
        else:
            assert language in ['EN', 'FR', 'IT', 'NL'] , "Language not supported. Aborting..."
            #model_poincare = PoincareModel.load('embeddings/poincare_common_domains_5_3_' + language + '_' + domain + '_50')
            model_poincare = PoincareModel.load('embeddings/poincare_common_domains_5_3_' + language + '_50')
            print("Poincare vocab size", len(model_poincare.kv.vocab))

        #print(model_poincare.kv.vocab)
        #wordlist = ["volcanic_eruption", "whipped_cream", 'ordinary_differential_equations', "Atlantic_Ocean", "electrical_engineering", "vanilla_extract", "wastewater", "lake", "freshwater", "water"]
        #wordlist = ["international_relations", "second_language_acquisition", "botany", "sweet_potatoes"]
        # for word in wordlist:
        #     print(word)
        #     distances = list(model_poincare.kv.distances(word))
        #     pairs = list(zip(distances, list(model_poincare.kv.vocab)))
        #     pairs = sorted(pairs)
        #     closest = [element[1] for element in pairs[:5]]
        #     print(closest, '\n')

    return [model, model_poincare]

def main():
    parser = argparse.ArgumentParser(description="Embeddings for Taxonomy")
    parser.add_argument('-m', '--mode', type=str, default='preload', choices=["root", "distributed_semantics", "eval"], help="Mode of the system.")
    parser.add_argument('-d', '--domain', type=str, default='science', help="Domain")
    parser.add_argument('-l', '--language', type=str, default='EN', choices=["EN", "FR", "IT", "NL"], help="Embedding to use")
    parser.add_argument('-ep', '--exparent', action='store_true', help='Exclude "parent" relations')
    parser.add_argument('-sys','--system', type=str, choices =["TAXI", "USAAR", "QASSIT", "JUNLP", "NUIG-UNLP"])
    parser.add_argument('-ico', '--inco', action='store_true', help='Include "co-hypernym relations')
    parser.add_argument('-com', '--compound', action='store_true', help='Includes compound word in outlier removal')
    parser.add_argument('-wn', '--wordnet', action ='store_true', help= 'Use Wordnet instead of own embeddings')
    parser.add_argument('-es', '--exclude_sub', action='store_true', help="Uses substring method")
    parser.add_argument('-pin', '--input_path', type=str, help="Set path for input taxonomy")
    parser.add_argument('-pout', '--output_path', type=str, help="Set path for refined taxonomy")
    args = parser.parse_args()
    print("Mode: ", args.mode)
    run(args.mode, args.domain, args.language, args.input_path, args.output_path, args.exparent, args.inco, args.compound, args.wordnet, args.exclude_sub, args.system)


def run(mode, domain, language, path_in, path_out, exclude_parent = False, include_co = False, compound = False, wordnet = False, exclude_sub = False, system = "TAXI"):
    if mode == 'distributed_semantics':
        if domain in ["environment", "environnement", "ambiente", "milieu"]:
            domain_l = "environment"
        elif domain in ["science", "scienze", "wetenschap"]:
            domain_l = "science"
        elif domain in ["food", "alimentation", "alimenti", "voedsel"]:
            domain_l = "food"

        model, model_poincare = load_embeddings(include_co, exclude_parent, wordnet, domain_l, language)

        taxonomy = []
        outliers = []
        exclude_co = not include_co

        gold, relations = read_all_data(path_in, system, domain, language)
        voc = set([rel[0] for rel in relations] + [rel[1] for rel in relations])
        g_voc = set([rel[0] for rel in gold] + [rel[1] for rel in gold])
        diff = len(g_voc) - len(voc)
        print(len(voc))
        print("Orphans at start", diff)


        compare_to_gold(gold = gold, taxonomy = relations, model = model, model_poincare = model_poincare)

        outliers = calculate_outliers(relations, model, threshold = 2, model_poincare = model_poincare, compound = compound, no_parents = exclude_parent,
         no_co = exclude_co, wordnet = wordnet, exclude_sub = exclude_sub)
        #print(outliers)
        relations2 = compare_to_gold(gold = gold, taxonomy = relations, model = model, model_poincare = model_poincare, outliers = outliers)

        replaced_outliers = replace_outliers(taxonomy = relations2, outliers = outliers, domain = domain, model = model, model_poincare = model_poincare,
         no_parents = exclude_parent, no_co = exclude_co, wordnet = wordnet, exclude_sub = exclude_sub)

        relations3 = compare_to_gold(gold = gold, taxonomy = relations2, model = model, model_poincare = model_poincare, new_nodes =  replaced_outliers)

        new_nodes = connect_new_nodes(taxonomy = relations3, gold = gold,  model = model, model_poincare = model_poincare, threshold = 2,
         no_parents = exclude_parent, no_co = exclude_co, wordnet = wordnet, exclude_sub = exclude_sub, outliers = outliers, domain = domain)
        #print(new_nodes)
        relations_final = compare_to_gold(gold = gold, taxonomy = relations3, new_nodes = new_nodes, model = model, model_poincare = model_poincare,  write_file = path_out)



    elif mode == 'root':
        gold, relations = read_all_data(path_in, system, domain)
        root = connect_to_root(taxonomy = relations, gold =  gold, domain = domain)
        compare_to_gold(gold = gold, taxonomy = relations, new_nodes = root, write_file = path_out)


if __name__ == '__main__':
    main()
