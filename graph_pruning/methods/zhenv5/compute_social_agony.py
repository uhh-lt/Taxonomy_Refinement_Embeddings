from datetime import datetime
import os.path
from helper_funs import dir_tail_name
import csv


def compute_social_agony_script(graph_file, output, agony_path):
    command = agony_path + " " + graph_file + " " + output
    from helper_funs import run_command
    print("Running command to compute social agony: %s" % command)
    start = datetime.now()
    run_command(command, is_print=True)
    end = datetime.now()
    time_used = end - start
    print("Time used in computing social agony: %0.4f s" % (time_used.seconds))


def read_dict_from_file(file_name, word_to_int, delimiter):
    input_file = open(file_name, "r")

    d = {}
    for line in input_file.readlines():
        k, v = line.split()

        try:
            key = word_to_int[int(k)]
            d[key] = int(v)
        except Exception as e:
            print e

    return d


def compute_social_agony(graph_file, delimiter='\t'):
    input_temp_file = os.path.abspath("socialagony_temp_transformed.csv")
    output_temp_file = os.path.abspath("socialagony_temp_out.txt")

    print("Write temporary input file to: %s" % input_temp_file)
    print("Write temporary social agony file to: %s" % output_temp_file)

    word_to_int = []

    with open(graph_file, "r") as f_in:
        with open(input_temp_file, "w+") as f_out:
            reader = csv.reader(f_in, delimiter=delimiter)
            writer = csv.writer(f_out, delimiter=" ")

            for i, line in enumerate(reader):
                if line[1] not in word_to_int:
                    word_to_int.append(line[1])

                if line[2] not in word_to_int:
                    word_to_int.append(line[2])

                key = word_to_int.index(line[1])
                value = word_to_int.index(line[2])

                writer.writerow((key, value))

    script_path = os.path.dirname(os.path.realpath(__file__))
    script_path += "/../agony/agony"

    compute_social_agony_script(input_temp_file, output_temp_file, script_path)
    agony_score = read_dict_from_file(output_temp_file, word_to_int, delimiter=" ")

    print("Remove temporary file: %s" % input_temp_file)
    os.remove(input_temp_file)

    print("Remove temporary file: %s" % output_temp_file)
    os.remove(output_temp_file)

    return agony_score


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_file", default=" ", help="input graph file name (edges list)")
    args = parser.parse_args()
    graph_file = args.graph_file
    compute_social_agony(graph_file)
