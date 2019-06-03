import os

def calculate_Nemar():
    systems = ["TAXI", "JUNLP", "USAAR"]
    domains = ["food"] #"environment", "science", "food"
    for i in range(3):
        domain = "food"
        system = systems[i]
        types = ["root", "w2v", "poincare_wordnet", "poincare_custom"]
        files = ["refinement_out/" + system + "/EN/" + type + "_refined_taxonomy.csv" for type in types]
        files.append("out/systems/" + system +"/EN/" + system + "_" + domain + ".taxo-pruned.csv-cleaned.csv")
        filename_gold = "data/EN/gold_" + domain+ ".taxo"
        files = [open(file, 'r').readlines() for file in files]
        gold_f = open(filename_gold, 'r').readlines()
        files.append(gold_f)
        files_results = []
        for l, file in enumerate(files):
            files_results.append([])
            for line in file:
                content = line.strip().split('\t')
                files_results[l].append((content[1], content[2]))
        gold = files_results[-1]
        baseline = files_results[-2]
        for l, system_result in enumerate(files_results[:-2]):
            yes_no =  0
            no_yes = 0
            has_header_system= [element[0] for element in system_result]
            has_header_baseline = [element[0] for element in baseline]
            for entry in gold:
                #if entry in system_result and (entry not in baseline or entry[0] not in has_header_baseline):
                #if (entry in system_result or entry[0] not in has_header_system) and entry not in baseline:
                if entry in system_result and entry not in baseline:
                    yes_no+=1
                #if entry in baseline and (entry not in system_result or entry[0] not in has_header_system):
                # if (entry in baseline or entry[0] not in has_header_baseline) and entry not in system_result:
                if entry in baseline and entry not in system_result:
                    no_yes+=1
            if yes_no + no_yes == 0:
                nemar = 0
            else:
                nemar = (yes_no - no_yes)**2 /(yes_no + no_yes)
            print(yes_no, no_yes)
            print(l, system, domain, nemar)
    return 0

def main():
    calculate_Nemar()

if __name__ == '__main__':
    main()
