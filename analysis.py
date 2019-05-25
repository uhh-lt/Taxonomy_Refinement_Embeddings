import os

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
    calculate_Nemar()
