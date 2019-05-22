
def measure_pairs_agreement(pairs,nodes_score):
	# whether nodes in pairs agree with their ranking scores
	num_correct_pairs = 0
	num_wrong_pairs = 0
	total_pairs = 0
	for u,v in pairs:
		if u in nodes_score and v in nodes_score:
			if nodes_score[u] <= nodes_score[v]:
				num_correct_pairs += 1
			else:
				num_wrong_pairs += 1
			total_pairs += 1
	acc = 0
	if total_pairs != 0:
		acc = num_correct_pairs * 1.0 / total_pairs
		#print("correct pairs: %d, wrong pairs: %d, total pairs: %d, accuracy: %0.4f" % (num_correct_pairs,num_wrong_pairs,total_pairs,num_correct_pairs*1.0/total_pairs))
	else:
		acc = 1
		#print("total pairs: 0, accuracy: 1")
	return acc

def F1(gt,predicted):
	'''
	inputs: gt, predicted (list)
	return: accuracy, recall, f1
	'''
	gt_set = set(gt)
	predicted_set = set(predicted)
	accurate_set = predicted_set & gt_set
	non_recall_set = gt_set - predicted_set
	#print("gt size: %d, predicted size: %d" % (len(gt_set),len(predicted_set)))
	#print("accurate size: %d, non-recall size: %d" % (len(accurate_set),len(non_recall_set)))
	try:
		accu = len(accurate_set)*1.0 / len(predicted_set)
		recall = 1 - len(non_recall_set)*1.0/len(gt_set)
		f1_score = 2.0*len(accurate_set)/(len(gt_set) + len(predicted_set))
		F1_score = 2.0 * accu * recall / (accu + recall)
		#print("Accu: %0.4f, Recall: %0.4f, f1 score: %0.4f, F1 score: %0.4f" % (accu,recall,f1_score,F1_score))
		return accu,recall,F1_score
	except Exception as e:
		print e
		return 0,0,0




import argparse
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-t","--ground_truth",default= " ", help = "ground truth edges")
	parser.add_argument("-p","--predicted",help = "predicted edges")
	args = parser.parse_args()
	gt_file = args.ground_truth
	predicted_file = args.predicted
	evaluation(gt_file,predicted_file)
