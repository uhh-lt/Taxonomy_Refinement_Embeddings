#!/bin/bash

SYSTEM="$1"
DOMAIN="$2"
LANG="$3"
METHOD="$4"

CYCLE_REMOVING_TOOL="graph_pruning/graph_pruning.py"
CYCLE_REMOVING_METHOD="tarjan"
CLEANING_TOOL="graph_pruning/cleaning.py"
EVAL_TOOL="eval/taxi_eval_archive/TExEval.jar"
EVAL_GOLD_STANDARD=eval/gold_${DOMAIN}.taxo
EVAL_JVM="-Xmx9000m"
OUTPUT_DIR=out
FILE_INPUT=systems/${SYSTEM}/EN/${SYSTEM}_${DOMAIN}.taxo
echo $FILE_INPUT
FILE_PRUNED_OUT=${FILE_INPUT}-pruned.csv
FILE_CLEANED_OUT=${FILE_PRUNED_OUT}-cleaned.csv
CYCLE_REMOVING_METHOD=tarjan

if [[ ! -e out/systems/${SYSTEM}/EN ]]; then
	mkdir -p out/systems/${SYSTEM}/EN
fi

echo "======================================================================================================================"
echo "Cycle removing: python3 $CYCLE_REMOVING_TOOL $FILE_INPUT $OUTPUT_DIR/$FILE_PRUNED_OUT $CYCLE_REMOVING_METHOD"

CYCLES=$(python3 $CYCLE_REMOVING_TOOL $FILE_INPUT $OUTPUT_DIR/$FILE_PRUNED_OUT $CYCLE_REMOVING_METHOD | tee /dev/tty)
echo "Cycle removing finished. Written to: $OUTPUT_DIR/$FILE_PRUNED_OUT"
echo

echo "======================================================================================================================"
echo "Cleaning: python3 $CLEANING_TOOL $OUTPUT_DIR/$FILE_PRUNED_OUT $OUTPUT_DIR/$FILE_CLEANED_OUT $DOMAIN"
python3 $CLEANING_TOOL $OUTPUT_DIR/$FILE_PRUNED_OUT $OUTPUT_DIR/$FILE_CLEANED_OUT $DOMAIN
echo "Finished cleaning. Write output to: $OUTPUT_DIR/$FILE_CLEANED_OUT"
echo
echo "======================================================================================================================"

if [ "$METHOD" -eq 0 ]; then
	VALUE="root"
  echo "Taxonomy refinement: Reconnect disconnected nodes to the root"
  python3 taxonomy_refinement.py -m root -l ${LANG} -d $DOMAIN -sys $SYSTEM -pin $OUTPUT_DIR/$FILE_CLEANED_OUT -pout refinement_out/${SYSTEM}/${LANG}/${VALUE}

elif [ "$METHOD" -eq 1 ]; then
	VALUE="w2v"
  echo "Taxonomy refinement: Employing word2vec embeddings"
  python3 taxonomy_refinement.py -m distributed_semantics -l ${LANG} -d $DOMAIN -sys $SYSTEM -ico -ep -com -pin $OUTPUT_DIR/$FILE_CLEANED_OUT -pout refinement_out/${SYSTEM}/${LANG}/${VALUE}

elif [ "$METHOD" -eq 2 ]; then
	VALUE="poincare_wordnet"
  echo "Taxonomy refinement: Employing wordnet poincaré embeddings"
  python3 taxonomy_refinement.py -m distributed_semantics -l ${LANG} -d $DOMAIN -sys $SYSTEM -com -wn -pin $OUTPUT_DIR/$FILE_CLEANED_OUT -pout refinement_out/${SYSTEM}/${LANG}/${VALUE}

elif [ "$METHOD" -eq 3 ]; then
	VALUE="poincare_custom"
  echo "Taxonomy refinment: Employing custom poincaré embeddings"
  python3 taxonomy_refinement.py -m distributed_semantics -l ${LANG} -d $DOMAIN -sys $SYSTEM -com -pin $OUTPUT_DIR/$FILE_CLEANED_OUT -pout refinement_out/${SYSTEM}/${LANG}/${VALUE}
fi

FILE_INPUT=refinement_out/${SYSTEM}/EN/${VALUE}_refined_taxonomy.csv
FILE_PRUNED_OUT=${FILE_INPUT}
FILE_CLEANED_OUT=${FILE_PRUNED_OUT}
FILE_EVAL_TOOL_RESULT=${FILE_CLEANED_OUT}-evalresul.txt

echo "======================================================================================================================"
echo "Cycle removing and cleaning"

CYCLES=$(python $CYCLE_REMOVING_TOOL $FILE_INPUT $FILE_PRUNED_OUT $CYCLE_REMOVING_METHOD | tee /dev/tty)
python3 $CLEANING_TOOL $FILE_PRUNED_OUT $FILE_CLEANED_OUT $DOMAIN
echo "Finished cleaning. Write output to: $FILE_CLEANED_OUT"
echo

echo "======================================================================================================================"
java $EVAL_JVM -jar $EVAL_TOOL $FILE_CLEANED_OUT $EVAL_GOLD_STANDARD $DOMAIN $FILE_EVAL_TOOL_RESULT 2> eval.out

L_GOLD="$(wc -l $EVAL_GOLD_STANDARD | grep -o -E '^[0-9]+').0"
L_INPUT="$(wc -l $FILE_CLEANED_OUT | grep -o -E '^[0-9]+').0"

RECALL="$(tail -n 1 $FILE_EVAL_TOOL_RESULT | grep -o -E '[0-9]+[\.]?[0-9]*')"
PRECISION=$(echo "print($RECALL * $L_GOLD / $L_INPUT)" | python)
F1=$(echo "print(2 * $RECALL * $PRECISION / ($PRECISION + $RECALL))" | python)
F_M=$(cat $FILE_EVAL_TOOL_RESULT | grep -o -E 'Cumulative Measure.*' | grep -o -E '0\.[0-9]+')

echo "Recall:    $RECALL"
echo "Precision: $PRECISION"
echo "F1:        $F1"
echo "F&M:       $F_M"
echo "$PRECISION	$RECALL	$F1	$F_M"| tr . ,
echo
