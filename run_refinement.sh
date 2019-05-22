#!/bin/bash

SYSTEM="$1"
DOMAIN="$2"
METHOD="$3"

echo $SYSTEM
CYCLE_REMOVING_TOOL="graph_pruning/graph_pruning.py"
CYCLE_REMOVING_METHOD="tarjan"
CLEANING_TOOL="graph_pruning/cleaning.py"
EVAL_TOOL="eval/taxi_eval_archive/TExEval.jar"
EVAL_GOLD_STANDARD=eval/gold_${DOMAIN}.taxo
EVAL_JVM="-Xmx9000m"
OUTPUT_DIR="out"
FILE_INPUT=systems/$SYSTEM/EN/${SYSTEM}_${DOMAIN}.taxo
FILE_PRUNED_OUT=${FILE_INPUT}-pruned.csv
FILE_CLEANED_OUT=${FILE_PRUNED_OUT}-cleaned.csv
CYCLE_REMOVING_METHOD=tarjan

if [[ ! -e $OUTPUT_DIR ]]; then
	mkdir $OUTPUT_DIR
fi

echo Reading file: $FILE_INPUT
echo Output directory: $OUTPUT_DIR
echo Domain: $DOMAIN

echo "======================================================================================================================"
echo "Cycle removing: python $CYCLE_REMOVING_TOOL $FILE_INPUT $OUTPUT_DIR/$FILE_PRUNED_OUT $CYCLE_REMOVING_METHOD"

CYCLES=$(python $CYCLE_REMOVING_TOOL $1 $OUTPUT_DIR/$FILE_PRUNED_OUT $CYCLE_REMOVING_METHOD | tee /dev/tty)
echo "Cycle removing finished. Written to: $OUTPUT_DIR/$FILE_PRUNED_OUT"
echo

echo "======================================================================================================================"
echo "Cleaning: python $CLEANING_TOOL $OUTPUT_DIR/$FILE_PRUNED_OUT $OUTPUT_DIR/$FILE_CLEANED_OUT $DOMAIN"
python3 $CLEANING_TOOL $OUTPUT_DIR/$FILE_PRUNED_OUT $OUTPUT_DIR/$FILE_CLEANED_OUT $DOMAIN
echo "Finished cleaning. Write output to: $OUTPUT_DIR/$FILE_CLEANED_OUT"
echo

echo "======================================================================================================================"


if [ "$METHOD" -eq 0 ]; then
  echo "Taxonomy refinement: Reconnect disconnected nodes to the root"
  python taxonomy_refinement.py -m root -e own_and_poincare -d $DOMAIN -sys $SYSTEM
  VALUE="root"

elif [ "$METHOD" -eq 1 ]; then
  echo "Taxonomy refinement: Employing word2vec embeddings"
  python taxonomy_refinement.py -m combined_embeddings_removal_and_new -e own_and_poincare -d $DOMAIN -sys $SYSTEM -ico -ep -com
  VALUE="False"

elif [ "$METHOD" -eq 2 ]; then
  echo "Taxonomy refinement: Employing wordnet poincaré embeddings"
  python taxonomy_refinement.py -m combined_embeddings_removal_and_new -e own_and_poincare -d $DOMAIN -sys $SYSTEM -com -wn
  VALUE="WN"

elif [ "$METHOD" -eq 3 ]; then
  echo "Taxonomy refinment: Employing custom poincaré embeddings"
  python taxonomy_refinement.py -m combined_embeddings_removal_and_new -e own_and_poincare -d $DOMAIN -sys $SYSTEM -com
  VALUE="True"
fi

FILE_INPUT=refinement_out/distributed_semantics_${DOMAIN}_${SYSTEM}_$VALUE.csv
FILE_PRUNED_OUT=${FILE_INPUT}-pruned.csv
FILE_CLEANED_OUT=${FILE_PRUNED_OUT}-refined.csv
FILE_EVAL_TOOL_RESULT=${FILE_CLEANED_OUT}-evalresul.txt

echo "======================================================================================================================"
echo "Cycle removing: python $CYCLE_REMOVING_TOOL $FILE_INPUT $OUTPUT_DIR/$FILE_PRUNED_OUT $CYCLE_REMOVING_METHOD"

CYCLES=$(python $CYCLE_REMOVING_TOOL $1 $OUTPUT_DIR/$FILE_PRUNED_OUT $CYCLE_REMOVING_METHOD | tee /dev/tty)
echo "Cycle removing finished. Written to: $OUTPUT_DIR/$FILE_PRUNED_OUT"
echo

echo "======================================================================================================================"
echo "Cleaning: python $CLEANING_TOOL $OUTPUT_DIR/$FILE_PRUNED_OUT $OUTPUT_DIR/$FILE_CLEANED_OUT $DOMAIN"
python3 $CLEANING_TOOL $OUTPUT_DIR/$FILE_PRUNED_OUT $OUTPUT_DIR/$FILE_CLEANED_OUT $DOMAIN
echo "Finished cleaning. Write output to: $OUTPUT_DIR/$FILE_CLEANED_OUT"
echo

echo "======================================================================================================================"

L_GOLD="$(wc -l $EVAL_GOLD_STANDARD | grep -o -E '^[0-9]+').0"
L_INPUT="$(wc -l $OUTPUT_DIR/$FILE_CLEANED_OUT | grep -o -E '^[0-9]+').0"

RECALL="$(tail -n 1 $OUTPUT_DIR/$FILE_EVAL_TOOL_RESULT | grep -o -E '[0-9]+[\.]?[0-9]*')"
PRECISION=$(echo "print($RECALL * $L_GOLD / $L_INPUT)" | python)
F1=$(echo "print(2 * $RECALL * $PRECISION / ($PRECISION + $RECALL))" | python)
F_M=$(cat $OUTPUT_DIR/$FILE_EVAL_TOOL_RESULT | grep -o -E 'Cumulative Measure.*' | grep -o -E '0\.[0-9]+')

echo "Recall:    $RECALL"
echo "Precision: $PRECISION"
echo "F1:        $F1"
echo "F&M:       $F_M"
echo "$PRECISION	$RECALL	$F1	$F_M"| tr . ,
echo
