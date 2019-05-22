#!/bin/bash
CYCLE_REMOVING_TOOL="graph_pruning/graph_pruning.py"
CYCLE_REMOVING_METHOD="tarjan"

CLEANING_TOOL="graph_pruning/cleaning.py"

EVAL_TOOL="eval/taxi_eval_archive/TExEval.jar"


EVAL_ROOT="$3"
EVAL_GOLD_STANDARD=eval/gold_${EVAL_ROOT}.taxo
EVAL_JVM="-Xmx9000m"

OUTPUT_DIR="out"

FILE_INPUT=$(basename "$1")
FILE_PRUNED_OUT=${FILE_INPUT}-pruned.csv
FILE_CLEANED_OUT=${FILE_PRUNED_OUT}-cleaned.csv

FILE_EVAL_TOOL_RESULT=${FILE_CLEANED_OUT}-evalresul.txt




DOMAIN=$EVAL_ROOT
if [ -n "$2" ]; then
	CYCLE_REMOVING_METHOD=$2
fi

echo Reading input: $1
echo Reading file: $FILE_INPUT
echo Output directory: $OUTPUT_DIR
echo Cycle removing method: $CYCLE_REMOVING_METHOD
echo Domain: $DOMAIN

echo

if [[ ! -e $OUTPUT_DIR ]]; then
	mkdir $OUTPUT_DIR
fi

echo "======================================================================================================================"
echo "Cycle removing: python $CYCLE_REMOVING_TOOL $1 $OUTPUT_DIR/$FILE_PRUNED_OUT $CYCLE_REMOVING_METHOD"

CYCLES=$(python $CYCLE_REMOVING_TOOL $1 $OUTPUT_DIR/$FILE_PRUNED_OUT $CYCLE_REMOVING_METHOD | tee /dev/tty)
echo "Cycle removing finished. Written to: $OUTPUT_DIR/$FILE_PRUNED_OUT"
echo

echo "======================================================================================================================"
echo "Cleaning: python $CLEANING_TOOL $OUTPUT_DIR/$FILE_PRUNED_OUT $OUTPUT_DIR/$FILE_CLEANED_OUT $DOMAIN"
python3 $CLEANING_TOOL $OUTPUT_DIR/$FILE_PRUNED_OUT $OUTPUT_DIR/$FILE_CLEANED_OUT $DOMAIN
echo "Finished cleaning. Write output to: $OUTPUT_DIR/$FILE_CLEANED_OUT"
echo


echo "======================================================================================================================"

echo "Running eval-tool: java $EVAL_JVM -jar $EVAL_TOOL $OUTPUT_DIR/$FILE_CLEANED_OUT $EVAL_GOLD_STANDARD $EVAL_ROOT $OUTPUT_DIR/$FILE_EVAL_TOOL_RESULT"
java $EVAL_JVM -jar $EVAL_TOOL $OUTPUT_DIR/$FILE_CLEANED_OUT $EVAL_GOLD_STANDARD $EVAL_ROOT $OUTPUT_DIR/$FILE_EVAL_TOOL_RESULT 2> $OUTPUT_DIR/eval.out

echo "Result of eval-tool written to: $OUTPUT_DIR/$FILE_EVAL_TOOL_RESULT"


echo "======================================================================================================================"

L_GOLD="$(wc -l $EVAL_GOLD_STANDARD | grep -o -E '^[0-9]+').0"
L_INPUT="$(wc -l $OUTPUT_DIR/$FILE_CLEANED_OUT | grep -o -E '^[0-9]+').0"


#echo $L_INPUT
#echo $L_GOLD
RECALL="$(tail -n 1 $OUTPUT_DIR/$FILE_EVAL_TOOL_RESULT | grep -o -E '[0-9]+[\.]?[0-9]*')"
PRECISION=$(echo "print($RECALL * $L_GOLD / $L_INPUT)" | python)
F1=$(echo "print(2 * $RECALL * $PRECISION / ($PRECISION + $RECALL))" | python)
F_M=$(cat $OUTPUT_DIR/$FILE_EVAL_TOOL_RESULT | grep -o -E 'Cumulative Measure.*' | grep -o -E '0\.[0-9]+')
CYCLES_REMOVED=$(echo $CYCLES | grep -o -E 'Removed: [0-9]+' | grep -o -E '[0-9]+') # Really dirty: First get the part with the cycle output and then parse the actual cycles

echo "Recall:    $RECALL"
echo "Precision: $PRECISION"
echo "F1:        $F1"
echo "F&M:       $F_M"
echo "$PRECISION	$RECALL	$F1	$F_M"| tr . ,
echo

echo "Copy to https://docs.google.com/spreadsheets/d/1cTUfm97m3vhnOOzYvbqzhJhFQSyWySszLcA6PYf8wFY/edit?usp=sharing"
echo -e "$(date +%F)\t$(whoami)\ttaxi\t$1\t$DOMAIN\t$CYCLE_REMOVING_METHOD\t$CYCLES_REMOVED\t$RECALL\t$PRECISION\t$F1\t$F_M"
echo
echo "Script finished."
