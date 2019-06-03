#!/bin/bash
Domain="$1"
Language="$2"
for num in 0 1 2 3
  do
  ./run.sh JUNLP $Domain $Language $num
  ./run.sh USAAR $Domain $Language $num
  ./run.sh TAXI $Domain $Language $num
 done
