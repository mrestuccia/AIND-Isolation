#!/bin/bash
COUNTER=8
while [  $COUNTER -lt 25 ]; do
  echo The counter is $COUNTER
  python tournament.py > $COUNTER.log  
  let COUNTER=COUNTER+1 
done