#!/bin/bash
COUNTER=2
while [  $COUNTER -lt 20 ]; do
  echo The counter is $COUNTER
  python tournament.py > $COUNTER.log  
  let COUNTER=COUNTER+1 
done