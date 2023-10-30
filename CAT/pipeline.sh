#!/bin/sh

set -e 

echo Pipeline Start
echo Replacement of words........
sh gentest.sh
echo Translate mutants.........
python generate_lookup.py
echo Analyze and compare translations........
echo desp.sh
sh desp.sh
echo read2diff
python3 read2diff.py
echo read_diff
python3 read_diff.py
echo readbugs
python3 readbugs.py
echo read_human
python3 read_human.py
