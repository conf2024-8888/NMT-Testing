#!/bin/sh
set -e 
echo Pipeline Start
echo Replacement of words........
python llama3_gri.py
python llama3_wali.py
python llama3.py

