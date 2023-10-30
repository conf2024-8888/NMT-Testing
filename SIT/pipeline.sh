#!/bin/sh
set -e 
echo Pipeline Start
echo Vulnerable token identification......
python GRI.py
python WALI.py
echo Replacement of words........
python SIT.py