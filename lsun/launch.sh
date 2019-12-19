#!/bin/sh
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute lsun.ipynb
jupyter nbconvert lsun.ipynb