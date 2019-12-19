#!/bin/sh
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute mnist.ipynb
jupyter nbconvert mnist.ipynb