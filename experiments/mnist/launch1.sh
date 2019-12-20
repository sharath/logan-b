#!/bin/sh
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute mnist_bigan.ipynb
jupyter nbconvert mnist_bigan.ipynb