#!/bin/sh
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute cifar10.ipynb
jupyter nbconvert cifar10.ipynb