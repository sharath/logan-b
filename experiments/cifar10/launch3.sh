#!/bin/sh
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute cifar10_logan_b.ipynb
jupyter nbconvert cifar10_logan_b.ipynb