#!/bin/sh
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute cifar10_dcgan.ipynb
jupyter nbconvert cifar10_dcgan.ipynb