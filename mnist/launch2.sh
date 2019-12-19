#!/bin/sh
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute mnist_dcgan.ipynb
jupyter nbconvert mnist_dcgan.ipynb