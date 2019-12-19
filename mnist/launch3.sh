#!/bin/sh
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute mnist_logan_b.ipynb
jupyter nbconvert mnist_logan_b.ipynb