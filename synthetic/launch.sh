#!/bin/sh
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute gan.ipynb
jupyter nbconvert gan.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute bigan.ipynb
jupyter nbconvert bigan.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute logan-b.ipynb
jupyter nbconvert logan-b.ipynb