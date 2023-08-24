#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_fasttext.py

Purpose:
    Run fasttext model on occupation data

Version:
    1       First start

Date:
    2023/08/23

Author:
    Diego Dabed
"""
###########################################################
### Imports
import fasttext
import os
import pandas as pd
import numpy as np

###########################################################
### main
def main():
    # Magic numbers
    path = "C:\\Users\\Dabed001\\Dropbox\\occ-auto-coding"
    os.chdir(path)
    
    model = fasttext.train_supervised(input = 'AL_ft_train.txt',
                                          dim = 300,
                                          neg = 10,
                                          wordNgrams = 2,
                                          loss = 'softmax',
                                          thread = 4,
                                          minCount = 1,
                                          pretrainedVectors = 'crawl-300d-2M-subword.vec')
    
    agg_results = model.test("AL_ft_test.txt")
    
    print(agg_results)
    
    occupation_results = model.test_label("AL_ft_test.txt")
    
    occs = pd.DataFrame.from_dict(occupation_results, orient = "index")
    
    occs.to_pickle("./Output/AL_ISCO_f1_ft_20230823.pkl")
    
    print(np.mean(occs.f1score))
    print(np.std(occs.f1score))
    
    model.save_model('./Output/AL_ISCO_ft_dim_300_20230823.bin')
    
###########################################################
### start main
if __name__ == "__main__":
    main()
