# -*- coding: utf-8 -*-
"""

prepare_data_fasttext.py

Purpose:
    Prepare the data for fasttext use, write into txt file

Version:
    1       First start

Date:
    2023/08/23

Author:
    Diego Dabed

"""

import pandas as pd
import os
import sys
sys.path.append("C:\\Users\\Dabed001\\Documents\\GitHub\\lorentz-center-workshop\\")
from preprocessing_Asialymph import preprocess_text, load_and_preprocess_csv, train_split

###########################################################
### write_for_fasttext(df, name_col, name)
def write_for_fasttext(df, name_col_text, name_col_label, name):
    """
    Purpose:
        Write down on a txt file the data for fasttext estimation

    Inputs:
        df               dataframe with a label and text column
        name_col_text    str, name of the text column
        name_col_label   str, name of the labels column
        name             str, name of the file
        

    Return nothing
    """
    
    f = open(name, "w+", encoding="utf-8")
    
    for i in range(len(df)):
        label = str(df.iloc[i].loc[name_col_label])
        senten = "__label__"+label+ " "+str(df.iloc[i].loc[name_col_text])
        senten = senten + "\n"
        f.write(senten)
    f.close()
    
    return



###########################################################
### main
def main():
    
    # Directory of the data files
    os.chdir("C:\\Users\\Dabed001\\Dropbox\\occ-auto-coding")
    
    # Load all datasets
    soc_alt_titles = pd.read_csv("SOC2010_Alternate_Titles.txt", sep='\t')
    ISCO_all_groups = pd.read_excel("ISCO88_all_groups.xlsx")
    soc_definitions = pd.read_excel("soc_2010_definitions.xls")
    soc_train_data = pd.read_csv("SOCtrainingdata_workshop.csv")
    ISCO_alt_titles = pd.read_excel('ISCO-88 EN Structure and defnitions.xlsx')
    valid = load_and_preprocess_csv('ALvalidationdata_workshop_nocode.csv', min_combined_length = 0)
    
    valid = valid[valid['isco88'].str[0]!= str(0)]
    
    AL_train_data = load_and_preprocess_csv("ALtrainingdata_workshop.csv")
    AL_train, AL_test = train_split(AL_train_data)
    
    write_for_fasttext(AL_train, 'combined_text', 'isco88', 'AL_raw_ft_train.txt')
    write_for_fasttext(AL_test, 'combined_text', 'isco88', 'AL_ft_test.txt')
    write_for_fasttext(valid, 'combined_text', 'isco88', 'AL_ft_validation.txt')
    
    ISCO_all_groups['Desc'] = ISCO_all_groups['Desc'].apply(preprocess_text) 
    ISCO_all_groups = ISCO_all_groups[ISCO_all_groups.ISCO88 // 1000 > 0].rename({'ISCO88':'isco88', 'Desc':'combined_text'}, axis = 1)
    
    ISCO_alt_titles['combined_text'] = str(ISCO_alt_titles['Title EN']) + ' ' + str(ISCO_alt_titles['Definition']) + ' ' + str(ISCO_alt_titles['Tasks include']) + ' ' + str(ISCO_alt_titles['Included occupations'])
    ISCO_alt_titles['combined_text'] = ISCO_alt_titles['combined_text'].apply(preprocess_text) 
    ISCO_alt_titles = ISCO_alt_titles[ISCO_alt_titles['ISCO 88 Code'] // 1000 > 0].rename({'ISCO 88 Code':'isco88', 'Desc':'combined_text'}, axis = 1)
    
    AL_train_extended = pd.concat([AL_train, ISCO_all_groups, ISCO_alt_titles], ignore_index=True)
    write_for_fasttext(AL_train_extended, 'combined_text', 'isco88', 'AL_extended_ft_train.txt')
    
    valid[''].apply(lambda x: preprocess_text(x, )).rename('')

    

###########################################################
### start main
if __name__ == "__main__":
    main()


























