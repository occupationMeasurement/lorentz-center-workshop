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
    
    AL_train_data = load_and_preprocess_csv("ALtrainingdata_workshop.csv")
    AL_train, AL_test = train_split(AL_train_data)
    
    # Prepare data
    soc_alt = pd.melt(soc_alt_titles, id_vars=['O*NET-SOC Code'], 
                      value_vars=['Alternate Title', 'Short Title'],
                      value_name="job-title").dropna()
    
    soc_alt["SOC"] = soc_alt["O*NET-SOC Code"].str[0:-3]
    
    soc_def = pd.melt(soc_definitions, id_vars=["SOC Code"], 
                      value_vars=['SOC Title', 'SOC Definition'],
                      value_name="job-title").dropna().rename({'SOC Code':'SOC'}, axis = 1)
    
    soc_train = pd.melt(soc_train_data, id_vars=["soc2010_1"],
                        value_vars=['JobTitle','JobTask'],
                        value_name='job-title').dropna().rename({"soc2010_1":'SOC'}, axis = 1)
    
    ISCO_all = ISCO_all_groups.rename({'ISCO88':'isco88', 'Desc':'job_title'}, axis = 1)
    ISCO_all['job_title'] = ISCO_all.job_title.apply(preprocess_text)
    ISCO_all = ISCO_all[ISCO_all['isco88']// 1000 > 0]
    
    ISCO_alt = ISCO_alt_titles.rename({'ISCO 88 Code':'isco88', 'Included occupations':'job_title'}, axis = 1)
    ISCO_all['job_title'] = ISCO_all['job_title'] + ' ' + ISCO_alt['Tasks include']
    
    AL_df = pd.concat([AL_train, ISCO_all_groups], ignore_index=True)
    
    # Divide into train and test
    soc_train_test = train_test_split(soc_train, test_size=0.2, random_state=3)
    
    # Merge with extra data
    soc_df = pd.concat([soc_alt,soc_def,soc_train_test[0]], ignore_index=True)
    
    # Apply text preprocessing
    soc_df["job-title-ready"] = soc_df["job-title"].apply(lambda x: preprocess_text(x))
    
    AL_df['job-title-ready'] = AL_df["job-title"].apply(lambda x: preprocess_text(str(x)))
    
    soc_test = soc_train_test[1]
    soc_test["job-title-ready"] = soc_test["job-title"].apply(lambda x: preprocess_text(x))
    
    AL_test = AL_test_train[1]
    AL_test['job-title-ready'] = AL_test["job-title"].apply(lambda x: preprocess_text(str(x)))

    # Write to file
    write_for_fasttext(soc_df, 'job-title-ready', 'SOC', 'soc_ft_train.txt')
    write_for_fasttext(soc_test, 'job-title-ready', 'SOC', 'soc_ft_test.txt')
    
    write_for_fasttext(AL_df, 'job-title-ready', 'ISCO88', 'AL_ft_train.txt')
    write_for_fasttext(AL_test, 'job-title-ready', 'ISCO88', 'AL_ft_test.txt')
    

###########################################################
### start main
if __name__ == "__main__":
    main()


























