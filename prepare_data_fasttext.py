# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os
import sys
sys.path.append("C:\\Users\\Dabed001\\Documents\\GitHub\\lorentz-center-workshop\\")
from preprocessing_Asialymph.py import preprocess_text


os.chdir("C:\\Users\\Dabed001\\Dropbox\\occ-auto-coding")


soc_alt_titles = pd.read_csv("SOC2010_Alternate_Titles.txt", sep='\t')
ISCO_all_groups = pd.read_excel("ISCO88_all_groups.xlsx")
soc_definitions = pd.read_excel("soc_2010_definitions.xls")
soc_train_data = pd.read_csv("SOCtrainingdata_workshop.csv")
AL_train_data = pd.read_csv("ALtrainingdata_workshop.csv")



soc_alt = pd.melt(soc_alt_titles, id_vars=['O*NET-SOC Code'], 
                  value_vars=['Alternate Title', 'Short Title'],
                  value_name="job-title").dropna()

soc_alt["SOC"] = soc_alt["O*NET-SOC Code"].str[0:-3]

soc_def = pd.melt(soc_definitions, id_vars=["SOC Code"], 
                  value_vars=['SOC Title', 'SOC Definition'],
                  value_name="job-title").dropna()


