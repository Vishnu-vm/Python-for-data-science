#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 10:43:17 2020

@author: Vishnu Mohan
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


data=pd.read_csv('/Volumes/Data/Course Content/DS content/A.Rules/Titanic.csv')
data=pd.DataFrame(data)

data=data.iloc[:,1:]

data_dum=pd.get_dummies(data)

freq_items=apriori(data_dum, min_support=0.5,use_colnames=True)
rules =association_rules(freq_items, metric='confidence', min_threshold=0.8, support_only=False)
rules.head()

rules.columns

sorted_rules=rules.sort_values(by='lift')
