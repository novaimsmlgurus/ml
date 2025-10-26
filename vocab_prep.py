#!/usr/bin/env python
# coding: utf-8

# # Building a Clean Vocabulary of Car Brands and Models

# ## Loading the data and data overview

# In[55]:


import numpy as np
import pandas as pd


# In[56]:


# this loads the train data 
train_raw = pd.read_csv('project_data/train.csv')


# In[57]:


pd.set_option('display.max_rows', 500)


# In[58]:


train = train_raw.copy()


# ## Brands

# In[59]:


# display all unique values 
train['Brand'].unique()


# In[60]:


# change all values to lower case and strip starting and ending spaces
train['Brand'] = train['Brand'].str.lower().str.strip()
np.array(sorted(train['Brand'].dropna().unique()))


# In[61]:


# Correct misspelled values in 'Brand' column
brand_corrections = {
    'aud': 'audi',
    'udi': 'audi',
    'ud': 'audi',
    'mw': 'bmw',
    'bm': 'bmw',
    'for': 'ford',
    'ord': 'ford', 
    'or': 'ford',
    'hyunda': 'hyundai',
    'yundai': 'hyundai',
    'yunda': 'hyundai',
    'mercedes': 'mercedes-benz',
    'mercede': 'mercedes-benz',
    'ercedes': 'mercedes-benz',
    'ercede': 'mercedes-benz',
    'mercedes benz': 'mercedes-benz',
    'koda': 'skoda',
    'skod': 'skoda',
    'kod': 'skoda',
    'toyot': 'toyota',
    'oyota': 'toyota',
    'pel': 'opel',
    'pe': 'opel',
    'ope': 'opel',
    'vw': 'volkswagen',
    'v': 'volkswagen',
    'w': 'volkswagen'
}

train['Brand'] = train['Brand'].replace(brand_corrections)
train['Brand'].unique()


# ## model

# In[62]:


train['model'].unique()


# In[63]:


train['model'].unique().shape


# In[64]:


# change all values to lower case and strip starting and ending spaces
train['model'] = train['model'].str.lower().str.strip()
np.array(sorted(train['model'].dropna().unique()))


# In[65]:


uniq_pairs = (
    train[['Brand', 'model']]
    .dropna()
    .drop_duplicates()
    .sort_values(['Brand','model'])
    .reset_index(drop=True)
)
uniq_pairs


# In[66]:


brand_model = [
{"brand": "audi", "model_lower": "a", "model_correct": "delete"},
{"brand": "audi", "model_lower": "a1", "model_correct": ""},
{"brand": "audi", "model_lower": "a2", "model_correct": ""},
{"brand": "audi", "model_lower": "a3", "model_correct": ""},
{"brand": "audi", "model_lower": "a4", "model_correct": ""},
{"brand": "audi", "model_lower": "a5", "model_correct": ""},
{"brand": "audi", "model_lower": "a6", "model_correct": ""},
{"brand": "audi", "model_lower": "a7", "model_correct": ""},
{"brand": "audi", "model_lower": "a8", "model_correct": ""},
{"brand": "audi", "model_lower": "q", "model_correct": "delete"},
{"brand": "audi", "model_lower": "q2", "model_correct": ""},
{"brand": "audi", "model_lower": "q3", "model_correct": ""},
{"brand": "audi", "model_lower": "q5", "model_correct": ""},
{"brand": "audi", "model_lower": "q7", "model_correct": ""},
{"brand": "audi", "model_lower": "q8", "model_correct": ""},
{"brand": "audi", "model_lower": "r8", "model_correct": ""},
{"brand": "audi", "model_lower": "rs", "model_correct": "delete"},
{"brand": "audi", "model_lower": "rs3", "model_correct": ""},
{"brand": "audi", "model_lower": "rs4", "model_correct": ""},
{"brand": "audi", "model_lower": "rs5", "model_correct": ""},
{"brand": "audi", "model_lower": "rs6", "model_correct": ""},
{"brand": "audi", "model_lower": "s3", "model_correct": ""},
{"brand": "audi", "model_lower": "s4", "model_correct": ""},
{"brand": "audi", "model_lower": "s5", "model_correct": ""},
{"brand": "audi", "model_lower": "s8", "model_correct": ""},
{"brand": "audi", "model_lower": "sq5", "model_correct": ""},
{"brand": "audi", "model_lower": "sq7", "model_correct": ""},
{"brand": "audi", "model_lower": "t", "model_correct": "delete"},
{"brand": "audi", "model_lower": "tt", "model_correct": ""},
{"brand": "bmw", "model_lower": "1 serie", "model_correct": "1 series"},
{"brand": "bmw", "model_lower": "1 series", "model_correct": ""},
{"brand": "bmw", "model_lower": "2 serie", "model_correct": "2 series"},
{"brand": "bmw", "model_lower": "2 series", "model_correct": ""},
{"brand": "bmw", "model_lower": "3 serie", "model_correct": "3 series"},
{"brand": "bmw", "model_lower": "3 series", "model_correct": ""},
{"brand": "bmw", "model_lower": "4 serie", "model_correct": "4 series"},
{"brand": "bmw", "model_lower": "4 series", "model_correct": ""},
{"brand": "bmw", "model_lower": "5 serie", "model_correct": "5 series"},
{"brand": "bmw", "model_lower": "5 series", "model_correct": ""},
{"brand": "bmw", "model_lower": "6 serie", "model_correct": "6 series"},
{"brand": "bmw", "model_lower": "6 series", "model_correct": ""},
{"brand": "bmw", "model_lower": "7 serie", "model_correct": "7 series"},
{"brand": "bmw", "model_lower": "7 series", "model_correct": ""},
{"brand": "bmw", "model_lower": "8 serie", "model_correct": "8 series"},
{"brand": "bmw", "model_lower": "8 series", "model_correct": ""},
{"brand": "bmw", "model_lower": "i", "model_correct": "delete"},
{"brand": "bmw", "model_lower": "i3", "model_correct": ""},
{"brand": "bmw", "model_lower": "i8", "model_correct": ""},
{"brand": "bmw", "model_lower": "m", "model_correct": "delete"},
{"brand": "bmw", "model_lower": "m2", "model_correct": ""},
{"brand": "bmw", "model_lower": "m3", "model_correct": ""},
{"brand": "bmw", "model_lower": "m4", "model_correct": ""},
{"brand": "bmw", "model_lower": "m5", "model_correct": ""},
{"brand": "bmw", "model_lower": "m6", "model_correct": ""},
{"brand": "bmw", "model_lower": "x", "model_correct": "delete"},
{"brand": "bmw", "model_lower": "x1", "model_correct": ""},
{"brand": "bmw", "model_lower": "x2", "model_correct": ""},
{"brand": "bmw", "model_lower": "x3", "model_correct": ""},
{"brand": "bmw", "model_lower": "x4", "model_correct": ""},
{"brand": "bmw", "model_lower": "x5", "model_correct": ""},
{"brand": "bmw", "model_lower": "x6", "model_correct": ""},
{"brand": "bmw", "model_lower": "x7", "model_correct": ""},
{"brand": "bmw", "model_lower": "z", "model_correct": "delete"},
{"brand": "bmw", "model_lower": "z3", "model_correct": ""},
{"brand": "bmw", "model_lower": "z4", "model_correct": ""},
{"brand": "ford", "model_lower": "b-ma", "model_correct": "b-max"},
{"brand": "ford", "model_lower": "b-max", "model_correct": ""},
{"brand": "ford", "model_lower": "c-ma", "model_correct": "c-max"},
{"brand": "ford", "model_lower": "c-max", "model_correct": ""},
{"brand": "ford", "model_lower": "ecospor", "model_correct": "ecosport"},
{"brand": "ford", "model_lower": "ecosport", "model_correct": ""},
{"brand": "ford", "model_lower": "edg", "model_correct": "edge"},
{"brand": "ford", "model_lower": "edge", "model_correct": ""},
{"brand": "ford", "model_lower": "escort", "model_correct": ""},
{"brand": "ford", "model_lower": "fiest", "model_correct": "fiesta"},
{"brand": "ford", "model_lower": "fiesta", "model_correct": ""},
{"brand": "ford", "model_lower": "focu", "model_correct": "focus"},
{"brand": "ford", "model_lower": "focus", "model_correct": ""},
{"brand": "ford", "model_lower": "fusion", "model_correct": ""},
{"brand": "ford", "model_lower": "galax", "model_correct": "galaxy"},
{"brand": "ford", "model_lower": "galaxy", "model_correct": ""},
{"brand": "ford", "model_lower": "grand c-ma", "model_correct": "grand c-max"},
{"brand": "ford", "model_lower": "grand c-max", "model_correct": ""},
{"brand": "ford", "model_lower": "grand tourneo connec", "model_correct": "grand tourneo connect"},
{"brand": "ford", "model_lower": "grand tourneo connect", "model_correct": ""},
{"brand": "ford", "model_lower": "k", "model_correct": "ka"},
{"brand": "ford", "model_lower": "ka", "model_correct": ""},
{"brand": "ford", "model_lower": "ka+", "model_correct": ""},
{"brand": "ford", "model_lower": "kug", "model_correct": "kuga"},
{"brand": "ford", "model_lower": "kuga", "model_correct": ""},
{"brand": "ford", "model_lower": "monde", "model_correct": "mondeo"},
{"brand": "ford", "model_lower": "mondeo", "model_correct": ""},
{"brand": "ford", "model_lower": "mustang", "model_correct": ""},
{"brand": "ford", "model_lower": "puma", "model_correct": ""},
{"brand": "ford", "model_lower": "ranger", "model_correct": ""},
{"brand": "ford", "model_lower": "s-ma", "model_correct": "s-max"},
{"brand": "ford", "model_lower": "s-max", "model_correct": ""},
{"brand": "ford", "model_lower": "streetka", "model_correct": ""},
{"brand": "ford", "model_lower": "tourneo connect", "model_correct": ""},
{"brand": "ford", "model_lower": "tourneo custo", "model_correct": "tourneo custom"},
{"brand": "ford", "model_lower": "tourneo custom", "model_correct": ""},
{"brand": "hyundai", "model_lower": "accent", "model_correct": ""},
{"brand": "hyundai", "model_lower": "getz", "model_correct": ""},
{"brand": "hyundai", "model_lower": "i1", "model_correct": "i10"},
{"brand": "hyundai", "model_lower": "i10", "model_correct": ""},
{"brand": "hyundai", "model_lower": "i2", "model_correct": "i20"},
{"brand": "hyundai", "model_lower": "i20", "model_correct": ""},
{"brand": "hyundai", "model_lower": "i3", "model_correct": "i30"},
{"brand": "hyundai", "model_lower": "i30", "model_correct": ""},
{"brand": "hyundai", "model_lower": "i40", "model_correct": ""},
{"brand": "hyundai", "model_lower": "i80", "model_correct": "i800"},
{"brand": "hyundai", "model_lower": "i800", "model_correct": ""},
{"brand": "hyundai", "model_lower": "ioni", "model_correct": "ioniq"},
{"brand": "hyundai", "model_lower": "ioniq", "model_correct": ""},
{"brand": "hyundai", "model_lower": "ix2", "model_correct": "ix20"},
{"brand": "hyundai", "model_lower": "ix20", "model_correct": ""},
{"brand": "hyundai", "model_lower": "ix35", "model_correct": ""},
{"brand": "hyundai", "model_lower": "kon", "model_correct": "kona"},
{"brand": "hyundai", "model_lower": "kona", "model_correct": ""},
{"brand": "hyundai", "model_lower": "santa f", "model_correct": "santa fe"},
{"brand": "hyundai", "model_lower": "santa fe", "model_correct": ""},
{"brand": "hyundai", "model_lower": "terracan", "model_correct": ""},
{"brand": "hyundai", "model_lower": "tucso", "model_correct": "tucson"},
{"brand": "hyundai", "model_lower": "tucson", "model_correct": ""},
{"brand": "hyundai", "model_lower": "veloste", "model_correct": "veloster"},
{"brand": "mercedes-benz", "model_lower": "200", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "220", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "230", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "a clas", "model_correct": "a class"},
{"brand": "mercedes-benz", "model_lower": "a class", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "b clas", "model_correct": "b class"},
{"brand": "mercedes-benz", "model_lower": "b class", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "c clas", "model_correct": "c class"},
{"brand": "mercedes-benz", "model_lower": "c class", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "cl clas", "model_correct": "cl class"},
{"brand": "mercedes-benz", "model_lower": "cl class", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "cla class", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "clc class", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "clk", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "cls clas", "model_correct": "cls class"},
{"brand": "mercedes-benz", "model_lower": "cls class", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "e clas", "model_correct": "e class"},
{"brand": "mercedes-benz", "model_lower": "e class", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "g class", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "gl class", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "gla clas", "model_correct": "gla class"},
{"brand": "mercedes-benz", "model_lower": "gla class", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "glb class", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "glc clas", "model_correct": "glc class"},
{"brand": "mercedes-benz", "model_lower": "glc class", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "gle clas", "model_correct": "gle class"},
{"brand": "mercedes-benz", "model_lower": "gle class", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "gls clas", "model_correct": "gls class"},
{"brand": "mercedes-benz", "model_lower": "gls class", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "m clas", "model_correct": "m class"},
{"brand": "mercedes-benz", "model_lower": "m class", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "s clas", "model_correct": "s class"},
{"brand": "mercedes-benz", "model_lower": "s class", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "sl", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "sl clas", "model_correct": "sl class"},
{"brand": "mercedes-benz", "model_lower": "sl class", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "slk", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "v clas", "model_correct": "v class"},
{"brand": "mercedes-benz", "model_lower": "v class", "model_correct": ""},
{"brand": "mercedes-benz", "model_lower": "x-clas", "model_correct": "x-class"},
{"brand": "mercedes-benz", "model_lower": "x-class", "model_correct": ""},
{"brand": "opel", "model_lower": "ada", "model_correct": "adam"},
{"brand": "opel", "model_lower": "adam", "model_correct": ""},
{"brand": "opel", "model_lower": "agila", "model_correct": ""},
{"brand": "opel", "model_lower": "ampera", "model_correct": ""},
{"brand": "opel", "model_lower": "antara", "model_correct": ""},
{"brand": "opel", "model_lower": "astr", "model_correct": "astra"},
{"brand": "opel", "model_lower": "astra", "model_correct": ""},
{"brand": "opel", "model_lower": "cascada", "model_correct": ""},
{"brand": "opel", "model_lower": "combo lif", "model_correct": "combo life"},
{"brand": "opel", "model_lower": "combo life", "model_correct": ""},
{"brand": "opel", "model_lower": "cors", "model_correct": "corsa"},
{"brand": "opel", "model_lower": "corsa", "model_correct": ""},
{"brand": "opel", "model_lower": "crossland", "model_correct": ""},
{"brand": "opel", "model_lower": "crossland x", "model_correct": ""},
{"brand": "opel", "model_lower": "grandland", "model_correct": ""},
{"brand": "opel", "model_lower": "grandland x", "model_correct": ""},
{"brand": "opel", "model_lower": "gtc", "model_correct": ""},
{"brand": "opel", "model_lower": "insigni", "model_correct": "insignia"},
{"brand": "opel", "model_lower": "insignia", "model_correct": ""},
{"brand": "opel", "model_lower": "kadjar", "model_correct": "delete"},
{"brand": "opel", "model_lower": "meriv", "model_correct": "meriva"},
{"brand": "opel", "model_lower": "meriva", "model_correct": ""},
{"brand": "opel", "model_lower": "mokk", "model_correct": "mokka"},
{"brand": "opel", "model_lower": "mokka", "model_correct": ""},
{"brand": "opel", "model_lower": "mokka x", "model_correct": ""},
{"brand": "opel", "model_lower": "tigra", "model_correct": ""},
{"brand": "opel", "model_lower": "vectra", "model_correct": ""},
{"brand": "opel", "model_lower": "viv", "model_correct": "delete"},
{"brand": "opel", "model_lower": "viva", "model_correct": ""},
{"brand": "opel", "model_lower": "vivaro", "model_correct": ""},
{"brand": "opel", "model_lower": "zafir", "model_correct": "zafira"},
{"brand": "opel", "model_lower": "zafira", "model_correct": ""},
{"brand": "opel", "model_lower": "zafira toure", "model_correct": "zafira tourer"},
{"brand": "opel", "model_lower": "zafira tourer", "model_correct": ""},
{"brand": "skoda", "model_lower": "citig", "model_correct": "citigo"},
{"brand": "skoda", "model_lower": "citigo", "model_correct": ""},
{"brand": "skoda", "model_lower": "fabi", "model_correct": "fabia"},
{"brand": "skoda", "model_lower": "fabia", "model_correct": ""},
{"brand": "skoda", "model_lower": "kami", "model_correct": "kamiq"},
{"brand": "skoda", "model_lower": "kamiq", "model_correct": ""},
{"brand": "skoda", "model_lower": "karo", "model_correct": "karoq"},
{"brand": "skoda", "model_lower": "karoq", "model_correct": ""},
{"brand": "skoda", "model_lower": "kodia", "model_correct": "kodiaq"},
{"brand": "skoda", "model_lower": "kodiaq", "model_correct": ""},
{"brand": "skoda", "model_lower": "octavi", "model_correct": "octavia"},
{"brand": "skoda", "model_lower": "octavia", "model_correct": ""},
{"brand": "skoda", "model_lower": "rapi", "model_correct": "rapid"},
{"brand": "skoda", "model_lower": "rapid", "model_correct": ""},
{"brand": "skoda", "model_lower": "roomste", "model_correct": "roomster"},
{"brand": "skoda", "model_lower": "roomster", "model_correct": ""},
{"brand": "skoda", "model_lower": "scal", "model_correct": "scala"},
{"brand": "skoda", "model_lower": "scala", "model_correct": ""},
{"brand": "skoda", "model_lower": "super", "model_correct": "superb"},
{"brand": "skoda", "model_lower": "superb", "model_correct": ""},
{"brand": "skoda", "model_lower": "yet", "model_correct": "yeti"},
{"brand": "skoda", "model_lower": "yeti", "model_correct": ""},
{"brand": "skoda", "model_lower": "yeti outdoo", "model_correct": "yeti outdoor"},
{"brand": "skoda", "model_lower": "yeti outdoor", "model_correct": ""},
{"brand": "toyota", "model_lower": "auri", "model_correct": "auris"},
{"brand": "toyota", "model_lower": "auris", "model_correct": ""},
{"brand": "toyota", "model_lower": "avensis", "model_correct": ""},
{"brand": "toyota", "model_lower": "ayg", "model_correct": "aygo"},
{"brand": "toyota", "model_lower": "aygo", "model_correct": ""},
{"brand": "toyota", "model_lower": "c-h", "model_correct": "c-hr"},
{"brand": "toyota", "model_lower": "c-hr", "model_correct": ""},
{"brand": "toyota", "model_lower": "camry", "model_correct": ""},
{"brand": "toyota", "model_lower": "coroll", "model_correct": "corolla"},
{"brand": "toyota", "model_lower": "corolla", "model_correct": ""},
{"brand": "toyota", "model_lower": "gt86", "model_correct": ""},
{"brand": "toyota", "model_lower": "hilu", "model_correct": "hilux"},
{"brand": "toyota", "model_lower": "hilux", "model_correct": ""},
{"brand": "toyota", "model_lower": "iq", "model_correct": ""},
{"brand": "toyota", "model_lower": "land cruise", "model_correct": "land cruiser"},
{"brand": "toyota", "model_lower": "land cruiser", "model_correct": ""},
{"brand": "toyota", "model_lower": "prius", "model_correct": ""},
{"brand": "toyota", "model_lower": "proace verso", "model_correct": ""},
{"brand": "toyota", "model_lower": "rav", "model_correct": "rav4"},
{"brand": "toyota", "model_lower": "rav4", "model_correct": ""},
{"brand": "toyota", "model_lower": "suvra", "model_correct": "supra"},
{"brand": "toyota", "model_lower": "suvra", "model_correct": "supra"},
{"brand": "toyota", "model_lower": "supra", "model_correct": ""},
{"brand": "toyota", "model_lower": "urban cruise", "model_correct": "urban cruiser"},
{"brand": "toyota", "model_lower": "urban cruiser", "model_correct": ""},
{"brand": "toyota", "model_lower": "vers", "model_correct": "verso"},
{"brand": "toyota", "model_lower": "verso", "model_correct": ""},
{"brand": "toyota", "model_lower": "verso-s", "model_correct": ""},
{"brand": "toyota", "model_lower": "yari", "model_correct": "yaris"},
{"brand": "toyota", "model_lower": "yaris", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "amaro", "model_correct": "amarok"},
{"brand": "volkswagen", "model_lower": "amarok", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "arteo", "model_correct": "arteon"},
{"brand": "volkswagen", "model_lower": "arteon", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "beetl", "model_correct": "beetle"},
{"brand": "volkswagen", "model_lower": "beetle", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "caddy", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "caddy life", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "caddy maxi", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "caddy maxi lif", "model_correct": "caddy maxi life"},
{"brand": "volkswagen", "model_lower": "caddy maxi life", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "california", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "caravell", "model_correct": "caravelle"},
{"brand": "volkswagen", "model_lower": "caravelle", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "cc", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "eos", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "fox", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "gol", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "golf", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "golf s", "model_correct": "golf sv"},
{"brand": "volkswagen", "model_lower": "golf sv", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "jetta", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "passa", "model_correct": "passat"},
{"brand": "volkswagen", "model_lower": "passat", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "pol", "model_correct": "polo"},
{"brand": "volkswagen", "model_lower": "polo", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "scirocc", "model_correct": "scirocco"},
{"brand": "volkswagen", "model_lower": "scirocco", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "shara", "model_correct": "sharan"},
{"brand": "volkswagen", "model_lower": "sharan", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "shuttle", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "t-cros", "model_correct": "t-cross"},
{"brand": "volkswagen", "model_lower": "t-cross", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "t-ro", "model_correct": "t-roc"},
{"brand": "volkswagen", "model_lower": "t-roc", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "tigua", "model_correct": "tiguan"},
{"brand": "volkswagen", "model_lower": "tiguan", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "tiguan allspac", "model_correct": "tiguan allspace"},
{"brand": "volkswagen", "model_lower": "tiguan allspace", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "touare", "model_correct": "touareg"},
{"brand": "volkswagen", "model_lower": "touareg", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "toura", "model_correct": "touran"},
{"brand": "volkswagen", "model_lower": "touran", "model_correct": ""},
{"brand": "volkswagen", "model_lower": "u", "model_correct": "up"},
{"brand": "volkswagen", "model_lower": "up", "model_correct": ""}
]


# In[67]:


brand_model_df = pd.DataFrame(brand_model)
brand_model_df


# In[68]:


brand_model_df = brand_model_df[brand_model_df["model_correct"] != "delete"]


# In[69]:


brand_model_df


# In[70]:


# Boolean
brand_model_df = brand_model_df.copy()
brand_model_df.loc[:, "correction"] = brand_model_df["model_correct"].fillna("").ne("").astype("int8")


# In[71]:


brand_model_df


# In[72]:


# It keeps each current model_correct value if it’s not empty (ne("")), and otherwise replaces it with the corresponding model_lower value.
brand_model_df["model_correct"] = brand_model_df["model_correct"].where(brand_model_df["model_correct"].ne(""), brand_model_df["model_lower"])


# In[73]:


brand_model_df


# In[83]:


import json
from pathlib import Path

# 1) Build records in the right column order; convert NaN -> None
records = (
    brand_model_df[['brand','model_lower','model_correct','correction']]
    .where(brand_model_df.notna(), None)
    .to_dict(orient='records')
)

# records = list of dicts in desired key order (e.g., from df[['brand','model_lower','model_correct']].to_dict('records'))
line = lambda r: json.dumps(r, ensure_ascii=False, separators=(", ", ": "))
txt = "[\n  " + ",\n  ".join(line(r) for r in records) + "\n]"
#print(txt)


# In[94]:


BRAND_MODEL_VOCAB = [
  {"brand": "audi", "model_lower": "a1", "model_correct": "a1", "correction": 0},
  {"brand": "audi", "model_lower": "a2", "model_correct": "a2", "correction": 0},
  {"brand": "audi", "model_lower": "a3", "model_correct": "a3", "correction": 0},
  {"brand": "audi", "model_lower": "a4", "model_correct": "a4", "correction": 0},
  {"brand": "audi", "model_lower": "a5", "model_correct": "a5", "correction": 0},
  {"brand": "audi", "model_lower": "a6", "model_correct": "a6", "correction": 0},
  {"brand": "audi", "model_lower": "a7", "model_correct": "a7", "correction": 0},
  {"brand": "audi", "model_lower": "a8", "model_correct": "a8", "correction": 0},
  {"brand": "audi", "model_lower": "q2", "model_correct": "q2", "correction": 0},
  {"brand": "audi", "model_lower": "q3", "model_correct": "q3", "correction": 0},
  {"brand": "audi", "model_lower": "q5", "model_correct": "q5", "correction": 0},
  {"brand": "audi", "model_lower": "q7", "model_correct": "q7", "correction": 0},
  {"brand": "audi", "model_lower": "q8", "model_correct": "q8", "correction": 0},
  {"brand": "audi", "model_lower": "r8", "model_correct": "r8", "correction": 0},
  {"brand": "audi", "model_lower": "rs3", "model_correct": "rs3", "correction": 0},
  {"brand": "audi", "model_lower": "rs4", "model_correct": "rs4", "correction": 0},
  {"brand": "audi", "model_lower": "rs5", "model_correct": "rs5", "correction": 0},
  {"brand": "audi", "model_lower": "rs6", "model_correct": "rs6", "correction": 0},
  {"brand": "audi", "model_lower": "s3", "model_correct": "s3", "correction": 0},
  {"brand": "audi", "model_lower": "s4", "model_correct": "s4", "correction": 0},
  {"brand": "audi", "model_lower": "s5", "model_correct": "s5", "correction": 0},
  {"brand": "audi", "model_lower": "s8", "model_correct": "s8", "correction": 0},
  {"brand": "audi", "model_lower": "sq5", "model_correct": "sq5", "correction": 0},
  {"brand": "audi", "model_lower": "sq7", "model_correct": "sq7", "correction": 0},
  {"brand": "audi", "model_lower": "tt", "model_correct": "tt", "correction": 0},
  {"brand": "bmw", "model_lower": "1 serie", "model_correct": "1 series", "correction": 1},
  {"brand": "bmw", "model_lower": "1 series", "model_correct": "1 series", "correction": 0},
  {"brand": "bmw", "model_lower": "2 serie", "model_correct": "2 series", "correction": 1},
  {"brand": "bmw", "model_lower": "2 series", "model_correct": "2 series", "correction": 0},
  {"brand": "bmw", "model_lower": "3 serie", "model_correct": "3 series", "correction": 1},
  {"brand": "bmw", "model_lower": "3 series", "model_correct": "3 series", "correction": 0},
  {"brand": "bmw", "model_lower": "4 serie", "model_correct": "4 series", "correction": 1},
  {"brand": "bmw", "model_lower": "4 series", "model_correct": "4 series", "correction": 0},
  {"brand": "bmw", "model_lower": "5 serie", "model_correct": "5 series", "correction": 1},
  {"brand": "bmw", "model_lower": "5 series", "model_correct": "5 series", "correction": 0},
  {"brand": "bmw", "model_lower": "6 serie", "model_correct": "6 series", "correction": 1},
  {"brand": "bmw", "model_lower": "6 series", "model_correct": "6 series", "correction": 0},
  {"brand": "bmw", "model_lower": "7 serie", "model_correct": "7 series", "correction": 1},
  {"brand": "bmw", "model_lower": "7 series", "model_correct": "7 series", "correction": 0},
  {"brand": "bmw", "model_lower": "8 serie", "model_correct": "8 series", "correction": 1},
  {"brand": "bmw", "model_lower": "8 series", "model_correct": "8 series", "correction": 0},
  {"brand": "bmw", "model_lower": "i3", "model_correct": "i3", "correction": 0},
  {"brand": "bmw", "model_lower": "i8", "model_correct": "i8", "correction": 0},
  {"brand": "bmw", "model_lower": "m2", "model_correct": "m2", "correction": 0},
  {"brand": "bmw", "model_lower": "m3", "model_correct": "m3", "correction": 0},
  {"brand": "bmw", "model_lower": "m4", "model_correct": "m4", "correction": 0},
  {"brand": "bmw", "model_lower": "m5", "model_correct": "m5", "correction": 0},
  {"brand": "bmw", "model_lower": "m6", "model_correct": "m6", "correction": 0},
  {"brand": "bmw", "model_lower": "x1", "model_correct": "x1", "correction": 0},
  {"brand": "bmw", "model_lower": "x2", "model_correct": "x2", "correction": 0},
  {"brand": "bmw", "model_lower": "x3", "model_correct": "x3", "correction": 0},
  {"brand": "bmw", "model_lower": "x4", "model_correct": "x4", "correction": 0},
  {"brand": "bmw", "model_lower": "x5", "model_correct": "x5", "correction": 0},
  {"brand": "bmw", "model_lower": "x6", "model_correct": "x6", "correction": 0},
  {"brand": "bmw", "model_lower": "x7", "model_correct": "x7", "correction": 0},
  {"brand": "bmw", "model_lower": "z3", "model_correct": "z3", "correction": 0},
  {"brand": "bmw", "model_lower": "z4", "model_correct": "z4", "correction": 0},
  {"brand": "ford", "model_lower": "b-ma", "model_correct": "b-max", "correction": 1},
  {"brand": "ford", "model_lower": "b-max", "model_correct": "b-max", "correction": 0},
  {"brand": "ford", "model_lower": "c-ma", "model_correct": "c-max", "correction": 1},
  {"brand": "ford", "model_lower": "c-max", "model_correct": "c-max", "correction": 0},
  {"brand": "ford", "model_lower": "ecospor", "model_correct": "ecosport", "correction": 1},
  {"brand": "ford", "model_lower": "ecosport", "model_correct": "ecosport", "correction": 0},
  {"brand": "ford", "model_lower": "edg", "model_correct": "edge", "correction": 1},
  {"brand": "ford", "model_lower": "edge", "model_correct": "edge", "correction": 0},
  {"brand": "ford", "model_lower": "escort", "model_correct": "escort", "correction": 0},
  {"brand": "ford", "model_lower": "fiest", "model_correct": "fiesta", "correction": 1},
  {"brand": "ford", "model_lower": "fiesta", "model_correct": "fiesta", "correction": 0},
  {"brand": "ford", "model_lower": "focu", "model_correct": "focus", "correction": 1},
  {"brand": "ford", "model_lower": "focus", "model_correct": "focus", "correction": 0},
  {"brand": "ford", "model_lower": "fusion", "model_correct": "fusion", "correction": 0},
  {"brand": "ford", "model_lower": "galax", "model_correct": "galaxy", "correction": 1},
  {"brand": "ford", "model_lower": "galaxy", "model_correct": "galaxy", "correction": 0},
  {"brand": "ford", "model_lower": "grand c-ma", "model_correct": "grand c-max", "correction": 1},
  {"brand": "ford", "model_lower": "grand c-max", "model_correct": "grand c-max", "correction": 0},
  {"brand": "ford", "model_lower": "grand tourneo connec", "model_correct": "grand tourneo connect", "correction": 1},
  {"brand": "ford", "model_lower": "grand tourneo connect", "model_correct": "grand tourneo connect", "correction": 0},
  {"brand": "ford", "model_lower": "k", "model_correct": "ka", "correction": 1},
  {"brand": "ford", "model_lower": "ka", "model_correct": "ka", "correction": 0},
  {"brand": "ford", "model_lower": "ka+", "model_correct": "ka+", "correction": 0},
  {"brand": "ford", "model_lower": "kug", "model_correct": "kuga", "correction": 1},
  {"brand": "ford", "model_lower": "kuga", "model_correct": "kuga", "correction": 0},
  {"brand": "ford", "model_lower": "monde", "model_correct": "mondeo", "correction": 1},
  {"brand": "ford", "model_lower": "mondeo", "model_correct": "mondeo", "correction": 0},
  {"brand": "ford", "model_lower": "mustang", "model_correct": "mustang", "correction": 0},
  {"brand": "ford", "model_lower": "puma", "model_correct": "puma", "correction": 0},
  {"brand": "ford", "model_lower": "ranger", "model_correct": "ranger", "correction": 0},
  {"brand": "ford", "model_lower": "s-ma", "model_correct": "s-max", "correction": 1},
  {"brand": "ford", "model_lower": "s-max", "model_correct": "s-max", "correction": 0},
  {"brand": "ford", "model_lower": "streetka", "model_correct": "streetka", "correction": 0},
  {"brand": "ford", "model_lower": "tourneo connect", "model_correct": "tourneo connect", "correction": 0},
  {"brand": "ford", "model_lower": "tourneo custo", "model_correct": "tourneo custom", "correction": 1},
  {"brand": "ford", "model_lower": "tourneo custom", "model_correct": "tourneo custom", "correction": 0},
  {"brand": "hyundai", "model_lower": "accent", "model_correct": "accent", "correction": 0},
  {"brand": "hyundai", "model_lower": "getz", "model_correct": "getz", "correction": 0},
  {"brand": "hyundai", "model_lower": "i1", "model_correct": "i10", "correction": 1},
  {"brand": "hyundai", "model_lower": "i10", "model_correct": "i10", "correction": 0},
  {"brand": "hyundai", "model_lower": "i2", "model_correct": "i20", "correction": 1},
  {"brand": "hyundai", "model_lower": "i20", "model_correct": "i20", "correction": 0},
  {"brand": "hyundai", "model_lower": "i3", "model_correct": "i30", "correction": 1},
  {"brand": "hyundai", "model_lower": "i30", "model_correct": "i30", "correction": 0},
  {"brand": "hyundai", "model_lower": "i40", "model_correct": "i40", "correction": 0},
  {"brand": "hyundai", "model_lower": "i80", "model_correct": "i800", "correction": 1},
  {"brand": "hyundai", "model_lower": "i800", "model_correct": "i800", "correction": 0},
  {"brand": "hyundai", "model_lower": "ioni", "model_correct": "ioniq", "correction": 1},
  {"brand": "hyundai", "model_lower": "ioniq", "model_correct": "ioniq", "correction": 0},
  {"brand": "hyundai", "model_lower": "ix2", "model_correct": "ix20", "correction": 1},
  {"brand": "hyundai", "model_lower": "ix20", "model_correct": "ix20", "correction": 0},
  {"brand": "hyundai", "model_lower": "ix35", "model_correct": "ix35", "correction": 0},
  {"brand": "hyundai", "model_lower": "kon", "model_correct": "kona", "correction": 1},
  {"brand": "hyundai", "model_lower": "kona", "model_correct": "kona", "correction": 0},
  {"brand": "hyundai", "model_lower": "santa f", "model_correct": "santa fe", "correction": 1},
  {"brand": "hyundai", "model_lower": "santa fe", "model_correct": "santa fe", "correction": 0},
  {"brand": "hyundai", "model_lower": "terracan", "model_correct": "terracan", "correction": 0},
  {"brand": "hyundai", "model_lower": "tucso", "model_correct": "tucson", "correction": 1},
  {"brand": "hyundai", "model_lower": "tucson", "model_correct": "tucson", "correction": 0},
  {"brand": "hyundai", "model_lower": "veloste", "model_correct": "veloster", "correction": 1},
  {"brand": "hyundai", "model_lower": "veloster", "model_correct": "veloster", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "200", "model_correct": "200", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "220", "model_correct": "220", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "230", "model_correct": "230", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "a clas", "model_correct": "a class", "correction": 1},
  {"brand": "mercedes-benz", "model_lower": "a class", "model_correct": "a class", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "b clas", "model_correct": "b class", "correction": 1},
  {"brand": "mercedes-benz", "model_lower": "b class", "model_correct": "b class", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "c clas", "model_correct": "c class", "correction": 1},
  {"brand": "mercedes-benz", "model_lower": "c class", "model_correct": "c class", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "cl clas", "model_correct": "cl class", "correction": 1},
  {"brand": "mercedes-benz", "model_lower": "cl class", "model_correct": "cl class", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "cla class", "model_correct": "cla class", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "clc class", "model_correct": "clc class", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "clk", "model_correct": "clk", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "cls clas", "model_correct": "cls class", "correction": 1},
  {"brand": "mercedes-benz", "model_lower": "cls class", "model_correct": "cls class", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "e clas", "model_correct": "e class", "correction": 1},
  {"brand": "mercedes-benz", "model_lower": "e class", "model_correct": "e class", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "g class", "model_correct": "g class", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "gl class", "model_correct": "gl class", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "gla clas", "model_correct": "gla class", "correction": 1},
  {"brand": "mercedes-benz", "model_lower": "gla class", "model_correct": "gla class", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "glb class", "model_correct": "glb class", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "glc clas", "model_correct": "glc class", "correction": 1},
  {"brand": "mercedes-benz", "model_lower": "glc class", "model_correct": "glc class", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "gle clas", "model_correct": "gle class", "correction": 1},
  {"brand": "mercedes-benz", "model_lower": "gle class", "model_correct": "gle class", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "gls clas", "model_correct": "gls class", "correction": 1},
  {"brand": "mercedes-benz", "model_lower": "gls class", "model_correct": "gls class", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "m clas", "model_correct": "m class", "correction": 1},
  {"brand": "mercedes-benz", "model_lower": "m class", "model_correct": "m class", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "s clas", "model_correct": "s class", "correction": 1},
  {"brand": "mercedes-benz", "model_lower": "s class", "model_correct": "s class", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "sl", "model_correct": "sl", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "sl clas", "model_correct": "sl class", "correction": 1},
  {"brand": "mercedes-benz", "model_lower": "sl class", "model_correct": "sl class", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "slk", "model_correct": "slk", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "v clas", "model_correct": "v class", "correction": 1},
  {"brand": "mercedes-benz", "model_lower": "v class", "model_correct": "v class", "correction": 0},
  {"brand": "mercedes-benz", "model_lower": "x-clas", "model_correct": "x-class", "correction": 1},
  {"brand": "mercedes-benz", "model_lower": "x-class", "model_correct": "x-class", "correction": 0},
  {"brand": "opel", "model_lower": "ada", "model_correct": "adam", "correction": 1},
  {"brand": "opel", "model_lower": "adam", "model_correct": "adam", "correction": 0},
  {"brand": "opel", "model_lower": "agila", "model_correct": "agila", "correction": 0},
  {"brand": "opel", "model_lower": "ampera", "model_correct": "ampera", "correction": 0},
  {"brand": "opel", "model_lower": "antara", "model_correct": "antara", "correction": 0},
  {"brand": "opel", "model_lower": "astr", "model_correct": "astra", "correction": 1},
  {"brand": "opel", "model_lower": "astra", "model_correct": "astra", "correction": 0},
  {"brand": "opel", "model_lower": "cascada", "model_correct": "cascada", "correction": 0},
  {"brand": "opel", "model_lower": "combo lif", "model_correct": "combo life", "correction": 1},
  {"brand": "opel", "model_lower": "combo life", "model_correct": "combo life", "correction": 0},
  {"brand": "opel", "model_lower": "cors", "model_correct": "corsa", "correction": 1},
  {"brand": "opel", "model_lower": "corsa", "model_correct": "corsa", "correction": 0},
  {"brand": "opel", "model_lower": "crossland", "model_correct": "crossland", "correction": 0},
  {"brand": "opel", "model_lower": "crossland x", "model_correct": "crossland x", "correction": 0},
  {"brand": "opel", "model_lower": "grandland", "model_correct": "grandland", "correction": 0},
  {"brand": "opel", "model_lower": "grandland x", "model_correct": "grandland x", "correction": 0},
  {"brand": "opel", "model_lower": "gtc", "model_correct": "gtc", "correction": 0},
  {"brand": "opel", "model_lower": "insigni", "model_correct": "insignia", "correction": 1},
  {"brand": "opel", "model_lower": "insignia", "model_correct": "insignia", "correction": 0},
  {"brand": "opel", "model_lower": "meriv", "model_correct": "meriva", "correction": 1},
  {"brand": "opel", "model_lower": "meriva", "model_correct": "meriva", "correction": 0},
  {"brand": "opel", "model_lower": "mokk", "model_correct": "mokka", "correction": 1},
  {"brand": "opel", "model_lower": "mokka", "model_correct": "mokka", "correction": 0},
  {"brand": "opel", "model_lower": "mokka x", "model_correct": "mokka x", "correction": 0},
  {"brand": "opel", "model_lower": "tigra", "model_correct": "tigra", "correction": 0},
  {"brand": "opel", "model_lower": "vectra", "model_correct": "vectra", "correction": 0},
  {"brand": "opel", "model_lower": "viva", "model_correct": "viva", "correction": 0},
  {"brand": "opel", "model_lower": "vivaro", "model_correct": "vivaro", "correction": 0},
  {"brand": "opel", "model_lower": "zafir", "model_correct": "zafira", "correction": 1},
  {"brand": "opel", "model_lower": "zafira", "model_correct": "zafira", "correction": 0},
  {"brand": "opel", "model_lower": "zafira toure", "model_correct": "zafira tourer", "correction": 1},
  {"brand": "opel", "model_lower": "zafira tourer", "model_correct": "zafira tourer", "correction": 0},
  {"brand": "skoda", "model_lower": "citig", "model_correct": "citigo", "correction": 1},
  {"brand": "skoda", "model_lower": "citigo", "model_correct": "citigo", "correction": 0},
  {"brand": "skoda", "model_lower": "fabi", "model_correct": "fabia", "correction": 1},
  {"brand": "skoda", "model_lower": "fabia", "model_correct": "fabia", "correction": 0},
  {"brand": "skoda", "model_lower": "kami", "model_correct": "kamiq", "correction": 1},
  {"brand": "skoda", "model_lower": "kamiq", "model_correct": "kamiq", "correction": 0},
  {"brand": "skoda", "model_lower": "karo", "model_correct": "karoq", "correction": 1},
  {"brand": "skoda", "model_lower": "karoq", "model_correct": "karoq", "correction": 0},
  {"brand": "skoda", "model_lower": "kodia", "model_correct": "kodiaq", "correction": 1},
  {"brand": "skoda", "model_lower": "kodiaq", "model_correct": "kodiaq", "correction": 0},
  {"brand": "skoda", "model_lower": "octavi", "model_correct": "octavia", "correction": 1},
  {"brand": "skoda", "model_lower": "octavia", "model_correct": "octavia", "correction": 0},
  {"brand": "skoda", "model_lower": "rapi", "model_correct": "rapid", "correction": 1},
  {"brand": "skoda", "model_lower": "rapid", "model_correct": "rapid", "correction": 0},
  {"brand": "skoda", "model_lower": "roomste", "model_correct": "roomster", "correction": 1},
  {"brand": "skoda", "model_lower": "roomster", "model_correct": "roomster", "correction": 0},
  {"brand": "skoda", "model_lower": "scal", "model_correct": "scala", "correction": 1},
  {"brand": "skoda", "model_lower": "scala", "model_correct": "scala", "correction": 0},
  {"brand": "skoda", "model_lower": "super", "model_correct": "superb", "correction": 1},
  {"brand": "skoda", "model_lower": "superb", "model_correct": "superb", "correction": 0},
  {"brand": "skoda", "model_lower": "yet", "model_correct": "yeti", "correction": 1},
  {"brand": "skoda", "model_lower": "yeti", "model_correct": "yeti", "correction": 0},
  {"brand": "skoda", "model_lower": "yeti outdoo", "model_correct": "yeti outdoor", "correction": 1},
  {"brand": "skoda", "model_lower": "yeti outdoor", "model_correct": "yeti outdoor", "correction": 0},
  {"brand": "toyota", "model_lower": "auri", "model_correct": "auris", "correction": 1},
  {"brand": "toyota", "model_lower": "auris", "model_correct": "auris", "correction": 0},
  {"brand": "toyota", "model_lower": "avensis", "model_correct": "avensis", "correction": 0},
  {"brand": "toyota", "model_lower": "ayg", "model_correct": "aygo", "correction": 1},
  {"brand": "toyota", "model_lower": "aygo", "model_correct": "aygo", "correction": 0},
  {"brand": "toyota", "model_lower": "c-h", "model_correct": "c-hr", "correction": 1},
  {"brand": "toyota", "model_lower": "c-hr", "model_correct": "c-hr", "correction": 0},
  {"brand": "toyota", "model_lower": "camry", "model_correct": "camry", "correction": 0},
  {"brand": "toyota", "model_lower": "coroll", "model_correct": "corolla", "correction": 1},
  {"brand": "toyota", "model_lower": "corolla", "model_correct": "corolla", "correction": 0},
  {"brand": "toyota", "model_lower": "gt86", "model_correct": "gt86", "correction": 0},
  {"brand": "toyota", "model_lower": "hilu", "model_correct": "hilux", "correction": 1},
  {"brand": "toyota", "model_lower": "hilux", "model_correct": "hilux", "correction": 0},
  {"brand": "toyota", "model_lower": "iq", "model_correct": "iq", "correction": 0},
  {"brand": "toyota", "model_lower": "land cruise", "model_correct": "land cruiser", "correction": 1},
  {"brand": "toyota", "model_lower": "land cruiser", "model_correct": "land cruiser", "correction": 0},
  {"brand": "toyota", "model_lower": "prius", "model_correct": "prius", "correction": 0},
  {"brand": "toyota", "model_lower": "proace verso", "model_correct": "proace verso", "correction": 0},
  {"brand": "toyota", "model_lower": "rav", "model_correct": "rav4", "correction": 1},
  {"brand": "toyota", "model_lower": "rav4", "model_correct": "rav4", "correction": 0},
  {"brand": "toyota", "model_lower": "suvra", "model_correct": "supra", "correction": 1},
  {"brand": "toyota", "model_lower": "supra", "model_correct": "supra", "correction": 0},
  {"brand": "toyota", "model_lower": "urban cruise", "model_correct": "urban cruiser", "correction": 1},
  {"brand": "toyota", "model_lower": "urban cruiser", "model_correct": "urban cruiser", "correction": 0},
  {"brand": "toyota", "model_lower": "vers", "model_correct": "verso", "correction": 1},
  {"brand": "toyota", "model_lower": "verso", "model_correct": "verso", "correction": 0},
  {"brand": "toyota", "model_lower": "verso-s", "model_correct": "verso-s", "correction": 0},
  {"brand": "toyota", "model_lower": "yari", "model_correct": "yaris", "correction": 1},
  {"brand": "toyota", "model_lower": "yaris", "model_correct": "yaris", "correction": 0},
  {"brand": "volkswagen", "model_lower": "amaro", "model_correct": "amarok", "correction": 1},
  {"brand": "volkswagen", "model_lower": "amarok", "model_correct": "amarok", "correction": 0},
  {"brand": "volkswagen", "model_lower": "arteo", "model_correct": "arteon", "correction": 1},
  {"brand": "volkswagen", "model_lower": "arteon", "model_correct": "arteon", "correction": 0},
  {"brand": "volkswagen", "model_lower": "beetl", "model_correct": "beetle", "correction": 1},
  {"brand": "volkswagen", "model_lower": "beetle", "model_correct": "beetle", "correction": 0},
  {"brand": "volkswagen", "model_lower": "caddy", "model_correct": "caddy", "correction": 0},
  {"brand": "volkswagen", "model_lower": "caddy life", "model_correct": "caddy life", "correction": 0},
  {"brand": "volkswagen", "model_lower": "caddy maxi", "model_correct": "caddy maxi", "correction": 0},
  {"brand": "volkswagen", "model_lower": "caddy maxi lif", "model_correct": "caddy maxi life", "correction": 1},
  {"brand": "volkswagen", "model_lower": "caddy maxi life", "model_correct": "caddy maxi life", "correction": 0},
  {"brand": "volkswagen", "model_lower": "california", "model_correct": "california", "correction": 0},
  {"brand": "volkswagen", "model_lower": "caravell", "model_correct": "caravelle", "correction": 1},
  {"brand": "volkswagen", "model_lower": "caravelle", "model_correct": "caravelle", "correction": 0},
  {"brand": "volkswagen", "model_lower": "cc", "model_correct": "cc", "correction": 0},
  {"brand": "volkswagen", "model_lower": "eos", "model_correct": "eos", "correction": 0},
  {"brand": "volkswagen", "model_lower": "fox", "model_correct": "fox", "correction": 0},
  {"brand": "volkswagen", "model_lower": "gol", "model_correct": "gol", "correction": 0},
  {"brand": "volkswagen", "model_lower": "golf", "model_correct": "golf", "correction": 0},
  {"brand": "volkswagen", "model_lower": "golf s", "model_correct": "golf sv", "correction": 1},
  {"brand": "volkswagen", "model_lower": "golf sv", "model_correct": "golf sv", "correction": 0},
  {"brand": "volkswagen", "model_lower": "jetta", "model_correct": "jetta", "correction": 0},
  {"brand": "volkswagen", "model_lower": "passa", "model_correct": "passat", "correction": 1},
  {"brand": "volkswagen", "model_lower": "passat", "model_correct": "passat", "correction": 0},
  {"brand": "volkswagen", "model_lower": "pol", "model_correct": "polo", "correction": 1},
  {"brand": "volkswagen", "model_lower": "polo", "model_correct": "polo", "correction": 0},
  {"brand": "volkswagen", "model_lower": "scirocc", "model_correct": "scirocco", "correction": 1},
  {"brand": "volkswagen", "model_lower": "scirocco", "model_correct": "scirocco", "correction": 0},
  {"brand": "volkswagen", "model_lower": "shara", "model_correct": "sharan", "correction": 1},
  {"brand": "volkswagen", "model_lower": "sharan", "model_correct": "sharan", "correction": 0},
  {"brand": "volkswagen", "model_lower": "shuttle", "model_correct": "shuttle", "correction": 0},
  {"brand": "volkswagen", "model_lower": "t-cros", "model_correct": "t-cross", "correction": 1},
  {"brand": "volkswagen", "model_lower": "t-cross", "model_correct": "t-cross", "correction": 0},
  {"brand": "volkswagen", "model_lower": "t-ro", "model_correct": "t-roc", "correction": 1},
  {"brand": "volkswagen", "model_lower": "t-roc", "model_correct": "t-roc", "correction": 0},
  {"brand": "volkswagen", "model_lower": "tigua", "model_correct": "tiguan", "correction": 1},
  {"brand": "volkswagen", "model_lower": "tiguan", "model_correct": "tiguan", "correction": 0},
  {"brand": "volkswagen", "model_lower": "tiguan allspac", "model_correct": "tiguan allspace", "correction": 1},
  {"brand": "volkswagen", "model_lower": "tiguan allspace", "model_correct": "tiguan allspace", "correction": 0},
  {"brand": "volkswagen", "model_lower": "touare", "model_correct": "touareg", "correction": 1},
  {"brand": "volkswagen", "model_lower": "touareg", "model_correct": "touareg", "correction": 0},
  {"brand": "volkswagen", "model_lower": "toura", "model_correct": "touran", "correction": 1},
  {"brand": "volkswagen", "model_lower": "touran", "model_correct": "touran", "correction": 0},
  {"brand": "volkswagen", "model_lower": "u", "model_correct": "up", "correction": 1},
  {"brand": "volkswagen", "model_lower": "up", "model_correct": "up", "correction": 0}
]


# In[95]:


test_vocab_df = pd.DataFrame(BRAND_MODEL_VOCAB)
test_vocab_df


# In[96]:


# keep=False: mark every occurrence of a duplicated row
test_vocab_df[test_vocab_df[['brand','model_lower','model_correct']].duplicated(keep=False)]


# ### Conclusion
# We built a complete, cleaned vocabulary of brands and model mappings, including a flag for corrected entries. There are no duplicate rows in this dataset.

# In[ ]:




