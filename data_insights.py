#!/usr/bin/env python
# coding: utf-8


# In[2]:


filepath = "./data/"


# In[3]:


# Scikit-Learn ≥0.20 is required
import sklearn

# Common imports
import numpy as np
import pandas as pd
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
FOLDER_ID = "CKD"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", FOLDER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# ## Import Data

# In[4]:


T_demo = pd.read_csv(filepath + "T_demo.csv", sep = ",")
T_stage = pd.read_csv(filepath + "T_stage.csv", sep = ",")
T_meds = pd.read_csv(filepath + "T_meds.csv", sep = ",")

# assign dataset names
lab_data_names = ['T_creatinine','T_DBP', 'T_SBP', 'T_HGB', 'T_glucose', 'T_ldl']
 
# create empty list
dataframes_list = []
 
# append datasets into teh list
for i in range(len(lab_data_names)):
    temp_df = pd.read_csv(filepath + lab_data_names[i] + ".csv")
    dataframes_list.append(temp_df)


# In[5]:


T_demo_stage = pd.merge(T_demo, T_stage, on = ['id'])
T_demo_stage.head()


# ## Take a Quick Look at the Data Structure

# In[6]:


T_demo_stage.head()


# In[7]:


T_demo_stage.info()


# In[8]:


cat_feat = ['race', 'gender', 'Stage_Progress']
for i in cat_feat:
    print(T_demo_stage[i].value_counts(normalize=True)*100)


# ## Data Descriptive Statistics

# #### Patient Demographics Data

# In[13]:


labels = ['True', 'False']
False_ = T_demo_stage['Stage_Progress'].value_counts()[0]
True_ = T_demo_stage['Stage_Progress'].value_counts()[1]
sizes = [True_, False_]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels = labels, autopct = '%1.1f%%', shadow = True, startangle = 90)
ax1.axis('equal')
save_fig("class_label_distributioh")
plt.show()


# In[21]:


labels = ['White', 'Unknown', 'Black', 'Asian', 'Hispanic']
sizes = [T_demo_stage['race'].value_counts()[x] for x in range(len(labels))]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels = labels, autopct = '%1.1f%%', shadow = True, startangle = 90)
ax1.axis('equal')
save_fig("race_distributioh")
plt.show()
# labels, sizes


# In[22]:


labels = ['Female', 'Male']
female = T_demo_stage['gender'].value_counts()[0]
male = T_demo_stage['gender'].value_counts()[1]
sizes = [female, male]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels = labels, autopct = '%1.1f%%', shadow = True, startangle = 90)
ax1.axis('equal')

save_fig("gender_distributioh")
plt.show()


# In[23]:


T_demo_feat = T_demo_stage['age']

T_demo_feat.hist(bins = 20, figsize = (4,3))
save_fig("Age Distribution")
plt.show()


# #### Lab Longitudinal Data

# In[24]:


for i in range(len(dataframes_list)):
    print(str(lab_data_names[i]))
    print(dataframes_list[i][['time']].describe())
    print()
    print(dataframes_list[i].info())
    dataframes_list[i][['value', 'time']].hist(bins = 25, figsize=(8,3))
    save_fig("{}".format(lab_data_names[i]))
    plt.show()


# #### Medication Log Data

# In[32]:


T_meds.head()
T_meds['drug'].value_counts().plot(kind = "bar")
save_fig("meds1")
plt.show()

T_meds[['drug','daily_dosage']].hist(bins = 25, figsize = (8,5))
save_fig("meds2")
plt.show()

T_meds.describe()

