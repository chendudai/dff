import os 

import numpy as np
import shutil
import pandas as pd

df = pd.read_csv('/storage/morrisalper/notebooks/babel/pseudolabelling/output/pseudolabelled-unk.csv')
df = df[df.spl=='test']
df = df[df.building_type == 'mosque']
df_Badshahi = df[df.name == 'Badshahi Mosque']
print(df_Badshahi.name.value_counts())
df_Sultan = df[df.name == 'Sultan Ahmed I Mosque']
print(df_Sultan.name.value_counts())

def create_new_img_dir(df,dst_dir):
    for i,ind in enumerate(df. index):
        print(i)
        shutil.copy(df["fn"][ind], dst_dir)
print("Badshahi_Mosque")
img_dir_Badashahi = '/storage/hanibezalel/colmap_mosque/Badshahi_Mosque_new'
print("Sultan_Ahmed_Mosque")
img_dir_Sultan = '/storage/hanibezalel/colmap_mosque/Sultan_Ahmed_Mosque_new'

create_new_img_dir(df_Badshahi,img_dir_Badashahi)
create_new_img_dir(df_Sultan,img_dir_Sultan)


