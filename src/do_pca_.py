import pandas as pd
from sklearn.decomposition import PCA
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

train_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/train.csv')

df_train_file = pd.read_csv(train_file)
df_train_dropped = df_train_file.drop(['ID_code','target'],axis=1)

scaler = StandardScaler()
scaler.fit(df_train_dropped)
df_train_scaled = scaler.transform(df_train_dropped)


num_comps = 50
print('running pca ...')
p_comp = PCA(n_components=num_comps)
p_comp.fit(df_train_scaled)
print('done..')

print('transforming train file...')
df_transformed = p_comp.transform(df_train_scaled)

df_full = pd.concat([df_train_file['ID_code'],df_train_file['target'],pd.DataFrame(df_transformed,columns=list(df_train_dropped.columns.values)[0:num_comps])],axis=1)

output_folder = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/pca')
if os.path.exists(output_folder) is False:
    os.mkdir(output_folder)

df_full.to_csv(os.path.join(output_folder,'train_pca.csv'),index=None)
print('export and saved train file')

print('transforming test file...')
test_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/test.csv')
df_test_file = pd.read_csv(test_file)
df_test_dropped = df_test_file.drop(['ID_code'],axis=1)
df_test_scaled = scaler.transform(df_test_dropped)
df_test_transformed = p_comp.transform(df_test_scaled)
df_test_full = pd.concat([df_test_file['ID_code'],pd.DataFrame(df_test_transformed,columns=list(df_test_dropped.columns.values)[0:num_comps])],axis=1)
df_test_full.to_csv(os.path.join(output_folder,'test_pca.csv'),index=None)
print('export and saved test file')

print('pickling pca model')
from joblib import dump
dump(p_comp,os.path.join(output_folder,'pca_model.joblib'))
print('save')

print('pickling scaler')
dump(scaler,os.path.join(output_folder,'scaler.joblib'))
print('done')