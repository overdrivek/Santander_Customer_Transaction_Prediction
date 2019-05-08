import pandas as pd
import os

train_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/train.csv')

df_train_file = pd.read_csv(train_file)

df_dropped_file = df_train_file.drop(['ID_code','target'],axis=1)

column_names = list(df_dropped_file.columns.values)
#df_grouped = df_dropped_file.groupby(column_names).size().reset_index(name='Count')

# tests
std= df_dropped_file['var_0'].std()
mean = df_dropped_file['var_0'].mean()
import  numpy as np
np_var = np.asarray(df_dropped_file['var_0'])
#print(len(np.where(np.abs(np_var)< mean-1*std )[0]))
from sklearn.mixture import GaussianMixture as GMM

gaussian_mm = GMM(n_components=20)
gaussian_mm.fit(np_var.reshape(-1,1),np.asarray(df_train_file['target']).reshape(-1,1))
gaussian_mm.predict(np_var)