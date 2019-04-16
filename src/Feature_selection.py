import os
import pandas as pd

train_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/train.csv')
test_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/train.csv')

df_train_file = pd.read_csv(train_file)
df_pristine_file = pd.read_csv(test_file)

print("get Training and test file")
from sklearn.model_selection import train_test_split
df_input = df_train_file.drop(['ID_code','target'],axis=1)
df_target = df_train_file['target']
x_train,x_test,y_train,y_test = train_test_split(df_input,df_target,test_size=0.2,shuffle=True,stratify=df_target)

num_features = x_train.shape[1]

print("Performing mrmr feature selection")
import pymrmr
mrmr_output = pymrmr.mRMR(x_train,'MIQ',20)
print(mrmr_output)

