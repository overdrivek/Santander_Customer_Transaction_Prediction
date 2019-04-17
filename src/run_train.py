import os
import pandas as pd

train_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/train.csv')
test_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/test.csv')

df_train_file = pd.read_csv(train_file)
df_pristine_file = pd.read_csv(test_file)

print("get Training and test file")
from sklearn.model_selection import train_test_split
df_input = df_train_file.drop(['ID_code','target'],axis=1)
df_target = df_train_file['target']
x_train,x_test,y_train,y_test = train_test_split(df_input,df_target,test_size=0.2,shuffle=True,stratify=df_target)

num_features = x_train.shape[1]

selected_features = ['var_45', 'var_131', 'var_23', 'var_162', 'var_47', 'var_62', 'var_39', 'var_26', 'var_176', 'var_102', 'var_69', 'var_147', 'var_14', 'var_29', 'var_160', 'var_181', 'var_186', 'var_139', 'var_3', 'var_122']

x_train = x_train[selected_features]
x_test = x_test[selected_features]

from sklearn.ensemble import RandomForestClassifier as RF_Classifier
clf_rf = RF_Classifier(n_estimators=100,verbose=2)

clf_rf.fit(x_train,y_train)

train_score = clf_rf.score(x_train,y_train)
print("Training results: ",train_score)

test_score = clf_rf.score(x_test,y_test)
print("Test results: ",test_score)

import pickle
model_path = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Model/')
if os.path.exists(model_path) is False:
    os.mkdir(model_path)
pickle.dump(clf_rf,open(os.path.join(model_path,'random_forest_mrmr.sav'),'wb'))

pristine_query= df_pristine_file[selected_features].drop(['ID_code'],axis=1)
rf_predicted = clf_rf.predict(pristine_query)
df_output = pd.concat([df_pristine_file['ID_code'],pd.DataFrame(rf_predicted,columns=['target'])],axis=1)
df_output.to_csv('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/predicted_rf.csv',index=False)
