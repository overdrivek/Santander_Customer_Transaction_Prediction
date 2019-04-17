import os
import pickle
import pandas as pd

test_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/test.csv')

model_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Model/random_forest_mrmr.sav')

rf_clf = pickle.load(open(model_file,'rb'))

df_pristine = pd.read_csv(test_file)
selected_features = ['var_45', 'var_131', 'var_23', 'var_162', 'var_47', 'var_62', 'var_39', 'var_26', 'var_176', 'var_102', 'var_69', 'var_147', 'var_14', 'var_29', 'var_160', 'var_181', 'var_186', 'var_139', 'var_3', 'var_122']
df_pristine_dropped= df_pristine.drop('ID_code',axis=1)
df_pristine_dropped = df_pristine_dropped[selected_features]
rf_predicted = rf_clf.predict(df_pristine_dropped)
df_output = pd.concat([df_pristine['ID_code'],pd.DataFrame(rf_predicted,columns=['target'])],axis=1)
df_output.to_csv('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/predicted_rf.csv',index=False)