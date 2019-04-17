import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF_Classifier
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

train_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/train.csv')
test_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/test.csv')

df_train_file = pd.read_csv(train_file)
df_pristine_file = pd.read_csv(test_file)

print("get Training and test file")
from sklearn.model_selection import train_test_split
df_input = df_train_file.drop(['ID_code','target'],axis=1)
df_target = df_train_file['target']
# scaler = MinMaxScaler()
# scaler.fit(df_input)
# train_scaled = scaler.transform(df_input)
train_scaled = (df_input-df_input.min())/(df_input.max()-df_input.min())
x_train,x_test,y_train,y_test = train_test_split(train_scaled,df_target,test_size=0.2,shuffle=True,stratify=df_target)



num_features = x_train.shape[1]


print("Performing mrmr feature selection")

import pymrmr
log_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/log.txt')
train_accuracy = []
test_accuracy = []
num_features = 10
model_folder = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Model/mrmr_iteration/')
result_folder = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Model/mrmr_iteration/Results/')
if os.path.exists(model_folder) is False:
    os.mkdir(model_folder)

if os.path.exists(result_folder) is False:
    os.mkdir(result_folder)

from matplotlib import pyplot as plt
plt.figure()
df_pristine_dropped = df_pristine_file.drop('ID_code', axis=1)
# test_scaled = scaler.transform(df_pristine_dropped)
test_scaled = (df_pristine_dropped-df_pristine_dropped.min())/(df_pristine_dropped.max()-df_pristine_dropped.min())
feature_list = []
for i in range(num_features):
    with open(log_file,'a') as logger:
        logger.write("MRMR selection top"+str(i+1)+" features \n")


        print('MRMR selection top {} features'.format(i+1))
        mrmr_output = pymrmr.mRMR(x_train,'MIQ',i+1)
        for mrmr_feature in mrmr_output:
            logger.write(mrmr_feature+'\t')
        logger.write('\n')
        print(mrmr_output)

        print('************************************************')

        # Run RF
        clf_rf = RF_Classifier(n_estimators=100, verbose=2)
        clf_rf.fit(x_train[mrmr_output], y_train)
        train_score = clf_rf.score(x_train[mrmr_output], y_train)
        train_accuracy.append(train_score)

        test_score = clf_rf.score(x_test[mrmr_output],y_test)
        test_accuracy.append(test_score)
        feature_list.append(i)
        plt.clf()
        plt.ion()
        plt.show()
        plt.plot(np.asarray(feature_list), np.asarray(train_accuracy), 'r-', label='train accuracy')
        plt.plot(np.asarray(feature_list), np.asarray(test_accuracy), 'b-', label='test_accuracy')
        plt.grid(True)
        plt.xlabel('number of features')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.draw()
        plt.pause(0.0001)



        # pristine output
        df_pristine_query = test_scaled[mrmr_output]
        rf_predicted = clf_rf.predict(df_pristine_query)
        df_output = pd.concat([df_pristine_file['ID_code'], pd.DataFrame(rf_predicted, columns=['target'])], axis=1)
        df_output.to_csv(os.path.join(result_folder,'rf_'+str(i+1)+'_features.csv'),index=False)

        pickle.dump(clf_rf,open(os.path.join(model_folder,'rf_num_features_'+str(i+1)+'.sav'),'wb'))

    logger.close()
plt.show()






