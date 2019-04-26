import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF_Classifier
from sklearn import svm
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.metrics import accuracy_score as acc
import mldashboard
import tqdm

log_dir = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/log/')
if os.path.exists(log_dir) is False:
    os.mkdir(log_dir)
run_name = 'Santandar_RF_Forward_Chaining'
mllogger = mldashboard.Logger()
data_writer = mldashboard.JsonRecordFileWriter(log_dir,run_name,override=True)
mllogger.add_writer(data_writer)

train_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/train.csv')
test_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/test.csv')

df_train_file = pd.read_csv(train_file)
df_pristine_file = pd.read_csv(test_file)

print("get Training and test file")
from sklearn.model_selection import train_test_split
df_input = df_train_file.drop(['ID_code','target'],axis=1)
df_target = df_train_file['target']
#scaler = MinMaxScaler()
#train_scaled = pd.DataFrame(scaler.fit_transform(df_input),columns=df_input.columns)
train_scaled = df_input
# train_scaled = scaler.transform(df_input)
#train_scaled = (df_input-df_input.min())/(df_input.max()-df_input.min())
x_train,x_test,y_train,y_test = train_test_split(train_scaled,df_target,test_size=0.2,shuffle=True,stratify=df_target)



num_features = x_train.shape[1]

log_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/log.txt')

train_accuracy = []
test_accuracy = []
feature_list = []

plt.figure()
df_pristine_dropped = df_pristine_file.drop('ID_code', axis=1)
#test_scaled = pd.DataFrame(scaler.transform(df_pristine_dropped),columns=df_pristine_dropped.columns)
test_scaled = df_pristine_dropped
#test_scaled = (df_pristine_dropped-df_input.min())/(df_input.max()-df_input.min())
model_folder = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Model/forward_selected_rf')
result_folder = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Model/forward_selected_rf/Results/')
if os.path.exists(model_folder) is False:
    os.mkdir(model_folder)

if os.path.exists(result_folder) is False:
    os.mkdir(result_folder)

input_feature = None
query_feature = None
feature_list = []
feature_num_list = []
df_pristine_query = None
from sklearn.feature_selection import RFECV
train_errors = []
test_errors = []
max_features = 200
features_selected = []
total_num_features = x_train.shape[1]
print('total number of features = ',x_train.shape[1])
with tqdm.tqdm(total=max_features,desc='Feature selection ...') as timer:
    for num_features in range(1,max_features+1,1):

        print('************************************************')
        print('Selecting top {} features'.format(num_features))
        # Run RF
        classifier = RF_Classifier(n_estimators=total_num_features*num_features, verbose=2)
        #clf_rf.fit(x_train[mrmr_output], y_train)
        #classifier = svm.SVC(C=1.0,verbose=2)
        #clf_rf.fit(x_train[mrmr_output], y_train)

        #classifier = MLPClassifier(solver='adam', verbose=True, hidden_layer_sizes=(10,),momentum=0.0)
        #selector = RFECV(classifier,step=1,cv=5)
        selector = sfs(classifier,k_features=num_features,forward=True,floating=False,verbose=2,scoring='accuracy',cv=0)
        selector = selector.fit(x_train,y_train)

        print("selected features ")
        feature_cols = list(selector.k_feature_idx_)
        features_selected.append(feature_cols)

        print("Training with the selected number of features")
        classifier_rf = RF_Classifier(n_estimators=5000,verbose=2)
        x_train_selected = x_train[:,feature_cols]
        x_test_selected = x_test[:,feature_cols]

        print('Training ...')
        classifier_rf.fit(x_test_selected,x_test_selected)
        y_train_pred = classifier_rf.predict(x_train_selected)

        y_test_pred = classifier_rf.predict(x_test_selected)

        train_error = acc(y_train,y_train_pred)
        test_error = acc(y_test,y_test_pred)

        print('Training done...')
        print('Testing ...')
        # pristine output

        rf_predicted = classifier_rf.predict(df_pristine_query)

        df_output = pd.concat([df_pristine_file['ID_code'], pd.DataFrame(rf_predicted, columns=['target'])], axis=1)
        df_output.to_csv(os.path.join(result_folder,'Test_RF_'+str(num_features)+'_features.csv'),index=False)
        print('done..')
        pickle.dump(classifier_rf, open(os.path.join(model_folder, 'rf_num_features_' + str(num_features) + '.sav'), 'wb'))

        print('model saved. ')

        mllogger.scalar('Train_errors',train_error,step=num_features)
        mllogger.scalar('Test_errors', test_error, step=num_features)
        mllogger.text('Selected features are '+str(feature_cols),step=num_features)
        mllogger.text('Train error'+str(train_error),step=num_features)
        mllogger.text('Test error' + str(test_error), step=num_features)

        timer.update()
timer.close()






