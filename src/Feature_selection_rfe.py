import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF_Classifier
from sklearn import svm
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt

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
model_folder = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Model/mrmr_iteration/MLP/')
result_folder = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Model/mrmr_iteration/Results/')
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

with open(log_file, 'a') as logger:
    print('************************************************')

    # Run RF
    #clf_rf = RF_Classifier(n_estimators=100, verbose=2)
    #clf_rf.fit(x_train[mrmr_output], y_train)
    classifier = svm.SVC(C=1.0,verbose=2)
    #clf_rf.fit(x_train[mrmr_output], y_train)

    #classifier = MLPClassifier(solver='adam', verbose=True, hidden_layer_sizes=(10,),momentum=0.0)
    selector = RFECV(classifier,step=1,cv=5)
    selector = selector.fit(x_train,y_train)

    print("selected features ")
    print(selector.ranking_)

    train_score = selector.score(x_train, y_train)
    print('Train score = ',train_score)
    train_accuracy.append(train_score)

    test_score = classifier.score(x_test, y_test)
    test_accuracy.append(test_score)
    print('Validation score = ', test_score)
    # feature_num_list.append(i)
    # plt.clf()
    # plt.ion()
    # plt.show()
    # plt.plot(np.asarray(feature_num_list), np.asarray(train_accuracy), 'r-', label='train accuracy')
    # plt.plot(np.asarray(feature_num_list), np.asarray(test_accuracy), 'b-', label='test_accuracy')
    # plt.grid(True)
    # plt.xlabel('number of features')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.draw()
    # plt.pause(0.0001)

    print('Training done...')
    print('Testing ...')
    # pristine output

    rf_predicted = classifier.predict(df_pristine_query)

    df_output = pd.concat([df_pristine_file['ID_code'], pd.DataFrame(rf_predicted, columns=['target'])], axis=1)
    df_output.to_csv(os.path.join(result_folder,'MLP_'+str(i+1)+'_features.csv'),index=False)
    print('done..')
    pickle.dump(classifier, open(os.path.join(model_folder, 'rf_num_features_' + str(i + 1) + '.sav'), 'wb'))

    #pickle.dump(scaler,open(os.path.join(model_folder,'scaler_num_features_'+str(i+1)+'.sav'),'wb'))
    print('model saved. ')



plt.show()






