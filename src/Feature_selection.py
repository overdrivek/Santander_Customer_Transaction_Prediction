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
perform_feature_selection = False
train_accuracy = []
test_accuracy = []
feature_list = []
if perform_feature_selection is True:
    print("Performing mrmr feature selection")

    import pymrmr
    num_features = 50

    with open(log_file, 'a') as logger:
        mrmr_output = pymrmr.mRMR(x_train,'MIQ',num_features)
        for mrmr_feature in mrmr_output:
            logger.write(mrmr_feature+'\t')
        logger.write('\n')
        print(mrmr_output)

    logger.close()
else:
    mrmr_output = ['var_132', 'var_131', 'var_135', 'var_134', 'var_133', 'var_137', 'var_139', 'var_130', 'var_129', 'var_128', 'var_136', 'var_126', 'var_138', 'var_145', 'var_144', 'var_148', 'var_147', 'var_146', 'var_125', 'var_140', 'var_143', 'var_142', 'var_141', 'var_127', 'var_150', 'var_107', 'var_106', 'var_110', 'var_109', 'var_108', 'var_112', 'var_114', 'var_105', 'var_104', 'var_103', 'var_111', 'var_151', 'var_113', 'var_120', 'var_119', 'var_123', 'var_122', 'var_121', 'var_124', 'var_115', 'var_118', 'var_117', 'var_116', 'var_149', 'var_101']

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
for i,feature_selected in enumerate(mrmr_output):
    with open(log_file, 'a') as logger:
        print('************************************************')

        # Run RF
        #clf_rf = RF_Classifier(n_estimators=100, verbose=2)
        #clf_rf.fit(x_train[mrmr_output], y_train)
        #clf_rf = svm.SVC(C=1.0,verbose=2)
        #clf_rf.fit(x_train[mrmr_output], y_train)
        feature_list.append(feature_selected)
        print('Features for training are :',feature_list)
        if input_feature is None:
            input_feature = x_train[feature_selected]
        else:
            input_feature = pd.concat([input_feature,x_train[feature_selected]],axis=1)

        classifier = MLPClassifier(solver='adam', verbose=True, hidden_layer_sizes=(10,),momentum=0.0)
        try:
            classifier.fit(input_feature, y_train)
        except:
            classifier.fit(np.array(input_feature).reshape(-1,1), np.array(y_train).reshape(-1,1))
        train_score = classifier.score(x_train[feature_list], y_train)
        print('Train score = ',train_score)
        train_accuracy.append(train_score)

        test_score = classifier.score(x_test[feature_list], y_test)
        test_accuracy.append(test_score)
        print('Validation score = ', test_score)
        feature_num_list.append(i)
        plt.clf()
        plt.ion()
        plt.show()
        plt.plot(np.asarray(feature_num_list), np.asarray(train_accuracy), 'r-', label='train accuracy')
        plt.plot(np.asarray(feature_num_list), np.asarray(test_accuracy), 'b-', label='test_accuracy')
        plt.grid(True)
        plt.xlabel('number of features')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.draw()
        plt.pause(0.0001)

        print('Training done...')
        print('Testing ...')
        # pristine output
        if df_pristine_query is None:
            df_pristine_query = test_scaled[feature_selected]
        else:
            df_pristine_query = pd.concat([df_pristine_query,test_scaled[feature_selected]],axis=1)
        #df_pristine_query = test_scaled[mrmr_output]
        try:
            rf_predicted = classifier.predict(df_pristine_query)
        except:
            rf_predicted = classifier.predict(np.array(df_pristine_query).reshape(-1, 1))

        df_output = pd.concat([df_pristine_file['ID_code'], pd.DataFrame(rf_predicted, columns=['target'])], axis=1)
        df_output.to_csv(os.path.join(result_folder,'MLP_'+str(i+1)+'_features.csv'),index=False)
        print('done..')
        pickle.dump(classifier, open(os.path.join(model_folder, 'rf_num_features_' + str(i + 1) + '.sav'), 'wb'))

        #pickle.dump(scaler,open(os.path.join(model_folder,'scaler_num_features_'+str(i+1)+'.sav'),'wb'))
        print('model saved. ')



    logger.close()
plt.show()






