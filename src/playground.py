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
#np_var = np.asarray(df_dropped_file['var_0'])
#print(len(np.where(np.abs(np_var)< mean-1*std )[0]))

from sklearn.preprocessing import StandardScaler,Normalizer
scaler = StandardScaler()
df_dropped_file_scaled = scaler.fit_transform(np.asarray(df_dropped_file))

# get training and validation files
test_size = 0.2
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(df_dropped_file_scaled,np.asarray(df_train_file['target']),test_size=test_size,shuffle=True,stratify=np.asarray(df_train_file['target']))

print('training and validation files generated')
run_mrmr = True
selected_features = None
num_features = 200
if run_mrmr is True:
    import pymrmr
    mrmr_type = 'MIQ'
    print('Running mrmr',mrmr_type)
    mrmr_output = pymrmr.mRMR(pd.DataFrame(X_train,columns=column_names),mrmr_type,num_features)
    print('Selected feature order with mrmr{} is the following: \n {} '.format(mrmr_type,mrmr_output))
    selected_features = mrmr_output
else:
    selected_features = ['var_6', 'var_180', 'var_143', 'var_190', 'var_146', 'var_32', 'var_195', 'var_19', 'var_3', 'var_188', 'var_26', 'var_110', 'var_139', 'var_86', 'var_79', 'var_31', 'var_58', 'var_89', 'var_116', 'var_95', 'var_16', 'var_187', 'var_67', 'var_148', 'var_173', 'var_104', 'var_91', 'var_70', 'var_47', 'var_76', 'var_178', 'var_137', 'var_194', 'var_192', 'var_96', 'var_44', 'var_166', 'var_74', 'var_69', 'var_160', 'var_22', 'var_163', 'var_100', 'var_167', 'var_59', 'var_54', 'var_10', 'var_97', 'var_107', 'var_51', 'var_20', 'var_64', 'var_46', 'var_2', 'var_122', 'var_33', 'var_52', 'var_5', 'var_48', 'var_114', 'var_81', 'var_57', 'var_11', 'var_88', 'var_149', 'var_73', 'var_145', 'var_25', 'var_43', 'var_186', 'var_181', 'var_108', 'var_127', 'var_80', 'var_34', 'var_134', 'var_56', 'var_124', 'var_106', 'var_68', 'var_30', 'var_111', 'var_130', 'var_62', 'var_84', 'var_152', 'var_129', 'var_147', 'var_24', 'var_42', 'var_199', 'var_118', 'var_179', 'var_18', 'var_177', 'var_17', 'var_117', 'var_72', 'var_162', 'var_175']

plot_output = True
from sklearn.metrics import f1_score
train_accuracies = []
test_accuracies = []
base_estimator = 'decTrees'
if selected_features is not None: 
    from sklearn.ensemble import GradientBoostingClassifier as GradBoost
    from sklearn.linear_model import SGDClassifier as SGD
    feature_list = []
    feature_name_list = []
    for feature in selected_features: 
        print('*************************************************')
        feature_index = int(feature.split('var_')[1])+1
        feature_list.append(feature_index)
        feature_name_list.append(feature)
        print('Number of features = ', len(feature_name_list))
        print('selected feature combination ',feature_name_list)
        x_selected = X_train[:,np.array(feature_list)]
        if base_estimator == 'sgd':
            estimator = SGD(verbose=0,tol=1e-3,max_iter=1000,learning_rate='adaptive',eta0=0.1,warm_start=True)
            classifier_mm = GradBoost(base_estimator=estimator, algorithm='SAMME')
        else:
            classifier_mm = GradBoost(n_estimators=500)
        classifier_mm.fit(x_selected,Y_train)
        train_output = classifier_mm.predict(x_selected)
        train_accuracy = f1_score(Y_train,train_output)
        train_accuracies.append(train_accuracy)
        print('Train accuracy (F1) =',train_accuracy)
        from sklearn.metrics import confusion_matrix
        conf = confusion_matrix(Y_train,train_output)
        print('Train confusion matrix = \n',conf)
        
        x_test_selected = X_test[:,np.array(feature_list)]
        test_output = classifier_mm.predict(x_test_selected)
        test_accuracy = f1_score(Y_test,test_output)
        test_accuracies.append(test_accuracy)
        print('Test accuracy (F1) =',test_accuracy)
        conf = confusion_matrix(Y_test,test_output)
        print('Test confusion matrix = \n',conf)
        
        print('*************************************************')
else:
    from sklearn.mixture import GaussianMixture as GMM
     
    gaussian_mm = GMM(n_components=2,verbose=2)
    gaussian_mm.fit(df_dropped_file_scaled,np.asarray(df_train_file['target']).reshape(-1,1))
    out = gaussian_mm.predict(df_dropped_file_scaled)
    
    from sklearn.metrics import confusion_matrix
    conf = confusion_matrix(df_train_file['target'],out)
    print(conf)
 
if plot_output is True: 
    from matplotlib import pyplot as plt
    plt.plot(np.arange(num_features),train_accuracies,'r-')
    plt.plot(np.arange(num_features),test_accuracies,'b-')
    plt.grid(True)
    plt.xlabel('Features')
    plt.ylabel('Accuracies')
    plt.show()
