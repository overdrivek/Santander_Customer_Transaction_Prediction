import pandas as pd
import os
from sklearn import metrics
from matplotlib import pyplot as plt
#from h2o4gpu.ensemble import GradientBoostingClassifierSklearn as GradBoost
#from xgboost import XGBClassifier as xgb_classifier
#import xgboost as xgb
#from sklearn.linear_model import SGDClassifier as SGD
#from h2o4gpu.linear_model import SGDClassifier as SGD
import  numpy as np
import lightgbm_gpu as lgbm
from sklearn.preprocessing import StandardScaler,Normalizer, RobustScaler
from sklearn.metrics import confusion_matrix


train_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/train.csv')
df_train_file = pd.read_csv(train_file)
df_dropped_file = df_train_file.drop(['ID_code','target'],axis=1)
column_names = list(df_dropped_file.columns.values)

# tests
std= df_dropped_file['var_0'].std()
mean = df_dropped_file['var_0'].mean()

scaler = StandardScaler()
df_dropped_file_scaled = scaler.fit_transform(np.asarray(df_dropped_file))

# get training and validation files
test_size = 0.1
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(df_dropped_file_scaled,np.asarray(df_train_file['target']),test_size=test_size,shuffle=True,stratify=np.asarray(df_train_file['target']))

train_data = lgbm.Dataset(X_train,label=Y_train)
valid_data = train_data.create_valid(X_test,label=Y_test)
print('Training and validation data generated.')


plot_output = True
from sklearn.metrics import f1_score
train_accuracies = []
test_accuracies = []
base_estimator = 'decTrees'

useTrainCV = False
num_rounds = 10
cv_folds = 10
eval_metric = 'binary_error'
num_leaves = 50
num_trees = 1000
objective = 'binary'
learning_rate = 0.1
boosting = 'gbdt'
num_iterations = 1000
verbosity  = 1



param = {'num_leaves': num_leaves,
         'num_trees': num_trees,
         'objective': objective,
         'metric':eval_metric,
         'learning_rate':learning_rate,
         'boosting':boosting,
         'num_iterations':num_iterations,
         'verbosity':verbosity
         }

if useTrainCV:
    cv_results = lgbm.cv(param, train_data, num_rounds, nfold=cv_folds)
    lgbm_classifier = lgbm.train(param, train_data, num_rounds, valid_sets=[valid_data])
else:
    lgbm_classifier = lgbm.train(param,train_data,num_rounds,valid_sets=[valid_data])

train_output = np.round(lgbm_classifier.predict(X_train))
test_output = np.round(lgbm_classifier.predict(X_test))

print('\n Train')
print('Accuracy : %.4g'% metrics.accuracy_score(Y_train,train_output))
train_f1_score = f1_score(Y_train, train_output)
print('F1 score: %.4g' % train_f1_score)
conf = confusion_matrix(Y_train,train_output)
print('Train confusion matrix = \n',conf)

print('\n Test')
print('Accuracy : %.4g'% metrics.accuracy_score(Y_test,test_output))
test_f1_score = f1_score(Y_test, test_output)
print('F1 score: %.4g' % test_f1_score)
conf = confusion_matrix(Y_test,test_output)
print('Test confusion matrix = \n',conf)

print('*************************************************')
pristine_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/test.csv')
df_pristine_file = pd.read_csv(pristine_file)
df_dropped_file = df_pristine_file.drop(['ID_code'],axis=1)
test_output = lgbm_classifier.predict(scaler.transform(np.asarray(df_dropped_file)))
df_output = pd.concat([df_pristine_file['ID_code'], pd.DataFrame(test_output, columns=['target'])], axis=1)
output_folder = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Model/lgbm/')
if os.path.exists(output_folder) is False:
    os.mkdir(output_folder)
df_output.to_csv(os.path.join(output_folder,'pristine_result_.csv'),index=False)

#
#
# grid_search = False
#
# from sklearn.model_selection import GridSearchCV
# if grid_search is True:
#     param_test1 = {
#         #'max_depth': range(3, 10, 2),
#         #'min_child_weight': range(1, 6, 2),
#         #'gamma': [i / 10.0 for i in range(0, 5)]
#         #'subsample': [i / 10.0 for i in range(6, 10)],
#         #'colsample_bytree': [i / 10.0 for i in range(6, 10)]
#         'learning_rate':[0.15,0.1,0.09,0.08,0.07,0.06,0.05]
#     }
#     gsearch1 = GridSearchCV(estimator=xgb_classifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
#                                                     min_child_weight=min_child_weight, gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree,
#                                                     objective='binary:logistic', scale_pos_weight=scale_pos_weight,seed=seed,tree_method=tree_method,nthread=4),
#                             param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
#     gsearch1.fit(X_train, Y_train)
#
#     print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
# else:
#
#     classifier_mm = xgb_classifier(learning_rate=learning_rate,tree_method=tree_method,grow_policy=grow_policy, objective=objective,eval_metric=eval_metric,booster=booster,n_estimators=n_estimators,max_depth=max_depth,
#                                    min_child_weight=min_child_weight,gamma=gamma,colsample_bytree=colsample_bytree,seed=seed,subsample=subsample,scale_pos_weight=scale_pos_weight)
#
#     modelfit(classifier_mm,X_train,Y_train)
#
#     #xgb_param = classifier_mm.get_xgb_params()
#
#
#     train_output = classifier_mm.predict(X_train)
#     train_f1_score = f1_score(Y_train, train_output)
#     print('Train F1 score =', train_f1_score)
#     train_accuracy = metrics.accuracy_score(Y_train,train_output)
#     print('Train accuracy = ',train_accuracy)




    # test_output = classifier_mm.predict(X_test)
    # test_f1_score = f1_score(Y_test, test_output)
    # print('Test accuracy (F1) =', test_f1_score)
    # test_accuracy = metrics.accuracy_score(Y_test,test_output)
    # print('Test accuracy = ',test_accuracy)
    # conf = confusion_matrix(Y_test,test_output)
    # print('Test confusion matrix = \n',conf)

    # print('*************************************************')
    # pristine_file = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Data/test.csv')
    # df_pristine_file = pd.read_csv(pristine_file)
    # df_dropped_file = df_pristine_file.drop(['ID_code'],axis=1)
    # test_output = classifier_mm.predict(scaler.transform(np.asarray(df_dropped_file)))
    # df_output = pd.concat([df_pristine_file['ID_code'], pd.DataFrame(test_output, columns=['target'])], axis=1)
    # output_folder = os.path.normpath('/home/naraya01/AEN/GIT/Santander/Santander_Customer_Transaction_Prediction/Model/xgBoost/')
    # if os.path.exists(output_folder) is False:
    #     os.mkdir(output_folder)
    # df_output.to_csv(os.path.join(output_folder,'pristine_result_'+grow_policy+'_'+tree_method+'_'+booster+'.csv'),index=False)
