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
test_size = 0.05
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(df_dropped_file_scaled,np.asarray(df_train_file['target']),test_size=test_size,shuffle=True,stratify=np.asarray(df_train_file['target']))

print('training and validation files generated')

plot_output = True
from sklearn.metrics import f1_score
train_accuracies = []
test_accuracies = []
base_estimator = 'decTrees'


#from h2o4gpu.ensemble import GradientBoostingClassifierSklearn as GradBoost
from xgboost import XGBClassifier as xgb_classifier
import xgboost as xgb
#from sklearn.linear_model import SGDClassifier as SGD
from h2o4gpu.linear_model import SGDClassifier as SGD
feature_list = []
feature_name_list = []

num_round = 50
maxdepth = 5
tree_method = 'gpu_hist'
grow_policy= 'lossguide'
max_depth= maxdepth
random_state = 1234
objective= 'binary:logistic'  # Specify multiclass classification
num_class = 2  # Number of possible output classes
base_score = 0.5
booster= 'gbtree'
colsample_bylevel=1
colsample_bytree=0.8
gamma=0
learning_rate=0.1
max_delta_step=0
min_child_weight=1
missing=None
n_estimators=1000
scale_pos_weight=1
silent=True
subsample=0.8
verbose=True
n_jobs=-1
seed=27

eval_metric = 'error'
cv_folds = 20
early_stopping_rounds = 50

gpu_res = {}  # Store accuracy result
classifier_mm = xgb_classifier(learning_rate=learning_rate,tree_method=tree_method,grow_policy=grow_policy, objective=objective,eval_metric=eval_metric,booster=booster,n_estimators=n_estimators,max_depth=max_depth,
                               min_child_weight=min_child_weight,gamma=gamma,colsample_bytree=colsample_bytree,seed=seed)

xgb_param = classifier_mm.get_xgb_params()

xgtrain = xgb.DMatrix(X_train, label=Y_train)
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=classifier_mm.get_params()['n_estimators'], nfold=cv_folds,metrics=eval_metric, early_stopping_rounds=early_stopping_rounds)
classifier_mm.set_params(n_estimators=cvresult.shape[0])


classifier_mm.fit(X_train,Y_train.reshape(-1,1))

train_output = classifier_mm.predict(X_train)
train_accuracy = f1_score(Y_train,train_output)
train_accuracies.append(train_accuracy)
print('Train accuracy (F1) =',train_accuracy)
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(Y_train,train_output)
print('Train confusion matrix = \n',conf)



test_output = classifier_mm.predict(X_test)
test_accuracy = f1_score(Y_test,test_output)
test_accuracies.append(test_accuracy)
print('Test accuracy (F1) =',test_accuracy)
conf = confusion_matrix(Y_test,test_output)
print('Test confusion matrix = \n',conf)

print('*************************************************')

 
# if plot_output is True:
#     from matplotlib import pyplot as plt
#     plt.plot(np.arange(num_features),train_accuracies,'r-')
#     plt.plot(np.arange(num_features),test_accuracies,'b-')
#     plt.grid(True)
#     plt.xlabel('Features')
#     plt.ylabel('Accuracies')
#     plt.show()
