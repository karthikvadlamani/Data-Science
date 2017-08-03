
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
get_ipython().magic('matplotlib inline')


# In[2]:

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[3]:

train = train.drop(['Ticket', 'Cabin'], axis=1)
test = test.drop(['Ticket', 'Cabin'], axis=1)


# In[4]:

combine = [train, test]


# In[5]:

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])


# In[6]:

pd.crosstab(test['Title'], train['Sex'])


# In[7]:

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Dr', 'Rev', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Sir'], 'Mr')


# In[9]:

train['Title'].unique()


# In[10]:

train.head()


# In[11]:

test.head()


# In[12]:

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[13]:

title1 = pd.get_dummies(train['Title'], drop_first=True)
train.drop(['Title'], axis=1, inplace=True)
train = pd.concat([train, title1], axis=1)


# In[14]:

title2 = pd.get_dummies(test['Title'], drop_first=True)
test.drop(['Title'], axis=1, inplace=True)
test = pd.concat([test, title2], axis=1)


# In[15]:

train.head(2)


# In[16]:

test.head(2)


# In[17]:

train = train.drop(['Name', 'PassengerId'], axis=1)
test = test.drop(['Name'], axis=1)


# In[18]:

combine = [train, test]
train.shape, test.shape


# In[19]:

train.head()


# In[20]:

test.head()


# In[21]:

sex1 = pd.get_dummies(train['Sex'], drop_first=True)
train.drop(['Sex'], axis=1, inplace=True)
train = pd.concat([train, sex1], axis=1)


# In[22]:

sex2 = pd.get_dummies(test['Sex'], drop_first=True)
test.drop(['Sex'], axis=1, inplace=True)
test = pd.concat([test, sex2], axis=1)


# In[23]:

train.head()


# In[24]:

test.head()


# In[25]:

guess_ages = np.zeros((2,3))
guess_ages


# In[26]:

for i in range(0,2):
    for j in range(0,3):
        guess_df1 = train[(train['male'] == i) & (train['Pclass'] == j+1)]['Age'].dropna()
        age_guess1 = guess_df1.median()
        guess_ages[i,j] = int(age_guess1/0.5 + 0.5) * 0.5
            
for i in range(0,2):
    for j in range(0,3):
        train.loc[(train.Age.isnull()) & (train.male == i) & (train.Pclass == j+1), 'Age'] = guess_ages[i,j]
        
train['Age'] = train['Age'].astype(int)
train.head()


# In[27]:

for i in range(0,2):
    for j in range(0,3):
        guess_df2 = test[(test['male'] == i) & (test['Pclass'] == j+1)]['Age'].dropna()
        age_guess2 = guess_df2.median()
        guess_ages[i,j] = int(age_guess2/0.5 + 0.5) * 0.5
            
for i in range(0,2):
    for j in range(0,3):
        test.loc[(test.Age.isnull()) & (test.male == i) & (test.Pclass == j+1), 'Age'] = guess_ages[i,j]
        
test['Age'] = test['Age'].astype(int)
test.head()


# In[28]:

embarked1 = pd.get_dummies(train['Embarked'], drop_first=True)
train.drop(['Embarked'], axis=1, inplace=True)
train = pd.concat([train, embarked1], axis=1)


embarked2 = pd.get_dummies(test['Embarked'], drop_first=True)
test.drop(['Embarked'], axis=1, inplace=True)
test = pd.concat([test, embarked2], axis=1)


# In[29]:

train.head(1)


# In[30]:

test.head(1)


# In[31]:

train['FamilySize'] = train['SibSp'] + train['Parch'] +1
test['FamilySize'] = test['SibSp'] + test['Parch'] +1


# In[32]:

train.head(1)


# In[33]:

train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[34]:

train['IsAlone'] = 0
train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
test['IsAlone'] = 0
test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1

train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# In[35]:

train = train.drop(['SibSp', 'Parch', 'FamilySize'], axis=1)
test = test.drop(['SibSp', 'Parch', 'FamilySize'], axis=1)
combine = [train, test]


# In[36]:

pclass1 = pd.get_dummies(train['Pclass'], drop_first=True)
train.drop(['Pclass'], axis=1, inplace=True)
train = pd.concat([train, pclass1], axis=1)


# In[37]:

train.head(1)


# In[38]:

pclass2 = pd.get_dummies(test['Pclass'], drop_first=True)
test.drop(['Pclass'], axis=1, inplace=True)
test = pd.concat([test, pclass2], axis=1)


# In[39]:

test.head(1)


# In[40]:

test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)


# In[40]:

X_train = train.drop("Survived", axis=1)
y_train = train["Survived"]
X_test  = test.drop(["PassengerId", "Master"], axis=1)
X_train.shape, y_train.shape, X_test.shape


# In[41]:

X_test.isnull().any()


# In[41]:

train.to_csv('Ftrain.csv', index=False)
test.to_csv('Ftest.csv', index=False)


# In[150]:

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[151]:

lr = LogisticRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
acc_lr = round(lr.score(X_train, y_train) * 100, 2)
acc_lr


# In[152]:

svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100, 2)
acc_svc


# In[153]:

knn = KNeighborsClassifier(n_neighbors = 4, weights = 'distance')
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
acc_knn


# In[154]:

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)
acc_dt = round(dt.score(X_train, y_train) * 100, 2)
acc_dt


# In[155]:

rf = RandomForestClassifier(n_estimators=75)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
acc_rf = round(rf.score(X_train, y_train) * 100, 2)
acc_rf


# In[156]:

pred_rf


# In[157]:

submission2 = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred_rf})
submission2.to_csv('C:/Users/karth/Data Science/Datasets/Titanic/submission2.csv', index=False)


# In[158]:

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'multi:softmax',
    'num_class': 2,
    'eval_metric': 'mlogloss',
    'silent': 1
}


# In[ ]:




# In[159]:

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test)


# In[160]:

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20, verbose_eval=100, show_stdv=False)


# In[167]:

model = xgb.XGBClassifier(max_depth=3, n_estimators=1000, learning_rate=0.01).fit(X_train, y_train)
predictions = model.predict(X_test)


# In[168]:

y_predict = model.predict(X_test)
output = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": y_predict})
output.head()


# In[169]:

output.to_csv('xgbTitanic.csv', index=False)


# In[ ]:



