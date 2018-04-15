import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
hp_train = pd.read_csv('data/train.csv')
hp_test = pd.read_csv("data/test.csv")

hp_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
hp_train = hp_train.drop(hp_train[hp_train['Id'] == 1299].index)
hp_train = hp_train.drop(hp_train[hp_train['Id'] == 524].index)
# 正态化SalePrice
target = np.log(hp_train['SalePrice'])
train = hp_train.drop(['SalePrice'], axis=1)
# 合并测试集和训练集
combined = train.append(hp_test)
combined.reset_index(inplace=True)
combined.drop(['index','Id'], inplace=True, axis=1)

# 显示数据
""""
print (hp_train.columns)
print (hp_train['SalePrice'].describe() )
sns.distplot(hp_train['SalePrice'])
plt.show(sns.distplot(hp_train['SalePrice']))
# 对于houseprice和GrLivArea,发现两个异常点 
var = 'GrLivArea'
data = pd.concat([hp_train['SalePrice'], hp_train[var]], axis=1)
plt.show(data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000)))
"""

# 正态化GrLivArea
#transformed histogram and normal probability plot
combined['GrLivArea'] = np.log(combined['GrLivArea'])


print ("Unique values are:", combined.BsmtFinSF1.unique())
print (train.BsmtFinSF1.value_counts(), "\n")

# 将是否有basement 分为两类
def encode(x): return 0 if x == 0 else 1
combined['enc_TotalBsmtSF'] = combined.TotalBsmtSF.apply(encode)

# 将enc_condition减少为二值属性
def encode(x): return 1 if x == 'Partial' else 0
combined['enc_condition'] = combined.SaleCondition.apply(encode)

combined['GarageCars'].fillna(combined.iloc[:1458].GarageCars.median(),inplace=True)
combined['GarageArea'].fillna(combined.iloc[:1458].GarageArea.median(),inplace=True)
combined['LotFrontage'].fillna(combined.iloc[:1458].LotFrontage.median(),inplace=True)
combined['MasVnrArea'].fillna(combined.iloc[:1458].MasVnrArea.median(),inplace=True)
combined['TotalBsmtSF'].fillna(combined.iloc[:1458].TotalBsmtSF.median(),inplace=True)
# one-hot encoding
combined['enc_Street'] = pd.get_dummies(combined.Street, drop_first=True)

combined['GarageCond'].fillna('TA',inplace=True)
GarageCond_dummies = pd.get_dummies(combined['GarageCond'], prefix='GarageCond')
combined = pd.concat([combined,GarageCond_dummies], axis=1)
combined.drop('GarageCond', axis=1, inplace=True)

combined['GarageQual'].fillna('TA',inplace=True)
# one-hot 多值编码
GarageQual_dummies = pd.get_dummies(combined['GarageQual'], prefix='GarageQual')
combined = pd.concat([combined,GarageQual_dummies], axis=1)
combined.drop('GarageQual', axis=1, inplace=True)

combined['GarageYrBlt'].fillna(combined.iloc[:1458].GarageYrBlt.median(),inplace=True)

combined['GarageFinish'].fillna('Unf',inplace=True)
GarageFinish_dummies = pd.get_dummies(combined['GarageFinish'], prefix='GarageFinish')
combined = pd.concat([combined,GarageFinish_dummies], axis=1)
combined.drop('GarageFinish', axis=1, inplace=True)

combined['GarageType'].fillna('Attchd',inplace=True)
GarageType_dummies = pd.get_dummies(combined['GarageType'], prefix='GarageType')
combined = pd.concat([combined,GarageType_dummies], axis=1)
combined.drop('GarageType', axis=1, inplace=True)


combined['BsmtCond'].fillna('TA',inplace=True)
BsmtCond_dummies = pd.get_dummies(combined['BsmtCond'], prefix='BsmtCond')
combined = pd.concat([combined,BsmtCond_dummies], axis=1)
combined.drop('BsmtCond', axis=1, inplace=True)




#nulls = pd.DataFrame(combined.isnull().sum().sort_values(ascending=False)[:25])
#nulls.columns = ['Null Count']
#nulls.index.name = 'Feature'
#print(nulls)
# 删掉空缺值太多的5个属性
combined.drop(['PoolQC'], 1, inplace=True)
combined.drop(['MiscFeature'], 1, inplace=True)
combined.drop(['Alley'], 1, inplace=True)
combined.drop(['Fence'], 1, inplace=True)
combined.drop(['FireplaceQu'], 1, inplace=True)
"""
numeric_features = combined.select_dtypes(include=[np.number])
print(numeric_features.dtypes)
"""
train = combined.iloc[:1458]
test = combined.iloc[1458:]

predictors=['OverallQual','GrLivArea','TotRmsAbvGrd','enc_TotalBsmtSF','GarageCars','1stFlrSF','GarageArea','FullBath',
            'YearBuilt','LotFrontage','GarageCond_Ex','GarageCond_Fa','GarageCond_Gd','GarageCond_Po','GarageCond_TA',
            'GarageYrBlt','GarageQual_Ex','GarageQual_Fa','GarageQual_Gd','GarageQual_Po','GarageQual_TA','GarageFinish_Fin',
            'GarageFinish_RFn','GarageFinish_Unf','YearRemodAdd','MasVnrArea','Fireplaces','TotalBsmtSF']
"""
,'GarageType_2Types','GarageType_Attchd','GarageType_Basment','GarageType_BuiltIn',
            'GarageType_CarPort','GarageType_Detchd','BsmtCond_Fa','BsmtCond_Gd','BsmtCond_Po','BsmtCond_TA'
"""
numeric_features = hp_train.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print (corr['SalePrice'].sort_values(ascending=False)[:15], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])

x_train, x_test, y_train, y_test = train_test_split(train[predictors], target, random_state=42, test_size=.2)

lr = linear_model.LinearRegression()
model = lr.fit(x_train, y_train)
predictions = model.predict(x_test)
print ('RMSE is: \n', mean_squared_error(y_test, predictions))
model = lr.fit(train[predictors], target)
predictions=model.predict(test[predictors])

predictions = np.exp(predictions)
submission = pd.DataFrame({
        "Id": hp_test["Id"],
        "SalePrice": predictions
    })
print(submission)
submission.to_csv('/Users/martin_yan/Desktop/submission4.csv', index=False)