import pandas as pd
import numpy as np

hp_train = pd.read_csv('data/train.csv')
hp_test = pd.read_csv("data/test.csv")
#去掉异常点
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
# 正态化GrLivArea
combined['GrLivArea'] = np.log(combined['GrLivArea'])

#nulls = pd.DataFrame(combined.isnull().sum().sort_values(ascending=False)[:35])
#nulls.columns = ['Null Count']
#nulls.index.name = 'Feature'
#print(nulls)
#print ("Unique values are:", combined.Neighborhood.unique())
#print (train.Neighborhood.value_counts(), "\n")
#categoricals = train.select_dtypes(exclude=[np.number])
#print(categoricals.describe())

# 删掉空缺值太多的18个特征
na_count = combined.isnull().sum().sort_values(ascending=False)
na_rate = na_count / len(combined)
na_data = pd.concat([na_count,na_rate],axis=1,keys=['count','ratio'])
print(na_data.head(34))
combined = combined.drop(na_data[na_data['count']>4].index, axis=1)  # 删除上述前18个特征


# 将是否有basement 分为两类
def encode(x): return 0 if x == 0 else 1
combined['enc_TotalBsmtSF'] = combined.TotalBsmtSF.apply(encode)
# 将是否有porch 分为两类
def encode(x): return 0 if x == 0 else 1
combined['enc_OpenPorchSF'] = combined.OpenPorchSF.apply(encode)

#将月份映射为季度
month_mapping = {
    1:'one',
    2:'one',
    3:'one',
    4:'two',
    5:'two',
    6: 'two',
    7: 'three',
    8: 'three',
    9:'three',
    10:'four',
    11:'four',
    12:'four',
}
combined['MoSold'] = combined.MoSold.map(month_mapping)
MoSold_dummies = pd.get_dummies(combined['MoSold'], prefix='MoSold')
combined = pd.concat([combined,MoSold_dummies], axis=1)
combined.drop('MoSold', axis=1, inplace=True)

# 将enc_condition减少为二值属性，partial 作为特别属性
def encode(x): return 1 if x == 'Partial' else 0
combined['enc_condition'] = combined.SaleCondition.apply(encode)

#用train data的中位数来填补缺失信息
combined['GarageCars'].fillna(combined.iloc[:1458].GarageCars.median(),inplace=True)
combined['GarageArea'].fillna(combined.iloc[:1458].GarageArea.median(),inplace=True)
combined['TotalBsmtSF'].fillna(combined.iloc[:1458].TotalBsmtSF.median(),inplace=True)
combined['BsmtFullBath'].fillna(combined.iloc[:1458].BsmtFullBath.median(),inplace=True)
combined['BsmtUnfSF'].fillna(combined.iloc[:1458].BsmtUnfSF.median(),inplace=True)
combined['BsmtFinSF1'].fillna(combined.iloc[:1458].BsmtFinSF2.median(),inplace=True)
combined['BsmtFinSF2'].fillna(combined.iloc[:1458].BsmtFinSF2.median(),inplace=True)
combined['BsmtHalfBath'].fillna(combined.iloc[:1458].BsmtHalfBath.median(),inplace=True)

# one-hot encoding
combined['enc_Street'] = pd.get_dummies(combined.Street, drop_first=True)


combined['MSZoning'].fillna('RL',inplace=True)
MSZoning_dummies = pd.get_dummies(combined['MSZoning'], prefix='MSZoning')
combined = pd.concat([combined,MSZoning_dummies], axis=1)
combined.drop('MSZoning', axis=1, inplace=True)

def encode(x): return 1 if x == 'Regular' else 0
combined['enc_LotShape'] = combined.LotShape.apply(encode)

#无需填补，直接ont-hot
LandContour_dummies = pd.get_dummies(combined['LandContour'], prefix='LandContour')
combined = pd.concat([combined,LandContour_dummies], axis=1)
combined.drop('LandContour', axis=1, inplace=True)

combined['Utilities'].fillna('AllPub',inplace=True)
combined['enc_Utilities'] = pd.get_dummies(combined.Utilities, drop_first=True)

LotConfig_dummies = pd.get_dummies(combined['LotConfig'], prefix='LotConfig')
combined = pd.concat([combined,LotConfig_dummies], axis=1)
combined.drop('LotConfig', axis=1, inplace=True)

LandSlope_dummies = pd.get_dummies(combined['LandSlope'], prefix='LandSlope')
combined = pd.concat([combined,LandSlope_dummies], axis=1)
combined.drop('LandSlope', axis=1, inplace=True)

BldgType_dummies = pd.get_dummies(combined['BldgType'], prefix='BldgType')
combined = pd.concat([combined,BldgType_dummies], axis=1)
combined.drop('BldgType', axis=1, inplace=True)

HouseStyle_mapping = {
       '2.5Unf':1,
       '1.5Fin':1,
       '1.5Unf':1,
        'SFoyer': 1,
        'SLvl': 2,
       '1Story':2,
       '2.5Fin':3,
        '2Story':3,
}
combined['HouseStyle'] = combined.HouseStyle.map(HouseStyle_mapping)
HouseStyle_dummies = pd.get_dummies(combined['HouseStyle'], prefix='HouseStyle')
combined = pd.concat([combined,HouseStyle_dummies], axis=1)
combined.drop('HouseStyle', axis=1, inplace=True)


SaleType_mapping = {
        'WD':2,
        'CWD':2,
        'New':3,
        'COD':1,
        'Con':3,
       'ConLw':1,
       'ConLI':1,
       'ConLD':1,
        'Oth':1,
}
combined['SaleType'] = combined.SaleType.map(SaleType_mapping)
SaleType_dummies = pd.get_dummies(combined['SaleType'], prefix='SaleType')
combined = pd.concat([combined,SaleType_dummies], axis=1)
combined.drop('SaleType', axis=1, inplace=True)


RoofStyle_dummies = pd.get_dummies(combined['RoofStyle'], prefix='RoofStyle')
combined = pd.concat([combined,RoofStyle_dummies], axis=1)
combined.drop('RoofStyle', axis=1, inplace=True)

RoofMatl_dummies = pd.get_dummies(combined['RoofMatl'], prefix='RoofMatl')
combined = pd.concat([combined,RoofMatl_dummies], axis=1)
combined.drop('RoofMatl', axis=1, inplace=True)

#print(hp_train.groupby(['MSSubClass'])[['SalePrice']].agg(['mean','median','count']))
MSSubClass_mapping = {
          180 : 1,
          30 : 2,   45 : 2,
          190 : 3, 50 : 3, 90 : 3,
          85 : 4, 40 : 4, 160 : 4,
          70 : 5, 20 : 5, 75 : 5, 80 : 5, 150 : 5,
          120: 6, 60 : 6,

}
combined['MSSubClass'] = combined.MSSubClass.map(MSSubClass_mapping)
MSSubClass_dummies = pd.get_dummies(combined['MSSubClass'], prefix='MSSubClass')
combined = pd.concat([combined,MSSubClass_dummies], axis=1)
combined.drop('MSSubClass', axis=1, inplace=True)

Neighborhood_mapping = {
'MeadowV' :1, 'IDOTRR':1, 'BrDale' :1, 'OldTown' :1, 'Edwards'  :1, 'BrkSide'  :1,
'Sawyer'  :2,'Blueste' :2,'SWISU'  :2,'NAmes'  :2,'NPkVill' :2,'Mitchel':2,
'SawyerW':3,'Gilbert':3,'NWAmes':3,'Blmngtn':3,'CollgCr':3,
'ClearCr':4,'Crawfor':4,'Veenker':4,'Somerst':4,'Timber':4,
'StoneBr':5,      'NoRidge':5,      'NridgHt':5,
}
combined['Neighborhood'] = combined.Neighborhood.map(Neighborhood_mapping)
Neighborhood_dummies = pd.get_dummies(combined['Neighborhood'], prefix='Neighborhood')
combined = pd.concat([combined,Neighborhood_dummies], axis=1)
combined.drop('Neighborhood', axis=1, inplace=True)



Condition1_mapping = {
'Artery' :'one', 'Feedr':'one', 'RRAe' :'one', 'OldTown' :'one', 'Norm'  :'one', 'RRAn'  :'one',
'PosA'  :'two','PosN' :'two','RRNn'  :'two',
}
combined['Condition1'] = combined.Condition1.map(Condition1_mapping)
combined['enc_Condition1'] = pd.get_dummies(combined.Condition1, drop_first=True)
Condition2_mapping = {
'Artery' :'one', 'Feedr':'one', 'RRAe' :'one', 'OldTown' :'one', 'Norm'  :'one', 'RRAn'  :'one',
'PosA'  :'two','PosN' :'two','RRNn'  :'one',
}
combined['Condition2'] = combined.Condition2.map(Condition2_mapping)
combined['enc_Condition2'] = pd.get_dummies(combined.Condition2, drop_first=True)

ExterQual_dummies = pd.get_dummies(combined['ExterQual'], prefix='ExterQual')
combined = pd.concat([combined,ExterQual_dummies], axis=1)
combined.drop('ExterQual', axis=1, inplace=True)


ExterCond_dummies = pd.get_dummies(combined['ExterCond'], prefix='ExterCond')
combined = pd.concat([combined,ExterCond_dummies], axis=1)
combined.drop('ExterCond', axis=1, inplace=True)

HeatingQC_dummies = pd.get_dummies(combined['HeatingQC'], prefix='HeatingQC')
combined = pd.concat([combined,HeatingQC_dummies], axis=1)
combined.drop('HeatingQC', axis=1, inplace=True)

KitchenQual_dummies = pd.get_dummies(combined['KitchenQual'], prefix='KitchenQual')
combined = pd.concat([combined,KitchenQual_dummies], axis=1)
combined.drop('KitchenQual', axis=1, inplace=True)


Functional_dummies = pd.get_dummies(combined['Functional'], prefix='Functional')
combined = pd.concat([combined,Functional_dummies], axis=1)
combined.drop('Functional', axis=1, inplace=True)

PavedDrive_dummies = pd.get_dummies(combined['PavedDrive'], prefix='PavedDrive')
combined = pd.concat([combined,PavedDrive_dummies], axis=1)
combined.drop('PavedDrive', axis=1, inplace=True)

SaleCondition_dummies = pd.get_dummies(combined['SaleCondition'], prefix='SaleCondition')
combined = pd.concat([combined,SaleCondition_dummies], axis=1)
combined.drop('SaleCondition', axis=1, inplace=True)

combined['enc_CentralAir'] = pd.get_dummies(combined.CentralAir, drop_first=True)


Electrical_dummies = pd.get_dummies(combined['Electrical'], prefix='Electrical')
combined = pd.concat([combined,Electrical_dummies], axis=1)
combined.drop('Electrical', axis=1, inplace=True)

Heating_dummies = pd.get_dummies(combined['Heating'], prefix='Heating')
combined = pd.concat([combined,Heating_dummies], axis=1)
combined.drop('Heating', axis=1, inplace=True)
print(Heating_dummies)

print(hp_train.groupby(['MiscFeature'])[['SalePrice']].agg(['mean','median','count']))









