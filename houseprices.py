import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier,GradientBoostingRegressor,AdaBoostClassifier
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
import sys
sys.path.append("preprocessdata")
import preprocessdata as pre

"""
numeric_features = combined.select_dtypes(include=[np.number])
print(numeric_features.dtypes)
"""
train = pre.combined.iloc[:1458]
test = pre.combined.iloc[1458:]

predictors=['MSSubClass_1','MSSubClass_2','MSSubClass_3','MSSubClass_4','MSSubClass_5','MSSubClass_6',
            'MSZoning_C (all)','MSZoning_FV','MSZoning_RH','MSZoning_RL','MSZoning_RM','LotArea','enc_Street','enc_LotShape',
            'LandContour_Bnk','LandContour_HLS','LandContour_Low','LandContour_Lvl',
            'enc_Utilities','LotConfig_Corner','LotConfig_CulDSac','LotConfig_FR2','LotConfig_FR3','LotConfig_Inside',
            'LandSlope_Gtl', 'LandSlope_Mod', 'LandSlope_Sev','Neighborhood_1','Neighborhood_2','Neighborhood_3','Neighborhood_4','Neighborhood_5',
            'enc_Condition1','enc_Condition2',
            'BldgType_1Fam', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE',
            'HouseStyle_1', 'HouseStyle_2', 'HouseStyle_3',
            'OverallQual','OverallCond', 'YearBuilt','YearRemodAdd',
            'RoofStyle_Flat', 'RoofStyle_Gable', 'RoofStyle_Gambrel', 'RoofStyle_Hip', 'RoofStyle_Mansard','RoofStyle_Shed',
            'RoofMatl_CompShg', 'RoofMatl_Membran', 'RoofMatl_Metal', 'RoofMatl_Roll', 'RoofMatl_Tar&Grv','RoofMatl_WdShake','RoofMatl_WdShngl',
            'ExterQual_Ex','ExterQual_Fa','ExterQual_Gd','ExterQual_TA','ExterCond_Ex','ExterCond_Fa','ExterCond_Gd','ExterCond_Po','ExterCond_TA',
            'BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating_Floor','Heating_GasA','Heating_GasW','Heating_Grav','Heating_OthW','Heating_Wall',
            'HeatingQC_Ex','HeatingQC_Fa','HeatingQC_Gd','HeatingQC_Po','HeatingQC_TA','enc_CentralAir','Electrical_FuseA','Electrical_FuseF','Electrical_FuseP','Electrical_Mix','Electrical_SBrkr',
            '1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
            'FullBath','HalfBath','BsmtFullBath','BsmtHalfBath',
            'BedroomAbvGr','KitchenAbvGr','KitchenQual_Ex','KitchenQual_Fa','KitchenQual_Gd','KitchenQual_TA',
            'TotRmsAbvGrd','Functional_Maj1','Functional_Maj2','Functional_Min1','Functional_Min2','Functional_Mod','Functional_Sev','Functional_Typ',
            'Fireplaces','GarageCars','GarageArea','PavedDrive_N','PavedDrive_P','PavedDrive_Y',
            'WoodDeckSF','enc_OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',
            'SaleType_1.0','SaleType_2.0','SaleType_3.0',
            'SaleCondition_Abnorml','SaleCondition_AdjLand','SaleCondition_Alloca','SaleCondition_Family','SaleCondition_Normal','SaleCondition_Partial',
            'MoSold_four','MoSold_one','MoSold_three', 'MoSold_two','MiscVal','YrSold',

           ]


x_train, x_test, y_train, y_test = train_test_split(train[predictors], pre.target, random_state=42, test_size=.2)



#lr = linear_model.LinearRegression()
#lr=GradientBoostingRegressor()

algorithms=[
linear_model.LinearRegression(),
#linear_model.RidgeCV(alphas=np.logspace(-3, 2, 100)),
XGBRegressor(max_depth=5),
]

full_predictions = []
for lr in algorithms:
    model = lr.fit(x_train, y_train)
    predictions=model.predict(x_test)
    full_predictions.append(predictions)
predictions = (full_predictions[0]*0.54 + full_predictions[1]*0.46)
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

full_predictions = []
for lr in algorithms:
    model = lr.fit(train[predictors], pre.target)
    predictions=model.predict(test[predictors])
    full_predictions.append(predictions)
predictions = (full_predictions[0]*0.54 + full_predictions[1]*0.46)
predictions = np.exp(predictions)
submission = pd.DataFrame({
        "Id": pre.hp_test["Id"],
        "SalePrice": predictions
    })
print(submission)
submission.to_csv('/Users/martin_yan/Desktop/submission6.csv', index=False)
