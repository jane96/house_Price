import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost.sklearn import  XGBRegressor
from sklearn.linear_model import Lasso
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
##column: MSSubClass MSZoning   LotArea Street LandContour Utilities HouseStyle OverallQual OverallCond YearBuilt YearRemodAdd
# ExterQual ExterCond BsmtQual BsmtCond BsmtExposure BsmtFinType1 BsmtFinType2 BsmtFinSF2 HeatingQC CentralAir KitchenQual
# GarageFinish GarageQual GarageCond

##data fill NA:BsmtQual BsmtCond BsmtExposure BsmtFinType1 BsmtFinType2  GarageFinish GarageQual GarageCond
#data normalize: MSSubClass  LotArea
#
all_str = ['MSSubClass','MSZoning','LotFrontage','LotArea','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
           'Condition1','BldgType','HouseStyle','OverallQual','OverallCond','YearBuilt','YearRemodAdd',
           'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation',
           'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF',
           'TotalBsmtSF','Heating','HeatingQC','CentralAir','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
           'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional',
           'Fireplaces','GarageType','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond',
           'PavedDrive','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','SaleType','SaleCondition']
filter_str = ['MSSubClass','MSZoning','Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood',
           'Condition1','BldgType','OverallQual','OverallCond',
           'RoofStyle','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation',
           'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF',
           'HeatingQC','CentralAir',
           'KitchenQual','Functional','GarageQual',
           'GarageType','GarageFinish','GarageCars','GarageCond',
           'PavedDrive','MiscVal','SaleType','SaleCondition']
rest_str = ['TotalArea','LotFrontage','LotArea','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
            'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea']


def data_fill(data):
    data.loc[data['MSZoning'].isnull(),'MSZoning'] = 'RH'
    data.loc[data['LotFrontage'].isnull(),'LotFrontage'] = 0

    data.loc[data['MasVnrType'].isnull(),'MasVnrType'] = 'null'
    data.loc[data['MasVnrArea'].isnull(),'MasVnrArea'] = 0

    data.loc[data['BsmtQual'].isnull(), 'BsmtQual'] = 0
    data.loc[data['BsmtCond'].isnull(), 'BsmtCond'] = 0
    data.loc[data['BsmtExposure'].isnull(), 'BsmtExposure'] = 0
    data.loc[data['BsmtFinType1'].isnull(), 'BsmtFinType1'] = 0
    data.loc[data['BsmtFinType2'].isnull(), 'BsmtFinType2'] = 0
    data.loc[data['GarageFinish'].isnull(), 'GarageFinish'] = 0
    data.loc[data['GarageQual'].isnull(), 'GarageQual'] = 'NA'
    data.loc[data['GarageCond'].isnull(), 'GarageCond'] = 'NA'
    data.loc[data['GarageCars'].isnull(), 'GarageCars'] = 0
    data.loc[data['GarageArea'].isnull(), 'GarageArea'] = 0
    data.loc[data['BsmtFinSF1'].isnull(), 'BsmtFinSF1'] = 0
    data.loc[data['BsmtFinSF2'].isnull(),'BsmtFinSF2'] = 0
    data.loc[data['TotalBsmtSF'].isnull(),'TotalBsmtSF'] = 0
    data.loc[data['KitchenQual'].isnull(),'KitchenQual'] = 'Gd'
    data.loc[data['GarageType'].isnull(),'GarageType'] = 'null'
    data.loc[data['GarageFinish'].isnull(),'GarageFinish'] = 'null'

    data.loc[data['BsmtFullBath'].isnull(), 'BsmtFullBath'] = 0
    data.loc[data['BsmtHalfBath'].isnull(), 'BsmtHalfBath'] = 0
    data.loc[data['Functional'].isnull(), 'Functional'] = 'Typ'
    data.loc[data['BsmtUnfSF'].isnull(), 'BsmtUnfSF'] = 0
    data.loc[data['SaleType'].isnull(), 'SaleType'] = 'Oth'


    data.loc[data['GarageQual'] == 'Ex','GarageQual'] = 'Gd'

    data['TotalArea'] = data['LotFrontage'] + data['LotArea'] + data['1stFlrSF'] + data['2ndFlrSF']

    # data.loc[data['GarageQual'] == 'Gd','GarageQual'] = 8
    # data.loc[data['GarageQual'] == 'TA','GarageQual'] = 6
    # data.loc[data['GarageQual'] == 'Fa','GarageQual'] = 4
    # data.loc[data['GarageQual'] == 'Po','GarageQual'] = 2
    # data.loc[data['GarageQual'] == 'NA','GarageQual'] = -1
    # data.loc[data['GarageQual'] == 'null','GarageQual'] = 0
    #
    # data.loc[data['ExterQual'] == 'Ex','ExterQual'] = 10
    # data.loc[data['ExterQual'] == 'Gd','ExterQual'] = 8
    # data.loc[data['ExterQual'] == 'TA','ExterQual'] = 6
    # data.loc[data['ExterQual'] == 'Fa','ExterQual'] = 4
    # data.loc[data['ExterQual'] == 'Po','ExterQual'] = 2
    #
    # data.loc[data['ExterCond'] == 'Ex', 'ExterCond'] = 10
    # data.loc[data['ExterCond'] == 'Gd', 'ExterCond'] = 8
    # data.loc[data['ExterCond'] == 'TA', 'ExterCond'] = 6
    # data.loc[data['ExterCond'] == 'Fa', 'ExterCond'] = 4
    # data.loc[data['ExterCond'] == 'Po', 'ExterCond'] = 2
    #
    # data.loc[data['BsmtQual'] == 'Ex', 'BsmtQual'] = 10
    # data.loc[data['BsmtQual'] == 'Gd', 'BsmtQual'] = 8
    # data.loc[data['BsmtQual'] == 'TA', 'BsmtQual'] = 6
    # data.loc[data['BsmtQual'] == 'Fa', 'BsmtQual'] = 4
    # data.loc[data['BsmtQual'] == 'Po', 'BsmtQual'] = 2
    # data.loc[data['BsmtQual'] == 'null', 'BsmtQual'] = 0
    # data.loc[data['BsmtQual'] == 'NA', 'BsmtQual'] = -1
    #
    # data.loc[data['BsmtCond'] == 'Ex', 'BsmtCond'] = 10
    # data.loc[data['BsmtCond'] == 'Gd', 'BsmtCond'] = 8
    # data.loc[data['BsmtCond'] == 'TA', 'BsmtCond'] = 6
    # data.loc[data['BsmtCond'] == 'Fa', 'BsmtCond'] = 4
    # data.loc[data['BsmtCond'] == 'Po', 'BsmtCond'] = 2
    # data.loc[data['BsmtQual'] == 'null', 'BsmtQual'] = 0
    # data.loc[data['BsmtCond'] == 'NA', 'BsmtCond'] = -1
    #
    # data.loc[data['BsmtExposure'] == 'Gd', 'BsmtExposure'] = 10
    # data.loc[data['BsmtExposure'] == 'Av', 'BsmtExposure'] = 8
    # data.loc[data['BsmtExposure'] == 'Mn', 'BsmtExposure'] = 6
    # data.loc[data['BsmtExposure'] == 'No', 'BsmtExposure'] = 4
    # data.loc[data['BsmtExposure'] == 'null', 'BsmtExposure'] = 0
    # data.loc[data['BsmtExposure'] == 'NA', 'BsmtExposure'] = -2
    #
    # data.loc[data['BsmtFinType1'] == 'GLQ', 'BsmtFinType1'] = 10
    # data.loc[data['BsmtFinType1'] == 'ALQ', 'BsmtFinType1'] = 8
    # data.loc[data['BsmtFinType1'] == 'BLQ', 'BsmtFinType1'] = 6
    # data.loc[data['BsmtFinType1'] == 'Rec', 'BsmtFinType1'] = 4
    # data.loc[data['BsmtFinType1'] == 'LwQ', 'BsmtFinType1'] = 2
    # data.loc[data['BsmtFinType1'] == 'Unf', 'BsmtFinType1'] = 0
    # data.loc[data['BsmtFinType1'] == 'null', 'BsmtFinType1'] = 0
    # data.loc[data['BsmtFinType1'] == 'NA', 'BsmtFinType1'] = -2
    #
    # data.loc[data['BsmtFinType2'] == 'GLQ', 'BsmtFinType2'] = 10
    # data.loc[data['BsmtFinType2'] == 'ALQ', 'BsmtFinType2'] = 8
    # data.loc[data['BsmtFinType2'] == 'BLQ', 'BsmtFinType2'] = 6
    # data.loc[data['BsmtFinType2'] == 'Rec', 'BsmtFinType2'] = 4
    # data.loc[data['BsmtFinType2'] == 'LwQ', 'BsmtFinType2'] = 2
    # data.loc[data['BsmtFinType2'] == 'Unf', 'BsmtFinType2'] = 0
    # data.loc[data['BsmtFinType2'] == 'null', 'BsmtFinType2'] = 0
    # data.loc[data['BsmtFinType2'] == 'NA', 'BsmtFinType2'] = -2
    #
    # data.loc[data['HeatingQC'] == 'Ex','HeatingQC'] = 10
    # data.loc[data['HeatingQC'] == 'Gd', 'HeatingQC'] = 8
    # data.loc[data['HeatingQC'] == 'TA', 'HeatingQC'] = 6
    # data.loc[data['HeatingQC'] == 'Fa', 'HeatingQC'] = 4
    # data.loc[data['HeatingQC'] == 'Po', 'HeatingQC'] = 2




    return data
def data_transform(data):
    return 3
def normal(data,name_str):
    # data[name_str] = (data[name_str] - data[name_str].min()) / (data[name_str].max() - data[name_str].min())
    return
def data_normalize(data,normal_list):
    for str in normal_list:
        normal(data,str)


    return data
def data_dummies(data,isTrain):
    data['MSSubClass'] = data['MSSubClass'].astype('str')

    data_filter = data[filter_str]

    data_dum = pd.get_dummies(data[filter_str])
    col_name = data_dum.columns.tolist()
    if 'MSSubClass_150' not in data:
        data_dum['MSSubClass_150'] = 0
        col_name.insert(col_name.index('MSSubClass_160'),'MSSubClass_150')
    data_dum = data_dum.sort_index(axis=1)
    print(data_dum.columns.tolist())

    data_final = pd.concat([data_dum,data[rest_str]],axis=1)
    if isTrain:
        data_final = pd.concat([data[['SalePrice']],data_final],axis=1)
    return data_final
def data_split(data,size):
    split_train, split_val = train_test_split(data, test_size=size)
    return split_train,split_val
def data_process(data,isTrain = True):
    data_f = data_fill(data)
    normal_list = ['LotArea','YearBuilt','YearRemodAdd','GarageQual','BsmtFinSF2']
    data_n = data_normalize(data_f,normal_list)
    data_d = data_dummies(data_n,isTrain)

    train,test = data_split(data_d,0.01)

    return train,test
def validation_process(data):
    data_f = data_fill(data)
    normal_list = ['LotArea', 'YearBuilt', 'YearRemodAdd', 'GarageQual','BsmtFinSF2']
    data_n = data_normalize(data_f,normal_list)
    data_d = data_dummies(data_n, False)

    return data_d
def alg_model(train,test):
    # l_svr = SVR(kernel="linear")
    # l_svr.fit(train.as_matrix()[:, 1:], train.as_matrix()[:, 0])
    # print("linear svm regression score is : {}".format(l_svr.score(test.as_matrix()[:, 1:], test.as_matrix()[:, 0])))
    #
    # r_svr = SVR(kernel="rbf")
    # r_svr.fit(train.as_matrix()[:, 1:], train.as_matrix()[:, 0])
    # print("rbf svm regression score is : {}".format(r_svr.score(test.as_matrix()[:, 1:], test.as_matrix()[:, 0])))

    # n_svr = SVR(kernel='poly')
    # n_svr.fit(train.as_matrix()[:,1:],train.as_matrix()[:,0])
    # print("poly svm regrssion score is : {}".format(n_svr.score(test.as_matrix()[:,1:],test.as_matrix()[:,0])))

    knn = KNeighborsRegressor(weights='uniform')

    knn.fit(train.as_matrix()[:, 1:], train.as_matrix()[:, 0])
    print("knn  regrssion score is : {}".format(
        knn.score(test.as_matrix()[:, 1:], test.as_matrix()[:, 0])))

    dt = DecisionTreeRegressor()
    dt.fit(train.as_matrix()[:, 1:], train.as_matrix()[:, 0])
    print("decision tree  regrssion score is : {}".format(
        dt.score(test.as_matrix()[:, 1:], test.as_matrix()[:, 0])))

    rfr = RandomForestRegressor()
    rfr.fit(train.as_matrix()[:, 1:], train.as_matrix()[:, 0])
    print("random forest   regrssion score is : {}".format(
        rfr.score(test.as_matrix()[:, 1:], test.as_matrix()[:, 0])))

    gbr = GradientBoostingRegressor()
    gbr.fit(train.as_matrix()[:, 1:], train.as_matrix()[:, 0])
    print("gradient boosting   regrssion score is : {}".format(
        gbr.score(test.as_matrix()[:, 1:], test.as_matrix()[:, 0])))

    ada = AdaBoostRegressor()
    ada.fit(train.as_matrix()[:, 1:], train.as_matrix()[:, 0])
    print("ada boosting   regrssion score is : {}".format(
        ada.score(test.as_matrix()[:, 1:], test.as_matrix()[:, 0])))

    xgb = XGBRegressor()
    xgb.fit(train.as_matrix()[:, 1:], train.as_matrix()[:, 0])
    print("XGBoosting   regrssion score is : {}".format(
        xgb.score(test.as_matrix()[:, 1:], test.as_matrix()[:, 0])))
    return [knn,dt,rfr,gbr,ada,xgb]
def run(origin,data,algs):
    cols = list(data)
    print("nan is : {} ".format(data.columns[np.where(np.isnan(data))[1]]))
    # for col in cols:
    #     if np.isnan(data[col]):
    #         print("nan is : {}".format(col))
    for alg in algs:
        prediction = alg.predict(data.as_matrix()[:,:])
        print("current alg is : {}".format(alg.__class__.__name__))
        result = pd.DataFrame(
            {'Id':origin['Id'].as_matrix(),'SalePrice':prediction.astype(np.float32)}
        )
        result.to_csv("~/dataset/housePrice/result/{}.csv".format(alg.__class__.__name__),index=False)
def correspondCoeff(data):
    lasso = Lasso(alpha = 0.001)
    lasso.fit(data.as_matrix()[:,1:],data.as_matrix()[:,0])
    fitt_lasso = pd.DataFrame({"Feature Importance": lasso.coef_},index=data.columns[1:])
    print(fitt_lasso.sort_values('Feature Importance',ascending=False))
    fitt_lasso[fitt_lasso['Feature Importance'] != 0].sort_values("Feature Importance").plot(kind='barh',figsize=(15,25))
    plt.xticks(rotation = 90)
    plt.show()
def train():
    origin = pd.read_csv("~/dataset/housePrice/originData/train.csv",engine='python')

    train,test = data_process(origin)
    # correspondCoeff(train)
    return train,alg_model(train,test)
def validation():
    data_train,algs = train()
    origin = pd.read_csv("~/dataset/housePrice/originData/test.csv",engine='python')
    validation = validation_process(origin)
    list1 = list(data_train)
    list2 = list(validation)
    for i in range(len(list1)-1):
        if list1[i+1] != list2[i]:
            print("i is {}; list1 is : {} ; list2 is : {}".format(i,list1[i+1],list2[i]))
    run(origin,validation,algs)
def draw():
    origin = pd.read_csv("~/dataset/housePrice/originData/train.csv", engine='python')
    column_list = list(origin)
    col = 1
    row =  int(len(column_list) / col) if (len(column_list) % col) == 0 else int(len(column_list) / col) + 1
    row = 1

    # fig,ax_list = plt.subplots(row,col,figsize=(50,50))
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # for i in range(row):
    #     for j in range(col):
    #         print("i is {};j is {}".format(i,j))
    #         if(i * col + j >= len(column_list)):
    #             break
    #         sns.barplot(x = column_list[i * col + j],y = 'SalePrice',data=origin,ax=ax_list[i][j])
    sns.barplot(x='Foundation', y='SalePrice', data=origin)
    plt.show()
    plt.savefig("all_attribute.png",dpi=500)
if __name__ == "__main__":
    validation()