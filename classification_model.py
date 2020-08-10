# -*- coding: utf-8 -*-

from numpy import *
import pandas
import matplotlib.pyplot as plt
import data_preprocessing as dp
#from sklearn.cross_validation import cross_val_score, cross_val_predict #??
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
#分类器选择
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn import neighbors 

status = 'after_process'
#status = 'before_process'

def import_dataset(status):
    '''
    ds_name: Name of the dataset 
    Returns:
    wls: Numpy ndarray: List of wavelength
    xdata: Pandas DataFrame: Measurements
    ydata: Pandas Series
    '''
    if status == "before_process":#数据预处理前直接分割
        oct_df = pandas.read_excel("plastic.xlsx")
        oct_df1 = oct_df.iloc[:32,:]
        oct_df2 = oct_df.iloc[32:68,:]
        oct_df3 = oct_df.iloc[68:81,:]
        oct_df4 = oct_df.iloc[81:99,:]
        xdata1,xdata2,xdata3,xdata4 = oct_df1.iloc[:,3:],oct_df2.iloc[:,3:],oct_df3.iloc[:,3:],oct_df4.iloc[:,3:]
        #print(xdata1,xdata2,xdata3,xdata4)
        ydata1 = oct_df1['class']
        ydata2 = oct_df2['class']
        ydata3 = oct_df3['class']
        ydata4 = oct_df4['class']
        X_train1, X_test1, y_train1, y_test1 = train_test_split(xdata1, ydata1,test_size=0.25)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(xdata2, ydata2,test_size=0.25)
        X_train3, X_test3, y_train3, y_test3 = train_test_split(xdata3, ydata3,test_size=0.25)
        X_train4, X_test4, y_train4, y_test4 = train_test_split(xdata4, ydata4,test_size=0.25)
        X_train= pandas.concat([X_train1,X_train2,X_train3,X_train4])
        X_test= pandas.concat([X_test1,X_test2,X_test3,X_test4])
        y_train= pandas.concat([y_train1,y_train2,y_train3,y_train4])
        y_test= pandas.concat([y_test1,y_test2,y_test3,y_test4])
        #print(len(y_train),len(y_test))
    elif status == 'after_process':#数据预处理，只对X_train的部分进行处理，test不变
        oct_df = pandas.read_excel("plastic.xlsx")
        wls = array([int(float(i)) for i in oct_df.columns.values[3:]])
        oct_df1 = oct_df.iloc[:32,:]
        oct_df2 = oct_df.iloc[32:68,:]
        oct_df3 = oct_df.iloc[68:81,:]
        oct_df4 = oct_df.iloc[81:99,:]
        xdata1,xdata2,xdata3,xdata4 = oct_df1.iloc[:,3:],oct_df2.iloc[:,3:],oct_df3.iloc[:,3:],oct_df4.iloc[:,3:]
        #print(xdata1,xdata2,xdata3,xdata4)
        ydata1 = oct_df1['class']
        ydata2 = oct_df2['class']
        ydata3 = oct_df3['class']
        ydata4 = oct_df4['class']
        X_train1, X_test1, y_train1, y_test1 = train_test_split(xdata1, ydata1,test_size=0.25)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(xdata2, ydata2,test_size=0.25)
        X_train3, X_test3, y_train3, y_test3 = train_test_split(xdata3, ydata3,test_size=0.25)
        X_train4, X_test4, y_train4, y_test4 = train_test_split(xdata4, ydata4,test_size=0.25)
        X_train0= pandas.concat([X_train1,X_train2,X_train3,X_train4])
        X_test0= pandas.concat([X_test1,X_test2,X_test3,X_test4])
        y_train= pandas.concat([y_train1,y_train2,y_train3,y_train4])
        y_test= pandas.concat([y_test1,y_test2,y_test3,y_test4])
        #X_train预处理
        #X_train = dp.do_D1(wls,X_train0,68)
        #X_train = dp.do_standardize(wls,X_train0,70)
        #X_train = dp.do_mean(wls,X_train0,70)
        X_train = dp.do_msc(wls,X_train0,70)
        #X_train = dp.do_snv(wls,X_train0,70)
        #X_train = dp.do_sg(wls,X_train0,70)
        #X_train = dp.do_D2(wls,X_train0,70)
        # pca = PCA(n_components=15)
        # X_train = pca.fit_transform(X_train0)
        # X_train = pandas.DataFrame(X_train)
        X_train.index = X_train0.index
        #X_test预处理
        #X_test = dp.do_D1(wls,X_test0,20)
        #X_test = dp.do_standardize(wls,X_test0,30) 
        #X_test = dp.do_mean(wls,X_test0,30)
        X_test = dp.do_msc(wls,X_test0,30)
        #X_test = dp.do_snv(wls,X_test0,30)
        #X_test = dp.do_sg(wls,X_test0,30)
        #X_test = dp.do_D2(wls,X_test0,30)
        # pca = PCA(n_components=15)
        # X_test = pca.fit_transform(X_test0)
        # X_test = pandas.DataFrame(X_test)
        X_test.index = X_test0.index
        X_train = X_train.iloc[:,1:]
        X_test = X_test.iloc[:,1:]
    return X_train, X_test, y_train, y_test
    

def knn(X_train, X_test, y_train, y_test):
    knn = neighbors.KNeighborsClassifier(3)
    knn_model = knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    knn_score = knn.score(X_test,y_test)
    print("测试集的预测：\n{}".format(y_pred))
    print("testing data accuracy(knn):\n{}".format(knn_score))
    
    X_train, X_test, y_train, y_test = X_train.values, X_test.values,y_train.values, y_test.values
	
	#画图
    # plt.figure()
    # plt.subplot(121)
    # plt.scatter(X_test[:, 0], X_test[:, -1], c=y_test.reshape((-1)), edgecolors='k',s=50)
    # plt.subplot(122)
    # plt.scatter(X_test[:, 0], X_test[:, -1], c=y_pred.reshape((-1)), edgecolors='k',s=50)
    # plt.show()#左图为测试数据，右图为对测试数据的预测。
    return knn_score

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print('%s accuracy:%.3f' %(tip, mean(acc)))
    return mean(acc)

    
def svm(X_train, X_test, y_train, y_test):
    clf = SVC(C=256,                         #误差项惩罚系数,默认值是1
                  kernel='linear',               #线性核比较好
                  decision_function_shape='ovo') #决策函数 一对一比较好 
    clf.fit(X_train,         #训练集特征向量
            y_train.values.ravel()) #训练集目标值
    
    # 原始结果与预测结果进行对比   predict()表示对x_train样本进行预测，返回样本类别
    #show_accuracy(clf.predict(X_train), y_train, 'training data')
    svm_score = show_accuracy(clf.predict(X_test), y_test.values, 'testing data')
    # # 计算决策函数的值，表示x到各分割平面的距离
    # print('decision_function:\n', clf.decision_function(X_train))
    X_train, X_test, y_train, y_test = X_train.values, X_test.values,y_train.values, y_test.values
    print(clf.predict(X_test),y_test)
    
	#画图
    # plt.figure()
    # plt.subplot(121)
    # plt.scatter(X_test[:, 0], X_test[:, -1], c=y_test.reshape((-1)), edgecolors='k',s=50)
    # plt.subplot(122)
    # plt.scatter(X_test[:, 0], X_test[:, -1], c=clf.predict(X_test).reshape((-1)), edgecolors='k',s=50)
    # plt.show()#左图为测试数据，右图为对测试数据的预测。
    return svm_score


if __name__ == "__main__":
    cv = 20
    svm_cv = 0
    for i in range(cv):
         #========================
        X_train, X_test, y_train, y_test = import_same_dataset(status)
        #X_train, X_test, y_train, y_test = import_dataset(status)
        print(X_train, X_test, type(y_train), y_test)
        print('--------------------SVM:k = {}------------------------'.format(i))
        #print(y_test.values)
        svm_score = svm(X_train, X_test, y_train, y_test)
        svm_cv += svm_score
    print('SVM交叉验证后测试集准确率为：',svm_cv/cv)
    
    knn_cv = 0
    for i in range(cv):
         #========================
        X_train, X_test, y_train, y_test = import_same_dataset(status)
        #print(X_train, X_test, y_train, y_test)
        print('--------------------KNN:k = {}------------------------'.format(i))
        knn_score = knn(X_train, X_test, y_train, y_test)
        knn_cv += knn_score
    print('KNN交叉验证后测试集准确率为：',knn_cv/cv)
    
    
    