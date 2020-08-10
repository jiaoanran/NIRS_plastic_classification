# -*- coding: utf-8 -*-

import numpy as np  
import matplotlib.pyplot as plt  
import pandas
  
def plotSpectrum(wls,xdata,num):  
    for i in range(num):
        #print(xdata.iloc[i].head)
        xdata.iloc[i].plot(x = wls)
        plt.xticks(wls[::1000])
        plt.legend("", loc ='upper left', ncol = 3)  
        plt.xlabel('Wavenum', fontsize=8)
        plt.ylabel('absorbance', fontsize=8 ) 
        plt.grid(True)
    return plt  

def mean_centralization(sdata):  
    temp1 = np.mean(sdata, axis=0)  
    temp2 = np.tile(temp1, sdata.shape[0]).reshape((sdata.shape[0], sdata.shape[1]))  #重复
    return sdata - temp2  

def standardize(wls,sdata):  
    from sklearn import preprocessing  
    return pandas.DataFrame(preprocessing.scale(sdata),columns = wls)

def msc(wls,xdata): 
    sdata = xdata.values 
    n = sdata.shape[0]  # 样本数量  
    k = np.zeros(sdata.shape[0])  
    b = np.zeros(sdata.shape[0])  
  
    M = np.mean(sdata, axis=0)  
  
    from sklearn.linear_model import LinearRegression  
    for i in range(n):  
        y = sdata[i, :]  
        y = y.reshape(-1, 1)  
        M = M.reshape(-1, 1)  
        model = LinearRegression()  
        model.fit(M, y)  
        k[i] = model.coef_  
        b[i] = model.intercept_  
  
    spec_msc = np.zeros_like(sdata)  
    for i in range(n):  
        bb = np.repeat(b[i], sdata.shape[1])  
        kk = np.repeat(k[i], sdata.shape[1])  
        temp = (sdata[i, :] - bb)/kk  
        spec_msc[i, :] = temp  
    return pandas.DataFrame(spec_msc,columns = wls)

def snv(wls,sdata):
    sdata = sdata.values 
    temp1 = np.mean(sdata, axis=1)  
    temp2 = np.tile(temp1, sdata.shape[1]).reshape((sdata.shape[0], sdata.shape[1]))  
    temp3 = np.std(sdata, axis=1)  
    temp4 = np.tile(temp3, sdata.shape[1]).reshape((sdata.shape[0], sdata.shape[1]))  
    return pandas.DataFrame((sdata - temp2)/temp4,columns = wls)  

def D1(wls,sdata):   
    temp1 = sdata
    temp2 = temp1.diff(axis=1)  
    return temp2

def D2(wls,sdata):  
    temp2 = (sdata).diff(axis=1) 
    temp4 = temp2.diff(axis=1)  
    return temp4

def plot_original(wls,xdata,num):
    #原始光谱
    origin_fig = plotSpectrum(wls,xdata,num) 
    origin_fig.rcParams['font.sans-serif']=['SimHei']
    origin_fig.rcParams['axes.unicode_minus'] = False
    origin_fig.title("初始光谱图")
    origin_fig.show()

def plot_mean(wls,xdata,num):
    #数据中心化
    x= mean_centralization(xdata)
    mean_fig = plotSpectrum(wls,x,num) 
    plt.title("mean centralization")
    mean_fig.show()
    return x

def do_mean(wls,xdata,num):
    #数据中心化
    x= mean_centralization(xdata)
    return x

def plot_standardize(wls,xdata,num):
    #数据标准化
    x= standardize(wls,xdata)
    standard_fig = plotSpectrum(wls,x,num) 
    plt.title("standardize")
    standard_fig.show()
    return x

def do_standardize(wls,xdata,num):
    #数据标准化
    x= standardize(wls,xdata)
    return x
    
def plot_sg(wls,xdata,num):
    #S-G平滑化
    from scipy.signal import savgol_filter
    x = pandas.DataFrame(savgol_filter(xdata, axis=0, window_length=7, polyorder=3),columns = wls)
    sg_fig = plotSpectrum(wls,x,num)
    plt.title("S-G")
    sg_fig.show() 
    return x

def do_sg(wls,xdata,num):
    #S-G平滑化
    from scipy.signal import savgol_filter
    x = pandas.DataFrame(savgol_filter(xdata, axis=0, window_length=7, polyorder=3),columns = wls)
    return x
    
def plot_msc(wls,xdata,num):
     #多元散射处理
    x = msc(wls,xdata)
    msc_fig = plotSpectrum(wls,x,num)  
    msc_fig.rcParams['font.sans-serif']=['SimHei']
    msc_fig.rcParams['axes.unicode_minus'] = False
    msc_fig.title("msc处理后光谱图")
    msc_fig.show() 
    return x

def do_msc(wls,xdata,num):
     #多元散射处理
    x = msc(wls,xdata)
    return x

def plot_snv(wls,xdata,num):
    #标准正态变换
    x = snv(wls,xdata)
    snv_fig = plotSpectrum(wls,x,num)  
    plt.title("snv")
    snv_fig.show()
    return x

def do_snv(wls,xdata,num):
    #标准正态变换
    x = snv(wls,xdata)
    return x

def plot_D1(wls,xdata,num):
    #一阶差分处理
    x = D1(wls,xdata)
    D1_fig = plotSpectrum(wls,x,num)  
    plt.title("D1")
    D1_fig.show() 
    return x

def do_D1(wls,xdata,num):
    #一阶差分处理
    x = D1(wls,xdata)
    return x
    
def plot_D2(wls,xdata,num):
    #二阶差分处理
    x = D2(wls,xdata)
    D2_fig = plotSpectrum(wls,x,num)  
    plt.title("D2")
    D2_fig.show() 
    return x

def do_D2(wls,xdata,num):
    #二阶差分处理
    x = D2(wls,xdata)
    return x

def main():
    FILENAME = "plastic.xlsx" #文件名
    oct_df = pandas.read_excel(FILENAME)
    #print(oct_df.head())
    wls = np.array([ int(float(i)) for i in oct_df.columns.values[3:]]) #波数从第几列开始
    xdata = oct_df.iloc[:,3:]
    xdata.columns = wls
    print(np.array([ int(float(i)) for i in oct_df.iloc[:,1:2].values]))
    xdata.index = np.array([ int(float(i)) for i in oct_df.iloc[:,1:2].values])
    #print(wls) #波数 [12000 11998 11996 ...  4004  4002  4000]
    print(xdata) #68组有效数据吸光度与波数表

    num =98
    #plot_original(wls,xdata,num)
    #x1 = plot_mean(wls,xdata,num)
    #x2 = plot_standardize(wls,xdata,num)
    #x3 = plot_sg(wls,xdata,num)
    # x4 = plot_msc(wls,x3,num)
    # x5 = plot_snv(wls,xdata,num)
    x6 = plot_D1(wls,xdata,num)
    # x7 = plot_D2(wls,xdata,num)
    #将数据导出
    #print(x3)
    #x2.to_excel('standardlize.xlsx',encoding='utf-8', index=True, header=True)
    
    
if __name__ == '__main__':
    main()
    # FILENAME = "plastic.xlsx"
    # oct_df = pandas.read_excel(FILENAME,header = None)
    # from sklearn import preprocessing  
    # x2 = pandas.DataFrame(preprocessing.scale(sdata))
    # print(x2)
    # x2.to_excel('standardlize2.xlsx',encoding='utf-8', index=True, header=True)
