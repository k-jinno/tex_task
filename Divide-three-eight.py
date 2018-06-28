# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:50:37 2018

@author: jinno
"""

import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np

#手書きデータの読み込み
digits = datasets.load_digits()

#3と8のデータ位置を求める
flag_3_8 = (digits.target == 3) + (digits.target == 8)

def twogroupfunc(flag_3_8):
    #3と8のデータを取得
    images = digits.images[flag_3_8]
    print(images)
    labels = digits.target[flag_3_8]
    
    n_samples = len(flag_3_8[flag_3_8])
    train_size = int(n_samples * 2 / 3)
    train_images = images[:train_size]
    train_labels = images[:train_size]
    train_images_3 = []
    train_images_8 = []
    #3と8のデータを分離
    for i in range(len(train_images)):
        if labels[i]==3:
            train_images_3.append(train_images[i])
        else :
            train_images_8.append(train_images[i]) 
            
    n_3 = len(train_images_3[0])
    n_8 = len(train_images_8[0])
    #各行列の平均ベクトル
    three_ave = np.average(train_images_3,axis = 1)
    eight_ave = np.average(train_images_8,axis = 1)
    
    #各行列の共分散行列
    S_3 = np.cov(train_images_3, rowvar=1, bias=0)
    S_8 = np.cov(train_images_8, rowvar=1, bias=0)
    
    #プールされた共分散行列を求める
    S_pl = ((n_3 - 1) * S_3 + (n_8 - 1) * S_8)/(n_3 + n_8 - 2)
    print("S_pl =",S_pl)
    
    #プールされた共分散行列の逆行列を求める
    inv_S_pl =  np.linalg.inv(S_pl)
    
    #答えを求める
    a = np.dot(inv_S_pl,three_ave - eight_ave) 
    print("a=",a)
    
    #判別する！！
    for i in range(len(images[train_size:])):
        print(np.dot(np.dot(three_ave - eight_ave,S_pl),images[i]-(three_ave + eight_ave)))
#        if np.dot(np.dot(three_ave - eight_ave,S_pl),images[i]-(three_ave + eight_ave))>0:
#            print(3)
#        else:
#            print(8)
            
twogroupfunc(flag_3_8)            
    
    
    
   


    
    



