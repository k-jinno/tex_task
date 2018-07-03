# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 16:23:26 2018

@author: jinno
"""
#ランダムフォレスト
from sklearn.datasets import load_digits
#手書きデータの読み込み
digits=load_digits(return_X_y=True)
input,ground_truth = list(digits)
input2=[]
ground_truth2=[]
feature_names=["3","8"]
for i in range(len(input)):
    if(ground_truth[i] == 3 or ground_truth[i] == 8):
        input2.append(input[i])
        ground_truth2.append(ground_truth[i])
        
input,ground_truth = input2,ground_truth2

from sklearn.model_selection import train_test_split
input_train, input_test, ground_truth_train, ground_truth_test = train_test_split(
  input, ground_truth, test_size=0.33)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(input_train, ground_truth_train)

predicted_input = classifier.predict(input_test)
accuracy = classifier.score(input_test, ground_truth_test)
print(accuracy)

#可視化
import pydotplus as pdp
from IPython.display import Image
from graphviz import Digraph
from sklearn.externals.six import StringIO
from sklearn import tree
import pandas as pd
import numpy as np

#estimators = classifier.estimators_
#file_name = "./tree_visualization.png"
#dot_data = tree.export_graphviz(estimators[0], # 決定木オブジェクトを一つ指定する
#                                out_file=None, # ファイルは介さずにGraphvizにdot言語データを渡すのでNone
#                                filled=True, # Trueにすると、分岐の際にどちらのノードに多く分類されたのか色で示してくれる
#                                rounded=True, # Trueにすると、ノードの角を丸く描画する。
#                                #feature_names=features, # これを指定しないとチャート上で特徴量の名前が表示されない
#                                #class_names=digits.target, # これを指定しないとチャート上で分類名が表示されない
#                                special_characters=True # 特殊文字を扱えるようにする
#                                )
#graph = pdp.graph_from_dot_data(dot_data)
#graph.write_png(file_name)
i_tree = 0
for tree_in_forest in classifier.estimators_:
    file_name = "./tree_data/tree_visualization"+str(i_tree)+".pdf"
    dot_data = tree.export_graphviz(tree_in_forest, # 決定木オブジェクトを一つ指定する
                                out_file=None, # ファイルは介さずにGraphvizにdot言語データを渡すのでNone
                                filled=True, # Trueにすると、分岐の際にどちらのノードに多く分類されたのか色で示してくれる
                                rounded=True, # Trueにすると、ノードの角を丸く描画する。
                                #feature_names=features, # これを指定しないとチャート上で特徴量の名前が表示されない
                                class_names=feature_names, # これを指定しないとチャート上で分類名が表示されない
                                special_characters=True # 特殊文字を扱えるようにする
                                )
    graph = pdp.graph_from_dot_data(dot_data)
    graph.write_pdf(file_name)
    i_tree = i_tree + 1


#i_tree = 0
#for tree_in_forest in classifier.estimators_:
#    with open('tree_' + str(i_tree) + '.dot', 'w') as my_file:
#        dot_data = StringIO()
#        tree.export_graphviz(tree_in_forest, out_file=my_file, max_depth=3)
#        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#        graph.write_pdf("graph.pdf")
#        Image(graph.create_png())
#    i_tree = i_tree + 1
    
   