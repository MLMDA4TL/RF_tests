import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import graphviz 
from os import listdir
from os.path import join,isfile
import copy
import sys
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
sys.path.append("..")
from SER_rec import SER
from STRUT import STRUT

def STRUT_function(source_model, X_train, Y_train):
    STRUT(source_model, 0, X_train, Y_train, 1)

def SER_function(source_model, X_train, Y_train):
    SER(0, source_model, X_train,Y_train)

def plot_tree(clf,feature_names, class_names):
    dot_data = tree.export_graphviz(clf, out_file=None, 
                             feature_names=feature_names,  
                             class_names=class_names,  
                             filled=True,rounded=True,  
                             special_characters=True)  
    graph = graphviz.Source(dot_data)  
    return graph

def save_tree(graph,name):
    graph.render(name, view=False)

def full_evaluation(clf,X_test,Y_test):
    Y_pred = clf.predict(X_test)
    recall = recall_score(y_pred=Y_pred,y_true=Y_test,average='weighted')
    precision = precision_score(y_pred=Y_pred,y_true=Y_test,average='weighted')
    f1 = f1_score(y_pred=Y_pred,y_true=Y_test,average='weighted')
    accuracy = accuracy_score(y_pred=Y_pred,y_true=Y_test)
    return {"recall":recall,"precision":precision,"f1":f1,"accuracy":accuracy}



def test_transfer(source_model, target_set, size, features, transfer_function):
    # generate train set
    df_train = target_set.iloc[:size,:]
    X_train = df_train[features].values
    Y_train = df_train["cluster"].values
    # generate test set
    df_test = df.iloc[size:,:]
    X_test = df_test[features].values
    Y_test = df_test["cluster"].values
    # transfer
    target_model = copy.deepcopy(source_model)
    if transfer_function is not None:
        transfer_function(target_model, X_train, Y_train)
    return target_model,full_evaluation(target_model,X_test,Y_test)

def save_transfer_model(base_name,folder,model,features_names,classes_names):
    path = join(folder,base_name)
    pickle.dump(model, open(path+".pickle","wb"))
    try:
        graph = plot_tree(model,features_names, classes_names)
        save_tree(graph,path+"_graph.gv")
    except:
        print "graph error: "+ path

# Large Sample
########
# Source model
source_model = pickle.load(open("source_models/source_D10_C10_Projected_8-10_model.pickle","rb"))
# Target data
folder_target_datasets = "synthetic_datasets/target/"
# Evaluation DF
strut_results_large_sample = pd.DataFrame(columns=["recall","precision","f1","accuracy"])
ser_results_large_sample = pd.DataFrame(columns=["recall","precision","f1","accuracy"])
source_results_large_sample = pd.DataFrame(columns=["recall","precision","f1","accuracy"])
# param
training_set_size = 4000
# Saving folders
folder_target_models = "target_models/"
strut_folder = join(folder_target_models,"STRUT/large_sample/")
ser_folder = join(folder_target_models,"SER/large_sample/")


for fname in listdir(folder_target_datasets):
    print fname
    if ".csv" in fname and fname[0] != ".":
        path = join(folder_target_datasets,fname)
        base_name = fname.split(".")[0]
        if isfile(path):
            df = pd.read_csv(path,index_col=0,header=0)
            # STRUT
            target_model_strut, eval_strut = test_transfer(source_model,
                                                           df,
                                                           training_set_size,
                                                           list(map(str,range(10))),
                                                           STRUT_function)
            # SER
            target_model_ser, eval_ser = test_transfer(source_model,
                                                       df,
                                                       training_set_size,
                                                       list(map(str,range(10))),
                                                       SER_function)
            # Source
            target_model_source, eval_source = test_transfer(source_model,
                                                             df,
                                                             training_set_size,
                                                             list(map(str,range(10))),
                                                             None)
            # Save evaluations
            strut_results_large_sample.loc[fname] = eval_strut
            ser_results_large_sample.loc[fname] = eval_ser
            source_results_large_sample.loc[fname] = eval_source
            # Save results
            save_transfer_model(base_name,strut_folder,
                                target_model_strut,
                                ["f"+`i` for i in range(10)],
                                ["c"+`i` for i in range(10)])
            save_transfer_model(base_name,ser_folder,
                                target_model_ser,
                                ["f"+`i` for i in range(10)],
                                ["c"+`i` for i in range(10)])



# Small Sample
########
# Source model
source_model = pickle.load(open("source_models/source_D10_C10_Projected_8-10_model.pickle","rb"))
# Target data
folder_target_datasets = "synthetic_datasets/target/"
# Evaluation DF
strut_results_small_sample = pd.DataFrame(columns=["recall","precision","f1","accuracy"])
ser_results_small_sample = pd.DataFrame(columns=["recall","precision","f1","accuracy"])
source_results_small_sample = pd.DataFrame(columns=["recall","precision","f1","accuracy"])
# param
training_set_size = 50
# Saving folders
folder_target_models = "target_models/"
strut_folder = join(folder_target_models,"STRUT/small_sample/")
ser_folder = join(folder_target_models,"SER/small_sample/")


for fname in listdir(folder_target_datasets):
    print fname
    if ".csv" in fname and fname[0] != ".":
        path = join(folder_target_datasets,fname)
        base_name = fname.split(".")[0]
        if isfile(path):
            df = pd.read_csv(path,index_col=0,header=0)
            # STRUT
            target_model_strut, eval_strut = test_transfer(source_model,
                                                           df,
                                                           training_set_size,
                                                           list(map(str,range(10))),
                                                           STRUT_function)
            # SER
            target_model_ser, eval_ser = test_transfer(source_model,
                                                       df,
                                                       training_set_size,
                                                       list(map(str,range(10))),
                                                       SER_function)
            # Source
            target_model_source, eval_source = test_transfer(source_model,
                                                             df,
                                                             training_set_size,
                                                             list(map(str,range(10))),
                                                             None)
            # Save evaluations
            strut_results_small_sample.loc[fname] = eval_strut
            ser_results_small_sample.loc[fname] = eval_ser
            source_results_small_sample.loc[fname] = eval_source
            # Save results
            save_transfer_model(base_name,strut_folder,
                                target_model_strut,
                                ["f"+`i` for i in range(10)],
                                ["c"+`i` for i in range(10)])
            save_transfer_model(base_name,ser_folder,
                                target_model_ser,
                                ["f"+`i` for i in range(10)],
                                ["c"+`i` for i in range(10)])

strut_results_small_sample.to_csv("strut_results_small_sample.csv")
ser_results_small_sample.to_csv("ser_results_small_sample.csv")
source_results_small_sample.to_csv("source_results_small_sample.csv")

strut_results_large_sample.to_csv("strut_results_large_sample.csv")
ser_results_large_sample.to_csv("ser_results_large_sample.csv")
source_results_large_sample.to_csv("source_results_large_sample.csv")
