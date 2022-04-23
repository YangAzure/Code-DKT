'''Author: Yang Shi yshi26@ncsu.edu'''
import math
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

main_df = pd.read_csv("../data/MainTable.csv")

main_df = main_df[main_df["EventType"] == "Run.Program"]
main_df = main_df[main_df["AssignmentID"] == 439]

students = pd.unique(main_df["SubjectID"])

problems = pd.unique(main_df["ProblemID"])
problems_d = {k:v for (v,k) in enumerate(problems) }

d = {}
for s in students:
    d[s] = {}
    df = main_df[main_df["SubjectID"] == s]
    d[s]["length"] = len(df)
    d[s]["Problems"] = [str(problems_d[i]) for i in df["ProblemID"]]
    d[s]["Result"] = list((df["Score"]==1).astype(int).astype(str))
    d[s]["CodeStates"] = list(df["CodeStateID"])
    
train_val_s, test_s = train_test_split(students, test_size=0.2, random_state=1)

np.save("../data/training_students.npy", train_val_s)
np.save("../data/testing_students.npy", test_s)

if not os.path.isdir("../data/DKTFeatures"):
    os.mkdir("../data/DKTFeatures")

file_test = open("../data/DKTFeatures/test_data.csv","w")
for s in test_s:
    if d[s]['length']>0:
        file_test.write(str(d[s]['length']))
        file_test.write(",\n")
        file_test.write(",".join(d[s]['CodeStates']))
        file_test.write(",\n")
        file_test.write(",".join(d[s]['Problems']))
        file_test.write(",\n")
        file_test.write(",".join(d[s]['Result']))
        file_test.write(",\n")
        
for fold in range(100):
    train_s, val_s = train_test_split(train_val_s, test_size=0.25, random_state=fold)

    file_train = open("../data/DKTFeatures/train_firstatt_"+str(fold)+".csv","w")
    for s in train_s:
        if d[s]['length']>0:
            file_train.write(str(d[s]['length']))
            file_train.write(",\n")
            file_train.write(",".join(d[s]['CodeStates']))
            file_train.write(",\n")
            file_train.write(",".join(d[s]['Problems']))
            file_train.write(",\n")
            file_train.write(",".join(d[s]['Result']))
            file_train.write(",\n")


    file_val = open("../data/DKTFeatures/val_firstatt_"+str(fold)+".csv","w")
    for s in val_s:
        if d[s]['length']>0:
            file_val.write(str(d[s]['length']))
            file_val.write(",\n")
            file_val.write(",".join(d[s]['CodeStates']))
            file_val.write(",\n")
            file_val.write(",".join(d[s]['Problems']))
            file_val.write(",\n")
            file_val.write(",".join(d[s]['Result']))
            file_val.write(",\n")