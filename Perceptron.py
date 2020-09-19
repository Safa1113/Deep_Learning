# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 20:57:34 2020

@author: Safofoh
"""

import numpy as np

def extract(typ = "train"):

    if (typ == "test"):
        f = open('data/test.data', "r")
    else:
        f = open('data/train.data', "r")
    

    lines = f.readlines()

    f.close()
    
    dataset = np.zeros((len(lines),5))
    
    for row in range(len(lines)):
        data = lines[row].split(",")
        for clmn in range(5):
            if data[clmn] == "class-1\n":
                dataset[row][clmn] = -1
            elif data[clmn] == "class-2\n":
                dataset[row][clmn] = 0
            elif data[clmn] == "class-3\n":
                dataset[row][clmn] = 1
            else:
                dataset[row][clmn] = float(data[clmn])
    #print(len(lines))
    #print(lines)
    return dataset


def perceptronTrain(dataset, MaxIter, Regulization=False, Coefficient=0.01):
    
    accuracy = np.zeros((MaxIter))
    #w contains the bias term too
    w = np.zeros((1,dataset.shape[1]))
    #print (w.shape)
    #
    #b = np.zeros((dataset.shape[0], dataset.shape[1] - 1))
    #adding one column to dataset to introduce one more instance for the weight
    db= np.ones((dataset.shape[0],1))
    x = dataset[:,:-1]
    #print(x.shape)
    #print(db.shape)
    x = np.append(x, db, axis = 1)
    #print (x)
    y = dataset[:,-1]
    #print (y)
    #print (b.shape)
    #print (dataset[:,:-1].shape)
    for i in range(MaxIter):
        
        #print (y.shape)
        
         
        #print (a.shape)
        wrongPredictions = 0
#        print("next iteration")
        for row in range(x.shape[0]):
            a=0
            for clmn in range(x.shape[1]):
                a += w[0][clmn] * x[row][clmn]
            #if (clss == 12):
                #print(y[row])
                #print(a[row][clmn])
                #print()
                #condition = y[row]*a < 0
            #if (clss == 23):
                #condition = y[row]*a < 1
            #if (clss == 13):
            if (Regulization):
                l2 = w[0][clmn]*w[0][clmn]*Coefficient
                a += l2
                a= y[row]*a
            else:
                a= y[row]*a
            condition = a <= 0
            #print (y[row]*a)
            if (condition):
                
                for clmn in range(x.shape[1]):
                    #if (Regulization):
                        #print("000000000000000000000000000000000000000000000000000000")
                        #print(w[0][clmn])
#                        print(type( w[0][clmn]))
                        
                        #w[0][clmn] = w[0][clmn] * w[0][clmn]
                        
                        #w[0][clmn] = w[0][clmn]* Coefficient * w[0][clmn]
                        ##w[0][clmn] +=   y[row] * x[row][clmn]
                       # l2 = 
                       # w[0][clmn] =  w[0][clmn] +  Coefficient * (a - y[row] * x[row][clmn])
                    #else:
                
                    w[0][clmn] = w[0][clmn] + y[row] * x[row][clmn]
#                    print("--------yyyyyyyyyyy--------")
#                    print(y[row])
#                    print("-------xxxxxxxxxxxxx-------")
#                    print(x[row][clmn])
#                    print("-----wwwwwwwwwwwwww------")
#                    print(w[0][clmn])
#                    print("end of row")
                    #b[row,clmn] = b[row,clmn] + y[row]
                wrongPredictions +=1
        correct_answers = dataset.shape[0] - wrongPredictions
        accuracy[i]=  correct_answers * 100 / dataset.shape[0]
    
  
    return w, accuracy


def perceptronTest (w,dataset):
    accuracy = 0
    
    db= np.ones((dataset.shape[0],1))
    x = dataset[:,:-1]
    #print(x.shape)
    #print(db.shape)
    x = np.append(x, db, axis = 1)
    

    a = np.ones((dataset.shape[0]))
#    print(a.shape)
#    print(w.shape)
#    print(x.shape)
    
    for row in range(x.shape[0]):
        for clmn in range(x.shape[1]):
            
            a[row] += w[0][clmn] * x[row][clmn]
    
    for i in range(dataset.shape[0]):
        if np.sign(dataset[i][-1]) == np.sign(a[i]):
            #print(dataset[i][-1])
            #print(a[i])
            accuracy += 1
        
    accuracy = accuracy * 100 / dataset.shape[0]
    return np.sign(a), accuracy

def getClss1Clss2 (dataset):
    ls = []
  
    #print (dataset.shape)
    for row in range(dataset.shape[0]):
        #print (dataset.shape)
        if dataset[row,-1]== 1:
            ls.append(row)
    dataset = np.delete(dataset, ls, 0)
    
    for i in range(dataset.shape[0]):
        if dataset[i,-1]== 0:
            dataset[i,-1] = 1
#    print("class 12")
#    print(dataset)
    return dataset

def getClss2Clss3 (dataset):
    ls = []
  
    #print (dataset.shape)
    for row in range(dataset.shape[0]):
        #print (dataset.shape)
        if dataset[row,-1]== -1:
            ls.append(row)
    dataset = np.delete(dataset, ls, 0)
    
    for i in range(dataset.shape[0]):
        if dataset[i,-1]== 0:
            dataset[i,-1] = -1
#    print("class 23")
#    print(dataset)
    return dataset

def getClss1Clss3 (dataset):
    ls = []
  
    #print (dataset.shape)
    for row in range(dataset.shape[0]):
        #print (dataset.shape)
        if dataset[row,-1]== 0:
            ls.append(row)
    dataset = np.delete(dataset, ls, 0)
    
    
    
#    print("class 13")
#    print(dataset)
    return dataset

def printAcc (typ, acc, multi=False):
    if(multi):
        if (typ == "12"):
            st = "Accuracy for class 1"
        if (typ == "23"):
            st = "Accuracy for class 2"        
        if (typ == "13"):
            st = "Accuracy for class 3"  
    else:
        
        if (typ == "12"):
            st = "Accuracy for class 1 and class 2"
        if (typ == "23"):
            st = "Accuracy for class 2 and class 3"        
        if (typ == "13"):
            st = "Accuracy for class 1 and class 3"        
   
    for i in range(MaxIter):
        st = st + "\nIter: {} Accuracy: {}".format(i, acc[i])
    print (st)


def printAccTest (typ, acc):
    if (typ == "12"):
        st = "Accuracy for class 1 and class 2"
    if (typ == "23"):
        st = "Accuracy for class 2 and class 3"        
    if (typ == "13"):
        st = "Accuracy for class 1 and class 3"        
    
    st = st + "\nAccuracy: {}".format(acc)
    print (st)












def perceptronTrainMultiClass(dataset, test, MaxIter, Regulization=False, Coefficient=0.01):
    
    clss1dataset = getMultiClassifierData(dataset,1)
    clss2dataset = getMultiClassifierData(dataset,2)
    clss3dataset = getMultiClassifierData(dataset,3)
    
    w1, acc1 = perceptronTrain(clss1dataset, MaxIter,Regulization, Coefficient)
    w2, acc2 = perceptronTrain(clss2dataset, MaxIter,Regulization, Coefficient)
    w3, acc3 = perceptronTrain(clss3dataset, MaxIter,Regulization, Coefficient)
    
    printAcc("12", acc1,multi=True)
    printAcc("23", acc2,multi=True)
    printAcc("13", acc3,multi=True)
    
    
    clss1dataset = getMultiClassifierData(test,1)
    clss2dataset = getMultiClassifierData(test,2)
    clss3dataset = getMultiClassifierData(test,3)
    
    a1, ac1 = perceptronTest(w1, clss1dataset)
    a2, ac2 = perceptronTest(w2, clss2dataset)
    a3, ac3 = perceptronTest(w3, clss3dataset)
    
    printAccTest("12", ac1)
    printAccTest("23", ac2)
    printAccTest("13", ac3)
    
    predicted = []
    for i in range(test.shape[0]):
#        a1[i] = a1[i] * 100 / (a1[i]+a2[i]+a3[i])
#        a2[i] = a2[i] * 100 / (a1[i]+a2[i]+a3[i])
#        a3[i] = a3[i] * 100 / (a1[i]+a2[i]+a3[i])
        
        if (a1[i] == max(a1[i],a2[i],a3[i])):
            predicted.append(-1) 
        if (a2[i] == max(a1[i],a2[i],a3[i])):
            predicted.append(0) 
        if (a3[i] == max(a1[i],a2[i],a3[i])):
            predicted.append(1) 
    
    correct = 0
    for i in range(test.shape[0]):
        if test[i][-1] == predicted[i]:
            correct += 1

    accuracy = 100 * correct / test.shape[0]
    print("--------------------------------")
    print("Final Accuracy of The Multi Classifer: ")
    print(accuracy)
  
    return accuracy


def getMultiClassifierData (dataset, typ):
    
    if (typ == 1):
        
    
        
      
        #print (dataset.shape)
        for row in range(dataset.shape[0]):
            #print (dataset.shape)
            if dataset[row,-1]== 0:
                dataset[row,-1] == 1
   
    if (typ == 2):
        

        #print (dataset.shape)
        for row in range(dataset.shape[0]):
            #print (dataset.shape)
            if dataset[row,-1]== -1:
                dataset[row,-1] == 1     
        for row in range(dataset.shape[0]):
            #print (dataset.shape)
            if dataset[row,-1]== 0:
                dataset[row,-1] == -1     
                
    if (typ == 3):
        

        #print (dataset.shape)
        for row in range(dataset.shape[0]):
            #print (dataset.shape)
            if dataset[row,-1]== 0:
                dataset[row,-1] == -1     
    
        
        
        
#    print("class 12")
#    print(dataset)
    return dataset








def printLines ():
    print("Enter 1 to get Answer 4:a")
    print("Enter 2 to get Answer 4:b")
    print("Enter 3 to get Answer 4:c")
    print("Enter 4 to get Answer Which pair of classes is most difficult to seperate")
    print("Enter 5 to get feature 0 train and test accuracies for class 1 and 2")
    print("Enter 6 to get feature 1 train and test accuracies for class 1 and 2")
    print("Enter 7 to get feature 2 train and test accuracies for class 1 and 2")
    print("Enter 8 to get feature 3 train and test accuracies for class 1 and 2")
    print("to get features for class 2 and 3 enter 5c23, 6c23, 7c23, 8c23")
    print("to get features for class 1 and 3 enter 5c13, 6c13, 7c13, 8c13")
    print("Enter 9 to get Answer 5: which feature is most discruminative")
    print("Enter 10 to get Answer 6")
    print("Enter 11 to get Answer 7 coefficient 0.01")
    print("Enter 12 to get Answer 7 coefficient 0.1")
    print("Enter 13 to get Answer 7 coefficient 1")
    print("Enter 14 to get Answer 7 coefficient 10")
    print("Enter 15 to get Answer 7 coefficient 100")
    print("Enter 16 to exit\n")


def main ():

    

    
   
    
    while(True):
        
        printLines()
        choice = input()
    
        if (choice=="1"):
            
            clss12dataset = getClss1Clss2(dataset)
            w1, acc1 = perceptronTrain(clss12dataset, MaxIter)
            printAcc("12", acc1)
            clss12dataset = getClss1Clss2(datasetTest)
            a1, ac1 = perceptronTest(w1, clss12dataset)
            printAccTest("12", ac1)
            
        elif (choice=="2"):
    
            clss23dataset = getClss2Clss3(dataset)
            w2, acc2 = perceptronTrain(clss23dataset, MaxIter)
            printAcc("23", acc2)
            clss23dataset = getClss2Clss3(datasetTest)
            a2, ac2 = perceptronTest(w2, clss23dataset)
            printAccTest("23", ac2)
            
        elif (choice=="3"):
    
            clss13dataset = getClss1Clss3(dataset)
            w3, acc3 = perceptronTrain(clss13dataset, MaxIter)
            printAcc("13", acc3)
            clss13dataset = getClss1Clss3(datasetTest)
            a3, ac3 = perceptronTest(w3, clss13dataset)
            printAccTest("13", ac3)
            
        elif (choice=="4"):
            print ("class-2 and class-3 are the most hard to seperate pairs")
            
        elif (choice=="5"):
            
            
    
            print("Feature 1::::::::::::::::::::::::::::::::")
            clss12dataset = getClss1Clss2(dataset)
            w1, acc1 = perceptronTrain(clss12dataset[:,[0,-1]], MaxIter)
            printAcc("12", acc1)
            clss12dataset  = getClss1Clss2(datasetTest)
            a1, ac1 = perceptronTest(w1, clss12dataset[:,[0,-1]])
            printAccTest("12", ac1)
            
        elif (choice=="6"):
            
        
    
            print("Feature 2::::::::::::::::::::::::::::::::")
            clss12dataset = getClss1Clss2(dataset)
            w1, acc1 = perceptronTrain(clss12dataset[:,[1,-1]], MaxIter)
            printAcc("12", acc1)
            clss12dataset  = getClss1Clss2(datasetTest)
            a1, ac1 = perceptronTest(w1, clss12dataset[:,[1,-1]])
            printAccTest("12", ac1)
            
        elif (choice=="7"):
    
            print("Feature 3::::::::::::::::::::::::::::::::")
            clss12dataset = getClss1Clss2(dataset)
            w1, acc1 = perceptronTrain(clss12dataset[:,[2,-1]], MaxIter)
            printAcc("12", acc1)
            clss12dataset  = getClss1Clss2(datasetTest)
            a1, ac1 = perceptronTest(w1, clss12dataset[:,[2,-1]])
            printAccTest("12", ac1)
            
        elif (choice=="8"):
    
            print("Feature 4::::::::::::::::::::::::::::::::")
            clss12dataset = getClss1Clss2(dataset)
            w1, acc1 = perceptronTrain(clss12dataset[:,[3,-1]], MaxIter)
            printAcc("12", acc1)
            clss12dataset  = getClss1Clss2(datasetTest)
            a1, ac1 = perceptronTest(w1, clss12dataset[:,[3,-1]])
            printAccTest("12", ac1)
            
            
        ############################################################################33
        
        
        elif (choice=="5c23"):
            
            
    
            print("Feature 1::::::::::::::::::::::::::::::::")
            clss12dataset = getClss2Clss3(dataset)
            w1, acc1 = perceptronTrain(clss12dataset[:,[0,-1]], MaxIter)
            printAcc("23", acc1)
            clss12dataset  = getClss2Clss3(datasetTest)
            a1, ac1 = perceptronTest(w1, clss12dataset[:,[0,-1]])
            printAccTest("23", ac1)
            
        elif (choice=="6c23"):
            
        
    
            print("Feature 2::::::::::::::::::::::::::::::::")
            clss12dataset = getClss2Clss3(dataset)
            w1, acc1 = perceptronTrain(clss12dataset[:,[1,-1]], MaxIter)
            printAcc("23", acc1)
            clss12dataset  = getClss2Clss3(datasetTest)
            a1, ac1 = perceptronTest(w1, clss12dataset[:,[1,-1]])
            printAccTest("23", ac1)
            
        elif (choice=="7c23"):
    
            print("Feature 3::::::::::::::::::::::::::::::::")
            clss12dataset = getClss2Clss3(dataset)
            w1, acc1 = perceptronTrain(clss12dataset[:,[2,-1]], MaxIter)
            printAcc("23", acc1)
            clss12dataset  = getClss2Clss3(datasetTest)
            a1, ac1 = perceptronTest(w1, clss12dataset[:,[2,-1]])
            printAccTest("23", ac1)
            
        elif (choice=="8c23"):
    
            print("Feature 4::::::::::::::::::::::::::::::::")
            clss12dataset = getClss2Clss3(dataset)
            w1, acc1 = perceptronTrain(clss12dataset[:,[3,-1]], MaxIter)
            printAcc("23", acc1)
            clss12dataset  = getClss2Clss3(datasetTest)
            a1, ac1 = perceptronTest(w1, clss12dataset[:,[3,-1]])
            printAccTest("23", ac1)
            
            
            ##########################################################################
            
        elif (choice=="5c13"):
            
            
    
            print("Feature 1::::::::::::::::::::::::::::::::")
            clss12dataset = getClss1Clss3(dataset)
            w1, acc1 = perceptronTrain(clss12dataset[:,[0,-1]], MaxIter)
            printAcc("13", acc1)
            clss12dataset  = getClss1Clss3(datasetTest)
            a1, ac1 = perceptronTest(w1, clss12dataset[:,[0,-1]])
            printAccTest("13", ac1)
            
        elif (choice=="6c13"):
            
        
    
            print("Feature 2::::::::::::::::::::::::::::::::")
            clss12dataset = getClss1Clss3(dataset)
            w1, acc1 = perceptronTrain(clss12dataset[:,[1,-1]], MaxIter)
            printAcc("13", acc1)
            clss12dataset  = getClss1Clss3(datasetTest)
            a1, ac1 = perceptronTest(w1, clss12dataset[:,[1,-1]])
            printAccTest("13", ac1)
            
        elif (choice=="7c13"):
    
            print("Feature 3::::::::::::::::::::::::::::::::")
            clss12dataset = getClss1Clss3(dataset)
            w1, acc1 = perceptronTrain(clss12dataset[:,[2,-1]], MaxIter)
            printAcc("13", acc1)
            clss12dataset  = getClss1Clss3(datasetTest)
            a1, ac1 = perceptronTest(w1, clss12dataset[:,[2,-1]])
            printAccTest("13", ac1)
            
        elif (choice=="8c13"):
    
            print("Feature 4::::::::::::::::::::::::::::::::")
            clss12dataset = getClss1Clss3(dataset)
            w1, acc1 = perceptronTrain(clss12dataset[:,[3,-1]], MaxIter)
            printAcc("13", acc1)
            clss12dataset  = getClss1Clss3(datasetTest)
            a1, ac1 = perceptronTest(w1, clss12dataset[:,[3,-1]])
            printAccTest("13", ac1)
            
            
            
        elif (choice=="9"):
            print ("most discrimintive feature is the third feature")
            
        elif (choice=="10"):
            print ("Multiclassifier: 3 Classifier Accuracies During Training and Testing===================")
            print(perceptronTrainMultiClass(dataset, datasetTest, MaxIter))
            
        elif (choice=="11"):
            
    
    
            print ("Adding regulazation term===================")
            print ("Coefficient = 0.01 ===================")
            print(perceptronTrainMultiClass(dataset, datasetTest, MaxIter, Regulization=True, Coefficient=0.01))
            
        elif (choice=="12"):
    
            print ("Adding regulazation term===================")
            print ("Coefficient = 0.1 ===================")
            print(perceptronTrainMultiClass(dataset, datasetTest, MaxIter, Regulization=True, Coefficient=0.10))
            
        elif (choice=="13"):
    
            print ("Adding regulazation term===================")
            print ("Coefficient = 1 ===================")
            print(perceptronTrainMultiClass(dataset, datasetTest, MaxIter, Regulization=True, Coefficient=1.00))
            
        elif (choice=="14"):
    
            print ("Adding regulazation term===================")
            print ("Coefficient = 10 ===================")
            print(perceptronTrainMultiClass(dataset, datasetTest, MaxIter, Regulization=True, Coefficient=10.0))
            
        elif (choice=="15"):
    
            print ("Adding regulazation term===================")
            print ("Coefficient = 100 ===================")
            print(perceptronTrainMultiClass(dataset, datasetTest, MaxIter, Regulization=True, Coefficient=100.00))
            
        elif (choice=="16"):
            break
        
        else:
            print("Please only enter one of the numbers")


dataset = extract()
datasetTest = extract("test")
MaxIter = 20          
main()
    
