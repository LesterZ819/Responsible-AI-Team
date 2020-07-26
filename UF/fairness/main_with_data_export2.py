#conda env create -f environment.yml

import os
import sys
#os.system('python setup.py develop') # should run only once

import gensim
import gzip
import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", palette="muted", color_codes=True, context="talk")
from IPython import display
import matplotlib as plt

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

import keras as ke
import keras.backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model

from fairness.models import biased_classifier
from fairness.helpers import load_ICU_data,plot_distributions,p_rule,plot_distributions_IBM


create_gif = True

print(f"sklearn: {sk.__version__}")
print(f"pandas: {pd.__version__}")
print(f"kerads: {ke.__version__}")


rndseed = 7 #Sets the seed so that it is the same each time

np.random.seed(rndseed)

Attrition = True #This determines which dataset is used: if you switch True or False it switches between the two datasets Marc provided

#data splits for train/test data (X), labels (y) and sensitive variables (Z)
def splitdata(X,y,Z):
    # split into train/test set
    X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(X, y, Z, test_size=0.5, 
                                                                     stratify=y, random_state=7) #Uses sklearn to train dataset-->Split arrays or matrices into random train and test subsets

    # standardize the data
    scaler = StandardScaler().fit(X_train)  #Sklearn package -->Standardize features by removing the mean and scaling to unit variance
                                            #The standard score of a sample x is calculated as:
                                            #z = (x - u) / s
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    X_train = X_train.pipe(scale_df, scaler) #I believe the .pipe method allows the code to read the X_train data
    X_test = X_test.pipe(scale_df, scaler) #.pipe allows the code to read the X_test data
    return X_train,X_test, y_train, y_test, Z_train, Z_test

def train_biased(X_train,y_train):
    # initialise NeuralNet Classifier
    biased_model = biased_classifier(n_features=X_train.shape[1]) #Creates some kind of model using the keras package: the function is imported from the models file

    # train on train set
    history = biased_model.fit(X_train.values, y_train.values, epochs=20, verbose=1) #Trains the model

    return biased_model


def accuracy(y_test,y_pred,income):
    
    roc_score = roc_auc_score(y_test,y_pred) #Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    acc_score = accuracy_score(y_test,y_pred>income/100000) #Determines accuracy of the model
    return roc_score, acc_score
    
def accuracy_IBM(y_test,y_pred,TH): #Same as above function for IBM Dataset
    
    roc_score = roc_auc_score(y_test,y_pred)
    acc_score = accuracy_score(y_test,y_pred>TH/1)
    return roc_score, acc_score

def fairness_factor(y_pred,Z_test): #This function determines fairness by using the 80% rule
    print(Z_test)
    race_bias_factor = p_rule(y_pred, Z_test['race']) #p_rule function is imported from helpers file
    gender_bias_factor = p_rule(y_pred, Z_test['sex']) #I believe this p_rule function is looking for p values > 0.25
    return race_bias_factor, gender_bias_factor

def fairness_factor_IBM(y_pred,Z_test): #Does the same thing as above but for IBM dataset
    print(pd.concat([y_pred,Z_test],axis=1, ignore_index=True))
    #file = pd.concat([y_pred,Z_test],axis=1, ignore_index=True) #Creates the file object
    #file.to_csv('newIBMprobabilitieswithZtest.csv') #Converts the dataframe to a csv file #Creates the CSV for the file object
    
    race_bias_factor = p_rule(y_pred, Z_test['MaritalStatus'])
    gender_bias_factor = p_rule(y_pred, Z_test['Gender'])
    
    return race_bias_factor, gender_bias_factor


def load_UIC_data(path): #Loads the dataframe for the Adult dataset
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'martial_status', 'occupation', 'relationship', 'race', 'sex',
                    'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target']
    input_data = (pd.read_csv(path, names=column_names,
                              na_values="?", sep=r'\s*,\s*', engine='python')
                  .loc[lambda df: df['race'].isin(['White', 'Black'])])
    # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
    sensitive_attribs = ['race', 'sex']
    Z = (input_data.loc[:, sensitive_attribs]
         .assign(race=lambda df: (df['race'] == 'White').astype(int),
                 sex=lambda df: (df['sex'] == 'Male').astype(int)))

    # targets; 1 when someone makes over 50k , otherwise 0
    y = (input_data['target'] == '>50K').astype(int)

    # features; note that the 'target' and sentive attribute columns are dropped
    X = (input_data
         .drop(columns=['target', 'race', 'sex', 'fnlwgt'])
         .fillna('Unknown')
         .pipe(pd.get_dummies, drop_first=True))
    print(input_data['workclass'])
    print(f"features X: {X.shape[0]} samples, {X.shape[1]} attributes")
    print(f"targets y: {y.shape} samples")
    print(f"sensitives Z: {Z.shape[0]} samples, {Z.shape[1]} attributes")

    return X, y, Z

def load_IBM_data(path): #Loads the dataframe for the IBM dataset
    column_names = ['Age','Attrition','BusinessTravel','DailyRate','Department',
                    'DistanceFromHome','Education','EducationField','EmployeeCount','EmployeeNumber',
                    'EnvironmentSatisfaction','Gender','HourlyRate','JobInvolvement','JobLevel',
                    'JobRole','JobSatisfaction','MaritalStatus','MonthlyIncome','MonthlyRate',
                    'NumCompaniesWorked','Over18','OverTime','PercentSalaryHike','PerformanceRating',
                    'RelationshipSatisfaction','StandardHours','StockOptionLevel','TotalWorkingYears',
                    'TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole',
                    'YearsSinceLastPromotion','YearsWithCurrManager']
    input_data = (pd.read_csv(path, names=column_names,
                              na_values="?", sep=r'\s*,\s*', engine='python')
                  .loc[lambda df: df['MaritalStatus'].isin(['Married', 'Divorced','Single'])])
    print(input_data)
    for col in input_data.columns:   
        converted = pd.to_numeric(input_data[col],errors='coerce')  
        input_data[col] = converted if not pd.isnull(converted).all() else input_data[col]
    #input_data = input_data.convert_objects(convert_numeric=True)
    
    input_data['MonthlyIncome']=input_data['MonthlyIncome']/1000
    # sensitive attributes; we identify 'MaritalStatus' and 'Gender' as sensitive attributes
    sensitive_attribs = ['MaritalStatus', 'Gender']
    Z = (input_data.loc[:, sensitive_attribs]
         .assign(MaritalStatus=lambda df: (df['MaritalStatus'] == 'Married').astype(int),
                 Gender=lambda df: (df['Gender'] == 'Male').astype(int)))

    
    y = (input_data['Attrition'] =='Yes').astype(int)
    # features; note that the 'target' and sentive attribute columns are dropped
    X = (input_data
         .drop(columns=['Age','Attrition','BusinessTravel','DailyRate','Department',
                    'DistanceFromHome','Education','EducationField','EmployeeCount','EmployeeNumber',
                    'EnvironmentSatisfaction','Gender','HourlyRate','JobInvolvement','JobLevel',
                    'JobRole','JobSatisfaction','MaritalStatus',
                    'NumCompaniesWorked','Over18','PercentSalaryHike','PerformanceRating',
                    'RelationshipSatisfaction','StandardHours','StockOptionLevel','TotalWorkingYears',
                    'TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany'])
         .fillna('Unknown')
         .pipe(pd.get_dummies, drop_first=True))
    X = (input_data
         .drop(columns=['Attrition', 'MaritalStatus', 'Gender', 'EmployeeNumber'])
         .fillna('Unknown')
         .pipe(pd.get_dummies, drop_first=True))
    print(input_data.dtypes)
    print(X.head())
    print(X.dtypes)
    print(input_data.dtypes)
    #print(dds)
    
    print(f"features X: {X.shape[0]} samples, {X.shape[1]} attributes")
    print(f"targets y: {y.shape} samples")
    print(f"sensitives Z: {Z.shape[0]} samples, {Z.shape[1]} attributes")

    return X, y, Z


if __name__ == '__main__':

    #action=sys.argv[1]
    if Attrition:
        #prepare data:
        print('Loading data...')
        X, y, Z = load_IBM_data('data/IBM-HR-Employee-Attrition.csv')
        #print(X)#source data
        print(y)#what needs to be predicted (Attrition)
        #print(Z)#protected variables


        print('split train/test')
        X_train,X_test, y_train, y_test, Z_train, Z_test = splitdata(X,y,Z)
        # biased predictor
        biased_model = train_biased(X_train, y_train)
        y_pred = pd.Series(biased_model.predict(X_test).ravel(), index=y_test.index)
        roc_score,acc_score = accuracy_IBM(y_test,y_pred,TH=.25)
        print(f"ROC AUC: {roc_score:.2f}")
        print(f"Accuracy: {100*acc_score:.1f}%")
        
        
        y_pred.to_frame() #This line truns the y_pred variable from a pandas series into a pandas dataframe
        print("hh")
        #evaluate the fairness of the model by using the 80% rule from US EEOC
        MaritalStatus_bias_factor, Gender_bias_factor = fairness_factor_IBM(y_pred, Z_test)

        #The following two lines are optional and if you leave them commented out the csv will only contain the y_pred variable
        # y_pred['MaritalStatusBiasFactor'] = MaritalStatus_bias_factor #Adds the Marital Status to the dataframe
        # y_pred['GenderBiasFactor'] = Gender_bias_factor #Adds gender to the dataframe
        #y_pred.to_csv('newIBMprobabilities.csv') #Converts the dataframe to a csv file
        
        #!Note!: if generating the CSV file comment out all of the code below this line or else it might throw an error
        
        # print(f"\tgiven attribute MaritalStatus; {MaritalStatus_bias_factor:.0f}%-rule")
        # print(f"\tgiven attribute Gender;  {Gender_bias_factor:.0f}%-rule")
        # plot_distributions_IBM(y_pred, Z_test, fname='output/IBM_training_withbias.png')#Function for creating the plots are in the helpers file
        # y_predbiased=y_pred
        # import numpy as np
        # count, division = np.histogram(y_predbiased[Z_test['Gender']==0],bins=10)
        # count2, division = np.histogram(y_predbiased[Z_test['Gender']==1],bins=10)
        # print(['Gender',count,count2])
        # count, division = np.histogram(y_predbiased[Z_test['MaritalStatus']==0],bins=10)
        # count2, division = np.histogram(y_predbiased[Z_test['MaritalStatus']!=1],bins=10)
        # print(['MaritalStatus',count,count2])

        
        
    else:
        #prepare data:
        print('Loading data...')
        X, y, Z = load_ICU_data('data/adult.data')
        #print(X)#source data
        #print(y)#what needs to be predicted
        #print(Z)#protected variables


        print('split train/test')
        X_train,X_test, y_train, y_test, Z_train, Z_test = splitdata(X,y,Z)

        # biased predictor
        biased_model = train_biased(X_train, y_train)
        y_pred = pd.Series(biased_model.predict(X_test).ravel(), index=y_test.index)

        roc_score,acc_score = accuracy(y_test,y_pred,income=50000.)
        print(f"ROC AUC: {roc_score:.2f}")
        print(f"Accuracy: {100*acc_score:.1f}%")
        
        #evaluate the fairness of the model by using the 80% rule from US EEOC
        race_bias_factor, gender_bias_factor = fairness_factor(y_pred, Z_test)
        print(f"\tgiven attribute race; {race_bias_factor:.0f}%-rule")
        print(f"\tgiven attribute gender;  {gender_bias_factor:.0f}%-rule")
        fig = plot_distributions(y_pred, Z_test, fname='output/training_withbias.png') #Function for creating the plots are in the helpers file
