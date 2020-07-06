#import all required libraries
#pip install heatmapz


import pandas as pd
from sklearn import preprocessing
from heatmap import heatmap, corrplot
from matplotlib import pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 7,7 

df = pd.read_csv(r'/Users/Maeve/OneDrive - University of Florida/AI/UF/data/adult.data.null.csv')
print(df)
print('')

#view data types (categorical vs numerical)
data_type_chart=df.dtypes
print(data_type_chart)
print('')

#create list of categorical columns
categoricalCol = []
categoricalCol = df.select_dtypes(exclude='int64')
print(categoricalCol)
print('')


#view the different categorical variables
for i in categoricalCol:
    print(i,':')
    print(df[i].unique())
print('')

    
#check for missing values
print(df.isnull().sum())

#delete null
df_complete=df.dropna(how='any', axis=0)

#check for missing values
print(df_complete.isnull().sum())


#convert categorical to numerical 
#Deep copy the original data
df_encoded = df_complete.copy(deep=True)
#Use Scikit-learn label encoding to encode character data
le = preprocessing.LabelEncoder()
for col in categoricalCol:
        df_encoded[col] = le.fit_transform(df_complete[col])
        le_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print('Feature: ', col)
        print(le_mapping)

pd.set_option('display.max_columns', 10)
print(df_encoded.describe().transpose())
print('')



df_corr = df_encoded.corr()
plt.figure(figsize=(8,8))
corrplot(df_corr,size_scale=300)
