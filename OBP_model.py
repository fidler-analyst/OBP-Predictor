#OBP regression model


# Libraries ------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------


# Data retrieval and cleaning ------------------------------------------------
raw_data = pd.read_csv('batting.csv')
raw_data.drop(columns = ['playerid', 'Team'], inplace = True) #remove playerid and team columns
df1 = raw_data
df1.dropna(inplace = True)  #remove null rows

#remove percent signs % from data
columns = df1.columns.to_list()
percent_columns = [x for x in columns if '%' in x]
for col in percent_columns:
    df1[col]=df1[col].str[:-1].astype(float)
    df1 = df1.rename(columns={col:col[:-1]})
df1['MarApr_HR/FB']=df1['MarApr_HR/FB'].str[:-1].astype(float)
# ----------------------------------------------------------------------------


# First pass of data and visualization ---------------------------------------
description = raw_data.describe()                             
df2 = df1.drop(columns = ['Name'])
corrMatrix = df2.corr()
print(corrMatrix.FullSeason_OBP.sort_values(ascending = False))
#reveals strongest correlation with OBP, R, AVG, H, SLG, and BB% all above 40%

#plot top six correlating features against Full Season OBP
fig, axs = plt.subplots(2, 3)
axs[0,0].scatter(df1['MarApr_OBP'], df1['FullSeason_OBP'])
axs[0,0].set_title('OBP r = '+ str(round(corrMatrix.loc['MarApr_OBP', 'FullSeason_OBP'],3)))

axs[0,1].scatter(df1['MarApr_R'], df1['FullSeason_OBP'])
axs[0,1].set_title('Runs r = '+ str(round(corrMatrix.loc['MarApr_R', 'FullSeason_OBP'],3)))

axs[0,2].scatter(df1['MarApr_AVG'], df1['FullSeason_OBP'])
axs[0,2].set_title('AVG r = '+ str(round(corrMatrix.loc['MarApr_AVG', 'FullSeason_OBP'],3)))

axs[1,0].scatter(df1['MarApr_H'], df1['FullSeason_OBP'])
axs[1,0].set_title('Hits r = '+ str(round(corrMatrix.loc['MarApr_H', 'FullSeason_OBP'],3)))

axs[1,1].scatter(df1['MarApr_SLG'], df1['FullSeason_OBP'])
axs[1,1].set_title('SLG r = '+ str(round(corrMatrix.loc['MarApr_SLG', 'FullSeason_OBP'],3)))

axs[1,2].scatter(df1['MarApr_BB'], df1['FullSeason_OBP'])
axs[1,2].set_title('BB% r = '+ str(round(corrMatrix.loc['MarApr_BB', 'FullSeason_OBP'],3)))
# ----------------------------------------------------------------------------


# Regression Model -----------------------------------------------------------
df1.set_index('Name', inplace = True)
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df1[['MarApr_OBP', 'MarApr_R', 'MarApr_AVG', 'MarApr_H', 'MarApr_SLG', 'MarApr_BB']], 
                                                   df1.FullSeason_OBP, test_size=0.25, random_state=42)

knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute')     #number of groups

sfs1 = SFS(knn, 
           k_features=6,           #number of features to use
           forward=True,           #set to forawrds/backwards sequence selection
           floating=True, 
           verbose=2,
           scoring = 'neg_mean_squared_error',     #negative mean squared error to score
           cv=0)

sfs1 = sfs1.fit(X_train, y_train)     #fit test to target

print(sfs1.subsets_)      #print resulting features and score

X_train_sfs = sfs1.transform(X_train)
X_test_sfs = sfs1.transform(X_test)

knn.fit(X_train_sfs, y_train)
y_pred = knn.predict(X_test_sfs)
# ----------------------------------------------------------------------------

#output results of test set
results_df = pd.DataFrame(y_test)
results_df.insert(1, 'Predicted', y_pred)
results_df.insert(2, 'Difference', y_test-y_pred)
results_df.to_csv('results.csv')
