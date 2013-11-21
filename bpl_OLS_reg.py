#!/usr/bin/python
# Filename: reg.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt

# Helper Function
def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

  #####################
 ##### LOAD DATA #####
#####################

root = "/Users/triplec1988/GA_Data_Science/finalProject/csv/"

firstYear = ['09', '10', '11', '12', ]
secondYear = ['10', '11', '12', '13', ]

overall = {}
home = {}
away = {}

for i, j in zip(firstYear, secondYear):
    overall[i + '/' + j] = pd.read_csv(root + i + ':' + j + '_overall.csv')
    home[i + '/' + j] = pd.read_csv(root + i + ':' + j + '_home.csv')
    away[i + '/' + j] = pd.read_csv(root + i + ':' + j + '_away.csv')

  ######################
 ##### CLEAN DATA #####
######################
'''
Simply doing the minimum cleaning to remove irrelevant features,
obviouse features, and features that contribute to a high degree
of multicollinearity
'''

# Clean overall
for key, value in overall.items():
    overall[key] = overall[key].drop(['R', 'Team', 'P', 'Form', 'W', 'D', 'L', 'GF', 'GA', 'Pens Against', 'Pens For', ], axis = 1) 


# Clean home
for key, value in home.items():
    home[key] = home[key].drop(['R', 'Team', 'P', 'Form', 'W', 'D', 'L', 'GF', 'GA', 'Pens Against', 'Pens For', ], axis = 1)


# Clean away
for key, value in away.items():
    away[key] = away[key].drop(['R', 'Team', 'P', 'Form', 'W', 'D', 'L', 'GF', 'GA', 'Pens Against', 'Pens For', ], axis = 1)

  ########################
 ##### KITCHEN SINK #####
######################## 
'''
Here we want to create a list of all possible combinations
of features. We've already addressed multicollinearity the
best we can given the data, so now we want to see which of
the feature combinations yields the highest average r-sqd
'''

# Create list of all possible feature combinations
features = ['Pass Success%', 'SpG', 'Yellow', 'Red', 'Pens +/- (F - A)', 'Possession%', 'Fouls pg', ]
all_feat = [x for x in powerset(features)]
all_feat.sort(key = len)

# HOME
av_rsq = 0.0
for i in range(len(all_feat)):
    r_sq = []
    for key, value in overall.items():
        test = overall[key]
        home_train = home[key]
        home_x = np.array(home_train.drop(['Pts'] + all_feat[i], axis = 1))
        y = np.array(test['Pts'])
        
        model_home = sm.OLS(y, home_x)
        
        results_home = model_home.fit()
        r_sq.append(results_home.rsquared_adj)

        
    new_rsq = sum(r_sq)/len(r_sq)
    
    if new_rsq > av_rsq:
        av_rsq = new_rsq
        print "This list increased r_squared: " + str(all_feat[i])
        print "The average r_squared is now: " + str(av_rsq)


# AWAY
av_rsq = 0.0
for i in range(len(all_feat)):
    r_sq = []
    for key, value in overall.items():
        test = overall[key]
        away_train = away[key]
        away_x = np.array(away_train.drop(['Pts'] + all_feat[i], axis = 1))
        y = np.array(test['Pts'])
        
        model_away = sm.OLS(y, away_x)
            
        results_away = model_away.fit()
        r_sq.append(results_away.rsquared_adj)
        
    new_rsq = sum(r_sq)/len(r_sq)
    
    if new_rsq > av_rsq:
        av_rsq = new_rsq
        print "This list increased r_squared: " + str(all_feat[i])
        print "The average r_squared is now: " + str(av_rsq)



  #########################
 ##### RE-CLEAN DATA #####
#########################
'''
Now that we've figured out which feature combinations
yield the highest r-squared values for home and away,
we can reclean the data
'''


# Clean home
for key, value in home.items():
    home[key] = home[key].drop(['SpG', 'Pens +/- (F - A)', 'Possession%', ], axis = 1)


# Clean away
for key, value in away.items():
    away[key] = away[key].drop(['SpG', 'Yellow', 'Red', 'Possession%', ], axis = 1)


  ###########################
 ##### SEASON ANALYSIS #####
###########################
'''
Now that we have recleaned the data to optimize average r-squared
we can take a deeper look at home and away for each season by
examining the summary feature of StatsModels as well as plotting
the predicted points totals for home and away versus the actual 
points total from the overall season to look for outliers.
'''

# 2009/10 Season
test10 = overall['09/10']
home_train10 = home['09/10']
away_train10 = away['09/10']
away_x10 = np.array(away_train10.drop(['Pts', ], axis = 1))
home_x10 = np.array(home_train10.drop(['Pts', ], axis = 1))
y10 = np.array(test10['Pts'])

feat_list_home = list(home_train10.columns)
feat_list_home.pop(1)

feat_list_away = list(away_train10.columns)
feat_list_away.pop(1)

model_home10 = sm.OLS(y10, home_x10)
model_away10 = sm.OLS(y10, away_x10)

results_home10 = model_home10.fit()
results_away10 = model_away10.fit()

print results_home10.summary(xname=feat_list_home)
print results_away10.summary(xname=feat_list_away)

y_pred = results_home10.predict()
plt.figure()
plt.plot(range(1, 21), y_pred, 'o', range(1, 21), y10, 'b-')
prstd, iv_l, iv_u = wls_prediction_std(results_home10)
plt.plot(range(1, 21), results_home10.fittedvalues, 'r--.')
plt.plot(range(1, 21), iv_u, 'r--')
plt.plot(range(1, 21), iv_l, 'r--')
plt.title('blue: true,   red: OLS')
plt.show()


y_pred = results_away10.predict()
plt.figure()
plt.plot(range(1, 21), y_pred, 'o', range(1, 21), y10, 'b-')
prstd, iv_l, iv_u = wls_prediction_std(results_away10)
plt.plot(range(1, 21), results_away10.fittedvalues, 'r--.')
plt.plot(range(1, 21), iv_u, 'r--')
plt.plot(range(1, 21), iv_l, 'r--')
plt.title('blue: true,   red: OLS')
plt.show()

# 2010/11 Season
test11 = overall['10/11']
home_train11 = home['10/11']
away_train11 = away['10/11']
away_x11 = np.array(away_train11.drop(['Pts', ], axis = 1))
home_x11 = np.array(home_train11.drop(['Pts', ], axis = 1))
y11 = np.array(test11['Pts'])

feat_list_home = list(home_train11.columns)
feat_list_home.pop(1)

feat_list_away = list(away_train11.columns)
feat_list_away.pop(1)

model_home11 = sm.OLS(y11, home_x11)
model_away11 = sm.OLS(y11, away_x11)

results_home11 = model_home11.fit()
results_away11 = model_away11.fit()

print results_home11.summary(xname=feat_list_home)
print results_away11.summary(xname=feat_list_away)

y_pred = results_home11.predict()
plt.figure()
plt.plot(range(1, 21), y_pred, 'o', range(1, 21), y11, 'b-')
prstd, iv_l, iv_u = wls_prediction_std(results_home11)
plt.plot(range(1, 21), results_home11.fittedvalues, 'r--.')
plt.plot(range(1, 21), iv_u, 'r--')
plt.plot(range(1, 21), iv_l, 'r--')
plt.title('blue: true,   red: OLS')
plt.show()


y_pred = results_away11.predict()
plt.figure()
plt.plot(range(1, 21), y_pred, 'o', range(1, 21), y11, 'b-')
prstd, iv_l, iv_u = wls_prediction_std(results_away11)
plt.plot(range(1, 21), results_away11.fittedvalues, 'r--.')
plt.plot(range(1, 21), iv_u, 'r--')
plt.plot(range(1, 21), iv_l, 'r--')
plt.title('blue: true,   red: OLS')
plt.show()

# 2011/12 Season
test12 = overall['11/12']
home_train12 = home['11/12']
away_train12 = away['11/12']
away_x12 = np.array(away_train12.drop(['Pts', ], axis = 1))
home_x12 = np.array(home_train12.drop(['Pts', ], axis = 1))
y12 = np.array(test12['Pts'])

feat_list_home = list(home_train12.columns)
feat_list_home.pop(1)

feat_list_away = list(away_train12.columns)
feat_list_away.pop(1)

model_home12 = sm.OLS(y12, home_x12)
model_away12 = sm.OLS(y12, away_x12)

results_home12 = model_home12.fit()
results_away12 = model_away12.fit()

print results_home12.summary(xname=feat_list_home)
print results_away12.summary(xname=feat_list_away)

y_pred = results_home12.predict()
plt.figure()
plt.plot(range(1, 21), y_pred, 'o', range(1, 21), y12, 'b-')
prstd, iv_l, iv_u = wls_prediction_std(results_home12)
plt.plot(range(1, 21), results_home12.fittedvalues, 'r--.')
plt.plot(range(1, 21), iv_u, 'r--')
plt.plot(range(1, 21), iv_l, 'r--')
plt.title('blue: true,   red: OLS')
plt.show()


y_pred = results_away12.predict()
plt.figure()
plt.plot(range(1, 21), y_pred, 'o', range(1, 21), y12, 'b-')
prstd, iv_l, iv_u = wls_prediction_std(results_away12)
plt.plot(range(1, 21), results_away12.fittedvalues, 'r--.')
plt.plot(range(1, 21), iv_u, 'r--')
plt.plot(range(1, 21), iv_l, 'r--')
plt.title('blue: true,   red: OLS')
plt.show()

# 2012/13 Season
test13 = overall['12/13']
home_train13 = home['12/13']
away_train13 = away['12/13']
away_x13 = np.array(away_train13.drop(['Pts', ], axis = 1))
home_x13 = np.array(home_train13.drop(['Pts', ], axis = 1))
y13 = np.array(test13['Pts'])

feat_list_home = list(home_train13.columns)
feat_list_home.pop(1)

feat_list_away = list(away_train13.columns)
feat_list_away.pop(1)

model_home = sm.OLS(y13, home_x13)
model_away = sm.OLS(y13, away_x13)

results_home13 = model_home.fit()
results_away13 = model_away.fit()

print results_home13.summary(xname=feat_list_home)
print results_away13.summary(xname=feat_list_away)

y_pred = results_home13.predict()
plt.figure()
plt.plot(range(1, 21), y_pred, 'o', range(1, 21), y13, 'b-')
prstd, iv_l, iv_u = wls_prediction_std(results_home13)
plt.plot(range(1, 21), results_home13.fittedvalues, 'r--.')
plt.plot(range(1, 21), iv_u, 'r--')
plt.plot(range(1, 21), iv_l, 'r--')
plt.title('blue: true,   red: OLS')
plt.show()

y_pred = results_away13.predict()
plt.figure()
plt.plot(range(1, 21), y_pred, 'o', range(1, 21), y13, 'b-')
prstd, iv_l, iv_u = wls_prediction_std(results_away13)
plt.plot(range(1, 21), results_away13.fittedvalues, 'r--.')
plt.plot(range(1, 21), iv_u, 'r--')
plt.plot(range(1, 21), iv_l, 'r--')
plt.title('blue: true,   red: OLS')
plt.show()

if __name__ == '__main__':
    print 'This program is being run by itself'
else:
    print 'I am being imported from another module'
