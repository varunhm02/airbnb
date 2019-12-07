import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt 


#Performs k fold cross validation with the specified classifer and returns all the necessary metrics
def cross_val_scores(classifier, clf, X, y,cv=10,plot=False):
    #cv = 10
    k_fold = KFold(n_splits = cv, shuffle = True,random_state=0)
    k_scores = []
    rmse_scores = []
    r2_scores = []

    
    
    
    for i, (train, test) in enumerate(k_fold.split(X, y)):
        clf.fit(X[train], y[train])
        k_scores.append(clf.score(X[test], y[test]))
        rmse_scores.append(np.sqrt(mean_squared_error(y[test],clf.predict(X[test]))))

        if plot :
            # set width of bar
            barWidth = 0.25

            plt.figure(figsize=(20,10))

            # Set position of bar on X axis
            r1 = np.arange(1,50)
            r2 = [x + barWidth + 0.1 for x in r1]

            # Make the plot
            plt.bar(r1, y[test][1:50], width=barWidth, color='royalblue', label='Actual')
            plt.bar(r2, list(clf.predict(X[test])[1:50]), color='coral', width=barWidth, label='Predicted')
            #plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='var3')

            # Add xticks on the middle of the group bars
            plt.xlabel('Sample', fontweight='bold')
            plt.xticks(r2,r1)
            plt.title(r'%s model K fold = %d' % (classifier,i))
            # Create legend & Show graphic
            plt.legend()
            plt.show()
   
    
    mean_rmse = np.array(rmse_scores).mean()
    std_rmse =  np.array(rmse_scores).std()
    print("%s RMSE : %0.2f (+/- %0.2f)" % (classifier, mean_rmse, std_rmse))

    mean_r2 = np.array(k_scores).mean()
    std_r2 =  np.array(k_scores).std()
    print("%s R2 Score: %0.2f (+/- %0.2f)" % (classifier, mean_r2, std_r2))
    
    return rmse_scores, k_scores, mean_rmse, mean_r2
