import numpy as np
import pandas as pd
from time import time 
from IPython.display import display

# Import supplementary visualization code visuals.py
import visuals as vs
%matplotlib inline
data = pd.read_csv("census.csv")


display(data.head(n=10))

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)

# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))、
# Visualize the new log distributions
vs.distribution(features_log_transformed, transformed = True)

from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() 
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n = 5))

features_final = pd.get_dummies(features_log_minmax_transform,prefix='income')

#Encode the 'income_raw' data to numerical values
income = int(income_raw==">50K")





# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

print (encoded)

import math 
TP = np.sum(income) 
FP = income.count() - TP # Specific to the naive case

TN = 0 
FN = 0 

#Calculate accuracy, precision and recall
accuracy = (TP+TN)/(FP+FN+TP+TN)
recall = TP/(TP+FN)
precision = TP/(TP+FP)

print(recall)
print(precision)

# Calculate F-score 
beta=0.5
fscore =((1+math.pow(beta,2))*precision*recall)/((math.pow(beta,2)*precision)+recall)

# Print the results 
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))

# Import two metrics from sklearn - fbeta_score and accuracy_score

from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
  
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    ## find the equation that fit this features and labels.
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end-start
        
    # Get the predictions on the test set and train set
    
    start = time() 
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() 
    
    
    
    # Calculate the total prediction time
    results['pred_time'] = end-start
            
    # Compute accuracy on the first 300 training samples 
    results['acc_train'] = accuracy_score(predictions_train[:300], y_train[:300])
        
    # Compute accuracy on test set 
    results['acc_test'] = accuracy_score(predictions_test, y_test)
    
    # Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(predictions_train[:300], y_train[:300],beta=1)
    ##
        
    # Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(predictions_test, y_test,beta=1)
       
 
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    return results

from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm

#  Initialize the three models
clf_A = tree.DecisionTreeClassifier()
clf_B = AdaBoostClassifier(n_estimators=100)
clf_C = svm.SVC()

#Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_100 = len(y_train)
samples_10 = len(y_train)//10
samples_1 = len(y_train)//100
# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)




from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer,fbeta_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
# Initialize the classifier

clf = AdaBoostClassifier(base_estimator=(DecisionTreeClassifier(max_depth=4)),random_state=42) 


# Doing Gridserach to optimize the model
parameters = {'learning_rate':[0.01,0.1,1,],'n_estimators':[50,100,150,200,400]}

# Calculate fbeta_score  
scorer =make_scorer(fbeta_score, beta=1)

grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

grid_fit = grid_obj.fit(X_train, y_train)

best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

best_clf
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))


from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree

# Train the  model on the training set 
model = AdaBoostClassifier(random_state=42)
model.fit(X_train,y_train)
# Extract the feature importances 
importances = model.feature_importances_

vs.feature_plot(importances, X_train, y_train)

from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))
