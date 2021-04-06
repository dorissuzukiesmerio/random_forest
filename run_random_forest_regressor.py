import pandas
import numpy

from sklearn.model_selection import KFold
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyyplot as plt

from sklearn.tree import export_graphviz # drawing the decision tree . Do on linux - on the Linode. 


dataset = pandas.read_csv("temperature_data.csv")

#EXTRA : TRY IMPUTATION ON DATA

print(dataset)
print(dataset.shape)
print(dataset.describe())



dataset = pandas.get_dummies(dataset)
#recognizes the text and makes into dummies
print(dataset)
print(dataset.columns)


# hear().T

# We want to predict the actual temperature
target = dataset['actual'].values # array
print(target)

data = dataset.drop('actual', axis = 1).values # because we want to use the columns and then actual is in the middle

kfold_object = KFold(n_splits = 4) # Attentive ! K is capital letter
kfold_object.get_n_splits(data)

i = 0 # We could call i= test_case
for training_index, test_index in kfold_object.split(data):
	print(i)
	i = i + 1
	print("training:", training_index)
	print("test:",test_index)
	# data_training = data[training_index]
	# data_test = data[test_index]
	target_training = target[training_index]
	target_test = target[test_index]
	machine = RandomForestRegressor(n_estimators = 201, max_depth = 30) 
	machine.fit(data_training, target_training)
	new_target = machine.predict(data_test)#instead of new_data
	#We run pretending we don't know the predicted. But it is target_ . We want to see how it is 
	print("Mean Absolute Error", metrics.mean_absolute_error(target_test, new_target))

#We want the smallest possible Mean Abs Error


#Questions:
# 1. get dummies
#why # forloop
#choice of 201
#Why can't we plot the one that is not the raw?? sort than 

machine = RandomForestRegressor(n_estimators = 201, max_depth = 30) # We don't split , but use the whole dataset
machine.fit(data, target)
feature_list = dataset.drop('actual', axis = 1).columns # to keep it organized and clear
feature_importances_raw= list(machine.feature_importances_)  # internal use _ . High chance that the name may change in the near future
print(feature_importances)
# How to interpret 

feature_importances = [(feature, round(importance, 3)) for feature, importance in zip(feature_list, feature_importances_raw)]# zip is to pack the list together
print(feature_importances)
sorted(feature_importances, key= lambda x:x[1], reverse=True ) # more important first, lest latter. 
#Sort by number and not by name
# You can use operations inside the parenthesis [], like square, etc
print(feature_importances)

# print('{} {}'.format(*i)) for i in feature_importances
[print('{:13} : {}'.format(*i)) for i in feature_importances] # :13 is number of characters between . Just to make it make it clearer to see 

# So the first rows shows us the most important features. The higher the feature_ importances the most important it is a criteria to include in the analysis
#Given that we already now the historical temperature, then month is not important
#Given that we already now yesterday's temperature,     then two days ago temperature is not important


#Graph
x_values = list(range(len(feature_importances_raw))) # the explanatory variables, and it will be the number of bars that we will have
#lenght : 10 
#range: form an array 0 to 9 
# put in a list so matplot can read it
plt.ylabel('importance')
plt.xlabel('feature')
plt.title('Feature Importance !! ')
plt.bar(x_values, feature_importances_raw, orientation = 'vertical')
plt.xticks(x_values, feature_list , rotation = 'vertical') #xticks is the little 
plt.tight_layout() #makes sure everything is inside 
plt.savefig("feature_importances.png")
plt.close()


#Importance of 



# Drawing a tree to represent the decision tree