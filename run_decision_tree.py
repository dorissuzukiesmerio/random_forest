import pandas

from sklearn import tree
from sklearn.model_selection import KFold
from sklearn import metrics


dataset = pandas.read_csv("dataset.csv")

print(dataset)

target = dataset.iloc[:,30].values # Be careful to have the 
data = dataset.iloc[:,0:30].values

print(target)
print(data)

# machine = tree.DecisionTreeClassifier(criterion="gini", max_depth=10)
# machine = decisiontree.DecisionTreeClassifier(criterion="entropy", )
# Criterion chosen; then how many decision trees (in general, it will be relative to the number of columns that will be relevant to the prediction); 
#monotonic vs. strange pattern

#Gini : correct , size weighted -> but size is more important criteria ; faster
#Entropy : correct , size weighted -> but correct is more important criteria
#You do not decrease entropy when you chop the sample

kfold_object = KFold(n_splits = 4)
kfold_object.get_n_splits(data)

print(kfold_object) 

i = 0 # We could call i= test_case
for training_index, test_index in kfold_object.split(data):
	print(i)
	i = i + 1
	print("training:", training_index)
	print("test:",test_index)
	data_training = data[training_index]
	data_test = data[test_index]
	target_training = target[training_index]
	target_test = target[test_index]
	machine = tree.DecisionTreeClassifier(criterion="gini", max_depth=30) #copy, paste and adapt from kfold #try different max_depth = 10 , 30, 3
	machine.fit(data_training, target_training)
	new_target = machine.predict(data_test)
	print("Accuracy score:", metrics.accuracy_score(target_test, new_target))# use acc instead of r2
	print("Confusion matrix: \n ", metrics.confusion_matrix(target_test, new_target))# use acc instead of r2	

#Interpretation:
#Being too simple: see confusion matrix: model doesn't classify any as 0. Which is a problem. 
#Accuracy score decreases just a little bit when we change max_depth, but if we do the model too simple, then the 