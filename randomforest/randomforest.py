import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

data_path = "../"

# Swiss army knife function to organize the data

def encode(train_data, test_data):
	le = preprocessing.LabelEncoder().fit(train_data.species) 
	labels = le.transform(train_data.species)           # encode species strings
	classes = list(le.classes_)                    # save column names for submission
	test_ids = test_data.pop('id')                             # save test_data ids for submission
	train_data = train_data.drop(['species', 'id'], axis=1)

	train_data = preprocessing.StandardScaler().fit(train_data).transform(train_data)
	test_data = preprocessing.StandardScaler().fit(test_data).transform(test_data)

	return train_data, labels, test_data, test_ids, classes

train_data = pd.read_csv(data_path + "train.csv")
test_data = pd.read_csv(data_path + "test.csv")
train_data, labels, test_data, test_ids, classes = encode(train_data, test_data)

#split the training set (known classes) using StatifiedShuffleSplit method

n_splits = 10
sss = StratifiedShuffleSplit(n_splits, test_size=0.2, random_state=23) #test set size is 20% of total training data size

for train_index, test_index in sss.split(train_data, labels): #shuffling data n times and only keeping last shuffle
	X_train, X_test = train_data[train_index], train_data[test_index]
	y_train, y_test = labels[train_index], labels[test_index]

#train model
clf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, random_state=40, verbose = True)
clf.fit(X_train, y_train)

print('****Results****')
y_predict = clf.predict(X_test)
acc = accuracy_score(y_test, y_predict)
print("Accuracy: {:.4%}".format(acc))
  
y_predict_prob = clf.predict_proba(X_test)
ll = log_loss(y_test, y_predict_prob)
print("Log Loss: {}".format(ll))

#cm = confusion_matrix(y_test, y_predict)
#plt.imshow(cm)
#plt.show()

##compute submission

#use full training data set to fit this time
clf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, random_state=40, verbose = True)
clf.fit(train_data, labels)
y_predict_prob = clf.predict_proba(test_data)

# Get the names of the column headers
text_labels = sorted(pd.read_csv(data_path + "train.csv").species.unique())

## Converting the test predictions in a dataframe as depicted by sample submission
y_submission = pd.DataFrame(y_predict_prob, index=test_ids ,columns=text_labels)
print('Creating and writing submission...')
fp = open('submission.csv', 'w')
fp.write(y_submission.to_csv())
print('Finished writing submission')
## Display the submission
y_submission.tail()
