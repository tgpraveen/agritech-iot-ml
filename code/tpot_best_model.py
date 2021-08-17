import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv("../data/IMU.csv", dtype={'activity' : 'category'}, parse_dates=['UnixTime','gps_unixTime'], date_parser=lambda epoch: pd.to_datetime(float(epoch)/1000))
features = tpot_data.drop(['activity', 'UnixTime', 'gps_unixTime'], axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['activity'], random_state=1)

# Average CV score on the training set was: 0.9824570200573066
exported_pipeline = DecisionTreeClassifier(criterion="gini", max_depth=9, min_samples_leaf=2, min_samples_split=5)

# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 1)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

fig = plt.figure(figsize=(25,20))
tree.plot_tree(exported_pipeline, feature_names=features.columns, class_names=tpot_data['activity'], filled=True)
fig.savefig("decision_tree_1.pdf")
fig.savefig("decision_tree_1.png")        