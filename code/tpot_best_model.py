import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('../data/IMU.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('activity', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['activity'], random_state=1)

# Average CV score on the training set was: 0.9824570200573066
exported_pipeline = DecisionTreeClassifier(criterion="gini", max_depth=9, min_samples_leaf=2, min_samples_split=5)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 1)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)