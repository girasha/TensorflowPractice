import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics

iris = tf.contrib.learn.datasets.load_dataset('iris')
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 5, random_state = 42)

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train)
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 3, 20ls], n_classes=3)

classifier.fit(x_train,y_train, steps = 200)
predictions = list(classifier.predict(x_test, as_iterable=True))

score = metrics.accuracy_score(y_test, predictions)
print (score)