from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


fashion_mnist = keras.datasets.fashion_mnist

(trainFeatures, trainLabels), (testFeatures, testLabels) = fashion_mnist.load_data()
accuracies = []

trainFeatures = trainFeatures[:30000]
trainLabels = trainLabels[:30000]

for i in range(1, 26, 2):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    placeholderTrainFeatures = tf.compat.v1.placeholder(trainFeatures.dtype, 
                                                        shape=trainFeatures.shape)
    placeholderTrainLabels = tf.compat.v1.placeholder(trainLabels.dtype, 
                                                      shape=trainLabels.shape)
    placeholderTest = tf.compat.v1.placeholder(testFeatures.dtype, 
                                               (28, 28))
    x = tf.cast(placeholderTrainFeatures, 'float32')
    y = tf.cast(placeholderTest, 'float32')
    substracted = tf.subtract(x, y) 
    distance = tf.sqrt(tf.reduce_sum(tf.square(substracted), axis=(1, 2)))

    _, indices = tf.nn.top_k(tf.negative(distance), k=i, sorted=False)
    top_k_labels = tf.gather(placeholderTrainLabels, indices)
    labels, _, counts = tf.unique_with_counts(top_k_labels)
    prediction = tf.gather(labels, tf.argmax(counts))

    accuracy = 0.

    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for testFeature, testLabel in zip(testFeatures, testLabels):
            predicted = sess.run(prediction, 
                            feed_dict={placeholderTrainFeatures: trainFeatures[:],
                                       placeholderTrainLabels : trainLabels[:],
                                       placeholderTest: testFeature})

            if predicted == testLabel:
                accuracy += 1./ len(testFeatures)
        accuracies.append((i, accuracy))
        print("Done!")
        print("Accuracy:", accuracy)

print(accuracies)

plt.plot(*zip(*accuracies))
plt.xlabel('Accuracy')
plt.ylabel('k')
plt.show()
