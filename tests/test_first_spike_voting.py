import numpy as np

from spilearn.estimators import FirstSpikeVotingClassifier


X_train = np.array(
    [
        [1.0, np.nan, 1.0],
        [2.0, 8.0, np.nan],
        [5.0, 2.0, 4.0],
        [6.0, 3.0, 5.0],
    ]
)
y_train = np.array([10, 10, 20, 20])

classifier = FirstSpikeVotingClassifier(h_time=25.0).fit(X_train, y_train)

np.testing.assert_array_equal(classifier.assignments_, [10, 20, 10])
np.testing.assert_array_equal(
    classifier.predict([[1.0, np.nan, 2.0], [6.0, 1.0, 5.0]]),
    [10, 20],
)
