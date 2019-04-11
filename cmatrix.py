"""Calculate a Confusion Matrix for multi-class classification results
2019 Colin Dietrich
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ConfusionMatrix:
    """Calculate a Confusion Matrix

    Parameters
    ----------
    y : array-like, true labels
    p : array-like, predicted labels of same type and length as y

    Attributes
    ----------
    y : see above
    p : see above
    df : Pandas DataFrame, aligned y and p data
    dfg : Pandas DataFrame, grouped by (y, p) combinations and counts
    a : Numpy array, confusion matrix values
    df_cm : Pandas DataFrame, confusion matrix values with row/column labels
    """

    def __init__(self, y, p):

        assert type(y) == type(p)
        assert len(y) == len(p)

        self.categorical = False
        if isinstance(y[0], str):
            self.categorical = True

        if self.categorical:
            from sklearn import preprocessing
            le = preprocessing.LabelEncoder()
            le.fit(y)
            y = le.transform(y)
            p = le.transform(p)

        self.y = y
        self.p = p

        self.df = pd.DataFrame({'y': self.y, 'p': self.p})
        self.n_classes = len(self.df.y.unique())

        self.labels = range(self.n_classes)
        if self.categorical:
            self.labels = le.inverse_transform(self.labels)

        self.a = np.zeros((self.n_classes, self.n_classes))

        self.dfg = self.df.groupby(['p', 'y']).size().reset_index().rename(columns={0: 'n'})
        _ = self.dfg.apply(lambda x: self.assemble_cm(x), axis=1)

        self.df_cm = pd.DataFrame(self.a, self.labels, self.labels)

        self.cmap = 'Greens'

    def assemble_cm(self, row):
        """Assemble a confusion matrix

        Parameters
        ----------
        row : row in a Pandas DataFrame with columns 'y_true', 'y_pred', 'count'
            'y_true' = true value of input data sample
            'y_pred' = predicted value of input data sample
            'n' = number of times this prediction combination occurred
        """
        y = row.y
        p = row.p
        self.a[y, p] = self.a[y, p] + row.n

    def plot(self, font_scale=1.4, axis_labels=True, ticklabels=False, figsize=5,
             **kwargs):
        """Plot a Confusion Matrix

        Parameters
        ----------
        font_scale : float, font scale multiplier
        axis_labels : bool, show axis labels
        ticklabels : bool, show ticklabels
        figsize : float, size in inches to make square confusion matrix
        **kwargs : keyword arguments to pass to seaborn.heatmap method
        """

        xticklabels = False
        yticklabels = False

        if ticklabels:
            xticklabels = self.labels
            yticklabels = self.labels
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        sns.set(font_scale=font_scale)
        sns.heatmap(self.df_cm,
                    xticklabels=xticklabels, yticklabels=yticklabels,
                    square=True, ax=ax, **kwargs)
        if axis_labels:
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
        plt.show();
