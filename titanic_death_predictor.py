import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree


def convert_gender_to_numerical(column):
    column[np.argwhere(column == 'male')] = 0
    column[np.argwhere(column == 'female')] = 1


def fill_nan_from_distribution(df, column):
    # get distribution
    count = df.loc[:, column].value_counts(normalize=True)
    missing = df.loc[:, column].isnull()
    df.loc[missing, column] = np.random.choice(count.index, size=df[missing].shape[0])


if __name__ == '__main__':
    # read train dataset
    train_ds = pd.read_csv("train.csv", delimiter=",")
    # fill age nan from distribution
    fill_nan_from_distribution(train_ds, "Age")
    # convert to numpy array
    train_ds = train_ds.to_numpy()
    # convert 'male' and 'female' to 0 and 1
    convert_gender_to_numerical(train_ds[:, 4])
    # select only part of the columns for the training
    fit_df = np.vstack((train_ds[:, 2], train_ds[:, 4:6].T)).T
    fit_df = fit_df.astype(int)
    # create decision tree
    clf = tree.DecisionTreeClassifier()
    # train decision tree with train dataset
    clf.fit(fit_df, train_ds[:, 1].astype(int))
    # plot decision tree
    plt.figure(figsize=(2 ^ 16, 2 ^ 16))
    dot_data = tree.plot_tree(clf, feature_names=("class", "gender", "Age"), class_names=("dead", "survived"),
                              max_depth=3)

    # load test dataset
    test_ds = pd.read_csv("test.csv", delimiter=",")
    # fill age nan from distribution
    fill_nan_from_distribution(test_ds, "Age")
    # convert to numpy array
    test_ds = test_ds.to_numpy()
    # convert 'male' and 'female' to 0 and 1
    convert_gender_to_numerical(test_ds[:, 3:4])
    # select only some column from test dataset
    predict_df = np.vstack((test_ds[:, 1], test_ds[:, 3:5].T)).T
    # predict survival outcome
    survival = clf.predict(predict_df)

    # join survival outcome with passenger IDs
    result = np.vstack((test_ds[:, 0], survival))

    # save to csv
    np.savetxt("result.csv", result.T, fmt="%d", delimiter=",", header="PassengerId,Survived", comments="")

    # show dialog with decision tree diagram
    plt.show()
