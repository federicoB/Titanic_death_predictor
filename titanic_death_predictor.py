import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import KFold


def convert_gender_to_numerical(column):
    """ Converts string gender feature to boolean"""
    column[np.argwhere(column == 'male')] = 0
    column[np.argwhere(column == 'female')] = 1


def fill_nan_from_distribution(df, column):
    """ replace nan value with values sampled from a distribution built from non-nan values"""
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
    # class, sex, age, fare
    fit_df = np.vstack((train_ds[:, 2], train_ds[:, 4:6].T, train_ds[:, 9].astype(float))).T

    # deciding best depth for the trees

    cv = KFold(n_splits=10)
    # tree depth range from 1 to max number of features
    depth_range = range(1, fit_df.shape[1] + 1)
    # cycle invariants
    best_model = None
    best_accuracy = 0
    for depth in depth_range:
        # create decision tree
        clf = tree.DecisionTreeClassifier(max_depth=depth)
        # for each fold make split, train on subset and keep best model
        for train_fold, valid_fold in cv.split(fit_df):
            model = clf.fit(fit_df[train_fold], train_ds[train_fold, 1].astype(int))
            validation_accuracy = model.score(fit_df[valid_fold], train_ds[valid_fold, 1].astype(int))
            if validation_accuracy > best_accuracy:
                best_model = model
                best_accuracy = validation_accuracy

    print("accuracy " + str(best_accuracy))
    # plot decision tree
    plt.figure(figsize=(2 ^ 16, 2 ^ 16))
    dot_data = tree.plot_tree(best_model, feature_names=("class", "gender", "Age", "Fare"),
                              class_names=("dead", "survived"))

    # load test dataset
    test_ds = pd.read_csv("test.csv", delimiter=",")
    # fill age nan from distribution
    fill_nan_from_distribution(test_ds, "Age")
    fill_nan_from_distribution(test_ds, "Fare")
    # convert to numpy array
    test_ds = test_ds.to_numpy()
    # convert 'male' and 'female' to 0 and 1
    convert_gender_to_numerical(test_ds[:, 3:4])
    # select only some column from test dataset
    predict_df = np.vstack((test_ds[:, 1], test_ds[:, 3:5].T, test_ds[:, 8].astype(float))).T
    # predict survival outcome
    survival = best_model.predict(predict_df)

    # join survival outcome with passenger IDs
    result = np.vstack((test_ds[:, 0], survival))

    # save to csv
    np.savetxt("result.csv", result.T, fmt="%d", delimiter=",", header="PassengerId,Survived", comments="")

    # show dialog with decision tree diagram
    plt.show()
