import data

import numpy as np
import pickle
from sklearn import metrics, svm, tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier,\
                             GradientBoostingClassifier, AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, recall_score, precision_score,\
                            f1_score


def SVM_linear(X_train, X_test, y_train, y_test, C, run):
    SVM_linear = svm.SVC(kernel='linear', C=C)
    SVM_linear.fit(X_train, y_train)

    y_pred = SVM_linear.predict(X_test)
    print "SVM_linear_C" + str(C) + "_run" + str(run) + ":"
    print metrics.classification_report(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    res = { "cm" : cm, "recall" : recall, "precision" : precision, "f1" : f1 }
    pickle.dump(res,
                open("SVM_linear_C" + str(C) + "_run" + str(run) + ".p", "wb"))


def SVM_poly(X_train, X_test, y_train, y_test, C, deg, run):
    SVM_poly = svm.SVC(kernel='poly', C=C, degree=deg)
    SVM_poly.fit(X_train, y_train)

    y_pred = SVM_poly.predict(X_test)
    print "SVM_poly_C" + str(C) + "_deg" + str(deg) + "_run" + str(run) + ":"
    print metrics.classification_report(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    res = { "cm" : cm, "recall" : recall, "precision" : precision, "f1" : f1 }
    pickle.dump(res,
                open("SVM_poly_C" + str(C) + "_deg" + str(deg) + "_run" +\
                     str(run) + ".p", "wb"))


def SVM_rbf(X_train, X_test, y_train, y_test, C, gamma, run):
    SVM_rbf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
    SVM_rbf.fit(X_train, y_train)

    y_pred = SVM_rbf.predict(X_test)
    print "SVM_rbf_C" + str(C) + "_gamma" + str(gamma) + "_run" + str(run) + ":"
    print metrics.classification_report(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    res = { "cm" : cm, "recall" : recall, "precision" : precision, "f1" : f1 }
    pickle.dump(res,
                open("SVM_rbf_C" + str(C) + "_gamma" + str(gamma) + "_run" +\
                     str(run) + ".p", "wb"))


def decision_tree(X_train, X_test, y_train, y_test, max_depth, run):
    dt = tree.DecisionTreeClassifier()
    dt = dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)
    print "Decision_tree_run" + str(run) + ":"
    print metrics.classification_report(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    res = { "cm" : cm, "recall" : recall, "precision" : precision, "f1" : f1 }
    pickle.dump(res,
                open("Decision_tree_run" + str(run) + "_depth" +\
                     str(max_depth) + ".p", "wb"))


def bagging(X, y, num_trees):
    b = BaggingClassifier(base_estimator=None,
                          n_estimators=num_trees,
                          oob_score=True)
    b.fit(X, y)

    oob_error = 1 - b.oob_score_
    print "Bagging_trees" + str(num_trees) + ":", oob_error
    res = { "oob_error" : oob_error }
    pickle.dump(res, open("Bagging_trees" + str(num_trees) + ".p", "wb"))


def random_forest(X, y, num_trees, max_features):
    rf = RandomForestClassifier(n_estimators=num_trees,
                                max_features=max_features,
                                oob_score=True)
    rf.fit(X, y)

    oob_error = 1 - rf.oob_score_
    print "Random_forest_trees" + str(num_trees) + "_" + max_features + ":",\
          oob_error
    res = { "oob_error" : oob_error }
    pickle.dump(res,
                open("Random_forest_trees" + str(num_trees) + "_" +\
                     max_features + ".p", "wb"))


def gradient_boosting(X_train, X_test, y_train, y_test, max_depth, num_trees,
                      run):
    gb = GradientBoostingClassifier(n_estimators=num_trees,
                                    max_depth=max_depth)
    gb.fit(X_train, y_train)

    y_pred = gb.predict(X_test)
    print "Gradient_boosting_run" + str(run) + "_depth" + str(max_depth) +\
          "_trees" + str(num_trees) + ":"
    print metrics.classification_report(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    res = { "cm" : cm, "recall" : recall, "precision" : precision, "f1" : f1 }
    pickle.dump(res,
                open("Gradient_boosting_run" + str(run) + "_depth" +\
                     str(max_depth) + "_trees" + str(num_trees) + ".p", "wb"))


def adaptive_boosting(X_train, X_test, y_train, y_test, num_trees, run):
    adab = AdaBoostClassifier(n_estimators=num_trees)
    adab.fit(X_train, y_train)

    y_pred = adab.predict(X_test)
    print "Ada_boost_run" + str(run) + "_trees" + str(num_trees) + ":"
    print metrics.classification_report(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    res = { "cm" : cm, "recall" : recall, "precision" : precision, "f1" : f1 }
    pickle.dump(res,
                open("Ada_boost_run" + str(run) + "_trees" + str(num_trees) +\
                     ".p", "wb"))


def evaluation():
    (Xs, ys) = data.get_RA_data()
    test_count = Xs.shape[0] / 5

    for run in range(1, 21):
        indices = np.random.permutation(Xs.shape[0])

        X_train = Xs[indices[:-test_count]]
        y_train = ys[indices[:-test_count]]
        X_test = Xs[indices[-test_count:]]
        y_test = ys[indices[-test_count:]]

        # Support vector machines
        for C in [0.1, 1.0, 10.0]:
            # SVM with linear kernel
            SVM_linear(X_train, X_test, y_train, y_test, C, run)
            # SVM with polynomial kernel
            for deg in [2,9,10,11,12,13,14,15]:
                SVM_poly(X_train, X_test, y_train, y_test, C, deg, run)
            # SVM with RBF kernel
            for gamma in [0.1, 1.0, 10.0]:
                SVM_rbf(X_train, X_test, y_train, y_test, C, gamma, run)

        # Decision trees
        for max_depth in [1,2,4,7]:
            decision_tree(X_train, X_test, y_train, y_test, max_depth, run)

        # Bagging
        for num_trees in range(10, 201):
            bagging(Xs, ys, num_trees)

        # Random forests
        for num_trees in range(10, 501):
            random_forest(Xs, ys, num_trees, "sqrt")
            random_forest(Xs, ys, num_trees, "log2")

        # Gradient boosting
        for max_depth in range(2, 11):
            for num_trees in range(10,100):
                gradient_boosting(X_train, X_test, y_train, y_test, max_depth,
                                  num_trees, run)

        # Adaptive boosting
        for num_trees in range(10,201):
            adaptive_boosting(X_train, X_test, y_train, y_test, num_trees, run)


if __name__=='__main__':
    evaluation()
