import matplotlib.pyplot as plt
import numpy as np
import pickle


def results_ada_boost(dir):

    # Prepare containers for results
    recall = np.empty(shape=(191,), dtype=float)
    precision = np.empty(shape=(191,), dtype=float)
    f1 = np.empty(shape=(191,), dtype=float)

    # Parse result files for each number of trees and average F1, recall,
    # precision and confusion matrices
    for num_trees in range(10, 201):

        # Prepare results container
        adab = { "cm": np.zeros((2,2), dtype=float),
                 "recall": 0.0,
                 "precision": 0.0,
                 "f1": 0.0 }

        for run in range(1, 21):
            res = pickle.load(open(dir + "Ada_boost_run" + str(run) +\
                                   "_trees" + str(num_trees) + ".p", "rb"))

            cm = np.array(res["cm"], dtype=float)
            cm[0,:] /= np.sum(cm[0,:])
            cm[1,:] /= np.sum(cm[1,:])
            for key in adab:
                if key != "cm":
                    adab[key] += res[key]
            adab["cm"] += cm

        # Display results for adaptive boosting with current number of trees
        """print "Results for adaptive boosting with " + str(num_trees) + " trees:"
        for key in adab:
            adab[key] /= 20.0
            print key, ":", adab[key]"""

        recall[num_trees - 10] = adab["recall"] / 20.0
        precision[num_trees - 10] = adab["precision"] / 20.0
        f1[num_trees - 10] = adab["f1"] / 20.0

    t = np.arange(10, 201, 1)
    plt.plot(t, recall, color='blue')
    plt.plot(t, precision, color='green')
    plt.plot(t, f1, color='red')
    plt.xlabel('t = Number of trees')
    plt.ylabel('Recall - blue, precision - green, f1 - red')
    plt.tight_layout()
    plt.show()


def results_bagging(dir):

    # Prepare results container
    bagging_errors = np.empty(shape=(191,), dtype=float)

    # Parse result files for each number of trees and record OOB error rate
    for num_trees in range(10, 201):
        res = pickle.load(open(dir + "Bagging_trees" + str(num_trees) + ".p",\
                               "rb"))
        bagging_errors[num_trees - 10] = res["oob_error"]

    t = np.arange(10, 201, 1)
    plt.plot(t, bagging_errors)
    plt.xlabel('t = Number of trees')
    plt.ylabel('Out-of-bag error rate')
    plt.tight_layout()
    plt.show()


def results_DT(dir, max_depth):

    # Prepare results container
    dt = { "cm": np.zeros((2,2), dtype=float),
           "recall": 0.0,
           "precision": 0.0,
           "f1": 0.0 }

    # Parse result files for each run and average F1, recall, precision and
    # confusion matrices
    for run in range(1, 21):
        if max_depth != "":
            res = pickle.load(open(dir + "Decision_tree_run" + str(run) +\
                                   "_depth" + max_depth + ".p", "rb"))
        else:
            res = pickle.load(open(dir + "Decision_tree_run" + str(run) + ".p",\
                                   "rb"))

        cm = np.array(res["cm"], dtype=float)
        cm[0,:] /= np.sum(cm[0,:])
        cm[1,:] /= np.sum(cm[1,:])
        for key in dt:
            if key != "cm":
                dt[key] += res[key]
        dt["cm"] += cm

    # Display results for decision trees
    if max_depth != "":
        print "Results for decision trees with max depth " + max_depth + ":"
    else:
        print "Results for decision trees with no bound on depth:"
    for key in dt:
        dt[key] /= 20.0
        print key, ":", dt[key]


def results_RF(dir):

    # Prepare results container
    rf_errors_log2 = np.empty(shape=(491,), dtype=float)
    rf_errors_sqrt= np.empty(shape=(491,), dtype=float)

    # Parse result files for each number of trees and record OOB error rate
    for num_trees in range(10, 501):
        res = pickle.load(open(dir + "Random_forest_trees" + str(num_trees) +\
                               "_sqrt.p", "rb"))
        rf_errors_sqrt[num_trees - 10] = res["oob_error"]
        res = pickle.load(open(dir + "Random_forest_trees" + str(num_trees) +\
                               "_log2.p", "rb"))
        rf_errors_log2[num_trees - 10] = res["oob_error"]

    t = np.arange(10, 501, 1)
    plt.plot(t, rf_errors_sqrt, color='blue')
    plt.plot(t, rf_errors_log2, color='green')
    plt.xlabel('t = Number of trees')
    plt.ylabel('OOB error - RFs with sqrt (blue) and log2 (green) max features')
    plt.tight_layout()
    plt.show()


def results_SVM_linear(dir, C):

    # Prepare results container
    svmlin = { "cm": np.zeros((2,2), dtype=float),
               "recall": 0.0,
               "precision": 0.0,
               "f1": 0.0 }

    # Parse result files for each run and average F1, recall, precision and
    # confusion matrices
    for run in range(1, 21):
        res = pickle.load(open(dir + "SVM_linear_C" + str(C) + "_run" +\
                               str(run) + ".p", "rb"))

        cm = np.array(res["cm"], dtype=float)
        cm[0,:] /= np.sum(cm[0,:])
        cm[1,:] /= np.sum(cm[1,:])
        for key in svmlin:
            if key != "cm":
                svmlin[key] += res[key]
        svmlin["cm"] += cm

    # Display results for linear SVMs
    print "Results for linear SVMs with C = " + str(C) + ":"
    for key in svmlin:
        svmlin[key] /= 20.0
        print key, ":", svmlin[key]


def results_SVM_poly(dir, C, deg):

    # Prepare results container
    svmpoly = { "cm": np.zeros((2,2), dtype=float),
                "recall": 0.0,
                "precision": 0.0,
                "f1": 0.0 }

    # Parse result files for each run and average F1, recall, precision and
    # confusion matrices
    for run in range(1, 21):
        res = pickle.load(open(dir + "SVM_poly_C" + str(C) + "_deg" +\
                               str(deg) + "_run" + str(run) + ".p", "rb"))

        cm = np.array(res["cm"], dtype=float)
        cm[0,:] /= np.sum(cm[0,:])
        cm[1,:] /= np.sum(cm[1,:])
        for key in svmpoly:
            if key != "cm":
                svmpoly[key] += res[key]
        svmpoly["cm"] += cm

    # Display results for polynomial SVMs
    print "Results for polynomial kernel SVMs with C = " + str(C) +\
          "and deg = " + str(deg) + ":"
    for key in svmpoly:
        svmpoly[key] /= 20.0
        print key, ":", svmpoly[key]


def results_SVM_rbf(dir, C, gamma):

    # Prepare results container
    svmrbf = { "cm": np.zeros((2,2), dtype=float),
               "recall": 0.0,
               "precision": 0.0,
               "f1": 0.0 }

    # Parse result files for each run and average F1, recall, precision and
    # confusion matrices
    for run in range(1, 21):
        res = pickle.load(open(dir + "SVM_rbf_C" + str(C) + "_gamma" +\
                               str(gamma) + "_run" + str(run) + ".p", "rb"))

        cm = np.array(res["cm"], dtype=float)
        cm[0,:] /= np.sum(cm[0,:])
        cm[1,:] /= np.sum(cm[1,:])
        for key in svmrbf:
            if key != "cm":
                svmrbf[key] += res[key]
        svmrbf["cm"] += cm

    # Display results for SVMs with RBF kernel
    print "Results for RBF kernel SVMs with C = " + str(C) + "and gamma = " +\
          str(gamma) + ":"
    for key in svmrbf:
        svmrbf[key] /= 20.0
        print key, ":", svmrbf[key]


def results_grad_boost(dir, max_depth, num_trees):

    # Prepare results container
    gradb = { "cm": np.zeros((2,2), dtype=float),
             "recall": 0.0,
             "precision": 0.0,
             "f1": 0.0 }

    for run in range(1, 21):
        res = pickle.load(open(dir + "Gradient_boosting_run" + str(run) +\
                               "_depth" + str(max_depth) + "_trees" +\
                               str(num_trees) + ".p", "rb"))

        cm = np.array(res["cm"], dtype=float)
        cm[0,:] /= np.sum(cm[0,:])
        cm[1,:] /= np.sum(cm[1,:])
        for key in gradb:
            if key != "cm":
                gradb[key] += res[key]
        gradb["cm"] += cm

    # Display results for gradient boosting with current number of trees and
    # maximum depth
    print "Results for gradient boosting with " + str(num_trees) + " trees, " +\
          str(max_depth) + " depth:"
    for key in gradb:
        gradb[key] /= 20.0
        print key, ":", gradb[key]


if __name__=='__main__':

    for depth in ["", "1", "2", "4", "7"]:
        results_DT("./DT/", depth)

    results_bagging("./Bagging/")

    results_ada_boost("./AdaBoost/")

    results_RF("./RF/")

    for C in [0.1, 1.0, 10.0]:
        results_SVM_linear("./SVM/", C)

        for deg in range(2, 16):
            if deg != 8:
                results_SVM_poly("./SVM/", C, deg)

        for gamma in [0.1, 1.0, 10.0]:
            results_SVM_rbf("./SVM/", C, gamma)

    for max_depth in range(2, 11):
        for num_trees in range(10,100):
            results_grad_boost("./GradBoost/", max_depth, num_trees)
