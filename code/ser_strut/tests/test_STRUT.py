import sys
import graphviz
import pickle
import copy
import matplotlib.pyplot as plt
sys.path.append('../')
from STRUT import *


SAVE = 0
LOAD = 0

clf_init_name = 'DT_initial.pkl'
clf_out_name = 'DT_out.pkl'
Xs_name = 'X_s.npy'
ys_name = 'y_s.npy'
Xt_name = 'X_t.npy'
yt_name = 'y_t.npy'

dataset_length = 500
D = 2
IMBALANCE = False
# nb ones left in imbalanced target set
nb_ones = 10
MAX_DEPTH = None
PRUNING_UPDATED_NODE = False

REPETITION = 1

if LOAD:
    clf = pickle.load(open(clf_init_name, 'rb'))
    X = np.load(Xs_name)
    Y = np.load(ys_name)
    X_t = np.load(Xt_name)
    Y_t = np.load(yt_name)

else:
    for i in range(REPETITION):
        print("REP ", i+1)
        # Build a dataset
        X = np.random.randn(dataset_length, D) * 0.1
        # X[0:dataset_length // 2, 0] += 0.1
        # X[dataset_length // 2:, 0] -= 0.1
        # X[0:dataset_length // 2, 1] += 0.05
        # X[dataset_length // 2:, 1] -= 0.05
        Y = np.ones(dataset_length)
        Y[0:dataset_length // 2] *= 0

        # Train a Tree
        clf = tree.DecisionTreeClassifier(max_depth=MAX_DEPTH)
        clf = clf.fit(X, Y)

        # Plot the classification threshold
        plt.plot(X[0:dataset_length // 2, 0], X[0:dataset_length // 2, 1], "ro")
        plt.plot(X[dataset_length // 2:, 0], X[dataset_length // 2:, 1], "bo")
        for node, feature in enumerate(clf.tree_.feature):
            if feature == 0:
                plt.axvline(x=clf.tree_.threshold[node])
            elif feature == 1:
                plt.axhline(y=clf.tree_.threshold[node])
        # plt.show()

        # Build a new dataset
        X_t = np.random.randn(dataset_length, D) * 0.1
        #X[:,1] = np.nan
        # X_t[:, :] += 120
        X_t[0:dataset_length // 2, 0] += 3.2
        X_t[dataset_length // 2:, 0] += 3.1
        X_t[0:dataset_length // 2, 1] += 4.1
        X_t[dataset_length // 2:, 1] += 4.1
        Y_t = np.ones(dataset_length)
        Y_t[0:dataset_length // 2] *= 0
        if IMBALANCE:
            # Create imbalance
            X_t = X_t[0: dataset_length // 2 + nb_ones, :]
            Y_t = Y_t[0: dataset_length // 2 + nb_ones]
        # Plot the initial tree
        # dot_data = tree.export_graphviz(clf, out_file=None,
                                        # feature_names=["feature_" +
                                                       # str(i) for i in range(D)],
                                        # class_names=["class_0", "class_1"],
                                        # filled=True, rounded=True,
                                        # special_characters=True)
        # graph = graphviz.Source(dot_data)
        # graph.render('DT_init', view=False)

        # Save tree & data
        if SAVE:
            np.save(Xs_name, X)
            np.save(ys_name, Y)
            np.save(Xt_name, X_t)
            np.save(yt_name, Y_t)
            pickle.dump(clf, open(clf_init_name, 'wb'))


        # Call STRUT
        print("Applying STRUT")

        clf_out_noupdate = copy.deepcopy(clf)
        clf_out_update = copy.deepcopy(clf)

        print("STRUT 1")
        STRUT(clf_out_noupdate, 0, X_t, Y_t, X_t, Y_t,
              no_prune_on_cl=True,
              cl_no_prune=[1],
              pruning_updated_node=False)
        # print("STRUT 2")
        # STRUT(clf_out_update, 0, X_t, Y_t, X_t, Y_t,
              # no_prune_on_cl=True,
              # cl_no_prune=[1],
              # pruning_updated_node=True)
        # Plot the new thresholds

        # plt.plot(X_t[0:dataset_length // 2, 0], X_t[0:dataset_length // 2, 1], "ro")
        # plt.plot(X_t[dataset_length // 2:, 0], X_t[dataset_length // 2:, 1], "bo")
        # for node, feature in enumerate(clf_out.tree_.feature):
            # if feature == 0:
                # plt.axvline(x=clf_out.tree_.threshold[node])
            # elif feature == 1:
                # plt.axhline(y=clf_out.tree_.threshold[node])
        # plt.show()

if SAVE:
    pickle.dump(clf_out_noupdate, open(clf_out_name + '_noupdate', 'wb'))
    pickle.dump(clf_out_update, open(clf_out_name + '_update', 'wb'))

if PLOT:
    # Plot the new tree
    dot_data = tree.export_graphviz(clf_out_noupdate, out_file=None,
                                    feature_names=["feature_" +
                                                   str(i) for i in range(D)],
                                    class_names=["class_0", "class_1"],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render('DT_out_noupdate', view=False)
    dot_data = tree.export_graphviz(clf_out_update, out_file=None,
                                    feature_names=["feature_" +
                                                   str(i) for i in range(D)],
                                    class_names=["class_0", "class_1"],
                                    filled=True, rounded=True,
                                    special_characters=True)

    graph = graphviz.Source(dot_data)
    graph.render('DT_out_update', view=False)
