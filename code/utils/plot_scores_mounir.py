import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
sys.path.insert(0, "../data_mngmt/")
import lib_data


def all_comments_score(path_scores):
    df_scores = pd.read_csv(path_scores)
    comment_set = set(list(df_scores.comment))
    return comment_set

# =======================================================
#   Parameters
# =======================================================
# path_scores = '../../outputs/scores/scores.csv'
# path_scores = '../../outputs/scores/scores_5.csv'
# path_scores = '../../outputs/scores/scores_cmla_5_11_2018.csv'
path_scores = '../../outputs/scores/synth_data/scores_095.csv'

comment_set = all_comments_score(path_scores)
print("All diff comments : ")
for el in comment_set:
    print(el)


# wine, tarkett_transfer
# DATASET = 'tarkett_transfer'
# score 075
# DATASET_LIST = ['SYNTH_IMBdrift5',
                # 'SYNTH_IMBn_clust(15;15)->(10;10)',
                # 'SYNTH_IMBn_clust(15;15)->(15;15)',
                # 'SYNTH_IMBn_clust(15;15)->(20;20)',
                # 'SYNTH_IMBvar_change0.5-2']

# score 095
DATASET_LIST = ['SYNTH_IMBdrift5',
                'SYNTH_IMBn_clust(15;15)->(10;10)',
                'SYNTH_IMBn_clust(15;15)->(15;15)',
                'SYNTH_IMBn_clust(15;15)->(20;20)',
                'SYNTH_IMBprop_only0.75',
                'SYNTH_IMBprop_only0.9',
                'SYNTH_IMBprop_only0.95',
                'SYNTH_IMBprop_only0.98',
                'SYNTH_IMBvar_change0.5-2']
# DATASET = 'SYNTH_IMBdrift5'
# DATASET = 'SYNTH_IMBn_clust(15,15)->(10,10)'
# DATASET = 'SYNTH_IMBn_clust(15,15)->(15,15)'
# DATASET = 'SYNTH_IMBn_clust(15,15)->(20,20)'
# DATASET = 'SYNTH_IMBprop_only0.75'
# DATASET = 'SYNTH_IMBprop_only0.9'
# DATASET = 'SYNTH_IMBprop_only0.95'
# DATASET = 'SYNTH_IMBprop_only0.98'
# DATASET = 'SYNTH_IMBvar_change0.5-2'
# DATASET = 'digits'
# DATASET = 'letter'
# DATASET = 'wine'
# DATASET = 'mushroom'
# DATASET = 'breast_cancer'
path_source = "/home/ludo/Documents/Thèse/Programmes/BDD/bdd_simulee_2015/features/Feat_blinesamp_100_norm_0_filt_1_cutoff_20_order_2_win_250/featmat/base_ech1nb_feat_872018_05_17.npy"
path_target = "/home/ludo/Documents/Thèse/Programmes/BDD/extracts/features/Feat_blinesamp_100_norm_0_filt_1_cutoff_20_order_2_win_250/featmat/base_ech1nb_feat_87_2018_05_03.npy"
SAME_SIZE = False
BALANCED_RF = False
BALANCED_SOURCE = False
BALANCED_TARGET = False
BL_METHOD = None
BL_RATIO = None
# TARGET_FOLD = None
# BL_METHOD = 'downsample'
# BL_RATIO = 1  # for tarkett_transfer : 1, 1.5, 2
TARGET_FOLD = 10  # for tarkett_transfer : 5, 10, 15, 20

MAX_DEPTH = 'None'
LEN_DEPTH_LIST = 15
# depth_list = list(np.arange(LEN_DEPTH_LIST) + 1) + [None]
# depth_list = [2, 4, 6, 8, 10, 12, None]
# depth_list = [2, 5, 10, 15, None]
depth_list = [3, None]

NB_TREE = 1


# ALGO = ['source_only', 'target_only', 'ser', 'mix', 'ser_nored',
# 'ser_noser', 'strut_noprune_update', 'strut_noprune_noupdate',
# 'strut_prune_update', 'strut_prune_noupdate']
# ALGO = ['source_only', 'target_only', 'ser', 'strut_prune_update', 'mix']
# ALGO = ['source_only', 'target_only', 'strut_prune_update',
# 'strut_noprune_update', 'strut_noprune_noupdate']
ALGO = ['source', 'target', 'ser', 'ser no red', 'ser no ext',
        'ser no ext cond', 'ser no red no ext', 'ser no red no ext cond']
COMPUTE_ADD_SCORES = []

# Plots
PLOT = 0
ERRBAR = 0
ERRBAR_FANCY = 0
FIG_TITLE = 1
FIGSIZE = (20, 10)
# FIGSIZE = None
LEGEND = 1
# Say, "the default sans-serif font is COMIC SANS"
# matplotlib.rcParams['font.sans-serif'] = "Comic Sans MS"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "serif"
matplotlib.rcParams['font.size'] = 11

SAVE = 1
path_save = '/home/ludo/Documents/Thèse/Programmes/PhD_code/outputs/plots/'

# =======================================================
#   ATTENTION
# =======================================================
CORRECTIF = 0

for DATASET in DATASET_LIST:
    if lib_data.class_nb(DATASET) == 2:
        SCORES = ['score_rate', 'tpr', 'fpr', 'f1', 'auc']
        # COMPUTE_ADD_SCORES = ['average', 'pos_likelihood']
        COMPUTE_ADD_SCORES = []
    else:
        SCORES = ['error_rate(perc)']

    # =======================================================
    #   Processing
    # =======================================================
    # if DATASET == 'tarkett_transfer':
    # comment = 'SOURCE_' + os.path.basename(path_source).split('.')[0] + \
    # '_TARGET_' + os.path.basename(path_target).split('.')[0]
    # else:
    # comment = ''
    # comment += '_SAMESIZE_' + str(SAME_SIZE) + \
    # '_BALANCEDRF_' + str(BALANCED_RF) + \
    # '_BALANCEDSOURCE' + str(BALANCED_SOURCE) + \
    # '_BALANCEDTARGET' + str(BALANCED_TARGET)
    # if BL_METHOD is not None:
    # comment += '_BLMETHOD' + str(BL_METHOD)
    # if BL_RATIO is not None:
    # comment += '_BLRATIO' + str(BL_RATIO)
    # if TARGET_FOLD is not None:
    # comment += '_TARGETFOLD' + str(TARGET_FOLD)

    df_scores = pd.read_csv(path_scores)
    df_scores = df_scores.fillna('')
    df_scores_filter = df_scores[(df_scores.data == DATASET)]

    if df_scores_filter.shape[0] == 0:
        raise ValueError("Error : filter resulted in empty data frame")
    if NB_TREE:
        df_scores_filter = df_scores_filter[(df_scores_filter.nb_tree == NB_TREE)]

    # =======================================================
    #   Creating Plot
    # =======================================================
    fig2, ax2 = plt.subplots(nrows=1, ncols=len(
        SCORES) + len(COMPUTE_ADD_SCORES), figsize=FIGSIZE)
    if len(SCORES) == 1:
        ax2 = [ax2]  # to allow indexing on Axe object
    bar_axes = []
    bar_figs = []
    for i_depth in range(len(depth_list)):
        fig_tmp, ax_tmp = plt.subplots(nrows=1, ncols=len(
            SCORES) + len(COMPUTE_ADD_SCORES), figsize=FIGSIZE)
        bar_axes.append(ax_tmp)
        bar_figs.append(fig_tmp)

    # Make title
    title = 'Data ' + DATASET + ' | ' +\
            'Score file ' + os.path.splitext(os.path.basename(path_scores))[0] +\
            '\n' + 'NB_TREE ' + str(NB_TREE)


    if FIG_TITLE:
        fig2.suptitle(title)
    path_save_name = path_save + title.replace(' | ', '_').replace(' ',
                                                                   '').replace(':', '').replace('\n', '')

    # fig, ax = plt.subplots()
    # title = 'Data : ' + DATASET + ' | ' + 'max depth : ' + MAX_DEPTH + '\n' + comment
    # fig.suptitle(title)

    nb_rep = df_scores_filter.shape[0] / len(ALGO)
    print('Mean number realisations : ', nb_rep)
    mean_arr = np.zeros((len(depth_list), len(SCORES)))
    std_arr = np.zeros((len(depth_list), len(SCORES)))
    scores_arr_mean = np.zeros((len(ALGO), len(depth_list), len(SCORES)))
    scores_arr_std = np.zeros((len(ALGO), len(depth_list), len(SCORES)))
    # time_arr = np.zeros((len(depth_list), len(ALGO)))

    for i_algo, algo in enumerate(ALGO):

        # tmp = df_scores_filter[(df_scores_filter.algo == algo)].loc[:, "time(s)"]

        for i_depth, max_depth in enumerate(depth_list):
            print("depth ", max_depth)
            # tmp_time = df_scores_filter[(df_scores_filter.algo == algo) &
            # (df_scores_filter.max_depth ==
            # str(max_depth))].loc[:, "time(s)"]
            tmp = df_scores_filter[(df_scores_filter.algo == algo) &
                                   (df_scores_filter.max_depth ==
                                       str(max_depth))].loc[:, SCORES]

            tmp = tmp.replace('', -1)
            print(algo, ' : ', tmp.shape[0], ' realisations')
            mean_arr[i_depth, :] = np.mean(tmp)
            std_arr[i_depth, :] = np.std(tmp)
            scores_arr_mean[i_algo, :, :] = mean_arr
            scores_arr_std[i_algo, :, :] = std_arr
            # time_arr[i_depth, :] = np.mean(tmp_time)

        # if algo == 'ser_nored':
            # algo = 'ser_nosercl'

        for i in range(len(SCORES)):
            # error_rate -> accuracy; tpr -> sensitivity; fpr -> specificity
            if SCORES[i] == 'score_rate':
                # mean_arr[:, i] = (100 - mean_arr[:, i])
                ax_title = 'Accuracy'
            elif SCORES[i] == 'tpr':
                ax_title = 'Sensitivity'
            elif SCORES[i] == 'fpr':
                mean_arr[:, i] = 1 - mean_arr[:, i]
                ax_title = 'Specificity'
            else:
                ax_title = SCORES[i]

            if i == 0:  #  legend only on first one
                if ERRBAR:
                    ax2[i].errorbar(range(len(depth_list)), mean_arr[:, i],
                                    yerr=std_arr[:, i], label=algo, fmt='o-',
                                    elinewidth=1, markeredgewidth=0.2, linewidth=2)
                # if ERRBAR_FANCY:
                    # ax2[i].fill_between(range(len(depth_list)), mean_arr[:,
                    # i] - std_arr[:, i], mean_arr[:, i] + std_arr[:, i],
                    # alpha=0.1)
                else:
                    ax2[i].errorbar(range(len(depth_list)), mean_arr[:, i],
                                    fmt='o-', label=algo,
                                    elinewidth=1, markeredgewidth=1, linewidth=2)
            else:
                if ERRBAR:
                    ax2[i].errorbar(range(len(depth_list)), mean_arr[:, i],
                                    yerr=std_arr[:, i], fmt='o-',
                                    elinewidth=1, markeredgewidth=1, linewidth=2)
                else:
                    ax2[i].errorbar(range(len(depth_list)), mean_arr[:, i],
                                    fmt='o-', markeredgewidth=1, linewidth=2)

            ax2[i].set_title(ax_title)
            ax2[i].set_xlabel('depth')
            if LEGEND:
                ax2[i].legend(loc='upper left')
            plt.sca(ax2[i])
            plt.xticks(range(len(depth_list)), [str(el) for el in depth_list])

        # additional scores
        fpr = df_scores_filter[(df_scores_filter.algo == algo) &
                               (df_scores_filter.max_depth ==
                                   str(max_depth))].loc[:, 'fpr']
        tpr = df_scores_filter[(df_scores_filter.algo == algo) &
                               (df_scores_filter.max_depth ==
                                   str(max_depth))].loc[:, 'tpr']
        fpr_mean = np.mean(fpr)
        tpr_mean = np.mean(tpr)
        for i in range(len(COMPUTE_ADD_SCORES)):
            # error_rate -> accuracy; tpr -> sensitivity; fpr -> specificity
            if COMPUTE_ADD_SCORES[i] == 'average':
                ax_title = 'Average'
                mean_arr[:, i] = (fpr_mean + tpr_mean) / 2
            elif COMPUTE_ADD_SCORES[i] == 'pos_likelihood':
                mean_arr[:, i] = 1 - mean_arr[:, i]
                ax_title = 'Pos likelihood'
            else:
                ax_title = COMPUTE_ADD_SCORES[i]

            if ERRBAR:
                ax2[i].errorbar(range(len(depth_list)), mean_arr[:, i],
                                yerr=std_arr[:, i], label=algo, fmt='o-',
                                elinewidth=1, markeredgewidth=0.2, linewidth=2)
            else:
                ax2[i].errorbar(range(len(depth_list)), mean_arr[:, i],
                                fmt='o-', label=algo,
                                elinewidth=1, markeredgewidth=1, linewidth=2)

            ax2[i].set_title(ax_title)
            ax2[i].set_xlabel('depth')
            if LEGEND:
                ax2[i].legend(loc='upper left')
            plt.sca(ax2[i])
            plt.xticks(range(len(depth_list)), [str(el) for el in depth_list])

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        scores_arr_mean.reshape((len(ALGO), len(depth_list), len(SCORES)))
        legend_elements = [Patch(facecolor=colors[i_algo], label=ALGO[i_algo]) for
                i_algo in range(len(ALGO))]
        for i_depth, max_depth in enumerate(depth_list):
            # Making title
            title_bar = title + ' | DEPTH ' + str(max_depth)

            for i_score in range(len(SCORES)):
                ax_title = SCORES[i_score]
                ax_tmp = bar_axes[i_depth][i_score]
                ax_tmp.bar(range(len(ALGO)), scores_arr_mean[:,
                                                             i_depth, i_score],
                           color=colors,
                           label=ALGO)
                low = np.min(scores_arr_mean[:, i_depth, i_score])
                high = np.max(scores_arr_mean[:, i_depth, i_score])
                y_low = low - 0.5 * (high - low)
                y_high = high + 0.5 * (high - low)
                ax_tmp.set_ylim([max(y_low, 0), y_high])
                # ax_tmp.set_xticks([r + BARWIDTH for r in range(len(ALGO))])
                # ax_tmp.set_xticklabels(ALGO)
                ax_tmp.set_title(ax_title)
                if i_score == 0:
                    # ax_tmp.legend()
                    ax_tmp.legend(handles=legend_elements)
            bar_figs[i_depth].suptitle(title_bar)

    if SAVE:
        print("saving fig at : ", path_save_name)
        fig2.savefig(path_save_name + '.png')
        fig2.savefig(path_save_name + '.pdf')
        for i_depth, max_depth in enumerate(depth_list):
            bar_figs[i_depth].savefig(path_save_name + '_BarPlot_' + 'depth' +
            str(max_depth) + '.pdf')
            bar_figs[i_depth].savefig(path_save_name + '_BarPlot_' + 'depth' +
            str(max_depth) + '.png')

    # plt.sca(ax)
    # plt.xticks(range(len(ALGO)), ALGO)

    if PLOT:
        plt.show()
