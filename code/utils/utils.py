import pandas as pd


def error_rate(er):
    """
    tranforms a sklearn score in error rate (in %)
    """
    return round((1 - er) * 100, 2)


def write_score(scores_file, score):
    # scores_df = pd.read_csv(scores_file, sep=',', header=0)
    new_score_df = pd.DataFrame([[score.algo, score.max_depth, score.nb_tree,
                                  score.data, score.error_rate, score.time]])
    with open(scores_file, 'a') as f:
        new_score_df.to_csv(f, header=False, index=False)
    f.close()


if __name__ == "__main__":
    scores_file = "output/scores.csv"
    # scores_df = pd.read_csv(scores_file, sep=',', header=0)
    # new_score_df = pd.DataFrame([[score.algo, score.max_depth, score.nb_tree,
                                  # score.data, score.error_rate, score.time]])
    # scores_df = scores_df.append(new_score_df, ignore_index=True)
