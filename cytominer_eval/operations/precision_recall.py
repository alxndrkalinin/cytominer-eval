"""Functions to calculate precision and recall at a given k."""

import pandas as pd
from typing import List, Union

from cytominer_eval.utils.precisionrecall_utils import calculate_precision_recall
from cytominer_eval.utils.transform_utils import set_pair_ids


def precision_recall(
    df: pd.DataFrame,
    groupby_columns: List[str],
    k: Union[int, List[int]],
) -> pd.DataFrame:
    """Determine the precision and recall at k for all unique groupby_columns samples
    based on a predefined similarity metric (see cytominer_eval.transform.metric_melt)

    Parameters
    ----------
    df : pandas.DataFrame
        An elongated symmetrical matrix indicating pairwise correlations between
        samples. Importantly, it must follow the exact structure as output from
        :py:func:`cytominer_eval.transform.transform.metric_melt`.
    groupby_columns : List of str
        Column by which the similarity matrix is grouped and by which the precision/recall is calculated.
        For example, if groupby_column = Metadata_sample then the precision is calculated for each sample.
        Calculating the precision by sample is the default
        but it is mathematically not incorrect to calculate the precision at the MOA level.
        This is just less intuitive to understand.
    k : List of ints or int
        an integer indicating how many pairwise comparisons to threshold.

    Returns
    -------
    pandas.DataFrame
        precision and recall metrics for all groupby_column groups given k
    """
    df.sort_values(by="similarity_metric", ascending=False)

    # Extract out specific columns
    pair_ids = set_pair_ids()
    suffix = pair_ids[list(pair_ids)[0]]["suffix"]
    groupby_cols_suffix = [f"{x}{suffix}" for x in groupby_columns]
    # iterate over all k
    precision_recall_all_ks = []
    if type(k) == int:
        k = [k]
    for k_ in k:
        # Calculate precision and recall for all groups
        precision_recall_df_at_k = df.groupby(groupby_cols_suffix).apply(
            lambda x: calculate_precision_recall(x, k=k_)
        )
        precision_recall_all_ks.append(precision_recall_df_at_k)

    precision_recall_all_ks = pd.concat(precision_recall_all_ks)
    # Rename the columns back to the replicate groups provided
    rename_cols = dict(zip(groupby_cols_suffix, groupby_columns))

    return precision_recall_all_ks.reset_index().rename(rename_cols, axis="columns")
