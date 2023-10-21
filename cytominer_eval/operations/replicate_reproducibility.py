"""Functions to calculate replicate reproducibility."""

import pandas as pd
from typing import List, Optional

from cytominer_eval.utils.operation_utils import set_pair_ids


def replicate_reproducibility(
    df: pd.DataFrame,
    replicate_groups: Optional[List[str]] = None,
    quantile_over_null: float = 0.95,
    return_median_correlations: bool = False,
) -> float:
    r"""Summarize pairwise replicate correlations

    For a given pairwise similarity matrix, replicate information, and specific options,
    output a replicate correlation summary.

    Parameters
    ----------
    df : pandas.DataFrame
        An elongated symmetrical matrix indicating pairwise correlations between
        samples. Importantly, it must follow the exact structure as output from
        :py:func:`cytominer_eval.transform.transform.metric_melt`.
    replicate_groups : list
        A list of metadata column names in the original profile dataframe to indicate
        replicate samples.
    quantile_over_null : float, optional
        A float between 0 and 1 indicating the threshold of nonreplicates to use when
        reporting percent matching or percent replicating. Defaults to 0.95.
    return_median_correlations : bool, optional
        If provided, also return median pairwise correlations per replicate.
        Defaults to False.

    Returns
    -------
    {float, (float, pd.DataFrame)}
        The replicate reproducibility of the profiles according to the replicate
        columns provided. If `return_median_correlations = True` then the function will
        return both the metric and a median pairwise correlation pandas.DataFrame.
    """
    if replicate_groups is None:
        raise ValueError("replicate_groups kwarg must be provided")

    assert (
        0 < quantile_over_null and 1 >= quantile_over_null
    ), "quantile_over_null must be between 0 and 1"

    # check that there are group_replicates (non-unique rows)
    replicate_df = df.query("group_replicate")
    denom = replicate_df.shape[0]

    assert denom != 0, f"no replicate groups identified in {replicate_groups} columns!"

    non_replicate_quantile = df.query("not group_replicate").similarity_metric.quantile(
        quantile_over_null
    )

    replicate_reproducibility = (
        replicate_df.similarity_metric > non_replicate_quantile
    ).sum() / denom

    if return_median_correlations:
        pair_ids = set_pair_ids()
        replicate_groups_for_groupby = {
            f"{x}{pair_ids['pair_a']['suffix']}": x for x in replicate_groups
        }

        median_cor_df = (
            replicate_df.groupby(list(replicate_groups_for_groupby))[
                "similarity_metric"
            ]
            .median()
            .reset_index()
            .rename(replicate_groups_for_groupby, axis="columns")
        )

        return (replicate_reproducibility, median_cor_df)

    return replicate_reproducibility
