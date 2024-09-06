"""Calculate mean average precision (mAP).
"""
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues
from statsmodels.stats.multitest import fdrcorrection

from copairs import compute
from copairs.map.average_precision import build_rank_lists

from cytominer_eval.utils.transform_utils import set_pair_ids


def get_p_value(map_score, indices, null_dists, null_size, rev_ix):
    null_dist = null_dists[rev_ix[indices]].mean(axis=0)
    num = (null_dist > map_score).sum()
    p_value = (num + 1) / (null_size + 1)
    return p_value


def rename_groupby_columns(ap_df, groupby_columns):
    """Rename index columns to original groupby columns names.

    Parameters
    ----------
    ap_df : pd.DataFrame
        Dataframe with mean_ap results.
    groupby_columns : list
        List of groupby columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with renamed columns.
    """
    if len(groupby_columns) > 1:
        rename_dict = {f"level_{i}": col for i, col in enumerate(groupby_columns)}
    else:
        rename_dict = {"index": groupby_columns[0], "level_0": groupby_columns[0]}

    return ap_df.rename(columns=rename_dict)


def mean_ap(
    df: pd.DataFrame,
    null_size: int = 1000,
    random_seed: int = 0,
    groupby_columns: List[str] = [],
    replicate_group_col: str = "group_replicate",
    kwargs: dict = {},
) -> pd.DataFrame:
    """Calculate multidimensional perturbation value (mp-value) [1].

    Parameters
    ----------
    df : pandas.DataFrame
        similarity dafarame computed with copairs.
    replicate_group_col : str
        The metadata identifier marking which column tracks control and replicate perts.
    null_size : int, optional
        Number of null samples to generate, by default 1000
    random_seed : int, optional
        Random seed for reproducibility, by default 0
    groupby_columns : list, optional
        List of groupby columns, by default []
    kwargs : dict, optional
        Optional parameters provided. See list of parameters in
        :py:func:`copairs.compute.compute_ap_contiguous`

    Returns
    -------
    pd.DataFrame
        AP scores per perturbation.
    """
    pair_ids = set_pair_ids()
    pair_indices = [pair_ids[x]["index"] for x in pair_ids]

    unique_ids, rel_k_list, counts = build_rank_lists(
        df.loc[df[replicate_group_col], pair_indices].values,
        df.loc[~df[replicate_group_col], pair_indices].values,
        df.loc[df[replicate_group_col], "dist"].values,
        df.loc[~df[replicate_group_col], "dist"].values,
    )

    ap_scores, null_confs = compute.ap_contiguous(rel_k_list, counts)
    null_confs_unique, rev_ix = np.unique(null_confs, axis=0, return_inverse=True)
    null_dists = compute.get_null_dists(null_confs_unique, null_size, seed=random_seed)

    suffix = pair_ids[list(pair_ids)[0]]["suffix"]
    groupby_cols_suffix = [f"{x}{suffix}" for x in groupby_columns]
    grouped = df.query(replicate_group_col).groupby(groupby_cols_suffix)

    map_dict = {}
    for group_name, group in grouped:
        # convert indices according to ap scores order
        node_indices = np.searchsorted(unique_ids, pd.unique(group[pair_indices].values.ravel()))
        map_score = ap_scores[node_indices].mean()
        p_value = get_p_value(map_score, node_indices, null_dists, null_size, rev_ix)
        map_dict[group_name] = {
            "mean_ap": map_score,
            "p_value": p_value,
            "n_pos_pairs": null_confs[node_indices, 0].mean(),
            "n_total_pairs": null_confs[node_indices, 1].mean(),
        }

    map_df = pd.DataFrame.from_dict(map_dict, orient="index").reset_index()
    # correct aggregated p-values
    map_df["p_value"] = fdrcorrection(map_df["p_value"], alpha=0.05)[1]
    return rename_groupby_columns(map_df, groupby_columns)
