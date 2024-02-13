"""Calculate evaluation metrics from profiling experiments.

The primary entrypoint into quickly evaluating profile quality.
"""
from collections import defaultdict
from typing import List, Union, Optional

import pandas as pd
import networkx as nx

from cytominer_eval.transform import metric_melt, get_copairs, copairs_similarity
from cytominer_eval.utils.operation_utils import assign_replicates
from cytominer_eval.utils.transform_utils import assert_melt
from cytominer_eval.operations import (
    replicate_reproducibility,
    precision_recall,
    grit,
    mp_value,
    enrichment,
    hitk,
    mean_ap,
)

from copairs.map import flatten_str_list, extract_filters


def mp_value_copairs(
    df,
    control_perts,
    replicate_id,
    features=None,
    replicate_groups=None,
    rescale_pca=True,
    nb_permutations=100,
    random_seed=0,
    use_copairs=False,
    control_pert_filter=None,
):
    if use_copairs:
        control_perts = [-1]
        replicate_id = "clique_id"

    metric_result = mp_value(
        df=df,
        control_perts=control_perts,
        replicate_id=replicate_id,
        features=features,
        control_pert_filter=control_pert_filter,
        kwargs={
            "rescale_pca": rescale_pca,
            "nb_permutations": nb_permutations,
            "random_seed": random_seed,
        },
    )

    if use_copairs:
        pos_sameby = replicate_groups.get("pos_sameby", [])
        pos_diffby = replicate_groups.get("pos_diffby", [])
        pos_meta_cols = flatten_str_list(pos_sameby, pos_diffby)

        metadata = df.loc[:, [col for col in df.columns if col not in features]]
        _, pos_meta_cols = extract_filters(pos_meta_cols, metadata.columns)
        metadata["clique_id"] = df["clique_id"].replace({-1: pd.NA})
        result_meta = metadata[pos_meta_cols + ["clique_id"]].dropna(
            axis=0, subset=["clique_id"]
        )

        metric_result = result_meta.merge(metric_result, on="clique_id", how="left")
        metric_result[pos_meta_cols + ["mp_value"]]

    return metric_result


def get_operation_fn(operation_name):
    """Get the function for a given operation name

    Parameters
    ----------
    operation_name : str
        The name of the operation to perform

    Returns
    -------
    function
        The function to perform the specified operation
    """

    operation_mapping = {
        "replicate_reproducibility": replicate_reproducibility,
        "precision_recall": precision_recall,
        "grit": grit,
        "mp_value": mp_value_copairs,
        "enrichment": enrichment,
        "hitk": hitk,
        "mean_ap": mean_ap,
    }

    try:
        return operation_mapping[operation_name]
    except KeyError:
        raise ValueError(f"Invalid operation name: {operation_name}")


def get_pos_neg_pairs(metadata, replicate_groups):
    pos_sameby = replicate_groups.get("pos_sameby", [])
    pos_diffby = replicate_groups.get("pos_diffby", [])
    neg_sameby = replicate_groups.get("neg_sameby", [])
    neg_diffby = replicate_groups.get("neg_diffby", [])

    replicate_groups = flatten_str_list(pos_sameby, pos_diffby, neg_sameby, neg_diffby)

    pos_pairs, neg_pairs = get_copairs(
        meta=metadata,
        pos_sameby=pos_sameby,
        pos_diffby=pos_diffby,
        neg_sameby=neg_sameby,
        neg_diffby=neg_diffby,
    )

    return pos_pairs, neg_pairs


def map_neg_pos_cliques(pos_cliques, neg_cliques):
    # create a reverse lookup dictionary for neg_cliques
    neg_lookup = defaultdict(list)
    for nc in neg_cliques:
        for elem in nc:
            neg_lookup[elem].extend(nc)

    # flatten the lists in neg_lookup and remove duplicates
    for key in neg_lookup:
        neg_lookup[key] = list(set(neg_lookup[key]))

    clique_dict = {}
    for index, pos_clique in enumerate(pos_cliques):
        # collect all unique neg_clique elements for each pos_clique
        neg_elements = set()
        for elem in pos_clique:
            neg_elements.update(neg_lookup.get(elem, []))
        # remove any element from neg_elements that is in pos_clique to avoid self-reference
        clique_dict[index] = sorted(neg_elements.difference(pos_clique))

    return clique_dict


def get_copairs_similarity_df(
    profiles, features, meta_features, replicate_groups, operation, distance_metric="cosine"
):
    metadata = profiles.loc[:, meta_features]
    feature_values = profiles.loc[:, features].values
    pos_pairs, neg_pairs = get_pos_neg_pairs(metadata, replicate_groups)

    if operation != "mp_value":
        pos_pairs, neg_pairs = copairs_similarity(pos_pairs, neg_pairs, feature_values, distance_metric=distance_metric)
        pos_pairs["group_replicate"] = True
        neg_pairs["group_replicate"] = False

        similarity_df = pd.concat([pos_pairs, neg_pairs])
        similarity_df.rename(
            {"ix1": "pair_a_index", "ix2": "pair_b_index"}, axis=1, inplace=True
        )
        similarity_df["similarity_metric"] = similarity_df["dist"]

        metadata_ix1 = metadata.loc[similarity_df["pair_a_index"]].reset_index(
            drop=True
        )
        metadata_ix1.columns = [f"{col}_pair_a" for col in metadata_ix1.columns]
        metadata_ix2 = metadata.loc[similarity_df["pair_b_index"]].reset_index(
            drop=True
        )
        metadata_ix2.columns = [f"{col}_pair_b" for col in metadata_ix2.columns]

        similarity_df = pd.concat(
            [similarity_df.reset_index(drop=True), metadata_ix1, metadata_ix2],
            axis=1,
        )
    else:
        pos_graph = nx.Graph()
        pos_graph.add_edges_from(pos_pairs[["ix1", "ix2"]].values.tolist())
        pos_cliques = list(nx.find_cliques(pos_graph))
        assert (
            len(set.intersection(*map(set, pos_cliques))) == 0
        ), "Error! Overlapping positive cliques found."

        neg_graph = nx.Graph()
        neg_graph.add_edges_from(neg_pairs[["ix1", "ix2"]].values.tolist())
        neg_cliques = list(nx.find_cliques(neg_graph))
        pos_to_neg_map = map_neg_pos_cliques(pos_cliques, neg_cliques)

        profiles["clique_id"] = pd.NA
        # TODO: consider multithreading this
        for idx, clique in enumerate(pos_cliques):
            profiles.loc[clique, "clique_id"] = idx
            profiles.loc[pos_to_neg_map[idx], "clique_id"] = -1

        similarity_df = profiles.dropna(axis=0, subset=["clique_id"]), pos_to_neg_map

    return similarity_df


def build_similarity_df(
    profiles,
    features,
    meta_features,
    replicate_groups,
    distance_metric,
    operation,
    use_copairs,
    copairs_kwargs=None,
    upper_triagonal=False,
):
    if not use_copairs:
        if operation == "mp_value":
            similarity_df = profiles
        else:
            similarity_df = metric_melt(
                df=profiles,
                features=features,
                metadata_features=meta_features,
                similarity_metric=distance_metric,
                upper_triagonal=upper_triagonal,
            )

            if "group_replicate" not in similarity_df.columns:
                similarity_df = assign_replicates(
                    similarity_melted_df=similarity_df,
                    replicate_groups=replicate_groups,
                )
                assert_melt(similarity_df, eval_metric=operation)

    elif use_copairs:
        similarity_df = get_copairs_similarity_df(
            profiles, features, meta_features, replicate_groups, operation, distance_metric=distance_metric
        )

    return similarity_df


def evaluate_metric(
    profiles: pd.DataFrame,
    features: List[str],
    meta_features: List[str],
    replicate_groups: Union[List[str], dict],
    operation: str,
    operation_kwargs: dict,
    distance_metric: str = "pearson",
    use_copairs: bool = False,
    similarity_df: Optional[pd.DataFrame] = None,
):
    metric_fn = get_operation_fn(operation)
    upper_triagonal = operation == "replicate_reproducibility"

    if similarity_df is None or operation == "mp_value":
        print("\nCalculating distances.")
        similarity_df = build_similarity_df(
            profiles=profiles,
            features=features,
            meta_features=meta_features,
            replicate_groups=replicate_groups,
            distance_metric=distance_metric,
            operation=operation,
            use_copairs=use_copairs,
            upper_triagonal=upper_triagonal,
        )

    if operation == "mp_value":
        operation_kwargs["features"] = features
        operation_kwargs["use_copairs"] = use_copairs
        operation_kwargs["replicate_groups"] = replicate_groups

        if use_copairs and len(similarity_df) == 2:
            operation_kwargs["control_pert_filter"] = similarity_df[1]
            similarity_df = similarity_df[0]

    print(f"\nCalculating metric: {operation}")
    metric_result = metric_fn(df=similarity_df, **operation_kwargs)

    if operation == "replicate_reproducibility" and not use_copairs:
        similarity_df = None
    return metric_result, similarity_df


def evaluate_metrics(
    profiles: pd.DataFrame,
    features: List[str],
    meta_features: List[str],
    replicate_groups: Union[List[str], dict],
    metrics_config: dict,
    distance_metric: str = "pearson",
    use_copairs: bool = False,
    similarity_df: Optional[pd.DataFrame] = None,
    return_similarity_df: bool = False,
):
    r"""Evaluate profile quality and strength.

    For a given profile dataframe containing both metadata and feature measurement
    columns, use this function to calculate profile quality metrics. The function
    contains all the necessary arguments for specific evaluation operations.

    Parameters
    ----------
    profiles : pandas.DataFrame
        profiles must be a pandas DataFrame with profile samples as rows and profile
        features as columns. The columns should contain both metadata and feature
        measurements.
    features : list
        A list of strings corresponding to feature measurement column names in the
        `profiles` DataFrame. All features listed must be found in `profiles`.
    meta_features : list
        A list of strings corresponding to metadata column names in the `profiles`
        DataFrame. All features listed must be found in `profiles`.
    replicate_groups : {str, list, dict}
        An important variable indicating which metadata columns denote replicate
        information. All metric operations require replicate profiles.
        `replicate_groups` indicates a str or list of columns to use. For
        `operation="grit"`, `replicate_groups` is a dict with two keys: "profile_col"
        and "replicate_group_col". "profile_col" is the column name that stores
        identifiers for each profile (can be unique), while "replicate_group_col" is the
        column name indicating a higher order replicate information. E.g.
        "replicate_group_col" can be a gene column in a CRISPR experiment with multiple
        guides targeting the same genes. See also
        :py:func:`cytominer_eval.operations.grit` and
        :py:func:`cytominer_eval.transform.util.check_replicate_groups`.
    operation : {'replicate_reproducibility', 'precision_recall', 'grit', 'mp_value'}, optional
        The specific evaluation metric to calculate. The default is
        "replicate_reproducibility".
    groupby_columns : List of str
        Only used for operation = 'precision_recall' and 'hitk'
        Column by which the similarity matrix is grouped and by which the operation is calculated.
        For example, if groupby_column = "Metadata_broad_sample" then precision/recall is calculated for each sample.
        Note that it makes sense for these columns to be unique or to span a unique space
        since precision and hitk may otherwise stop making sense.
    similarity_metric: {'pearson', 'spearman', 'kendall'}, optional
        How to calculate pairwise similarity. Defaults to "pearson". We use the input
        in pandas.DataFrame.cor(). The default is "pearson".

    Returns
    -------
    float, pd.DataFrame
        The resulting evaluation metric. The return is either a single value or a pandas
        DataFrame summarizing the metric as specified in `operation`.

    Other Parameters
    -----------------------------
    replicate_reproducibility_quantile : {0.95, ...}, optional
        Only used when `operation='replicate_reproducibility'`. This indicates the
        percentile of the non-replicate pairwise similarity to consider a reproducible
        phenotype. Defaults to 0.95.
    replicate_reproducibility_return_median_cor : bool, optional
        Only used when `operation='replicate_reproducibility'`. If True, then also
        return pairwise correlations as defined by replicate_groups and
        similarity metric
    precision_recall_k : int or list of ints {10, ...}, optional
        Only used when `operation='precision_recall'`. Used to calculate precision and
        recall considering the top k profiles according to pairwise similarity.
    grit_control_perts : {None, ...}, optional
        Only used when `operation='grit'`. Specific profile identifiers used as a
        reference when calculating grit. The list entries must be found in the
        `replicate_groups[replicate_id]` column.
    grit_replicate_summary_method : {"mean", "median"}, optional
        Only used when `operation='grit'`. Defines how the replicate z scores are
        summarized. see
        :py:func:`cytominer_eval.operations.util.calculate_grit`
    mp_value_params : {{}, ...}, optional
        Only used when `operation='mp_value'`. A key, item pair of optional parameters
        for calculating mp value. See also
        :py:func:`cytominer_eval.operations.util.default_mp_value_parameters`
    enrichment_percentile : float or list of floats, optional
        Only used when `operation='enrichment'`. Determines the percentage of top connections
        used for the enrichment calculation.
    hitk_percent_list : list or "all"
        Only used when operation='hitk'. Default : [2,5,10]
        A list of percentages at which to calculate the percent scores, ie the amount of indexes below this percentage.
        If percent_list == "all" a full dict with the length of classes will be created.
        Percentages are given as integers, ie 50 means 50 %.
    """
    # make `replicate_reproducibility` first and `mp_value` last for better reuse of similarity_df
    first_last_keys = ["replicate_reproducibility", "mp_value"]
    key_order = sorted(k for k in metrics_config if k not in first_last_keys)
    key_order = [first_last_keys[0], *key_order, first_last_keys[-1]]
    metrics_config = {k: metrics_config[k] for k in key_order if k in metrics_config}

    metric_results = {}
    for operation, operation_kwargs in metrics_config.items():
        metric_result, similarity_df = evaluate_metric(
            profiles=profiles,
            features=features,
            meta_features=meta_features,
            replicate_groups=replicate_groups,
            operation=operation,
            operation_kwargs=operation_kwargs,
            distance_metric=distance_metric,
            use_copairs=use_copairs,
            similarity_df=similarity_df,
        )
        metric_results[operation] = metric_result

    return (metric_result, similarity_df) if return_similarity_df else metric_results
