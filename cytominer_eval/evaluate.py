"""Calculate evaluation metrics from profiling experiments.

The primary entrypoint into quickly evaluating profile quality.
"""
import pandas as pd
import networkx as nx
from typing import List, Union, Optional
from itertools import chain

from cytominer_eval.transform import metric_melt, get_copairs, copairs_similarity
from cytominer_eval.operations import (
    replicate_reproducibility,
    precision_recall,
    grit,
    mp_value,
    enrichment,
    hitk,
)

from copairs.map import flatten_str_list, extract_filters


def mp_value_copairs(
    df,
    features,
    control_perts,
    replicate_groups,
    replicate_id,
    rescale_pca=True,
    nb_permutations=100,
    use_copairs=False,
):
    if use_copairs:
        control_perts = [-1]
        replicate_id = "clique_id"

    metric_result = mp_value(
        df=df,
        control_perts=control_perts,
        replicate_id=replicate_id,
        features=features,
        kwargs={"rescale_pca": rescale_pca, "nb_permutations": nb_permutations},
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


def get_copairs_similarity_df(
    profiles, features, meta_features, replicate_groups, operation
):
    metadata = profiles.loc[:, meta_features]
    feature_values = profiles.loc[:, features].values
    pos_pairs, neg_pairs = get_pos_neg_pairs(metadata, replicate_groups)

    if operation != "mp_value":
        pos_pairs, neg_pairs = copairs_similarity(pos_pairs, neg_pairs, feature_values)
        pos_pairs["group_replicate"] = True
        neg_pairs["group_replicate"] = False

        similarity_df = pd.concat([pos_pairs, neg_pairs])
        similarity_df.rename(
            {"ix1": "pair_a_index", "ix2": "pair_b_index"}, axis=1, inplace=True
        )
        similarity_df["similarity_metric"] = 1 - similarity_df["dist"]
        print(similarity_df)

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
        neg_clique_intersect = set.intersection(*map(set, neg_cliques))
        assert (
            len(neg_clique_intersect) > 0
        ), "Error! No overlapping negative cliques found."
        assert (
            len(set(chain.from_iterable(pos_cliques)) & neg_clique_intersect) == 0
        ), "Error! Positive and negative cliques overlap."

        profiles["clique_id"] = pd.NA
        for idx, clique in enumerate(pos_cliques):
            profiles.loc[profiles.index.isin(clique), "clique_id"] = idx
        profiles.loc[profiles.index.isin(neg_clique_intersect), "clique_id"] = -1
        similarity_df = profiles.dropna(axis=0, subset=["clique_id"])

    return similarity_df


def build_similarity_df(
    profiles,
    features,
    meta_features,
    replicate_groups,
    similarity_metric,
    operation,
    use_copairs,
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
                similarity_metric=similarity_metric,
                upper_triagonal=upper_triagonal,
            )

    elif use_copairs:
        similarity_df = get_copairs_similarity_df(
            profiles, features, meta_features, replicate_groups, similarity_metric
        )

    return similarity_df


def evaluate_operation(
    profiles: pd.DataFrame,
    features: List[str],
    meta_features: List[str],
    replicate_groups: Union[List[str], dict],
    operation: str,
    operation_kwargs: dict,
    similarity_metric: str = "pearson",
    use_copairs: bool = False,
    similarity_df: Optional[pd.DataFrame] = None,
):
    metric_fn = get_operation_fn(operation)
    upper_triagonal = operation == "replicate_reproducibility"
    similarity_df = similarity_df if operation != "mp_value" else None

    if similarity_df is None:
        similarity_df = build_similarity_df(
            profiles=profiles,
            features=features,
            meta_features=meta_features,
            replicate_groups=replicate_groups,
            similarity_metric=similarity_metric,
            operation=operation,
            use_copairs=use_copairs,
            upper_triagonal=upper_triagonal,
        )

    metric_result = metric_fn(
        df=similarity_df,
        features=features,
        replicate_groups=replicate_groups,
        **operation_kwargs,
    )

    return metric_result, similarity_df


def evaluate(
    profiles: pd.DataFrame,
    features: List[str],
    meta_features: List[str],
    replicate_groups: Union[List[str], dict],
    operation_list: List[str],
    operation_kwargs: Optional[List[dict]] = None,
    similarity_metric: str = "pearson",
    use_copairs: bool = False,
    copairs_kwargs: Optional[dict] = None,
    # groupby_columns: List[str] = ["Metadata_broad_sample"],
    # replicate_reproducibility_quantile: float = 0.95,
    # replicate_reproducibility_return_median_cor: bool = False,
    # precision_recall_k: Union[int, List[int]] = 10,
    # grit_control_perts: List[str] = ["None"],
    # grit_replicate_summary_method: str = "mean",
    # mp_value_params: dict = {},
    # enrichment_percentile: Union[float, List[float]] = 0.99,
    # hitk_percent_list=[2, 5, 10],
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
    # Check replicate groups input
    # check_replicate_groups(eval_metric=operation, replicate_groups=replicate_groups)

    # make `replicate_reproducibility` first for easier reuse of similarity_df
    if "replicate_reproducibility" in operation_list:
        operation_list = ["replicate_reproducibility"] + [
            op for op in operation_list if op != "replicate_reproducibility"
        ]
    # make `mp_values` last for easier reuse of similarity_df
    if "mp_value" in operation_list:
        operation_list = [op for op in operation_list if op != "mp_value"] + [
            "mp_value"
        ]

    operation_kwargs = (
        operation_kwargs if operation_kwargs is not None else [{}] * len(operation_list)
    )

    similarity_df = None
    metric_results = []
    for operation in operation_list:
        metric_result, similarity_df = evaluate_operation(
            profiles=profiles,
            features=features,
            meta_features=meta_features,
            replicate_groups=replicate_groups,
            operation=operation,
            operation_kwargs=operation_kwargs[operation_list.index(operation)],
            similarity_metric=similarity_metric,
            use_copairs=use_copairs,
            similarity_df=similarity_df,
        )
        metric_results.append(metric_result)

        similarity_df = (
            None if operation == "replicate_reproducibility" else similarity_df
        )

    return metric_results

    # Perform the input operation
    # if operation == "replicate_reproducibility":
    #     metric_result = replicate_reproducibility(
    #         df=similarity_melted_df,
    #         replicate_groups=replicate_groups,
    #         quantile_over_null=replicate_reproducibility_quantile,
    #         return_median_correlations=replicate_reproducibility_return_median_cor,
    #         use_copairs=use_copairs,
    #     )
    # elif operation == "precision_recall":
    #     metric_result = precision_recall(
    #         df=similarity_melted_df,
    #         replicate_groups=replicate_groups,
    #         groupby_columns=groupby_columns,
    #         k=precision_recall_k,
    #         use_copairs=use_copairs,
    #     )
    # elif operation == "grit":
    #     metric_result = grit(
    #         df=similarity_melted_df,
    #         control_perts=grit_control_perts,
    #         profile_col=replicate_groups["profile_col"],
    #         replicate_group_col=replicate_groups["replicate_group_col"],
    #         replicate_summary_method=grit_replicate_summary_method,
    #     )
    # elif operation == "mp_value" and not use_copairs:
    #     metric_result = mp_value(
    #         df=profiles,
    #         control_perts=grit_control_perts,
    #         replicate_id=replicate_groups,
    #         features=features,
    #         params=mp_value_params
    #     )
    # elif operation == "enrichment":
    #     metric_result = enrichment(
    #         df=similarity_melted_df,
    #         replicate_groups=replicate_groups,
    #         percentile=enrichment_percentile,
    #     )
    # elif operation == "hitk":
    #     metric_result = hitk(
    #         df=similarity_melted_df,
    #         replicate_groups=replicate_groups,
    #         groupby_columns=groupby_columns,
    #         percent_list=hitk_percent_list,
    #     )
    # return metric_result
