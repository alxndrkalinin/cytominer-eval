import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
from itertools import chain

from cytominer_eval.utils.availability_utils import check_similarity_metric
from cytominer_eval.utils.transform_utils import (
    assert_pandas_dtypes,
    get_upper_matrix,
    set_pair_ids,
)

from copairs.matching import dict_to_dframe
from copairs.map import (
    create_matcher,
    compute_similarities,
    flatten_str_list,
    extract_filters,
    apply_filters,
)


def get_pairwise_metric(df: pd.DataFrame, similarity_metric: str) -> pd.DataFrame:
    """Helper function to output the pairwise similarity metric for a feature-only
    dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Samples x features, where all columns can be coerced to floats
    similarity_metric : str
        The pairwise comparison to calculate

    Returns
    -------
    pandas.DataFrame
        A pairwise similarity matrix
    """
    # Check that the input data is in the correct format
    check_similarity_metric(similarity_metric)
    df = assert_pandas_dtypes(df=df, col_fix=float)

    # faster alternatives:
    # 1) scipy.spatial.distance.pdist - compatible with all SciPy distances
    #  it returns a condensed vector-form distance matrix,
    #  which can be converted to the square format by scipy.spatial.distance.squareform
    # 2) numpy.corrcoef – fastest for calculating correlations
    pair_df = df.transpose().corr(method=similarity_metric)

    # Check if the metric calculation went wrong
    # (Current pandas version makes this check redundant)
    if pair_df.shape == (0, 0):
        raise TypeError(
            "Something went wrong - check that 'features' are profile measurements"
        )

    return pair_df


def process_melt(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
    upper_triagonal: bool = False,
) -> pd.DataFrame:
    """Helper function to annotate and process an input similarity matrix

    Parameters
    ----------
    df : pandas.DataFrame
        A similarity matrix output from
        :py:func:`cytominer_eval.transform.transform.get_pairwise_metric`
    meta_df : pandas.DataFrame
        A wide matrix of metadata information where the index aligns to the similarity
        matrix index
    eval_metric : str, optional
        Which metric to ultimately calculate. Determines whether or not to keep the full
        similarity matrix or only one diagonal. Defaults to "replicate_reproducibility".

    Returns
    -------
    pandas.DataFrame
        A pairwise similarity matrix
    """
    # Confirm that the user formed the input arguments properly
    assert df.shape[0] == df.shape[1], "Matrix must be symmetrical"

    # Get identifiers for pairing metadata
    pair_ids = set_pair_ids()

    np.fill_diagonal(df.values, np.nan)
    if upper_triagonal:
        upper_tri = get_upper_matrix(df)
        df = df.where(upper_tri)

    # Convert pairwise matrix to melted (long) version based on index value
    metric_unlabeled_df = (
        pd.melt(
            df.reset_index(),
            id_vars="index",
            value_vars=df.columns,
            var_name=pair_ids["pair_b"]["index"],
            value_name="similarity_metric",
        )
        .dropna()
        .reset_index(drop=True)
        .rename({"index": pair_ids["pair_a"]["index"]}, axis="columns")
    )

    # Merge metadata on index for both comparison pairs
    output_df = meta_df.merge(
        meta_df.merge(
            metric_unlabeled_df,
            left_index=True,
            right_on=pair_ids["pair_b"]["index"],
        ),
        left_index=True,
        right_on=pair_ids["pair_a"]["index"],
        suffixes=[pair_ids["pair_a"]["suffix"], pair_ids["pair_b"]["suffix"]],
    ).reset_index(drop=True)

    return output_df


def metric_melt(
    df: pd.DataFrame,
    features: List[str],
    metadata_features: List[str],
    upper_triagonal: bool = False,
    similarity_metric: str = "pearson",
) -> pd.DataFrame:
    """Helper function to fully transform an input dataframe of metadata and feature
    columns into a long, melted dataframe of pairwise metric comparisons between
    profiles.

    Parameters
    ----------
    df : pandas.DataFrame
        A profiling dataset with a mixture of metadata and feature columns
    features : list
        Which features make up the profile; included in the pairwise calculations
    metadata_features : list
        Which features are considered metadata features; annotate melted dataframe and
        do not use in pairwise calculations.
    eval_metric : str, optional
        Which metric to ultimately calculate. Determines whether or not to keep the full
        similarity matrix or only one diagonal. Defaults to "replicate_reproducibility".
    similarity_metric : str, optional
        The pairwise comparison to calculate

    Returns
    -------
    pandas.DataFrame
        A fully melted dataframe of pairwise correlations and associated metadata
    """
    # Subset dataframes to specific features
    df = df.reset_index(drop=True)

    assert all(
        [x in df.columns for x in metadata_features]
    ), "Metadata feature not found"
    assert all([x in df.columns for x in features]), "Profile feature not found"

    meta_df = df.loc[:, metadata_features]
    df = df.loc[:, features]

    # Convert pandas column types and assert conversion success
    meta_df = assert_pandas_dtypes(df=meta_df, col_fix=str)
    df = assert_pandas_dtypes(df=df, col_fix=float)

    # Get pairwise metric matrix
    pair_df = get_pairwise_metric(df=df, similarity_metric=similarity_metric)

    # Convert pairwise matrix into metadata-labeled melted matrix
    output_df = process_melt(
        df=pair_df, meta_df=meta_df, upper_triagonal=upper_triagonal
    )

    return output_df


def get_copairs_similarity(
    meta: pd.DataFrame,
    feats: np.ndarray,
    pos_sameby: List[str],
    pos_diffby: List[str],
    neg_sameby: List[str],
    neg_diffby: List[str],
    multilabel_col: Optional[str] = None,
    batch_size: int = 20000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get similarity scores for positive and negative pairs

    Parameters
    ----------
    meta : pandas.DataFrame
        Metadata dataframe
    feats : numpy.ndarray
        Feature matrix
    pos_sameby : list
        List of columns that have to match to make a positive pair
    pos_diffby : list
        List of columns that have to differ to make a positive pair
    neg_sameby : list
        List of columns that have to match to make a negative pair
    neg_diffby : list
        List of columns that have to differ to make a negative pair
    multilabel_col : str, optional
        Column to use for multilabel matching
    batch_size : int, optional
        Batch size for similarity calculation

    Returns
    -------
    tuple(pandas.DataFrame, pandas.DataFrame)
        Positive and negative similarity
    """
    pos_pairs, neg_pairs = get_copairs(
        meta,
        pos_sameby,
        pos_diffby,
        neg_sameby,
        neg_diffby,
        multilabel_col,
    )

    pos_pairs, neg_pairs = copairs_similarity(pos_pairs, neg_pairs, feats, batch_size)
    return pos_pairs, neg_pairs


def copairs_similarity(
    pos_pairs: pd.DataFrame,
    neg_pairs: pd.DataFrame,
    feats: np.ndarray,
    batch_size: int = 20000,
):
    """Compute similarity for positive and negative pairs

    Parameters
    ----------
    pos_pairs : pandas.DataFrame
        Positive pairs
    neg_pairs : pandas.DataFrame
        Negative pairs
    feats : numpy.ndarray
        Feature matrix
    batch_size : int
        Batch size for similarity calculation

    Returns
    -------
    tuple(pandas.DataFrame, pandas.DataFrame)
        Positive and negative similarity
    """
    pos_pairs = compute_similarities(pos_pairs, feats, batch_size)
    neg_pairs = compute_similarities(neg_pairs, feats, batch_size)
    return pos_pairs, neg_pairs


def get_copairs(
    meta: pd.DataFrame,
    pos_sameby: List[str],
    pos_diffby: List[str],
    neg_sameby: List[str],
    neg_diffby: List[str],
    filters: Optional[Union[List[str], str]] = None,
    multilabel_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get positive and negative pairs

    Parameters
    ----------
    meta : pandas.DataFrame
        Metadata dataframe
    pos_sameby : list
        List of columns that have to match to make a positive pair
    pos_diffby : list
        List of columns that have to differ to make a positive pair
    neg_sameby : list
        List of columns that have to match to make a negative pair
    neg_diffby : list
        List of columns that have to differ to make a negative pair
    filters : list, optional
        List of filters to apply to the metadata dataframe
    multilabel_col : str, optional
        Column to use for multilabel matching

    Returns
    -------
    tuple(pandas.DataFrame, pandas.DataFrame)
        Positive and negative pairs
    """
    meta = meta.reset_index(drop=True).copy()

    # generic filters that do not affect matching
    if filters is not None:
        query_list, _ = extract_filters(filters, meta.columns)
        meta = apply_filters(meta, query_list)

    pos_columns = flatten_str_list(pos_sameby, pos_diffby)
    neg_columns = flatten_str_list(neg_sameby, neg_diffby)

    if all(c in meta.columns for c in pos_columns + neg_columns):
        matcher = create_matcher(
            meta, pos_sameby, pos_diffby, neg_sameby, neg_diffby, multilabel_col
        )

        pos_pairs = matcher.get_all_pairs(sameby=pos_sameby, diffby=pos_diffby)
        pos_pairs = dict_to_dframe(pos_pairs, pos_sameby)

        neg_pairs = matcher.get_all_pairs(sameby=neg_sameby, diffby=neg_diffby)
        neg_pairs = set(chain.from_iterable(neg_pairs.values()))
        neg_pairs = pd.DataFrame(neg_pairs, columns=["ix1", "ix2"])

    else:
        matcher = create_matcher(meta, pos_sameby, pos_diffby, [], [], multilabel_col)
        pos_pairs = matcher.get_all_pairs(sameby=pos_sameby, diffby=pos_diffby)
        pos_pairs = dict_to_dframe(pos_pairs, pos_sameby)

        matcher = create_matcher(meta, [], [], neg_sameby, neg_diffby, multilabel_col)
        neg_pairs = matcher.get_all_pairs(sameby=neg_sameby, diffby=neg_diffby)
        neg_pairs = set(chain.from_iterable(neg_pairs.values()))
        neg_pairs = pd.DataFrame(neg_pairs, columns=["ix1", "ix2"])

    return pos_pairs, neg_pairs
