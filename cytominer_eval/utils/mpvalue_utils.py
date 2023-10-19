import numpy as np
import pandas as pd
from typing import Union

from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance


class MahalanobisEstimator:
    """
    Store location and dispersion estimators of the empirical distribution of data
    provided in an array and allow computation of statistical distances.

    Parameters
    ----------
    arr : {pandas.DataFrame, np.ndarray}
        the matrix used to calculate covariance

    Attributes
    ----------
    sigma : np.array
        Fitted covariance matrix of sklearn.covariance.EmpiricalCovariance()

    Methods
    -------
    mahalanobis(X)
        Computes mahalanobis distance between the input array (self.arr) and the X
        array as provided
    """

    def __init__(self, arr: Union[pd.DataFrame, np.ndarray]):
        self.sigma = EmpiricalCovariance().fit(arr)

    def mahalanobis(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Compute the mahalanobis distance between the empirical distribution described
        by this object and points in an array `X`.

        Parameters
        ----------
        X : {pandas.DataFrame, np.ndarray}
            A samples by features array-like matrix to compute mahalanobis distance
            between self.arr

        Returns
        -------
        numpy.array
            Mahalanobis distance between the input array and the original sigma
        """
        return self.sigma.mahalanobis(X)


def calculate_mahalanobis(pert_df: pd.DataFrame, control_df: pd.DataFrame) -> pd.Series:
    """Given perturbation and control dataframes, calculate mahalanobis distance per
    perturbation

    Usage: Designed to be called within a pandas.DataFrame().groupby().apply(). See
    :py:func:`cytominer_eval.operations.util.calculate_mp_value`.

    Parameters
    ----------
    pert_df : pandas.DataFrame
        A pandas dataframe of replicate perturbations (samples by features)
    control_df : pandas.DataFrame
        A pandas dataframe of control perturbations (samples by features). Must have the
        same feature measurements as pert_df

    Returns
    -------
    float
        The mahalanobis distance between perturbation and control
    """
    assert len(control_df) > 1, "Error! No control perturbations found."

    # Get dispersion and center estimators for the control perturbations
    control_estimators = MahalanobisEstimator(control_df)

    # Distance between mean of perturbation and control
    maha = control_estimators.mahalanobis(np.array(np.mean(pert_df, 0)).reshape(1, -1))[
        0
    ]
    return maha


def calculate_mp_value(
    pert_df: pd.DataFrame,
    control_df: pd.DataFrame,
    rescale_pca: bool = True,
    nb_permutations: int = 100,
) -> pd.Series:
    """Given perturbation and control dataframes, calculate mp-value per perturbation

    Usage: Designed to be called within a pandas.DataFrame().groupby().apply(). See
    :py:func:`cytominer_eval.operations.mp_value.mp_value`.

    Parameters
    ----------
    pert_df : pandas.DataFrame
        A pandas dataframe of replicate perturbations (samples by features)
    control_df : pandas.DataFrame
        A pandas dataframe of control perturbations (samples by features). Must have the
        same feature measurements as pert_df
    params : {dict}, optional
        the parameters to use when calculating mp value. See
        :py:func:`cytominer_eval.operations.util.default_mp_value_parameters`.

    Returns
    -------
    float
        The mp value for the given perturbation

    """
    assert len(control_df) > 1, "Error! No control perturbations found."

    merge_df = pd.concat([pert_df, control_df]).reset_index(drop=True)

    # We reduce the dimensionality with PCA
    # so that 90% of the variance is conserved
    pca = PCA(n_components=0.9, svd_solver="full")

    try:
        pca_array = pca.fit_transform(merge_df)
    except np.linalg.LinAlgError as err:
        if "SVD did not converge" in str(err):
            print(
                "SVD did not converge: check that merged dataframe does not contain duplicate rows or columns."
            )
        raise err

    # We scale columns by the variance explained
    if rescale_pca:
        pca_array = pca_array * pca.explained_variance_ratio_
    # This seems useless, as the point of using the Mahalanobis
    # distance instead of the Euclidean distance is to be independent
    # of axes scales

    # Distance between mean of perturbation and control
    obs = calculate_mahalanobis(
        pert_df=pca_array[: pert_df.shape[0]],
        control_df=pca_array[-control_df.shape[0] :],
    )
    # In the paper's methods section it mentions the covariance used
    # might be modified to include variation of the perturbation as well.

    # Permutation test
    sim = np.zeros(nb_permutations)
    pert_mask = np.zeros(pca_array.shape[0], dtype=bool)
    pert_mask[: pert_df.shape[0]] = 1
    for i in range(nb_permutations):
        pert_mask_perm = np.random.permutation(pert_mask)
        pert_perm = pca_array[pert_mask_perm]
        control_perm = pca_array[np.logical_not(pert_mask_perm)]
        sim[i] = calculate_mahalanobis(pert_df=pert_perm, control_df=control_perm)

    return np.mean(sim >= obs)
