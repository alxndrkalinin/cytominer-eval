"""Functions to calculate multidimensional perturbation values (mp-value)

mp-value describes the distance, in dimensionality-reduced space, between a perturbation
and a control [1]_.

References
----------

.. [1] Hutz, J. et al. "The Multidimensional Perturbation Value: A Single Metric to
   Measure Similarity and Activity of Treatments in High-Throughput Multidimensional
   Screens" Journal of Biomolecular Screening, Volume: 18 issue: 4, page(s): 367-377.
   doi: 10.1177/1087057112469257
"""

import pandas as pd
from typing import List

from cytominer_eval.utils.mpvalue_utils import calculate_mp_value


def mp_value(
    df: pd.DataFrame,
    features: List[str],
    control_perts: List[str],
    replicate_id: str,
    kwargs: dict = {},
) -> pd.DataFrame:
    """Calculate multidimensional perturbation value (mp-value) [1].

    Parameters
    ----------
    df : pandas.DataFrame
        profiles with measurements per row and features or metadata per column.
    control_perts : list
        The control perturbations against which the distances will be computed.
    replicate_id : str
        The metadata identifier marking which column tracks control and replicate perts.
    features : list
        columns containing numerical features to be used for the mp-value computation
    params : dict, optional
        Optional parameters provided. See list of parameters in
        :py:func:`cytominer_eval.operations.util.default_mp_value_parameters`

    Returns
    -------
    pd.DataFrame
        mp-values per perturbation.
    """
    assert isinstance(
        replicate_id, str
    ), "replicate_id must be a string with column name"
    assert replicate_id in df.columns, "replicate_id not found in dataframe columns"

    # Extract features for control rows
    control_df = df[df[replicate_id].isin(control_perts)]
    replicate_df = df[~df[replicate_id].isin(control_perts)]

    # Calculate mp_value for each perturbation
    mp_value_dict = {}
    for replicate_id_name, group_df in replicate_df.groupby(replicate_id):
        mp_value = calculate_mp_value(
            group_df[features], control_df[features], **kwargs
        )
        mp_value_dict[replicate_id_name] = mp_value

    mp_value_df = pd.DataFrame(
        list(mp_value_dict.items()), columns=[replicate_id, "mp_value"]
    )
    mp_value_df.reset_index(inplace=True)

    return mp_value_df
