#
# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import division

from collections import OrderedDict
from functools import partial

import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as stats

from . import utils
from .utils import APPROX_BDAYS_PER_MONTH, APPROX_BDAYS_PER_YEAR
from .utils import DAILY, WEEKLY, MONTHLY, YEARLY, ANNUALIZATION_FACTORS
from .interesting_periods import PERIODS


def portfolio_returns(holdings_returns, exclude_non_overlapping=True):
    """Generates an equal-weight portfolio.
    Parameters
    ----------
    holdings_returns : list
       List containing each individual holding's daily returns of the
       strategy, noncumulative.
    exclude_non_overlapping : boolean, optional
       If True, timeseries returned will include values only for dates
       available across all holdings_returns timeseries If False, 0%
       returns will be assumed for a holding until it has valid data
    Returns
    -------
    pd.Series
        Equal-weight returns timeseries.
    """
    port = holdings_returns[0]
    for i in range(1, len(holdings_returns)):
        port = port + holdings_returns[i]

    if exclude_non_overlapping:
        port = port.dropna()
    else:
        port = port.fillna(0)

    return port / len(holdings_returns)


def portfolio_returns_metric_weighted(holdings_returns,
                                      exclude_non_overlapping=True,
                                      weight_function=None,
                                      weight_function_window=None,
                                      inverse_weight=False,
                                      portfolio_rebalance_rule='q',
                                      weight_func_transform=None):
    """
    Generates an equal-weight portfolio, or portfolio weighted by
    weight_function
    Parameters
    ----------
    holdings_returns : list
       List containing each individual holding's daily returns of the
       strategy, noncumulative.
    exclude_non_overlapping : boolean, optional
       (Only applicable if equal-weight portfolio, e.g. weight_function=None)
       If True, timeseries returned will include values only for dates
       available across all holdings_returns timeseries If False, 0%
       returns will be assumed for a holding until it has valid data
    weight_function : function, optional
       Function to be applied to holdings_returns timeseries
    weight_function_window : int, optional
       Rolling window over which weight_function will use as its input values
    inverse_weight : boolean, optional
       If True, high values returned from weight_function will result in lower
       weight for that holding
    portfolio_rebalance_rule : string, optional
       A pandas.resample valid rule. Specifies how frequently to compute
       the weighting criteria
    weight_func_transform : function, optional
       Function applied to value returned from weight_function
    Returns
    -------
    (pd.Series, pd.DataFrame)
        pd.Series : Portfolio returns timeseries.
        pd.DataFrame : All the raw data used in the portfolio returns
           calculations
    """

    if weight_function is None:
        if exclude_non_overlapping:
            holdings_df = pd.DataFrame(holdings_returns).T.dropna()
        else:
            holdings_df = pd.DataFrame(holdings_returns).T.fillna(0)

        holdings_df['port_ret'] = holdings_df.sum(axis=1)/len(holdings_returns)
    else:
        holdings_df_na = pd.DataFrame(holdings_returns).T
        holdings_cols = holdings_df_na.columns
        holdings_df = holdings_df_na.dropna()
        holdings_func = pd.rolling_apply(holdings_df,
                                         window=weight_function_window,
                                         func=weight_function).dropna()
        holdings_func_rebal = holdings_func.resample(
            rule=portfolio_rebalance_rule,
            how='last')
        holdings_df = holdings_df.join(
            holdings_func_rebal, rsuffix='_f').fillna(method='ffill').dropna()
        if weight_func_transform is None:
            holdings_func_rebal_t = holdings_func_rebal
            holdings_df = holdings_df.join(
                holdings_func_rebal_t,
                rsuffix='_t').fillna(method='ffill').dropna()
        else:
            holdings_func_rebal_t = holdings_func_rebal.applymap(
                weight_func_transform)
            holdings_df = holdings_df.join(
                holdings_func_rebal_t,
                rsuffix='_t').fillna(method='ffill').dropna()
        transform_columns = list(map(lambda x: x+"_t", holdings_cols))
        if inverse_weight:
            inv_func = 1.0 / holdings_df[transform_columns]
            holdings_df_weights = inv_func.div(inv_func.sum(axis=1),
                                               axis='index')
        else:
            holdings_df_weights = holdings_df[transform_columns] \
                .div(holdings_df[transform_columns].sum(axis=1), axis='index')

        holdings_df_weights.columns = holdings_cols
        holdings_df = holdings_df.join(holdings_df_weights, rsuffix='_w')
        holdings_df_weighted_rets = np.multiply(
            holdings_df[holdings_cols], holdings_df_weights)
        holdings_df_weighted_rets['port_ret'] = holdings_df_weighted_rets.sum(
            axis=1)
        holdings_df = holdings_df.join(holdings_df_weighted_rets,
                                       rsuffix='_wret')

    return holdings_df['port_ret'], holdings_df
