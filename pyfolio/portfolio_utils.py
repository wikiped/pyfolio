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


def bucket_std(value, bins=[0.12, 0.15, 0.18, 0.21], max_default=0.24):
    """
    Simple quantizing function. For use in binning stdevs into a "buckets"
    Parameters
    ----------
    value : float
       Value corresponding to the the stdev to be bucketed
    bins : list, optional
       Floats used to describe the buckets which the value can be placed
    max_default : float, optional
       If value is greater than all the bins, max_default will be returned
    Returns
    -------
    float
        bin which the value falls into
    """

    annual_vol = value * np.sqrt(252)

    for i in bins:
        if annual_vol <= i:
            return i

    return max_default


def min_max_vol_bounds(value, lower_bound=0.12, upper_bound=0.24):
    """
    Restrict volatility weighting of the lowest volatility asset versus the
    highest volatility asset to a certain limit.
    E.g. Never allocate more than 2x to the lowest volatility asset.
    round up all the asset volatilities that fall below a certain bound
    to a specified "lower bound" and round down all of the asset
    volatilites that fall above a certain bound to a specified "upper bound"
    Parameters
    ----------
    value : float
       Value corresponding to a daily volatility
    lower_bound : float, optional
       Lower bound for the volatility
    upper_bound : float, optional
       Upper bound for the volatility
    Returns
    -------
    float
        The value input, annualized, or the lower_bound or upper_bound
    """

    annual_vol = value * np.sqrt(252)

    if annual_vol < lower_bound:
        return lower_bound

    if annual_vol > upper_bound:
        return upper_bound

    return annual_vol
