import pandas as pd
import numpy as np


def get_hold_time_dollar_volume(pos, market_data):
	DV = mkt_data['volume'] * mkt_data['close_price']
	DV = DV.rename(columns=data['sid_symbol_mapping']).replace(0, np.nan)

	positions_alloc = pf.pos.get_percent_alloc(pos)
	positions_alloc = positions_alloc.drop('cash', axis=1)

	max_exposure_per_ticker = abs(positions_alloc).max()

	vol_analysis = pd.DataFrame()
	vol_analysis['algo_max_exposure_pct'] = max_exposure_per_ticker
	vol_analysis.loc[:, 'avg_daily_dollar_volume'] = np.round(DV.mean() / 1000000, 2)
	vol_analysis['10th_%_daily_dollar_volume'] = np.round(DV.apply(
		lambda x: np.nanpercentile(x, 10, )) / 1000000, 2)
	vol_analysis['90th_%_daily_dollar_volume'] = np.round(DV.apply(
		lambda x: np.nanpercentile(x, 90)) / 1000000, 2)

	return vol_analysis

def get_portfolio_size_constraints(vol_analysis):
	constraints = pd.DataFrame()
	constraints['algo_max_capacity_at_adtv ($mm)'] = np.round((vol_analysis.avg_daily_dollar_volume * daily_volume_limit)\
                                                          / vol_analysis.algo_max_exposure_pct, 2)
	constraints['algo_max_capacity_at_10th% ($mm)'] = np.round((vol_analysis['10th_%_daily_dollar_volume'] * daily_volume_limit)\
                                                          / vol_analysis.algo_max_exposure_pct, 2)
	constraints['algo_max_capacity_at_90th% ($mm)'] = np.round((vol_analysis['90th_%_daily_dollar_volume'] * daily_volume_limit)\
                                                          / vol_analysis.algo_max_exposure_pct, 2)

	constraints.sort('algo_max_capacity_at_adtv ($mm)')

	return constraints
