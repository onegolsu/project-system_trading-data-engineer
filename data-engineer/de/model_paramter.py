import pandas as pd
from itertools import combinations
from .processor.factor.parameter import FACTOR_PARAMETER_PROCESSOR


class MODEL_PARAMTER_PROCESSOR:
    def __init__(self, FACTOR_ANALYSIS_CFG) -> None:
        self.FACTOR_ANALYSIS_CFG = FACTOR_ANALYSIS_CFG

    @staticmethod
    def get_factor_combs(factor_analyser):
        factors = [col for col in factor_analyser.factors_df.columns if col != "StockCode"]
        factor_combs = list(combinations(factors, 2))
        return factor_combs

    def get_params_dict(self, future_ohlcv_df, factor_analyser):
        factor_combs = self.get_factor_combs(factor_analyser)
        params_dict = dict()

        for factor_comb in factor_combs:
            profit_analysis_2d_df = factor_analyser.get_profit_analysis_2d_df(
                future_ohlcv_df, self.FACTOR_ANALYSIS_CFG, factor_comb
            )
            profit_analysis_2d_df.fillna(profit_analysis_2d_df.mean().mean(), inplace=True)
            factor_parameter_processor = FACTOR_PARAMETER_PROCESSOR(profit_analysis_2d_df, 3, 3)
            for i in range(5):
                param = factor_parameter_processor.get_param(n=i + 1)
                param_value = factor_parameter_processor.get_param_value(n=i + 1)
                param_variance = factor_parameter_processor.get_param_variance(n=i + 1)
                params_dict[param] = (param_value, param_variance)
        return params_dict

    def get_params(self, future_ohlcv_df, factor_analyser, n=5):
        params_dict = self.get_params_dict(future_ohlcv_df, factor_analyser)
        params_df = pd.DataFrame().from_dict(params_dict, orient="index", columns=["Value", "Variance"])
        params = list(params_df.nlargest(10, "Value").nsmallest(n, "Variance").index)
        return params
