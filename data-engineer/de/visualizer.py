import numpy as np
import seaborn as sns

class ANALYSIS_VISUALIZER:
    def __init__(self,future_ohlcv_df) -> None:
        self.future_ohlcv_df = future_ohlcv_df
    
    def show_1d_heatmap(self, factor_analyser, FACTOR_ANALYSIS_CFG):
        profit_1d_df = factor_analyser.get_profit_analysis_1d_df(self.future_ohlcv_df, FACTOR_ANALYSIS_CFG)
        sns.heatmap(
            profit_1d_df,
            annot=True,
            cmap="RdBu_r",
            vmin=np.quantile(profit_1d_df.to_numpy().flatten(), 0.2),
            vmax=np.quantile(profit_1d_df.to_numpy().flatten(), 0.8),
            center=np.median(profit_1d_df.to_numpy()),
            fmt=".2f",
        )
        return None

    
    def show_2d_heatmap(self, factor_analyser, FACTOR_ANALYSIS_CFG, factors):
        profit_2d_df = factor_analyser.get_profit_analysis_2d_df(self.future_ohlcv_df, FACTOR_ANALYSIS_CFG, factors=factors)
        profit_2d_df.fillna(profit_2d_df.mean(), inplace=True)
        sns.heatmap(
            profit_2d_df,
            annot=True,
            cmap="RdBu_r",
            vmin=np.quantile(profit_2d_df.to_numpy().flatten(), 0.2),
            vmax=np.quantile(profit_2d_df.to_numpy().flatten(), 0.8),
            center=np.median(profit_2d_df.to_numpy()),
            fmt=".2f",
        )
        return None