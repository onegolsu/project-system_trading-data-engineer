import numpy as np


class FACTOR_PARAMETER_PROCESSOR:
    def __init__(self, analysis_2d_df, n=3, m=3) -> None:
        self.analysis_2d_df = analysis_2d_df
        self.average_array = self.get_average_array(analysis_2d_df, n, m)
        self.variance_array = self.get_variance_array(analysis_2d_df, n, m)
        self.n = n
        self.m = m

    def get_average_array(self, analysis_2d_df, n, m):
        windows = np.lib.stride_tricks.sliding_window_view(analysis_2d_df.to_numpy(), (n, m))
        average_array = np.mean(windows, axis=(2, 3))
        return average_array

    def get_variance_array(self, analysis_2d_df, n, m):
        windows = np.lib.stride_tricks.sliding_window_view(analysis_2d_df.to_numpy(), (n, m))
        variance_array = np.var(windows, axis=(2, 3))
        return variance_array

    def get_param(self, n=1):
        analysis_2d_df = self.analysis_2d_df
        average_array = self.average_array
        selected_arg = np.unravel_index(average_array.flatten().argsort()[::-1][n], average_array.shape)

        factor_1 = list(analysis_2d_df.index[selected_arg[0] : selected_arg[0] + self.n])
        factor_1_key = factor_1[0].split("_")[0]
        factor_1_lower_pct = factor_1[0].split("_")[-1].split("~")[0]
        factor_1_upper_pct = factor_1[-1].split("_")[-1].split("~")[-1]

        factor_2 = list(analysis_2d_df.columns[selected_arg[1] : selected_arg[1] + self.m])
        factor_2_key = factor_2[0].split("_")[0]
        factor_2_lower_pct = factor_2[0].split("_")[-1].split("~")[0]
        factor_2_upper_pct = factor_2[-1].split("_")[-1].split("~")[-1]

        param = (
            (factor_1_key, float(factor_1_lower_pct), float(factor_1_upper_pct)),
            (factor_2_key, float(factor_2_lower_pct), float(factor_2_upper_pct)),
        )
        return param

    def get_param_value(self, n=1):
        param_value = np.sort(self.average_array.flatten())[::-1][n]
        return param_value

    def get_param_variance(self, n=1):
        param_variance = np.sort(self.variance_array.flatten())[::-1][n]
        return param_variance
