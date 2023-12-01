from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .preprocessor import OHLCV_PREPROCESSOR


class FACTOR_PROCESSOR(ABC):
    @abstractmethod
    def get_factor_df(self):
        pass


class FUNDAMENTAL_FACTOR_PROCESSOR(FACTOR_PROCESSOR, OHLCV_PREPROCESSOR):
    def __init__(self, ohlcv_df, dart_fundamental_processor, fdr_info_processor) -> None:
        self.ohlcv_df = ohlcv_df
        self.dart_fundamental_processor = dart_fundamental_processor
        self.fdr_info_processor = fdr_info_processor

    def get_factor_df(self, processor_cfg):
        price_window = processor_cfg["ohlcv_recent_n"]
        main_df = super().get_recent_n_mean(self.ohlcv_df, price_window)
        main_df = self.append_dart_fundamental(main_df, self.dart_fundamental_processor)
        main_df = self.append_fdr_info(main_df, self.fdr_info_processor)
        factor_df = self._get_factor_df(main_df)
        factor_df.dropna(inplace=True)
        return factor_df

    @staticmethod
    def append_dart_fundamental(df, dart_fundamental_processor):
        # profit
        df["NetProfit"] = df["StockCode"].map(dart_fundamental_processor.get_NetProfit_dict())
        df["OperationProfit"] = df["StockCode"].map(dart_fundamental_processor.get_OperationProfit_dict())

        # current
        df["CurrentAssets"] = df["StockCode"].map(dart_fundamental_processor.get_CurrentAssets_dict())
        df["CurrentLiabilities"] = df["StockCode"].map(dart_fundamental_processor.get_CurrentLiabilities_dict())

        # total
        df["TotalAssets"] = df["StockCode"].map(dart_fundamental_processor.get_TotalAssets_dict())
        df["TotalLiabilities"] = df["StockCode"].map(dart_fundamental_processor.get_TotalLiabilities_dict())
        return df

    @staticmethod
    def append_fdr_info(df, fdr_info_processor):
        # shares
        df["Shares"] = df["StockCode"].map(fdr_info_processor.get_Shares_dict())
        return df

    @staticmethod
    def _get_factor_df(main_df):
        factor_df = main_df.loc[:, ["StockCode"]].copy()
        # my Factors
        # Current Liabilities Ratio
        factor_df["CLR"] = main_df["CurrentLiabilities"] / main_df["CurrentAssets"]
        # Total Liabilities Ratio
        factor_df["TLR"] = main_df["TotalLiabilities"] / main_df["TotalAssets"]

        # Netprofit Per Price
        factor_df["NPP"] = (main_df["Close"] * main_df["Shares"]) / main_df["NetProfit"]
        # OperationProfit Per Price
        factor_df["OPP"] = (main_df["Close"] * main_df["Shares"]) / main_df["OperationProfit"]

        # TotalAssets Per Price
        factor_df["TAPP"] = (main_df["Close"] * main_df["Shares"]) / main_df["TotalAssets"]
        # TotalEquity Per Price
        factor_df["TEPP"] = (main_df["Close"] * main_df["Shares"]) / (
            main_df["TotalAssets"] - main_df["TotalLiabilities"]
        )
        # CurrentAssets Per Price
        factor_df["CAPP"] = (main_df["Close"] * main_df["Shares"]) / main_df["CurrentAssets"]
        # CurrentEquity Per Price
        factor_df["CEPP"] = (main_df["Close"] * main_df["Shares"]) / (
            main_df["CurrentAssets"] - main_df["CurrentLiabilities"]
        )
        return factor_df


lr = LinearRegression()


class LINEAR_COEF_FACTOR_PROCESSOR(FACTOR_PROCESSOR, OHLCV_PREPROCESSOR):
    def __init__(self, ohlcv_df, fdr_info_processor) -> None:
        self.ohlcv_df = ohlcv_df
        self.fdr_info_processor = fdr_info_processor

    def get_factor_df(self, LINEAR_COEF_CFG):
        ohlcv_df = self.get_pps_ohlcv_df(self.ohlcv_df, self.fdr_info_processor)
        linear_coef_dict = dict()
        for factor in LINEAR_COEF_CFG["factors"]:
            moving_average_df = (
                ohlcv_df.sort_values("Date")
                .set_index("Date")
                .groupby("StockCode")[factor]
                .rolling(window=LINEAR_COEF_CFG["moving_average_window"])
                .mean()
                .reset_index()
            )
            recent_moving_average_df = self.get_recent_n_df(moving_average_df, LINEAR_COEF_CFG["coef_recent_n"])
            coef_dict = (
                recent_moving_average_df.sort_values("Date")
                .groupby("StockCode")[factor]
                .apply(lambda x: self.get_coef(x))
            )
            linear_coef_dict[factor] = coef_dict
        linear_coef_df = pd.DataFrame(linear_coef_dict)
        linear_coef_df.reset_index(names="StockCode", inplace=True)
        return linear_coef_df

    def get_pps_ohlcv_df(self, ohlcv_df, fdr_info_processor):
        pps_ohlcv_df = super().filter_zero(ohlcv_df, "Volume")
        pps_ohlcv_df = super().append_VolumeRotation(pps_ohlcv_df, fdr_info_processor)
        pps_ohlcv_df = super().filter_cnt(pps_ohlcv_df)
        return pps_ohlcv_df

    @staticmethod
    def get_recent_n_df(df, n):
        recent_dates = df["Date"].drop_duplicates().nlargest(n)
        recent_n_df = df[df["Date"].isin(recent_dates)]
        return recent_n_df

    @staticmethod
    def get_coef(array):
        x = np.arange(1, len(array) + 1).reshape(-1, 1)
        y = np.array(array).reshape(-1, 1)
        lr.fit(x, y)
        coef = lr.coef_[0][0]
        scaled_coef = coef / np.abs(np.mean(y))
        return scaled_coef


class MOVING_AVERAGE_FACTOR_PROCESSOR(FACTOR_PROCESSOR, OHLCV_PREPROCESSOR):
    def __init__(self, ohlcv_df, fdr_info_processor) -> None:
        super().__init__()
        self.ohlcv_df = ohlcv_df
        self.fdr_info_processor = fdr_info_processor

    def get_factor_df(self, MOVING_AVERAGE_CFG):
        try:
            moving_average_df = self.moving_average_df
        except:
            moving_average_df = self.get_moving_average_df(MOVING_AVERAGE_CFG)
            self.moving_average_df = moving_average_df

        recent_n_moving_average_df = (
            moving_average_df.sort_values("Date").groupby("StockCode").tail(MOVING_AVERAGE_CFG["signal_recent_n"])
        )
        factors = MOVING_AVERAGE_CFG["factors"]
        factor_signals = [f"{factor}Signal" for factor in factors]
        factor_dict_list = list()
        for factor_signal in factor_signals:
            factor_dict = recent_n_moving_average_df.groupby("StockCode")[factor_signal].sum().to_dict()
            factor_dict_list.append(factor_dict)
        recent_moving_average_df = pd.DataFrame(factor_dict_list).T
        recent_moving_average_df.columns = factors
        recent_moving_average_df.reset_index(names="StockCode", inplace=True)
        return recent_moving_average_df

    def get_moving_average_df(self, MOVING_AVERAGE_CFG):
        moving_average_df = self.get_pps_ohlcv_df(self.ohlcv_df, self.fdr_info_processor)
        for factor in MOVING_AVERAGE_CFG["factors"]:
            moving_average_df = self.append_moving_average(
                moving_average_df, factor, MOVING_AVERAGE_CFG["short_term_window"]
            )
            moving_average_df = self.append_moving_average(
                moving_average_df, factor, MOVING_AVERAGE_CFG["long_term_window"]
            )
            moving_average_df = self.append_signal(
                moving_average_df,
                factor,
                MOVING_AVERAGE_CFG["short_term_window"],
                MOVING_AVERAGE_CFG["long_term_window"],
            )
        return moving_average_df

    def get_pps_ohlcv_df(self, ohlcv_df, fdr_info_processor):
        pps_ohlcv_df = super().filter_zero(ohlcv_df, "Volume")
        pps_ohlcv_df = super().append_VolumeRotation(pps_ohlcv_df, fdr_info_processor)
        pps_ohlcv_df = super().filter_cnt(pps_ohlcv_df)
        return pps_ohlcv_df

    @staticmethod
    def append_moving_average(df, column, window):
        moving_average_row = (
            df.set_index("Date").sort_index().groupby("StockCode")[column].rolling(window=window).mean()
        )
        moving_average_row.name = f"{column}_{window}"
        appended_df = pd.merge(df, moving_average_row, on=["StockCode", "Date"])
        return appended_df

    @staticmethod
    def append_signal(df, column, st_window, lt_window):
        df[f"{column}Signal"] = (df[f"{column}_{st_window}"] - df[f"{column}_{lt_window}"]) / df[
            f"{column}_{st_window}"
        ]
        return df


class TRADER_FACTOR_PROCESSOR(FACTOR_PROCESSOR):
    def __init__(self, pykrx_loader) -> None:
        self.pykrx_loader = pykrx_loader

    def get_factor_df(self, stockcodes, TRADER_CFG):
        trader_factor_dict = dict()
        for stockcode in stockcodes:
            trader_df = self.load_trader_df(stockcode, TRADER_CFG["start"], TRADER_CFG["end"])
            pps_trader_df = self.get_pps_trader_df(trader_df)
            weighted_dict = self.get_weighted_dict(pps_trader_df)
            trader_factor_dict[stockcode] = weighted_dict
        trader_factor_df = pd.DataFrame(trader_factor_dict).T.reset_index(names="StockCode")
        return trader_factor_df

    def load_trader_df(self, stockcode, start, end):
        trader_df = self.pykrx_loader.get_stock_trader_df(StockCode=stockcode, start=start, end=end)
        return trader_df

    @staticmethod
    def get_pps_trader_df(trader_df):
        format_dict = {
            "Corp": ["금융투자", "보험", "투신", "사모"],
            "Indivisual": ["개인"],
            "Foreign": ["외국인"],
        }
        for key, value in format_dict.items():
            trader_df[key] = trader_df.loc[:, value].sum(axis=1)
        pps_trader_df = trader_df.loc[:, list(format_dict.keys())]
        return pps_trader_df

    @staticmethod
    def get_weighted_dict(pps_trader_df):
        weighted_trader_df = pps_trader_df.reset_index(drop=True).multiply(
            pd.Series(range(1, len(pps_trader_df) + 1)), axis=0
        )
        weighted_dict = weighted_trader_df.mean(axis=0).to_dict()
        return weighted_dict
