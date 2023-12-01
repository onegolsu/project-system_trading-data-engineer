import os
from glob import glob
import pandas as pd


class KRX_LOADER:
    def __init__(self, path) -> None:
        self.path = path

    def get_kospi_info_df(self):
        kospi_path = os.path.join(self.path, "kospi_*.csv")
        latest_kospi_path = sorted(glob(kospi_path))[-1]
        kospi_info_df = pd.read_csv(latest_kospi_path, encoding="cp949")
        return kospi_info_df

    def get_kosdaq_info_df(self):
        kosdaq_path = os.path.join(self.path, "kosdaq_*.csv")
        latest_kosdaq_path = sorted(glob(kosdaq_path))[-1]
        kosdaq_info_df = pd.read_csv(latest_kosdaq_path, encoding="cp949")
        return kosdaq_info_df

    @staticmethod
    def concat_kospi_kosdaq(kospi_info_df, kosdaq_info_df):
        krx_info_df = pd.concat([kospi_info_df, kosdaq_info_df], axis=0)
        return krx_info_df

    def get_krx_info_df(self):
        kospi_info_df = self.get_kospi_info_df()
        kosdaq_info_df = self.get_kosdaq_info_df()
        krx_info_df = self.concat_kospi_kosdaq(kospi_info_df, kosdaq_info_df)
        return krx_info_df
    
