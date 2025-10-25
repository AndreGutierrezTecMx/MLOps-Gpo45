import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataAnalysis:
    """A class for get visualizations from cleaned dataframe."""

    def __init__(self, dataframe):
        """
        Initializes the DataExplorer with a pandas DataFrame.

        Parameters:
        dataframe (pd.DataFrame): The DataFrame to be explored.
        """
        self.dataframe = dataframe
        self.categorical_cols = self.dataframe.select_dtypes(include=['object']).columns.tolist()
        self.numeric_cols = self.dataframe.select_dtypes(include=[np.number]).columns.tolist()