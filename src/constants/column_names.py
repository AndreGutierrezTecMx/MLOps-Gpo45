from enum import Enum

class ColumnNames(Enum):
    """
    Enum con los nombres de columnas usados en el dataset.
    """
    DATA_CHANNEL_IS = "data_channel_is_"
    NO_CHANNEL = "'no_channel'"
    SHARES = "shares"
    WEEKDAY_IS = "weekday_is_"
    CHANNEL = "Channel"
    COUNT = "Count"
    TOTAL_SHARES = "Total Shares"
    DAY = "Day"
    URL = "url"
    COMPARTIDOS = "Compartidos"
    FRECUENCIA = "Frecuencia"

    @classmethod
    def list(cls):
        """Devuelve una lista con los valores válidos."""
        return [t.value for t in cls]
