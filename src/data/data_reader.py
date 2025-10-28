from utils.logger import get_logger
import pandas as pd
logger = get_logger(__name__)

class DataReader:
    """Class to read the data from a CSV file."""
    
    def __init__(self, file_path: str = None, repo_url: str = None, revision: str = None):
        self.file_path = file_path
        self.repo_url = repo_url
        self.revision = revision

    def read_data(self) -> pd.DataFrame:
        """Reads the CSV file and returns a DataFrame."""
        if self.file_path:
            try:
                dataframe = pd.read_csv(self.file_path)
                logger.info(f"✅ Datos cargados desde {self.file_path}")
                return dataframe
            except Exception as e:
                logger.exception(f"❌ Error al leer los datos desde {self.file_path}")
                raise
        elif self.repo_url and self.revision:
            try:
                import dvc.api
                data_url = dvc.api.get_url(
                    path='data/raw/online_news_modified.csv',
                    repo=self.repo_url,
                    rev=self.revision
                )
                dataframe = pd.read_csv(data_url)
                logger.info(f"✅ Datos cargados desde DVC repo {self.repo_url} at revision {self.revision}")
                return dataframe
            except Exception as e:
                logger.exception(f"❌ Error al leer los datos desde DVC repo {self.repo_url} at revision {self.revision}")
                raise