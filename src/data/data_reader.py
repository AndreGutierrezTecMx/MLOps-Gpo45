from src.utils.logger import get_logger
import pandas as pd
logger = get_logger(__name__)

class DataReader:
    """
    Unified data reader for loading CSV datasets from multiple sources.

    Supports reading from local files and DVC repositories with automatic
    source detection based on provided parameters.
    """

    def __init__(self, file_path: str = None, repo_url: str = None, revision: str = None):
        """
        Initialize DataReader with data source configuration.

        Args:
            file_path (str, optional): Path to local CSV file
            repo_url (str, optional): URL of the DVC repository
            revision (str, optional): Git revision/branch for DVC repo

        Note:
            Must provide either file_path OR both repo_url and revision.
        """
        self.file_path = file_path
        self.repo_url = repo_url
        self.revision = revision

    def read_data(self) -> pd.DataFrame:
        """
        Read data from the configured source.

        Automatically detects and uses the appropriate loading method based
        on initialization parameters.

        Returns:
            pd.DataFrame: Loaded dataset

        Raises:
            ValueError: If no valid data source configuration is provided
        """
        if self.file_path:
            return self._load_from_file()

        if self.repo_url and self.revision:
            return self._load_from_repo()

        raise ValueError("No se proporcionó file_path o configuración de DVC válida")

    def _load_from_file(self) -> pd.DataFrame:
        """
        Load data from local CSV file.

        Returns:
            pd.DataFrame: Dataset loaded from local file

        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.EmptyDataError: If file is empty
            Exception: For other file reading errors
        """
        try:
            dataframe = pd.read_csv(self.file_path)
            logger.info(f"✅ Datos cargados desde {self.file_path}")
            return dataframe
        except Exception as e:
            logger.exception(f"❌ Error al leer los datos desde {self.file_path}")
            raise

    def _load_from_repo(self) -> pd.DataFrame:
        """
        Load data from DVC repository.

        Retrieves data URL from DVC repository at specified revision
        and loads the CSV file from remote location.

        Returns:
            pd.DataFrame: Dataset loaded from DVC repository

        Raises:
            ImportError: If DVC is not installed
            dvc.exceptions.PathMissingError: If path doesn't exist in repo
            Exception: For other DVC or network errors
        """
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