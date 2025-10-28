from utils.logger import get_logger
from data.data_reader import DataReader
from data.data_explorer import DataExplorer
from data.data_cleaning import DataCleaning
from constants.dvc_remote_type_enums import DvcRemoteType
from versioning.version_tracker import VersionTracker
from versioning.version_control import VersionControl

logger = get_logger(__name__)

class ModelPipeline:
    def __init__(
        self,
        file_path: str = None,
        repo_url: str = None,
        revision: str = None,
        mlflow_tracking_uri: str = "http://127.0.0.1:5000",
        mlflow_port: int = 5000,
        mlflow_experiment: str = "default",
        dvc_remote_type: DvcRemoteType = DvcRemoteType.LOCAL,
        dvc_remote_name: str = "myremote",
        dvc_remote_path: str = "../../dvc_remote",
        output_dir: str = "data/processed/",
        metadata_path: str = "registry/data_versions.json"):
        self.file_path = file_path
        self.repo_url = repo_url
        self.revision = revision
        self.dr = DataReader(file_path=self.file_path, repo_url=self.repo_url, revision=self.revision)
        vc = VersionControl(mlflow_experiment=mlflow_experiment, mlflow_port=mlflow_port,
                            mlflow_tracking_uri=mlflow_tracking_uri,
                            dvc_remote_type=dvc_remote_type, dvc_remote_name=dvc_remote_name,
                            dvc_remote_path=dvc_remote_path)
        self.vt = VersionTracker(version_control=vc, output_dir=output_dir, metadata_path=metadata_path)
    
    def load_data(self):
        """Loads data using DataReader and returns a DataFrame."""
        self.df = self.dr.read_data()
        logger.info("Datos cargados correctamente en el pipeline.")
        self.de = DataExplorer(self.df)
        return self
    
    def explore_data(self):
        """Performs data exploration using DataExplorer."""
        self.de.full_report()
        return self
    
    def clean_data(self):
        """Performs data cleaning using DataCleaning."""
        self.dc = DataCleaning(self.de.dataframe)
        logger.info("Limpieza de datos completada.")
        return self
    
    def preprocess_data(self):
        """Placeholder for data preprocessing steps."""
        # Implement preprocessing steps here
        logger.info("Preprocesamiento de datos completado.")
        return self
    
    def modeling_data(self):
        """Placeholder for modeling steps."""
        # Implement modeling steps here
        logger.info("Modelado de datos completado.")
        return self