from typing import Optional
from pathlib import Path
import utils.logger as lg
from data.data_reader import DataReader
from data.data_explorer import DataExplorer
from data.data_cleaning import DataCleaning
from constants.dvc_remote_type_enums import DvcRemoteType
from versioning.version_tracker import VersionTracker
from versioning.version_control import VersionControl
from data.data_analysis import DataAnalysis
from data.data_preprocessing import Preprocessor
from modeling.data_modeling import ModelTrainer
from constants.column_names import ColumnNames
from mlflow.entities import RunInfo as run_info
import json
from utils.seed import set_seeds 

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
        output_dir: str = "data/",
        metadata_path: str = "data/interim/registry/data_versions.json",
        ):
        lg.setup_logging()
        self.logger = lg.get_logger(__name__)
        self.logger.info("Inicializando el pipeline de modelado...")
        self.file_path = file_path
        self.repo_url = repo_url
        self.revision = revision
        self.output_dir = output_dir
        self.vc = VersionControl(mlflow_experiment=mlflow_experiment, mlflow_port=mlflow_port,
                            mlflow_tracking_uri=mlflow_tracking_uri,
                            dvc_remote_type=dvc_remote_type, dvc_remote_name=dvc_remote_name,
                            dvc_remote_path=dvc_remote_path)
        self.vt = VersionTracker(version_control=self.vc, output_dir=f'{self.vc.project_root}/{output_dir}interim', metadata_path=f'{self.vc.project_root}/{metadata_path}')
        self.dr = DataReader(file_path=f'{self.vc.project_root}/{self.file_path}', repo_url=self.repo_url, revision=self.revision)
         # --- leer semilla desde configs/experiment.json ---
        config_path = Path(self.vc.project_root) / "configs" / "experiment.json"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        seed = int(cfg.get("seed", 42))
        set_seeds(seed)  # << fijamos semillas aquí para paso 3 
        self.seed = int(seed) #Semilla para repetibilidad
        
    
    def load_data(self):
        """Loads data using DataReader and returns a DataFrame."""
        self.df = self.dr.read_data()
        self.logger.info("Datos cargados correctamente en el pipeline.")
        self.de = DataExplorer(self.df)
        return self
    
    def explore_data(self):
        """Performs data exploration using DataExplorer."""
        self.de.full_report()
        return self
    
    def clean_data(self, strategy:str='drop', method:str='iqr', threshold:float=1.5):
        """Performs data cleaning using DataCleaning."""
        self.dc = DataCleaning(self.de.dataframe, tracker=self.vt)
        (self.dc
                .convert_data_types()
                .remove_duplicates()
                .handle_missing_values(strategy=strategy)
                .handle_outliers(method=method, threshold=threshold)
                .save_cleaned_data(f'{self.vc.project_root}/{self.output_dir}/processed/online_news_cleaned.csv'))
        self.logger.info(f"✅ Limpieza de datos completada. Reporte de limpieza: {self.dc.cleaning_report}")
        self.df = self.dc.df_clean
        return self
    
    def plot_analysis(self, top_n:int=20):
        """Plot the data analysis using DataAnalysis."""
        self.dc = DataAnalysis(self.de.dataframe)
        self.logger.info("Imprimiendo gráficas para el análisis de datos…")
        self.dc.plot_bar_charts()
        self.dc.print_top_shared_articles(top_n=top_n)
        self.dc.print_scatter_plot()
        self.dc.print_histograms()
        self.logger.info("✅ Impresion de graficas para el analisis de datos completada.")
        return self
    
    def preprocess_data(self):
        self.logger.info("Iniciando etapa de preprocesamiento…")
        pre = Preprocessor(df_clean=self.df, target_col=ColumnNames.SHARES.value, test_size=0.2, random_state=self.seed).run() #Semilla configurable
        self.X_train, self.X_test, self.y_train, self.y_test = pre.get_splits()
        self.preprocess_ct = pre.get_preprocess()
        self.feature_groups = pre.get_feature_groups()
        self.logger.info(f"Feature groups: {self.feature_groups}")
        self.logger.info("✅ Preprocesamiento de datos completado.")
        return self
    
    def train_models(self):
        """Steps to train models using ModelTrainer."""
        required = ["X_train", "X_test", "y_train", "y_test", "preprocess_ct"]
        missing = [k for k in required if not hasattr(self, k)]
        if missing:
            self.logger.error(f"Falta correr preprocess_data() antes de modelar. Faltan: {missing}")
            raise RuntimeError(f"Falta correr preprocess_data() antes de modelar. Faltan: {missing}")
        self.logger.info("Iniciando etapa de modelado…")
        self.trainer = ModelTrainer(preprocess=self.preprocess_ct,X_train=self.X_train, X_test=self.X_test,
                               y_train=self.y_train, y_test=self.y_test, version_tracker=self.vt,
                               cv_splits=5, random_state=self.seed) #Semilla Configurable
        metrics = {}
        best_estimators = {}
        # HGB (Poisson)
        best_hgb, m_hgb = self.trainer.fit_hgb_poisson()
        best_estimators["HistGradientBoosting (Poisson)"] = best_hgb
        metrics["HistGradientBoosting (Poisson)"] = m_hgb
        # Ridge (TTR)
        best_ridge, m_ridge = self.trainer.fit_ridge()
        best_estimators["Ridge (tuned)"] = best_ridge
        metrics["Ridge (tuned)"] = m_ridge
        # RF 
        best_rf, m_rf = self.trainer.fit_random_forest_fast()
        best_estimators["RandomForest (tuned fast)"] = best_rf
        metrics["RandomForest (tuned fast)"] = m_rf
        # XGBoost 
        try:
            best_xgb, m_xgb = self.trainer.fit_xgboost_fast()
            best_estimators["XGBoost (tuned fast)"] = best_xgb
            metrics["XGBoost (tuned fast)"] = m_xgb
        except Exception as e:
            self.logger.warning(f"No se corrió XGBoost: {e}")
        self.logger.info("✅ Modelado completado.")
        return self
    
    def get_best_model_info(self, metric: str = "R2") -> Optional[run_info]:
        """Gets the best tracked model info from VersionTracker."""
        return self.vt.get_best_tracked_model(metric=metric)
    
    def save_model(self):
        """Saves the best tracked model locally."""
        self.vt.save_best_tracked_model()

