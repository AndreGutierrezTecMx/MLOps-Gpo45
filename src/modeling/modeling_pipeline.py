import utils.logger as lg
from utils.dependency_checker import DependencyChecker
from data.data_reader import DataReader
from data.data_explorer import DataExplorer
from data.data_cleaning import DataCleaning
from constants.dvc_remote_type_enums import DvcRemoteType
from versioning.version_tracker import VersionTracker
from versioning.version_control import VersionControl
from data.data_analysis import DataAnalysis
from utils.logger import get_logger
from data.data_preprocessing import Preprocessor
from modeling.data_modeling import ModelTrainer
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
        lg.setup_logging()
        self.logger = lg.get_logger(__name__)
        deps = DependencyChecker("configs/dependencies.json")
        deps.ensure_dependencies()
        self.file_path = file_path
        self.repo_url = repo_url
        self.revision = revision
        vc = VersionControl(mlflow_experiment=mlflow_experiment, mlflow_port=mlflow_port,
                            mlflow_tracking_uri=mlflow_tracking_uri,
                            dvc_remote_type=dvc_remote_type, dvc_remote_name=dvc_remote_name,
                            dvc_remote_path=dvc_remote_path)
        self.vt = VersionTracker(version_control=vc, output_dir=output_dir, metadata_path=metadata_path)
        self.dr = DataReader(file_path=f'{vc.project_root}/{self.file_path}', repo_url=self.repo_url, revision=self.revision)
        
    
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
    
    def clean_data(self):
        """Performs data cleaning using DataCleaning."""
        self.dc = DataCleaning(self.de.dataframe)
        # TODO: Implement cleaning steps here
        self.logger.info("Limpieza de datos completada.")
        return self
    
    def plot_analysis(self):
        """Plot the data analysis using DataAnalysis."""
        self.dc = DataAnalysis(self.de.dataframe)
        # TODO: Implement cleaning steps here
        self.logger.info("Impresion de graficas para el analisis de datos completada.")
        self.dc.plot_bar_charts()
        self.logger.info("Impresion de top 20 artículos más compartidos.")
        self.dc.print_top_shared_articles(top_n=20)
        return self
    
    def preprocess_data(self):
        """Placeholder for data preprocessing steps."""
        df_source = getattr(self.dc, "df_clean", None)
        if df_source is None:
            df_source = self.de.dataframe
            self.logger.info("ℹ️ No se encontró df_clean; se usa el DF del explorer.")
        # 2) Ejecutar el preprocesamiento (idéntico al notebook)
        self.logger.info("Iniciando etapa de preprocesamiento…")
        pre = Preprocessor(df_clean=df_source,target_col="shares",test_size=0.2,random_state=42).run()
        self.logger.info("Preprocesamiento de datos completado.")
        # 3) Guardar salidas para la etapa de modelado
        self.X_train, self.X_test, self.y_train, self.y_test = pre.get_splits()
        self.preprocess_ct = pre.get_preprocess()
        self.feature_groups = pre.get_feature_groups()
        self.logger.info(f"Feature groups: {self.feature_groups}")
        self.logger.info("Preprocesamiento de datos completado.")
        return self
    
    def modeling_data(self):
        """Placeholder for modeling steps."""
        # TODO: Implement modeling steps here
        # Obtener hiperparametros 
        # Obtener metricas a utilizar
        # Entrenar modelo de RandomForest
        # Save model RandomForest
        # TODO: Hacer la comparación de modelos y guardar el mejor modelo
        required = ["X_train", "X_test", "y_train", "y_test", "preprocess_ct"]
        missing = [k for k in required if not hasattr(self, k)]
        if missing:
            raise RuntimeError(f"Falta correr preprocess_data() antes de modelar. Faltan: {missing}")
        self.logger.info("Iniciando etapa de modelado…")
        trainer = ModelTrainer(preprocess=self.preprocess_ct,X_train=self.X_train, X_test=self.X_test
                               ,y_train=self.y_train, y_test=self.y_test,
                               cv_splits=5, random_state=42)
        metrics = {}
        best_estimators = {}
        # HGB (Poisson)
        best_hgb, m_hgb = trainer.fit_hgb_poisson()
        best_estimators["HistGradientBoosting (Poisson)"] = best_hgb
        metrics["HistGradientBoosting (Poisson)"] = m_hgb
        # Ridge (TTR)
        best_ridge, m_ridge = trainer.fit_ridge()
        best_estimators["Ridge (tuned)"] = best_ridge
        metrics["Ridge (tuned)"] = m_ridge
        # RF 
        best_rf, m_rf = trainer.fit_random_forest_fast()
        best_estimators["RandomForest (tuned fast)"] = best_rf
        metrics["RandomForest (tuned fast)"] = m_rf
        # XGBoost 
        try:
            best_xgb, m_xgb = trainer.fit_xgboost_fast()
            best_estimators["XGBoost (tuned fast)"] = best_xgb
            metrics["XGBoost (tuned fast)"] = m_xgb
        except Exception as e:
            self.logger.warning(f"No se corrió XGBoost: {e}")
        # Elegir mejor por R²
        best_name, best_model = trainer.best_by_r2()
        # Exponer resultados para MLflow
        self.model_metrics = metrics
        self.best_model_name = best_name
        self.best_model = best_model
        self.best_models_all = best_estimators
        self.logger.info("Modelado de datos completado.")
        return self
