from utils.dependency_checker import DependencyChecker

def main():
    # Verificar dependencias antes de iniciar el pipeline
    deps = DependencyChecker("configs/dependencies.json")
    deps.ensure_dependencies()

    from modeling.modeling_pipeline import ModelPipeline
    # Inicializar y ejecutar el pipeline de modelado
    pipeline = ModelPipeline(file_path="MLOps-Gpo45/data/raw/online_news_modified.csv",
                              mlflow_experiment="Modeling_Experiment")
    (pipeline.load_data()
        .explore_data()
        .clean_data()
        # .plot_analysis() # Opcional: Descomentar para analizar gráficos.
        .preprocess_data()
        .train_models()
        .get_best_model())

if __name__ == "__main__":
    main()