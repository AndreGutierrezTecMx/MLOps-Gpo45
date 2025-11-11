# src/main.py
import json
from utils.dependency_checker import DependencyChecker
from utils.seed import set_seeds 
from modeling.modeling_pipeline import ModelPipeline

def main():
    # Dependencias
    DependencyChecker.ensure_dependencies("configs/dependencies.json")

    # --- leer semilla desde configs/experiment.json ---
    with open("configs/experiment.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    seed = int(cfg.get("seed", 42))
    set_seeds(seed)  # << fijamos semillas aquí para paso 3 

    # Pipeline (pasamos seed para usarla en splits/modelos)
    pipeline = ModelPipeline(
        file_path="MLOps-Gpo45/data/raw/online_news_modified.csv",
        mlflow_experiment="Modeling_Experiment",
        seed=seed,                    # << semilla para reproducibilidad de resultados
    )

    (pipeline
        .load_data()
        .explore_data()
        .clean_data()
        # .plot_analysis()
        .preprocess_data()
        .train_models()
        .get_best_model_info()
    )

if __name__ == "__main__":
    main()
