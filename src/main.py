# src/main.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.modeling.modeling_pipeline import ModelPipeline

def main():
    # Pipeline
    pipeline = ModelPipeline(
        file_path="data/raw/online_news_modified.csv",
        mlflow_experiment="Modeling_Experiment",
    )

    (pipeline
        .load_data()
        .explore_data()
        .clean_data()
        .plot_analysis()
        .preprocess_data()
        .train_models()
        .get_best_model_info()
    )

if __name__ == "__main__":
    main()
