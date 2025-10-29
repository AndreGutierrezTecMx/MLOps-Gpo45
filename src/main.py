from modeling.modeling_pipeline import ModelPipeline

def main():
    pipeline = ModelPipeline(file_path="/MLOps-Gpo45/data/raw/online_news_modified.csv",
                              mlflow_experiment="Modeling_Experiment")
    (pipeline.load_data()
        .explore_data()
        .clean_data()
        .preprocess_data()
        .modeling_data())

if __name__ == "__main__":
    main()