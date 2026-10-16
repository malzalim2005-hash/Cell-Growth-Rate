from src import constants as c
from src import functions as f


def main():
    c.FIGURES_DIR.mkdir(exist_ok=True)
    c.TABLES_DIR.mkdir(exist_ok=True)

    df = f.load_data()
    X_train, X_test, y_train, y_test = f.split_data(df)

    models = f.build_candidates()
    results = f.compare_models(models, X_train, y_train)
    f.plot_model_comparison(results)

    best_name = results.iloc[0]["Model"]
    print(f"\nSelected model: {best_name}")
    best_model = f.evaluate_on_test(best_name, models[best_name], X_train, X_test, y_train, y_test)
    f.plot_feature_effects(best_model, X_train)


if __name__ == "__main__":
    main()
