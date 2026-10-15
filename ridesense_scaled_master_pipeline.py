# ============================================================
# RIDESENSE AI - FINAL MASTER PIPELINE
# REGRESSION PROBLEM
# ============================================================
 
import pandas as pd
import numpy as np
import joblib
 
import matplotlib
matplotlib.use("Agg")   # save plots as files (no popup window)
import matplotlib.pyplot as plt
 
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_validate
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
 
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
 
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
 
 
# ============================================================
# 1. LOAD DATA
# ============================================================
 
def load_data(path):
 
    df = pd.read_csv(path)
 
    print("\n[DATA LOADED]", df.shape)
 
    return df
 
 
# ============================================================
# 2. MISSING VALUE HANDLING
# ============================================================
 
def handle_missing(df):
 
    df = df.copy()
 
    for col in df.columns:
 
        if not pd.api.types.is_numeric_dtype(df[col]):
 
            df[col] = df[col].fillna(
                df[col].mode()[0]
            )
 
        else:
 
            df[col] = df[col].fillna(
                df[col].median()
            )
 
    print("\n[MISSING VALUES HANDLED]")
 
    return df
 
 
# ============================================================
# 3. WINSORIZATION
# ============================================================
 
def winsorize(df, num_cols):
 
    df = df.copy()
 
    for col in num_cols:
 
        q1 = df[col].quantile(0.25)
 
        q3 = df[col].quantile(0.75)
 
        lower = q1 - 1.5 * (q3 - q1)
 
        upper = q3 + 1.5 * (q3 - q1)
 
        df[col] = np.clip(
            df[col],
            lower,
            upper
        )
 
    print("\n[WINSORIZATION COMPLETED]")
 
    return df
 
 
# ============================================================
# 4. LOG TRANSFORMATION
# ============================================================
 
def log_transform(df, num_cols):
 
    df = df.copy()
 
    for col in num_cols:
 
        if (df[col] >= 0).all():
 
            df[col] = np.log1p(df[col])
 
    print("\n[LOG TRANSFORMATION COMPLETED]")
 
    return df
 
 
# ============================================================
# 5. FEATURE SELECTION
# ============================================================
 
def feature_selection(X, y):
 
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
 
    model.fit(X, y)
 
    importance = model.feature_importances_
 
    threshold = np.mean(importance)
 
    selected_cols = X.columns[
        importance >= threshold
    ]
 
    print("\nSelected Features:", len(selected_cols))
 
    print("Selected Feature Names:", list(selected_cols))
 
    return X[selected_cols]
 
 
# ============================================================
# 6. ENCODING
# ============================================================
 
def encode(df, target):
 
    X = df.drop(columns=[target])
 
    y = df[target]
 
    X = pd.get_dummies(
        X,
        drop_first=True
    )
 
    return X, y
 
 
# ============================================================
# 7. TRAIN TEST SPLIT
# ============================================================
 
def split(X, y):
 
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
 
 
# ============================================================
# 8. SCALING
# ============================================================
 
def scale(X_train, X_test):
 
    scaler = StandardScaler()
 
    X_train_scaled = scaler.fit_transform(X_train)
 
    X_test_scaled = scaler.transform(X_test)
 
    print("\n[SCALING COMPLETED]")
 
    return X_train_scaled, X_test_scaled
 
 
# ============================================================
# 9. MODELS
# ============================================================
 
def get_models():
 
    return {
 
        "LinearRegression": LinearRegression(),
 
        "KNNRegressor": KNeighborsRegressor(),
 
        "DecisionTree": DecisionTreeRegressor(
            random_state=42
        ),
 
        "RandomForest": RandomForestRegressor(
            random_state=42
        ),
 
        "AdaBoost": AdaBoostRegressor(
            random_state=42
        )
    }
 
 
# ============================================================
# 10. EVALUATION
# ============================================================
 
def evaluate(
        model,
        X_train,
        X_test,
        y_train,
        y_test
):
 
    model.fit(X_train, y_train)
 
    pred = model.predict(X_test)
 
    train_pred = model.predict(X_train)
 
    mse = mean_squared_error(y_test, pred)
 
    rmse = np.sqrt(mse)
 
    mae = mean_absolute_error(y_test, pred)
 
    mape = mean_absolute_percentage_error(
        y_test,
        pred
    )
 
    r2 = r2_score(y_test, pred)
 
    train_r2 = r2_score(y_train, train_pred)
 
    result = {
 
        "MSE": round(mse, 2),
 
        "RMSE": round(rmse, 2),
 
        "MAE": round(mae, 2),
 
        "MAPE": round(mape, 4),
 
        "R2": round(r2, 4),
 
        "Train_R2": round(train_r2, 4)
    }
 
    return result
 
 
# ============================================================
# 11. FINAL MODEL (TUNING + CV + PLOTS + SAVE PKL)
# ============================================================
 
def final_model(df, target):
 
    print("\n\n========== FINAL MODEL ==========")
 
    X = df.drop(columns=[target])
 
    y = df[target]
 
    X_train, X_test, y_train, y_test = split(X, y)
 
    num_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()
 
    cat_features = [
        col for col in X.columns
        if col not in num_features
    ]
 
    # ---------- preprocessing inside the pipeline ----------
    # (so the saved .pkl can predict directly on raw data)
 
    preprocessor = ColumnTransformer([
 
        ("num", SimpleImputer(strategy="median"), num_features),
 
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_features)
    ])
 
    pipe = Pipeline([
 
        ("prep", preprocessor),
 
        ("model", RandomForestRegressor(random_state=42))
    ])
 
    # ---------- hyperparameter tuning ----------
 
    param_dist = {
 
        "model__n_estimators": [100, 200, 300],
 
        "model__max_depth": [None, 10, 20, 30],
 
        "model__min_samples_split": [2, 5, 10],
 
        "model__min_samples_leaf": [1, 2, 4],
 
        "model__max_features": ["sqrt", 0.5, 1.0]
    }
 
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=15,
        cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1
    )
 
    print("\n[TUNING STARTED - please wait 1 to 3 minutes]")
 
    search.fit(X_train, y_train)
 
    best = search.best_estimator_
 
    print("\nBest Parameters:")
 
    for k, v in search.best_params_.items():
 
        print("  ", k, "=", v)
 
    # ---------- cross validation (5-fold on train data) ----------
 
    cv = cross_validate(
        best,
        X_train,
        y_train,
        cv=5,
        scoring=[
            "r2",
            "neg_root_mean_squared_error",
            "neg_mean_absolute_error"
        ]
    )
 
    print("\n===== 5-FOLD CROSS VALIDATION =====")
 
    print(
        "R2   : %.4f (+/- %.4f)" %
        (cv["test_r2"].mean(), cv["test_r2"].std())
    )
 
    print(
        "RMSE : %.2f (+/- %.2f)" %
        (-cv["test_neg_root_mean_squared_error"].mean(),
         cv["test_neg_root_mean_squared_error"].std())
    )
 
    print(
        "MAE  : %.2f (+/- %.2f)" %
        (-cv["test_neg_mean_absolute_error"].mean(),
         cv["test_neg_mean_absolute_error"].std())
    )
 
    # ---------- final test evaluation ----------
 
    pred = best.predict(X_test)
 
    train_pred = best.predict(X_train)
 
    mse = mean_squared_error(y_test, pred)
 
    print("\n===== FINAL TEST RESULTS (TUNED RANDOM FOREST) =====")
 
    print("MAE        :", round(mean_absolute_error(y_test, pred), 2))
 
    print("RMSE       :", round(np.sqrt(mse), 2))
 
    print("R2 (test)  :", round(r2_score(y_test, pred), 4))
 
    print("R2 (train) :", round(r2_score(y_train, train_pred), 4))
 
    # ---------- feature importance ----------
 
    names = best.named_steps["prep"].get_feature_names_out()
 
    names = [n.split("__", 1)[-1] for n in names]
 
    importances = best.named_steps["model"].feature_importances_
 
    imp_df = pd.DataFrame({
 
        "Feature": names,
 
        "Importance": importances
 
    }).sort_values("Importance", ascending=False)
 
    print("\n===== TOP 10 FEATURE IMPORTANCES =====")
 
    print(imp_df.head(10).to_string(index=False))
 
    top = imp_df.head(10).iloc[::-1]
 
    plt.figure(figsize=(8, 5))
 
    plt.barh(top["Feature"], top["Importance"])
 
    plt.xlabel("Importance")
 
    plt.title("Top 10 Feature Importances - RandomForest")
 
    plt.tight_layout()
 
    plt.savefig("feature_importance.png", dpi=150)
 
    plt.close()
 
    # ---------- actual vs predicted + residuals ----------
 
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
 
    ax[0].scatter(y_test, pred, alpha=0.4, s=10)
 
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
 
    ax[0].plot(lims, lims, "r--")
 
    ax[0].set_xlabel("Actual Fare")
 
    ax[0].set_ylabel("Predicted Fare")
 
    ax[0].set_title("Actual vs Predicted")
 
    ax[1].scatter(pred, y_test - pred, alpha=0.4, s=10)
 
    ax[1].axhline(0, color="r", linestyle="--")
 
    ax[1].set_xlabel("Predicted Fare")
 
    ax[1].set_ylabel("Residual")
 
    ax[1].set_title("Residual Plot")
 
    plt.tight_layout()
 
    plt.savefig("actual_vs_predicted.png", dpi=150)
 
    plt.close()
 
    print("\n[PLOTS SAVED] feature_importance.png, actual_vs_predicted.png")
 
    # ---------- save model ----------
 
    joblib.dump(best, "ridesense_model.pkl")
 
    print("[MODEL SAVED] ridesense_model.pkl")
 
    # ---------- quick check: load pkl and predict on raw rows ----------
 
    loaded = joblib.load("ridesense_model.pkl")
 
    sample = X_test.head(3)
 
    print("\n===== SAMPLE PREDICTIONS FROM SAVED MODEL =====")
 
    print(pd.DataFrame({
 
        "Actual": y_test.head(3).values,
 
        "Predicted": np.round(loaded.predict(sample), 2)
    }))
 
    return best
 
 
# ============================================================
# 12. RUN PIPELINE
# ============================================================
 
def run_pipeline(path):
 
    df = load_data(path)
 
    target = "ride_fare"
 
    # ========================================================
    # STEP 1 - MISSING VALUES
    # ========================================================
 
    df_raw = df.copy()   # keep raw copy for the final saved model
 
    df = handle_missing(df)
 
    # ========================================================
    # NUMERICAL COLUMNS
    # ========================================================
 
    num_cols = df.select_dtypes(
        include=np.number
    ).columns
 
    num_cols = [
        col for col in num_cols
        if col != target
    ]
 
    # ========================================================
    # STEP 2 - WINSORIZATION
    # ========================================================
 
    df_w = winsorize(df, num_cols)
 
    # ========================================================
    # STEP 3 - LOG TRANSFORMATION
    # ========================================================
 
    df_l = log_transform(df, num_cols)
 
    # ========================================================
    # EXPERIMENT 1 - BASE DATA
    # ========================================================
 
    X, y = encode(df, target)
 
    X_train, X_test, y_train, y_test = split(X, y)
 
    models = get_models()
 
    print("\n===== BASE DATA RESULTS =====")
 
    base_results = []
 
    for name, model in models.items():
 
        res = evaluate(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )
 
        res["Model"] = name
 
        base_results.append(res)
 
    base_df = pd.DataFrame(base_results)
 
    # ========================================================
    # EXPERIMENT 2 - SCALED DATA
    # ========================================================
 
    X, y = encode(df, target)
 
    X_train, X_test, y_train, y_test = split(X, y)
 
    X_train_scaled, X_test_scaled = scale(
        X_train,
        X_test
    )
 
    print("\n===== SCALED DATA RESULTS =====")
 
    scaled_results = []
 
    for name, model in models.items():
 
        res = evaluate(
            model,
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test
        )
 
        res["Model"] = name
 
        scaled_results.append(res)
 
    scaled_df = pd.DataFrame(scaled_results)
 
    # ========================================================
    # EXPERIMENT 3 - WINSORIZED DATA
    # ========================================================
 
    X, y = encode(df_w, target)
 
    X_train, X_test, y_train, y_test = split(X, y)
 
    X_train_scaled, X_test_scaled = scale(
        X_train,
        X_test
    )
 
    print("\n===== WINSORIZED DATA RESULTS =====")
 
    win_results = []
 
    for name, model in models.items():
 
        res = evaluate(
            model,
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test
        )
 
        res["Model"] = name
 
        win_results.append(res)
 
    win_df = pd.DataFrame(win_results)
 
    # ========================================================
    # EXPERIMENT 4 - LOG TRANSFORM
    # ========================================================
 
    X, y = encode(df_l, target)
 
    X_train, X_test, y_train, y_test = split(X, y)
 
    X_train_scaled, X_test_scaled = scale(
        X_train,
        X_test
    )
 
    print("\n===== LOG TRANSFORM RESULTS =====")
 
    log_results = []
 
    for name, model in models.items():
 
        res = evaluate(
            model,
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test
        )
 
        res["Model"] = name
 
        log_results.append(res)
 
    log_df = pd.DataFrame(log_results)
 
    # ========================================================
    # EXPERIMENT 5 - FEATURE SELECTION
    # ========================================================
 
    X, y = encode(df, target)
 
    X = feature_selection(X, y)
 
    X_train, X_test, y_train, y_test = split(X, y)
 
    X_train_scaled, X_test_scaled = scale(
        X_train,
        X_test
    )
 
    print("\n===== FEATURE SELECTION RESULTS =====")
 
    fs_results = []
 
    for name, model in models.items():
 
        res = evaluate(
            model,
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test
        )
 
        res["Model"] = name
 
        fs_results.append(res)
 
    fs_df = pd.DataFrame(fs_results)
 
    # ========================================================
    # FINAL OUTPUT
    # ========================================================
 
    print("\n\n========== FINAL COMPARISON ==========\n")
 
    print(
        "\nBASE DATA\n",
        base_df.sort_values(
            "RMSE",
            ascending=True
        )
    )
 
    print(
        "\nSCALED DATA\n",
        scaled_df.sort_values(
            "RMSE",
            ascending=True
        )
    )
 
    print(
        "\nWINSORIZED DATA\n",
        win_df.sort_values(
            "RMSE",
            ascending=True
        )
    )
 
    print(
        "\nLOG TRANSFORM DATA\n",
        log_df.sort_values(
            "RMSE",
            ascending=True
        )
    )
 
    print(
        "\nFEATURE SELECTION DATA\n",
        fs_df.sort_values(
            "RMSE",
            ascending=True
        )
    )
 
    # ========================================================
    # BEST MODEL - TUNING, CV, PLOTS, SAVE PKL
    # ========================================================
 
    final_model(df_raw, target)
 
    return (
        base_df,
        scaled_df,
        win_df,
        log_df,
        fs_df
    )
 
 
# ============================================================
# EXECUTION
# ============================================================
 
if __name__ == "__main__":
 
    run_pipeline(
        "ridesense_dataset.csv"
    )