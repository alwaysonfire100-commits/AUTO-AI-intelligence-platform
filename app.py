
# ============================================================
# AVN 💞🧸🧸 AUTO AI INTELLIGENCE PLATFORM
# Complete Data Science Dashboard
# Supports CSV + PDF
# ============================================================

# -------------------------------
# IMPORT REQUIRED LIBRARIES
# -------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import fitz  # PyMuPDF for PDF reading
import joblib
from collections import Counter
import re

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="AVN AI Platform",
    layout="wide"
)

# -------------------------------
# CUSTOM LIGHT STYLING
# -------------------------------

st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.sidebar .sidebar-content {
    background-color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR SECTION
# -------------------------------

st.sidebar.markdown("## 🔎 Global Search")

search_term = st.sidebar.text_input("Search anything (column/value)")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📂 Navigate",
    ["Data Overview", "Visualization", "Machine Learning", "PDF Summary"]
)

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or PDF",
    type=["csv", "pdf"]
)

# -------------------------------
# GLOBAL VARIABLES
# -------------------------------

df = None
pdf_text = None

# -------------------------------
# FILE PROCESSING
# -------------------------------

if uploaded_file is not None:

    # CSV Handling
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("CSV Loaded Successfully")

    # PDF Handling
    elif uploaded_file.name.endswith(".pdf"):
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        full_text = ""

        for page_num in range(len(pdf_document)):
            page_obj = pdf_document.load_page(page_num)
            full_text += page_obj.get_text()

        pdf_text = full_text
        st.sidebar.success("PDF Loaded Successfully")

# -------------------------------
# LOGO & TITLE
# -------------------------------

st.markdown("# AVN 💞🧸🧸")
st.title("AUTO AI Intelligence Platform")

# ============================================================
# DATA OVERVIEW PAGE
# ============================================================

if page == "Data Overview":

    if df is not None:

        st.header("📊 Dataset Preview")

        # Apply search filter
        if search_term:
            filtered_df = df[df.astype(str).apply(
                lambda row: row.str.contains(search_term, case=False).any(),
                axis=1
            )]
            st.dataframe(filtered_df)
        else:
            st.dataframe(df.head())

        st.markdown("### 📈 Dataset Info")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        st.markdown("### 📉 Missing Values")
        st.dataframe(df.isnull().sum())

        st.download_button(
            "⬇ Download Dataset",
            df.to_csv(index=False),
            file_name="dataset.csv"
        )

    elif pdf_text is not None:
        st.info("PDF Loaded. Go to PDF Summary page.")
    else:
        st.warning("Please upload CSV or PDF file.")

# ============================================================
# VISUALIZATION PAGE
# ============================================================

elif page == "Visualization":

    if df is None:
        st.warning("Upload CSV file for visualization.")
        st.stop()

    st.header("📊 Visualization Dashboard")

    plot_type = st.selectbox(
        "Select Plot Type",
        ["Histogram", "Line Plot", "Scatter Plot",
         "Box Plot", "Bar Plot", "Correlation Heatmap"]
    )

    if plot_type == "Histogram":
        column = st.selectbox("Select Column", df.select_dtypes(include=np.number).columns)
        fig, ax = plt.subplots()
        ax.hist(df[column], bins=20)
        st.pyplot(fig)

    elif plot_type == "Line Plot":
        column = st.selectbox("Select Column", df.select_dtypes(include=np.number).columns)
        fig, ax = plt.subplots()
        ax.plot(df[column])
        st.pyplot(fig)

    elif plot_type == "Scatter Plot":
        col1 = st.selectbox("X Axis", df.select_dtypes(include=np.number).columns)
        col2 = st.selectbox("Y Axis", df.select_dtypes(include=np.number).columns)
        fig, ax = plt.subplots()
        ax.scatter(df[col1], df[col2])
        st.pyplot(fig)

    elif plot_type == "Box Plot":
        column = st.selectbox("Select Column", df.select_dtypes(include=np.number).columns)
        fig, ax = plt.subplots()
        ax.boxplot(df[column])
        st.pyplot(fig)

    elif plot_type == "Bar Plot":
        column = st.selectbox("Select Column", df.columns)
        fig, ax = plt.subplots()
        df[column].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    elif plot_type == "Correlation Heatmap":
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, ax=ax)
        st.pyplot(fig)

# ============================================================
# MACHINE LEARNING PAGE
# ============================================================
elif page == "Machine Learning":

    import datetime
    import os
    import shap
    import joblib
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, r2_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.header("🚀 Pro-Level Auto Machine Learning Engine")

    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # Convert categorical features
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Detect problem type
    problem_type = "Classification" if y.nunique() <= 10 else "Regression"
    st.success(f"Detected Problem Type: {problem_type}")

    # ---------------------------------------
    # MODEL TRAINING + HYPERPARAMETER TUNING
    # ---------------------------------------
    if problem_type == "Classification":
        model = RandomForestClassifier()
    else:
        model = RandomForestRegressor()

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 5, 10]
    }

    grid = GridSearchCV(model, param_grid, cv=3)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # ---------------------------------------
    # PERFORMANCE METRICS
    # ---------------------------------------
    preds = best_model.predict(X_test)

    if problem_type == "Classification":
        score = accuracy_score(y_test, preds)
        metric_name = "Accuracy"
    else:
        score = r2_score(y_test, preds)
        metric_name = "R2 Score"

    cv_score = cross_val_score(best_model, X, y, cv=5).mean()

    leaderboard = pd.DataFrame({
        "Metric": [metric_name, "Cross Validation"],
        "Score": [score, cv_score]
    })

    st.subheader("📊 Model Performance Leaderboard")
    st.dataframe(leaderboard)

    st.success(f"Best Parameters: {grid.best_params_}")

    # ---------------------------------------
    # CONFUSION MATRIX (Classification Only)
    # ---------------------------------------
    if problem_type == "Classification":
        st.subheader("📌 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d")
        st.pyplot(fig)

    # ---------------------------------------
    # ACTUAL VS PREDICTED (Regression Only)
    # ---------------------------------------
    if problem_type == "Regression":
        st.subheader("📈 Actual vs Predicted")
        fig, ax = plt.subplots()
        ax.scatter(y_test, preds)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

    # ---------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------
    if hasattr(best_model, "feature_importances_"):
        st.subheader("📊 Feature Importance")
        importance = best_model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(feat_df.set_index("Feature"))

    # ---------------------------------------
    # SHAP EXPLAINABILITY
    # ---------------------------------------
    st.subheader("🧠 SHAP Explainability")

    try:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)

        fig2 = plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig2)
    except:
        st.warning("SHAP not supported for this model.")

    # ---------------------------------------
    # MODEL VERSIONING SYSTEM
    # ---------------------------------------
    st.subheader("💾 Model Versioning")

    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    if st.button("Save Model Version"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"saved_models/model_{timestamp}.pkl"
        joblib.dump(best_model, filename)
        st.success(f"Model saved as {filename}")

    # Show saved models
    saved_files = os.listdir("saved_models")
    if saved_files:
        st.subheader("📂 Saved Models")
        st.write(saved_files)

    # ---------------------------------------
    # MANUAL PREDICTION INTERFACE
    # ---------------------------------------
    st.subheader("🔮 Manual Prediction")

    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(f"Enter value for {col}", value=0.0)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = best_model.predict(input_df)
        st.success(f"Prediction: {prediction[0]}")
    # DETECT PROBLEM TYPE
    # -------------------------------------

    if y.nunique() <= 10:
        st.success("Detected: Classification")

        model = RandomForestClassifier()

        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5]
        }

        grid = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring="accuracy"
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        preds = best_model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        st.write("### Best Parameters:")
        st.json(grid.best_params_)

        st.write("### Final Accuracy:", acc)

    else:
        st.success("Detected: Regression")

        model = RandomForestRegressor()

        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5]
        }

        grid = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring="r2"
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        preds = best_model.predict(X_test)
        r2 = r2_score(y_test, preds)

        st.write("### Best Parameters:")
        st.json(grid.best_params_)

        st.write("### Final R2 Score:", r2)

    # -------------------------------------
    # SAVE BEST MODEL
    # -------------------------------------

    import io

    model_buffer = io.BytesIO()
    joblib.dump(best_model, model_buffer)

    st.download_button(
        "⬇ Download Best Model",
        model_buffer.getvalue(),
        file_name="best_model.pkl"
    )
    # Detect Problem Type
    if y.nunique() <= 10:
        st.success("Detected: Classification")

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results[name] = acc

        result_df = pd.DataFrame(results.items(), columns=["Model", "Accuracy"])
        st.dataframe(result_df)

    else:
        st.success("Detected: Regression")

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor()
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = r2_score(y_test, preds)
            results[name] = score

        result_df = pd.DataFrame(results.items(), columns=["Model", "R2 Score"])
        st.dataframe(result_df)

# ============================================================
# PDF SUMMARY PAGE
# ============================================================

elif page == "PDF Summary":

    if pdf_text is None:
        st.warning("Upload PDF file.")
        st.stop()

    st.header("📄 PDF Analysis")

    st.text_area("Extracted Text Preview", pdf_text[:2000], height=300)

    st.markdown("### 📊 PDF Statistics")
    st.write("Total Words:", len(pdf_text.split()))
    st.write("Total Characters:", len(pdf_text))

    words = re.findall(r'\b\w+\b', pdf_text.lower())
    word_freq = Counter(words)
    common_words = word_freq.most_common(10)

    freq_df = pd.DataFrame(common_words, columns=["Word", "Frequency"])
    st.dataframe(freq_df)

    st.download_button(
        "⬇ Download Extracted Text",
        pdf_text,
        file_name="pdf_text.txt"
    )