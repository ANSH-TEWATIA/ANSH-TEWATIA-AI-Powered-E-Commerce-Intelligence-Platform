import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart E-Commerce AI",
    layout="wide"
)

st.title("🛒 Smart E-Commerce AI System")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():

    product_df = joblib.load("models/product_df.pkl")
    product_index = joblib.load("models/product_index.pkl")
    tfidf = joblib.load("models/tfidf.pkl")

    demand_model = joblib.load("models/demand_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")

    rfm = joblib.load("models/rfm.pkl")

    return (
        product_df,
        product_index,
        tfidf,
        demand_model,
        feature_columns,
        rfm
    )

(
    product_df,
    product_index,
    tfidf,
    demand_model,
    feature_columns,
    rfm
) = load_models()

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Product Recommendation",
        "Demand Prediction",
        "Customer Segmentation"
    ]
)

# ===================================================
# PRODUCT RECOMMENDATION
# ===================================================
if page == "Product Recommendation":

    st.header("🎯 Product Recommendation")

    product_name = st.text_input("Enter Product Name")

    def recommend_product(name, top_n=5):

        matches = product_df[
            product_df["ProductName"].str.contains(
                name,
                case=False,
                na=False
            )
        ]

        if matches.empty:
            return None

        idx = matches.index[0]

        tfidf_matrix = tfidf.transform(product_df["review"])

        similarity_scores = cosine_similarity(
            tfidf_matrix[idx],
            tfidf_matrix
        ).flatten()

        similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]

        return product_df.iloc[similar_indices][
            ["ProductName"]
        ]

    if st.button("Recommend"):

        recommendations = recommend_product(product_name)

        if recommendations is not None:
            st.dataframe(recommendations)
        else:
            st.error("Product not found")

# ===================================================
# DEMAND PREDICTION
# ===================================================
elif page == "Demand Prediction":

    st.header("📈 Demand Prediction")

    previous_sales = st.number_input(
        "Previous Sales",
        value=100
    )

    # Create all required columns with default 0
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=feature_columns
    )

    # Fill important columns
    input_data["lag_1"] = previous_sales
    input_data["lag_3"] = previous_sales
    input_data["lag_7"] = previous_sales
    input_data["lag_14"] = previous_sales

    input_data["rolling_mean"] = previous_sales
    input_data["rolling_mean_7"] = previous_sales
    input_data["rolling_mean_14"] = previous_sales

    input_data["year"] = 2024
    input_data["month"] = 6
    input_data["day"] = 1
    input_data["day_of_week"] = 1

    if st.button("Predict Demand"):

        prediction = demand_model.predict(input_data)

        st.success(
            f"Predicted Demand: {prediction[0]:.2f}"
        )

# ===================================================
# CUSTOMER SEGMENTATION
# ===================================================
elif page == "Customer Segmentation":

    st.header("👥 Customer Segments")

    segment_counts = rfm["SegmentLabel"].value_counts()

    st.bar_chart(segment_counts)

    fig = px.bar(
        x=segment_counts.index,
        y=segment_counts.values,
        labels={
            "x": "Segment",
            "y": "Customers"
        },
        title="Customer Segments"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(rfm.head())