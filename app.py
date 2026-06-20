import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import base64

def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{encoded}");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
}}
</style>
""", unsafe_allow_html=True)
    st.markdown("""
<style>
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    
    /* 👇 brightness control */
    background: rgba(0, 0, 0, 0.35);  /* increase/decrease this */

    z-index: 0;
}

/* Ensure content stays above overlay */
.stApp > div {
    position: relative;
    z-index: 1;
}
</style>
""", unsafe_allow_html=True)
# ================================
# LOAD + CLEAN DATA
# ================================

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv", encoding="latin1")

    df = df.dropna()
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]

    return df

df = load_data()

# ================================
# CREATE MATRIX
# ================================

customer_product_matrix = df.pivot_table(
    index='CustomerID',
    columns='StockCode',
    values='Quantity',
    aggfunc='sum',
    fill_value=0
)

customer_product_matrix.columns = customer_product_matrix.columns.astype(str)

# Convert to binary
customer_product_matrix = (customer_product_matrix > 0).astype(int)

# ================================
# PRODUCT SIMILARITY
# ================================

product_similarity = cosine_similarity(customer_product_matrix.T)

similarity_df = pd.DataFrame(
    product_similarity,
    index=customer_product_matrix.columns,
    columns=customer_product_matrix.columns
)

# ================================
# PRODUCT NAME MAPPING
# ================================

product_mapping = df[['StockCode', 'Description']].drop_duplicates()
product_mapping['StockCode'] = product_mapping['StockCode'].astype(str)

product_dict = dict(zip(product_mapping['StockCode'], product_mapping['Description']))

# ================================
# RECOMMENDATION FUNCTION
# ================================

def recommend_products(product, top_n=5):
    similar_products = similarity_df[product].sort_values(ascending=False)[1:top_n+1]
    
    results = []
    for prod, score in similar_products.items():
        name = product_dict.get(prod, "Unknown Product")
        results.append((name, score))
    
    return results

# ================================
# STREAMLIT UI
# ================================

st.set_page_config(page_title="🛒 Product Recommendation System", layout="wide")
# ================================
# NAVIGATION
# ================================
page = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Home", "🎯 Recommend", "📊 Insights"]
)
# ================================
# HOME PAGE (Temporary)
# ================================
if page == "🏠 Home":

    # Set background image
    set_bg("homepage.png")

    # Title + subtitle on top of image
    st.markdown("""
    <h1 style='text-align: center; color: white; margin-top: 50px; text-shadow: 2px 2px 10px black;'>
    🛒 Smart Product Recommendation System
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h4 style='text-align: center; color: white; margin-bottom: 40px; text-shadow: 2px 2px 8px black;'>
    Discover what customers love together 🚀
    </h4>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Feature cards (center)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style='background-color: rgba(255,255,255,0.9); padding:20px; border-radius:10px; text-align:center;'>
            <h3>🎯 Smart Recommendations</h3>
            <p>Get AI-powered product suggestions instantly</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background-color: rgba(255,255,255,0.85); padding:20px; border-radius:10px; text-align:center;'>
            <h3>📊 Data Insights</h3>
            <p>Understand trends and customer behavior</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='background-color: rgba(255,255,255,0.85); padding:20px; border-radius:10px; text-align:center;'>
            <h3>⚡ Real-Time Results</h3>
            <p>Fast and interactive experience</p>
        </div>
        """, unsafe_allow_html=True)


# ================================
# RECOMMEND PAGE (your current app)
# ================================
# ================================
# 🎯 RECOMMEND PAGE
# ================================

elif page == "🎯 Recommend":

    # 🌐 Background
    set_bg("recommend.png")

    # 🎯 Header
    st.markdown("""
    <h1 style='text-align: center; color: white; text-shadow: 2px 2px 10px black;'>
    🎯 Product Recommendation Engine
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h4 style='text-align: center; color: white; margin-bottom: 30px; text-shadow: 2px 2px 8px black;'>
    Find products customers love together 💡
    </h4>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
<style>

/* Remove entire selectbox container background */
div[data-testid="stSelectbox"] {
    background-color: transparent !important;
}

/* Remove inner wrapper (this is the strip culprit) */
div[data-testid="stSelectbox"] > div {
    background: none !important;
    padding: 0px !important;
    border: none !important;
}

/* Style actual dropdown box */
div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.75) !important;
    border-radius: 8px !important;
    border: none !important;
}

/* Remove any shadow */
div[data-baseweb="select"] {
    box-shadow: none !important;
}

</style>
""", unsafe_allow_html=True)
    # 📦 Input Box (Styled)
    st.markdown("""
    <div style='background-color: rgba(255,255,255,0.75); padding:25px; border-radius:10px;'>
    """, unsafe_allow_html=True)

    product_list = list(product_dict.values())
    st.markdown("""
<style>

/* Target Streamlit selectbox label specifically */
div[data-baseweb="select"] label {
    color: white !important;
    font-weight: 500;
}

/* Also ensure all labels are white (backup) */
label, .stSelectbox label {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)
    selected_product_name = st.selectbox("🔍 Select a Product", product_list)

    st.markdown("</div>", unsafe_allow_html=True)
# 🔁 Reverse Mapping
    reverse_product_dict = {v: k for k, v in product_dict.items()}
    selected_product_code = reverse_product_dict[selected_product_name]

    st.markdown("<br>", unsafe_allow_html=True)

    # 🎯 Button Styling
    st.markdown("""
    <style>
    div.stButton > button {
        background-color: #4FC3F7;
        color: white;
        border-radius: 10px;
        height: 45px;
        width: 250px;
        font-size: 16px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #29B6F6;
    }
    </style>
    """, unsafe_allow_html=True)

    # 🎯 Center Button
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        recommend_btn = st.button("🎯 Recommend Products", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 🛍️ OUTPUT
    if recommend_btn:

        recommendations = recommend_products(selected_product_code)

        st.markdown("""
        <h2 style='color:white; text-align:center; text-shadow: 2px 2px 8px black;'>
        🛍️ Recommended Products
        </h2>
        """, unsafe_allow_html=True)

        for name, score in recommendations:
            st.markdown(f"""
            <div style="
                background-color: rgba(255,255,255,0.9);
                padding:15px;
                margin:10px 0;
                border-radius:10px;
                box-shadow:0 2px 5px rgba(0,0,0,0.2);
            ">
                <h4 style="color:#333;">🛒 {name}</h4>
                <p style="color:#555;">Similarity Score: {score:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

# ================================
# INSIGHTS PAGE (Temporary)
# ================================
# ================================
# 📊 INSIGHTS PAGE
# ================================

elif page == "📊 Insights":

    # 🌐 Background
    set_bg("insights.png")   

    # 📊 Header
    st.markdown("""
    <h1 style='text-align: center; color: white; text-shadow: 2px 2px 10px black;'>
    📊 Product Insights Dashboard
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h4 style='text-align: center; color: white; margin-bottom: 30px; text-shadow: 2px 2px 8px black;'>
    Understand trends and customer behavior 📈
    </h4>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================================
    # 🛒 TOP PRODUCTS
    # ================================

    st.markdown("""
    <h2 style='color:white; text-shadow: 2px 2px 8px black;'>
    🔥 Top Selling Products
    </h2>
    """, unsafe_allow_html=True)

    top_products = df['Description'].value_counts().head(5)

    for product, count in top_products.items():
        st.markdown(f"""
        <div style="
            background-color: rgba(255,255,255,0.85);
            padding:15px;
            margin:10px 0;
            border-radius:10px;
        ">
            <h4 style="color:#333;">🛍️ {product}</h4>
            <p style="color:#555;">Purchased {count} times</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================================
    # 📊 PRODUCT DEMAND INSIGHTS
    # ================================

    st.markdown("""
    <h2 style='color:white; text-shadow: 2px 2px 8px black;'>
    📈 Demand Insights
    </h2>
    """, unsafe_allow_html=True)

    avg_quantity = df['Quantity'].mean()
    avg_price = df['UnitPrice'].mean()

    st.markdown(f"""
    <div style="
        background-color: rgba(255,255,255,0.85);
        padding:20px;
        border-radius:10px;
    ">
        <p style="color:#333;">📦 Average Quantity Purchased: <b>{avg_quantity:.2f}</b></p>
        <p style="color:#333;">💰 Average Product Price: <b>£{avg_price:.2f}</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================================
    # 💡 BUSINESS INSIGHTS
    # ================================

    st.markdown("""
    <h2 style='color:white; text-shadow: 2px 2px 8px black;'>
    💡 Business Insights
    </h2>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background-color: rgba(255,255,255,0.85);
        padding:20px;
        border-radius:10px;
    ">
        <ul style="color:#333;">
            <li>🔥 High demand products drive repeat purchases</li>
            <li>📊 Customers tend to buy similar product groups</li>
            <li>🛒 Recommendation systems can increase sales</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
