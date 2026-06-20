import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# ================================
# STEP 1: LOAD DATA
# ================================

def load_data():
    df = pd.read_csv("data.csv",encoding="latin1")
    return df


# ================================
# STEP 2: UNDERSTAND DATA (EDA)
# ================================

def explore_data(df):
    print("\n📊 Dataset Shape:")
    print(df.shape) # (541909, 8)

    print("\n📌 Columns:")
    print(df.columns.tolist()) # (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country)

    print("\n🧠 Data Types:")
    print(df.dtypes) # (str, str, str, int64, str, float64, float64, str)

    print("\n🔍 First 5 Rows:")
    print(df.head())

    print("\n⚠️ Missing Values:")
    print(df.isnull().sum()) #
     # missing values in Description(1454) and CustomerID(135080)

    print("\n📈 Summary Statistics:")
    print(df.describe()) # prints the statistical measures for the numerical features


# ================================
# MAIN EXECUTION
# ================================

if __name__ == "__main__":
    df = load_data()
    explore_data(df)
 # Remove missing values
df = df.dropna()

# Remove negative or zero quantity
df = df[df['Quantity'] > 0]

# Remove zero or negative price
df = df[df['UnitPrice'] > 0]

print("Cleaned shape:", df.shape) # (397884, 8)
# Create Customer-Product matrix
customer_product_matrix = df.pivot_table(
    index='CustomerID',
    columns='StockCode',
    values='Quantity',
    aggfunc='sum',
    fill_value=0
)
customer_product_matrix.columns = customer_product_matrix.columns.astype(str)
# Rows -> customers , column-> products, values-> purchase quantity
print("Matrix shape:", customer_product_matrix.shape) # Matrix shape : (4338, 3665)
print(type(customer_product_matrix))
# convert to binary 
# we only care if user bought or not
customer_product_matrix = (customer_product_matrix > 0).astype(int)
# similarity calculation - finds similar customers
product_similarity = cosine_similarity(customer_product_matrix.T)
print("Similarity matrix shape:", product_similarity.shape) # (4338,4338)
# convert to dataframe
similarity_df = pd.DataFrame(
    product_similarity,
    index=customer_product_matrix.columns,
    columns=customer_product_matrix.columns
)
similarity_df.columns = similarity_df.columns.astype(str)
similarity_df.index = similarity_df.index.astype(str)
# Create product mapping (StockCode → Description)
product_mapping = df[['StockCode', 'Description']].drop_duplicates()
product_mapping['StockCode'] = product_mapping['StockCode'].astype(str)

product_dict = dict(zip(product_mapping['StockCode'], product_mapping['Description']))
# recommendation function
def recommend_products(product, top_n=5):
    
    similar_products = similarity_df[product].sort_values(ascending=False)[1:top_n+1]
    
    print("\n🛒 Recommended Products:\n")
    
    for prod, score in similar_products.items():
        name = product_dict.get(prod, "Unknown Product")
        print(f"{name} (Score: {score:.2f})")
# ---------------- TEST OUTPUT ----------------
product = customer_product_matrix.columns[1]

print("\nTesting for product:", product)
print("Product Name:", product_dict.get(product))

recommend_products(product)
