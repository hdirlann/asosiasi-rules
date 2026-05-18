import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules

# =====================================
# CONFIG PAGE
# =====================================
st.set_page_config(
    page_title="Association Rule Mining",
    layout="wide"
)

# =====================================
# TITLE
# =====================================
st.title("Association Rule Mining - Algoritma Apriori")
st.write("Analisis pola pembelian pelanggan menggunakan dataset Online Retail")

# =====================================
# LOAD DATASET
# =====================================
file_path = "online_retail_II.xlsx"


@st.cache_data
def load_data():
    # Ambil sebagian data agar lebih cepat
    df = pd.read_excel(file_path, nrows=5000)
    return df


try:
    # =====================================
    # READ DATA
    # =====================================
    df = load_data()

    st.subheader("Dataset Awal")
    st.dataframe(df.head())

    # =====================================
    # DATA CLEANING
    # =====================================
    st.subheader("Data Cleaning")

    # Hapus data kosong
    df = df.dropna(subset=["Description"])

    # Hapus quantity negatif
    df = df[df["Quantity"] > 0]

    st.write("Jumlah data setelah cleaning:", df.shape)

    # =====================================
    # BASKET TRANSACTION
    # =====================================
    st.subheader("Basket Transaction")

    basket = (
        df.groupby(["Invoice", "Description"])["Quantity"]
        .sum()
        .unstack()
        .fillna(0)
    )

    # Hapus produk yang terlalu jarang muncul
    basket = basket.loc[:, (basket.sum(axis=0) > 10)]

    # Ubah menjadi biner
    basket = (basket > 0).astype(int)

    st.dataframe(basket.head())

    # =====================================
    # SIDEBAR PARAMETER
    # =====================================
    st.sidebar.header("Parameter Apriori")

    min_support = st.sidebar.slider(
        "Minimum Support",
        min_value=0.01,
        max_value=1.0,
        value=0.02,
        step=0.01
    )

    min_confidence = st.sidebar.slider(
        "Minimum Confidence",
        min_value=0.01,
        max_value=1.0,
        value=0.30,
        step=0.01
    )

    # =====================================
    # APRIORI
    # =====================================
    st.subheader("Frequent Itemsets")

    frequent_itemsets = apriori(
        basket,
        min_support=min_support,
        use_colnames=True
    )

    st.write("Jumlah Frequent Itemsets:", len(frequent_itemsets))
    st.dataframe(frequent_itemsets.head(20))

    # =====================================
    # ASSOCIATION RULES
    # =====================================
    st.subheader("Association Rules")

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    # Urutkan berdasarkan lift tertinggi
    rules = rules.sort_values(by="lift", ascending=False)

    # Ambil kolom penting
    result = rules[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]
    ]

    st.write("Jumlah Rules:", len(result))
    st.dataframe(result.head(20))

    # =====================================
    # KESIMPULAN
    # =====================================
    st.subheader("Kesimpulan")

    if len(result) > 0:
        top_rule = result.iloc[0]

        st.success(
            f"Produk {list(top_rule['antecedents'])} "
            f"sering dibeli bersama dengan "
            f"{list(top_rule['consequents'])} "
            f"dengan confidence "
            f"{round(top_rule['confidence'] * 100, 2)}%"
        )

    else:
        st.warning("Tidak ditemukan association rules.")

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")