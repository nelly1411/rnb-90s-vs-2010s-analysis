import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="Dataset-Übersicht",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_data():
    return pd.read_csv("data/SpotifyFeatures.csv")

df = load_data()

@st.cache_data
def load_data2():
    return pd.read_csv("data/RnB_merged_clean.csv")

df2 = load_data2()

# Title & Intro
st.title("Dataset-Übersicht 🏠")
st.write("Überblick über den RnB-Datensatz und zentrale Kennzahlen.")

# Kurzinfo / KPIs
st.subheader("Kurzinfo")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Songs", len(df))
col2.metric("Spalten", df.shape[1])
col3.metric("Eras", df["era"].nunique() if "era" in df.columns else "—")
col4.metric("Artists", df["artist_name"].nunique() if "artist_name" in df.columns else "—")

total_missing = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())

col5, col6 = st.columns(2)
col5.metric("Missing Values (gesamt)", total_missing)
col6.metric("Duplikate (Rows)", duplicate_rows)


# Beispiel-Rohdaten
st.subheader("Beispiel-Rohdaten")

n = st.slider(
    "Wie viele Zeilen anzeigen?",
    min_value=5,
    max_value=100,
    value=20,
    step=5
)

st.dataframe(df.head(n), use_container_width=True)

st.markdown("---")


# Era-Verteilung
st.subheader("Era-Verteilung")

if "era" in df2.columns:
    era_counts = df2["era"].value_counts(dropna=False).rename("count")
    st.dataframe(era_counts, use_container_width=True)
else:
    st.info("Spalte 'era' nicht gefunden.")

# Beispiel-CleanedRohdaten
st.subheader("Beispiel-Cleaned Rohdaten (nach Bereinigung)")

n_clean = st.slider(
    "Wie viele Zeilen der bereinigten Daten anzeigen?",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
    key="clean_slider"
)

st.dataframe(df2.head(n_clean), use_container_width=True)

st.markdown("---")

# Dataset Info
st.subheader("Dataset Info")

info_df = pd.DataFrame({
    "Spalte": df.columns,
    "Datentyp": [str(dtype) for dtype in df.dtypes],
    "Missing": df.isna().sum().values
})

st.dataframe(info_df, use_container_width=True)

st.markdown("---")


# Deskriptive Statistik
st.subheader("Deskriptive Statistik")

numeric_df = df.select_dtypes(include="number")

if not numeric_df.empty:
    st.dataframe(numeric_df.describe().T, use_container_width=True)
else:
    st.info("Keine numerischen Spalten gefunden → describe() nicht verfügbar.")

# Footer
st.markdown("---")
st.caption("Datenquelle: Spotify Audio Features (Kaggle)")
