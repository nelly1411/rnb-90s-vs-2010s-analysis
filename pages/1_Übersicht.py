import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="Dataset-Übersicht",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Daten laden
@st.cache_data
def load_data():
    return pd.read_csv("data/RnB_merged_clean.csv")


df = load_data()


# Titel & Intro
st.title("Dataset-Übersicht 🏠")
st.write(
    "Überblick über den bereinigten R&B-Datensatz "
    "für den Vergleich der 1990er- und 2010er-Jahre."
)


# Kurzinfo / KPIs
st.subheader("Kurzinfo")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Songs", len(df))
col2.metric("Spalten", df.shape[1])
col3.metric(
    "Eras",
    df["era"].nunique() if "era" in df.columns else "—"
)
col4.metric(
    "Artists",
    df["artist_name"].nunique()
    if "artist_name" in df.columns else "—"
)


total_missing = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())

col5, col6 = st.columns(2)

col5.metric("Missing Values (gesamt)", total_missing)
col6.metric("Duplikate (Rows)", duplicate_rows)


st.markdown("---")


# Era-Verteilung
st.subheader("Era-Verteilung")

if "era" in df.columns:
    era_counts = (
        df["era"]
        .value_counts(dropna=False)
        .rename("Songs")
    )

    st.dataframe(
        era_counts,
        use_container_width=True
    )
else:
    st.info("Spalte 'era' nicht gefunden.")


st.markdown("---")


# Beispieldaten
st.subheader("Beispieldaten")

n = st.slider(
    "Wie viele Zeilen anzeigen?",
    min_value=5,
    max_value=100,
    value=20,
    step=5
)

st.dataframe(
    df.head(n),
    use_container_width=True
)


st.markdown("---")


# Dataset Info
st.subheader("Dataset Info")

info_df = pd.DataFrame({
    "Spalte": df.columns,
    "Datentyp": [str(dtype) for dtype in df.dtypes],
    "Missing": df.isna().sum().values
})

st.dataframe(
    info_df,
    use_container_width=True
)


st.markdown("---")


# Deskriptive Statistik
st.subheader("Deskriptive Statistik")

numeric_df = df.select_dtypes(include="number")

if not numeric_df.empty:
    st.dataframe(
        numeric_df.describe().T,
        use_container_width=True
    )
else:
    st.info(
        "Keine numerischen Spalten gefunden → "
        "describe() nicht verfügbar."
    )


# Footer
st.markdown("---")
st.caption("Datenquelle: Spotify Audio Features (Kaggle)")
