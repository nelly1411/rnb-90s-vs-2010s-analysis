import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Visualisierung",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    return pd.read_csv("data/RnB_merged_clean.csv")
    



def main():
    st.title("Visualisierung 📊")
    st.write("Filtere nach Era (90s / 10s) und vergleiche Audio-Features.")

    df = load_data()

    # Sidebar: Filter
    st.sidebar.title("Filter ⚙️")

    eras = sorted(df["era"].dropna().unique()) if "era" in df.columns else []
    default_eras = [e for e in ["90s", "10s"] if e in eras] or eras

    selected_eras = st.sidebar.multiselect("Era:", options=eras, default=default_eras)

    # Fallback: wenn alles abgewählt
    if not selected_eras:
        st.sidebar.warning("Keine Era ausgewählt → zeige wieder alle.")
        selected_eras = eras

    filtered_df = df[df["era"].isin(selected_eras)].copy() if "era" in df.columns else df.copy()

    #Artist Suche
    if "artist_name" in filtered_df.columns:
        artist_query = st.sidebar.text_input("Artist suchen (optional):", "")
        if artist_query.strip():
            filtered_df = filtered_df[
                filtered_df["artist_name"].str.contains(artist_query, case=False, na=False)
            ]

    
    st.subheader("Gefilterte Daten")
    st.write(f"Zeige **{len(filtered_df)}** von **{len(df)}** Songs")

    col1, col2, col3 = st.columns(3)
    col1.metric("Songs", len(filtered_df))
    col2.metric("Artists", filtered_df["artist_name"].nunique() if "artist_name" in filtered_df.columns else "—")
    col3.metric("Ø Tempo (BPM)", f"{filtered_df['tempo'].mean():.1f}" if "tempo" in filtered_df.columns else "—")

    # Feature Auswahl
    st.subheader("Charts")
    feature = st.selectbox(
        "Feature wählen:",
        ["danceability", "energy", "acousticness", "loudness", "tempo", "speechiness"]
    )

    eras_selected = sorted(filtered_df["era"].dropna().unique()) if "era" in filtered_df.columns else []

    
    if set(["90s", "10s"]).issubset(set(eras_selected)):
        df_plot = filtered_df[filtered_df["era"].isin(["90s", "10s"])][["era", feature]].dropna()

        # Reihenfolge fixieren, damit Farben sicher stimmen
        df_plot["era"] = pd.Categorical(df_plot["era"], categories=["90s", "10s"], ordered=True)

        fig, ax = plt.subplots(figsize=(8, 5))

        df_plot.boxplot(
            column=feature,
            by="era",
            ax=ax,
            grid=True,
            patch_artist=True,              # <- wichtig für Füllfarben
            medianprops=dict(color="black", linewidth=2),
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
        )

        # Farben setzen: 90s = blau, 10s = soft-rot
        colors = ["#4A90E2", "#E57373"]
        for patch, color in zip(ax.patches, colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f"{feature} – 90s vs 10s (Boxplot)")
        ax.set_xlabel("Era")
        ax.set_ylabel(feature)
        plt.suptitle("")

        st.pyplot(fig)

    # --------------------------------------------------
    # FALL 2: nur 1 Era -> Histogramm
    # --------------------------------------------------
    elif len(eras_selected) == 1:
        era = eras_selected[0]
        df_one = filtered_df[filtered_df["era"] == era]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df_one[feature].dropna(), bins=20, edgecolor="black")
        ax.set_title(f"{feature} – {era} (Histogramm)")
        ax.set_xlabel(feature)
        ax.set_ylabel("Häufigkeit")
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

        if df_one[feature].dropna().shape[0] > 0:
            median = df_one[feature].median()
            q1 = df_one[feature].quantile(0.25)
            q3 = df_one[feature].quantile(0.75)
            iqr = q3 - q1
            st.caption(f"Median: {median:.3f} | IQR: {iqr:.3f} (Q1: {q1:.3f}, Q3: {q3:.3f})")

    else:
        st.info("Bitte mindestens eine Era auswählen (z.B. 90s oder 10s).")

    st.markdown("---")
    st.caption("Datenquelle: Spotify Audio Features (Kaggle)")


if __name__ == "__main__":
    main()