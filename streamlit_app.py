from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from traffic_sign_pipeline import (
    get_model_path,
    get_reference_image_path,
    load_inference_bundle,
    load_trained_model,
    load_training_metadata,
    predict_image,
    top_predictions,
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(214, 94, 52, 0.14), transparent 26%),
                radial-gradient(circle at top right, rgba(25, 109, 136, 0.10), transparent 24%),
                linear-gradient(180deg, #f7f1e5 0%, #efe6d7 100%);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 2rem;
        }

        .hero-card {
            background: rgba(255, 251, 245, 0.84);
            border: 1px solid rgba(23, 32, 38, 0.08);
            border-radius: 28px;
            padding: 1.45rem 1.55rem;
            box-shadow: 0 18px 40px rgba(23, 32, 38, 0.08);
            backdrop-filter: blur(12px);
        }

        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.75rem;
            color: #8d5b44;
            margin-bottom: 0.7rem;
            font-weight: 700;
        }

        .hero-title {
            font-size: 2.4rem;
            line-height: 1.05;
            color: #172026;
            font-weight: 700;
            margin: 0;
        }

        .hero-copy {
            margin-top: 0.9rem;
            color: #4d5457;
            font-size: 1rem;
            line-height: 1.55;
        }

        .status-pill {
            display: inline-block;
            margin-top: 1rem;
            padding: 0.5rem 0.8rem;
            border-radius: 999px;
            font-size: 0.9rem;
            font-weight: 700;
            background: rgba(24, 135, 87, 0.12);
            color: #166b47;
            border: 1px solid rgba(24, 135, 87, 0.18);
        }

        .control-card {
            background: rgba(255, 251, 245, 0.78);
            border: 1px solid rgba(23, 32, 38, 0.08);
            border-radius: 24px;
            padding: 1.1rem 1.2rem 0.8rem 1.2rem;
            box-shadow: 0 16px 32px rgba(23, 32, 38, 0.06);
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #172026;
            margin-bottom: 0.1rem;
        }

        .section-copy {
            color: #596164;
            font-size: 0.95rem;
            margin-bottom: 0.8rem;
        }

        .soft-card {
            background: rgba(255, 251, 245, 0.74);
            border: 1px solid rgba(23, 32, 38, 0.08);
            border-radius: 24px;
            padding: 1rem 0.95rem 1.05rem 0.95rem;
            box-shadow: 0 14px 32px rgba(23, 32, 38, 0.06);
            height: 100%;
            min-height: 170px;
        }

        .metric-label {
            display: block;
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: #8d5b44;
            font-weight: 700;
        }

        .metric-value {
            display: block;
            margin-top: 0.35rem;
            font-size: clamp(1.55rem, 1.8vw, 2.35rem);
            line-height: 1.05;
            color: #172026;
            font-weight: 700;
            letter-spacing: -0.03em;
            white-space: nowrap;
        }

        .metric-note {
            display: block;
            margin-top: 0.6rem;
            font-size: 0.88rem;
            color: #596164;
        }

        .result-banner {
            border-radius: 20px;
            padding: 0.95rem 1rem;
            font-weight: 700;
            margin: 1rem 0 0.2rem 0;
        }

        .result-ok {
            background: rgba(198, 233, 206, 0.72);
            color: #185d33;
        }

        .result-ko {
            background: rgba(245, 213, 205, 0.78);
            color: #8b3427;
        }

        .top-card {
            background: rgba(255, 251, 245, 0.72);
            border: 1px solid rgba(23, 32, 38, 0.08);
            border-radius: 20px;
            padding: 0.95rem 1rem;
            min-height: 126px;
            box-shadow: 0 12px 24px rgba(23, 32, 38, 0.05);
        }

        .top-rank {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: #8d5b44;
            font-weight: 700;
        }

        .top-class {
            margin-top: 0.45rem;
            font-size: 2rem;
            line-height: 1;
            color: #172026;
            font-weight: 700;
        }

        .top-confidence {
            margin-top: 0.5rem;
            color: #596164;
            font-size: 0.95rem;
        }

        [data-testid="stImage"] img {
            border-radius: 22px;
        }

        [data-testid="stExpander"] {
            background: rgba(255, 251, 245, 0.58);
            border-radius: 18px;
            border: 1px solid rgba(23, 32, 38, 0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_bundle_cached():
    return load_inference_bundle()


@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str, modified_at: float):
    return load_trained_model(model_path=model_path)


def format_percentage(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def render_metric_card(title: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="soft-card">
            <span class="metric-label">{title}</span>
            <span class="metric-value">{value}</span>
            <span class="metric-note">{note}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_prediction(rank: int, class_id: int, confidence: float) -> None:
    st.markdown(
        f"""
        <div class="top-card">
            <div class="top-rank">Top {rank}</div>
            <div class="top-class">{class_id}</div>
            <div class="top-confidence">Confiance : {confidence * 100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_reference_image(meta_df: pd.DataFrame, class_id: int, label: str) -> None:
    reference_path = get_reference_image_path(meta_df, class_id)
    if reference_path and reference_path.exists():
        st.image(str(reference_path), caption=label, use_container_width=True)
    else:
        st.info(f"Aucune image de reference disponible pour la classe {class_id}.")


def build_filtered_options(test_df: pd.DataFrame, query: str) -> list[str]:
    if not query.strip():
        return test_df["Path"].tolist()

    lowered_query = query.strip().lower()
    filtered_df = test_df[test_df["Path"].str.lower().str.contains(lowered_query, na=False)]
    if filtered_df.empty:
        return test_df["Path"].tolist()
    return filtered_df["Path"].tolist()


def pick_random_test_image(all_paths: list[str]) -> None:
    st.session_state.search_query = ""
    st.session_state.selected_test_path = random.choice(all_paths)


def main() -> None:
    st.set_page_config(
        page_title="Projet 2 - GTSRB",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()

    try:
        inference_bundle = load_bundle_cached()
    except Exception as exc:
        st.error(f"Impossible de charger le dataset : {exc}")
        st.stop()

    model_path = get_model_path()
    metadata = load_training_metadata()

    if not model_path.exists():
        st.error(
            "Le modele exporte est absent. Le deploiement doit inclure "
            "`artifacts/traffic_sign_cnn.keras`."
        )
        st.stop()

    try:
        model = load_model_cached(str(model_path), model_path.stat().st_mtime)
    except Exception as exc:
        st.error(f"Impossible de charger le modele sauvegarde : {exc}")
        st.stop()

    test_df = inference_bundle.test_df.reset_index(drop=True)
    meta_df = inference_bundle.meta_df
    all_paths = test_df["Path"].tolist()
    path_to_index = {path: index for index, path in enumerate(all_paths)}

    if "selected_test_path" not in st.session_state:
        st.session_state.selected_test_path = all_paths[0]
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Extension V</div>
            <h1 class="hero-title">Tester le modele sur les images du jeu de test</h1>
            <div class="hero-copy">
                Le modele sauvegarde est charge automatiquement au demarrage.
                Choisissez un fichier du dossier <strong>GTSRB/Test</strong>, lancez la prediction
                et comparez le resultat a la classe reelle.
            </div>
            <div class="status-pill">Modele charge automatiquement depuis artifacts/traffic_sign_cnn.keras</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="control-card">
            <div class="section-title">Choisir une image du test</div>
            <div class="section-copy">
                Le champ de recherche permet de retrouver rapidement un fichier comme
                00012.png ou 10532.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_columns = st.columns([2.3, 1.1, 0.9], gap="medium")
    with control_columns[0]:
        st.text_input(
            "Recherche",
            key="search_query",
            placeholder="Exemple : 00012.png ou 10532",
            label_visibility="collapsed",
        )

    filtered_paths = build_filtered_options(test_df, st.session_state.search_query)
    if st.session_state.selected_test_path not in filtered_paths:
        st.session_state.selected_test_path = filtered_paths[0]

    with control_columns[1]:
        st.selectbox(
            "Fichier du test",
            options=filtered_paths,
            key="selected_test_path",
            label_visibility="collapsed",
        )

    with control_columns[2]:
        st.button(
            "Image aleatoire",
            use_container_width=True,
            on_click=pick_random_test_image,
            args=(all_paths,),
        )

    selected_row = test_df.iloc[path_to_index[st.session_state.selected_test_path]]
    image_path = Path(selected_row["full_path"])
    real_class = int(selected_row["ClassId"])

    if not image_path.exists():
        st.error(f"Image introuvable : {image_path}")
        st.stop()

    with Image.open(image_path) as raw_image:
        selected_image = raw_image.convert("RGB")

    prediction = predict_image(model, image_path)
    top_3 = top_predictions(prediction.probabilities, top_k=3)
    is_correct = prediction.predicted_class == real_class

    image_column, results_column = st.columns([1.25, 0.95], gap="large")

    with image_column:
        st.image(selected_image, caption=selected_row["Path"], use_container_width=True)
        st.caption(
            f"Image selectionnee : `{selected_row['Path']}` | "
            f"Taille originale : `{selected_image.width} x {selected_image.height}`"
        )

    with results_column:
        metrics_row = st.columns([0.9, 0.9, 1.25], gap="small")
        with metrics_row[0]:
            render_metric_card("Classe reelle", str(real_class), "Etiquette du jeu de test")
        with metrics_row[1]:
            render_metric_card("Classe predite", str(prediction.predicted_class), "Sortie du modele")
        with metrics_row[2]:
            render_metric_card("Confiance", format_percentage(prediction.confidence), "Probabilite max")

        banner_class = "result-ok" if is_correct else "result-ko"
        banner_text = (
            "Prediction correcte : la classe predite correspond a la classe reelle."
            if is_correct
            else "Prediction incorrecte : la classe predite ne correspond pas a la classe reelle."
        )
        st.markdown(
            f'<div class="result-banner {banner_class}">{banner_text}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Top 3 predictions")
        top_columns = st.columns(3, gap="medium")
        for position, item in enumerate(top_3, start=1):
            with top_columns[position - 1]:
                render_top_prediction(position, int(item["ClassId"]), float(item["Confiance"]))

    with st.expander("Voir les panneaux de reference", expanded=False):
        ref_columns = st.columns(2, gap="large")
        with ref_columns[0]:
            st.markdown("**Classe reelle**")
            display_reference_image(meta_df, real_class, f"Reference classe {real_class}")
        with ref_columns[1]:
            st.markdown("**Classe predite**")
            display_reference_image(
                meta_df,
                prediction.predicted_class,
                f"Reference classe {prediction.predicted_class}",
            )

    with st.expander("Details du modele", expanded=False):
        if metadata:
            details_columns = st.columns(3, gap="medium")
            with details_columns[0]:
                render_metric_card(
                    "Accuracy test",
                    format_percentage(metadata.get("test_accuracy")),
                    "Performance sur Test.csv",
                )
            with details_columns[1]:
                render_metric_card(
                    "Accuracy validation",
                    format_percentage(metadata.get("validation_accuracy")),
                    "Performance sur le split validation",
                )
            with details_columns[2]:
                model_size_mb = model_path.stat().st_size / (1024 * 1024)
                render_metric_card(
                    "Modele",
                    f"{model_size_mb:.2f} MB",
                    "Fichier embarque dans artifacts",
                )
        else:
            st.info("Aucune metadonnee supplementaire disponible.")


if __name__ == "__main__":
    main()
