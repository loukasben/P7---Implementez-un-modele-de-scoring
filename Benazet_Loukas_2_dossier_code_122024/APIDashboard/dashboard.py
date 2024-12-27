import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import os
import plotly.graph_objects as go
import requests
import streamlit as st

# Obtenir le répertoire courant du script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construire les chemins vers les fichiers CSV dans le dossier Prediction
path_df_train = os.path.join(current_directory, "../Prediction/df_train.csv")
path_definition_features_df = os.path.join(current_directory, "../Prediction/definition_features.csv")

df_train = pd.read_csv(path_df_train)
definition_features_df = pd.read_csv(path_definition_features_df)

st.set_page_config(layout="wide")

def get_title_font_size(height):
    base_size = 12  # une taille de police de base
    scale_factor = height / 600.0  # supposons que 600 est la hauteur par défaut
    return base_size * scale_factor

def create_gauge(score, tolerance):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",  # Mode avec jauge, nombre et delta
        value=score,  # Valeur du score
        delta={'reference': tolerance, 'position': "top", 'increasing': {'color': "red"}},  # Référence pour le delta
        gauge={
            'axis': {
                'range': [0, 100], 
                'tickwidth': 2,  # Augmenter la largeur des ticks
                'tickcolor': "black",  # Couleur des ticks
                'tickfont': {'size': 14}  # Taille des labels des ticks
            },
            'bar': {'color': "black"},  # Curseur principal
            'steps': [
                {'range': [0, tolerance], 'color': "green"},  # Zone sécurisée
                {'range': [tolerance, 100], 'color': "red"}  # Zone à risque
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},  # Ligne du seuil
                'thickness': 1,
                'value': tolerance
            }
        },
        domain={'x': [0, 1], 'y': [0, 1]}  # Occupation totale du conteneur
    ))

    fig.update_layout(
        height=300,  # Ajustez la hauteur pour correspondre à votre mise en page
        margin={'t': 10, 'b': 10, 'l': 10, 'r': 10}, 
        font=dict(color="black", size=14)  # Taille et couleur de la police
    )
    return fig

def generate_figure(df, title_text, x_anchor, yaxis_categoryorder, yaxis_side):
    fig = go.Figure(data=[go.Bar(y=df["Feature"], x=df["SHAP Value"], orientation="h")])
    annotations = generate_annotations(df, x_anchor)

    title_font_size = get_title_font_size(600)
    fig.update_layout(
        annotations=annotations,
        title_text=title_text,
        title_x=0.25,
        title_y=0.88,
        title_font=dict(size=title_font_size),
        yaxis=dict(
            categoryorder=yaxis_categoryorder, side=yaxis_side, tickfont=dict(size=14)
        ),
        height=600,
    )
    fig.update_xaxes(title_text="Impact des fonctionnalités")
    return fig

def generate_figure_with_gradient(df, title_text, x_anchor, yaxis_categoryorder, yaxis_side, color_map, invert=False):
    # Check
    if df.empty or "Feature" not in df.columns or "SHAP Value" not in df.columns:
        st.error("Erreur : Les données pour le graphique sont invalides ou manquantes.")
        return go.Figure()
    
    # Normalisation des valeurs
    norm = mcolors.Normalize(vmin=df["SHAP Value"].min(), vmax=df["SHAP Value"].max())
    cmap = cm.get_cmap(color_map)

    # Inverser le dégradé si nécessaire
    if invert:
        cmap = cm.get_cmap(color_map).reversed()

    # Générer les couleurs en fonction des SHAP values
    colors = [cmap(norm(val)) for val in df["SHAP Value"]]
    colors = [
        "rgba({},{},{},{})".format(int(r * 255), int(g * 255), int(b * 255), a)
        for r, g, b, a in colors
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                y=df["Feature"],
                x=df["SHAP Value"],
                orientation="h",
                marker_color=colors,  # Utiliser les couleurs générées
            )
        ]
    )

    annotations = generate_annotations(df, x_anchor)

    title_font_size = get_title_font_size(600)
    fig.update_layout(
        annotations=annotations,
        title_text=title_text,
        title_x=0.25,
        title_y=0.88,
        title_font=dict(size=title_font_size),
        yaxis=dict(
            categoryorder=yaxis_categoryorder, side=yaxis_side, tickfont=dict(size=14)
        ),
        height=600,
    )
    fig.update_xaxes(title_text="Impact des fonctionnalités")
    return fig

def generate_annotations(df, x_anchor):
    annotations = []
    for y_val, x_val, feat_val in zip(
        df["Feature"], df["SHAP Value"], df["Feature Value"]
    ):
        formatted_feat_val = (
            feat_val
            if pd.isna(feat_val)
            else (int(feat_val) if feat_val == int(feat_val) else feat_val)
        )
        annotations.append(
            dict(
                x=x_val,
                y=y_val,
                text=f"<b>{formatted_feat_val}</b>",
                showarrow=False,
                xanchor=x_anchor,
                yanchor="middle",
                font=dict(color="white"),
            )
        )
    return annotations

def compute_color(value):
    if 0 <= value < 48:
        return "green"
    elif 48 <= value <= 100:
        return "red"

def format_value(val):
    if pd.isna(val):
        return val
    if isinstance(val, (float, int)):
        if val == int(val):
            return int(val)
        return round(val, 2)
    return val

def find_closest_description(feature_name, definitions_df):
    for index, row in definitions_df.iterrows():
        if row["Row"] in feature_name:
            return row["Description"]
    return None

def plot_distribution(selected_feature, col):
    if selected_feature:

        if selected_feature not in df_train.columns:
            st.error(f"Erreur : La fonctionnalité '{selected_feature}' est absente du DataFrame.")
            return

        data = df_train[selected_feature].dropna()

        if data.empty:
            st.warning(f"Pas de données disponibles pour la fonctionnalité '{selected_feature}'.")
            return
        
        data = df_train[selected_feature]

        # Trouver la valeur de la fonctionnalité pour le client actuel :
        client_feature_value = feature_values[feature_names.index(selected_feature)]

        fig = go.Figure()

        # Vérifier si la fonctionnalité est catégorielle :
        unique_values = sorted(data.dropna().unique())
        if set(unique_values) <= {0, 1, 2, 3, 4, 5, 6, 7}:
            # Compter les occurrences de chaque valeur :
            counts = data.value_counts().sort_index()

            # Assurez-vous que les longueurs correspondent
            assert len(unique_values) == len(counts)

            # Modifier la déclaration de la liste de couleurs pour correspondre à la taille de unique_values
            colors = ["blue"] * len(unique_values)

            # Mettre à jour client_value
            client_value = (
                unique_values.index(client_feature_value)
                if client_feature_value in unique_values
                else None
            )

            # Mettre à jour la couleur correspondante si client_value n'est pas None
            if client_value is not None:
                colors[client_value] = "red"

            # Modifier le tracé pour utiliser unique_values
            fig.add_trace(go.Bar(x=unique_values, y=counts.values, marker_color=colors))

        else:
            # Calculer les bins pour le histogramme :
            hist_data, bins = np.histogram(data.dropna(), bins=20)

            # Trouver le bin pour client_feature_value :
            client_bin_index = np.digitize(client_feature_value, bins) - 1

            # Créer une liste de couleurs pour les bins :
            colors = ["blue"] * len(hist_data)
            if (
                0 <= client_bin_index < len(hist_data)
            ):  # Vérifiez que l'index est valide
                colors[client_bin_index] = "red"

            # Tracer la distribution pour les variables continues :
            fig.add_trace(
                go.Histogram(
                    x=data,
                    marker=dict(color=colors, opacity=0.7),
                    name="Distribution",
                    xbins=dict(start=bins[0], end=bins[-1], size=bins[1] - bins[0]),
                )
            )

            # Utiliser une échelle logarithmique si la distribution est fortement asymétrique :
            mean_val = np.mean(hist_data)
            std_val = np.std(hist_data)
            if std_val > 3 * mean_val:  # Ce seuil peut être ajusté selon vos besoins
                fig.update_layout(yaxis_type="log")

        height = 600  # Ajuster cette valeur selon la hauteur par défaut de la figure ou l'obtenir d'une autre manière.
        title_font_size = get_title_font_size(height)

        fig.update_layout(
            title_text=f"Distribution pour {selected_feature}",
            title_font=dict(size=title_font_size),  # Ajoutez cette ligne
            xaxis_title=selected_feature,
            yaxis_title="Nombre de clients",
            title_x=0.3,
        )

        col.plotly_chart(fig, use_container_width=True)

        # Afficher la définition de la feature choisi :
        description = find_closest_description(selected_feature, definition_features_df)
        if description:
            col.write(f"**Definition:** {description}")

# Une fonction pour récupérer les états stockés :
def get_state():
    if "state" not in st.session_state:
        st.session_state["state"] = {
            "data_received": False,
            "data": None,
            "last_sk_id_curr": None,  # Stocker le dernier ID soumis
        }
    return st.session_state["state"]

state = get_state()  # Initialise ou récupère l'état

st.markdown(
    "<h1 style='text-align: center; color: black;'>Estimation du risque de non-remboursement</h1>",
    unsafe_allow_html=True,
)
sk_id_curr = st.text_input(
    "Entrez le SK_ID_CURR:", on_change=lambda: state.update(run=True)
)
col1, col2 = st.columns([1, 20])

st.markdown(
    """
    <style>
        /* Style pour le bouton */
        button {
            width: 60px !important;
            white-space: nowrap !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if col1.button("Run") or state["data_received"]:
    # Avant de traiter l'appel API, vérifier si l'ID actuel est différent du dernier ID
    if state["last_sk_id_curr"] != sk_id_curr:
        state["data_received"] = False
        state["last_sk_id_curr"] = sk_id_curr  # Mettre à jour le dernier ID

    if not state["data_received"]:
        response = requests.post("https://p7-implementez-un-modele-de-scoring.onrender.com/predict", json={"SK_ID_CURR": int(sk_id_curr)})
        
        if response.status_code != 200:
            st.error(f"Erreur lors de l'appel à l'API: {response.status_code}")
            st.stop()

        state["data"] = response.json()
        state["data_received"] = True

    data = state["data"]

    st.write("Données retournées par l'API :")
    st.json(data)

    proba = data["probability"]
    feature_names = data["feature_names"]
    shap_values = data["shap_values"]
    feature_values = data["feature_values"]
    shap_values = [val[0] if isinstance(val, list) else val for val in shap_values]
    shap_df = pd.DataFrame(
        list(
            zip(
                feature_names,
                shap_values,
                [format_value(val) for val in feature_values],
            )
        ),
        columns=["Feature", "SHAP Value", "Feature Value"],
    )

    st.write("DataFrame SHAP Values :")
    st.dataframe(shap_df)

    if shap_df.empty:
        st.error("Erreur : Le DataFrame SHAP est vide.")
        st.stop()

    color = compute_color(proba)
    st.empty()
    col2.markdown(
        f"<p style='margin: 10px;'>La probabilité que ce client ne puisse pas rembourser son crédit est de <span style='color:{color}; font-weight:bold;'>{proba:.2f}%</span> (tolérance max: <strong>48%</strong>)</p>",
        unsafe_allow_html=True,
    )

    # Ajouter une ligne pour afficher la jauge
    gauge_fig = create_gauge(proba, 48)
    st.plotly_chart(gauge_fig, use_container_width=True)

    decision_message = (
        "Le prêt est accordé" if proba < 48 else "Le prêt n'est pas accordé"
    )
    st.markdown(
        f"<div style='text-align: center; color:{color}; font-size:30px; border:2px solid {color}; padding:10px;'>{decision_message}</div>",
        unsafe_allow_html=True,
    )

    # Ici, nous définissons top_positive_shap et top_negative_shap
    top_positive_shap = shap_df.sort_values(by="SHAP Value", ascending=False).head(10)
    top_negative_shap = shap_df.sort_values(by="SHAP Value").head(10)

    fig_positive = generate_figure_with_gradient(
        top_positive_shap,
        "Top 10 des fonctionnalités augmentant le risque de non-remboursement",
        "right",
        "total ascending",
        "left",
        "Reds" # Dégradé de rouges
    )
    fig_negative = generate_figure_with_gradient(
        top_negative_shap,
        "Top 10 des fonctionnalités réduisant le risque de non-remboursement",
        "left",
        "total descending",
        "right",
        "Greens", # Dégradé de verts
        invert=True  # Inversion des couleurs
    )

    # Créer une nouvelle ligne pour les graphiques
    col_chart1, col_chart2 = st.columns(2)
    col_chart1.plotly_chart(fig_positive, use_container_width=True)
    col_chart2.plotly_chart(fig_negative, use_container_width=True)

    # Créez des colonnes pour les listes déroulantes
    col1, col2 = st.columns(2)

    # Mettez la première liste déroulante dans col1
    with col1:
        if top_positive_shap.empty:
            st.warning("Aucune fonctionnalité augmentant le risque n'est disponible.")
            selected_feature_positive = None
        else:
            selected_feature_positive = st.selectbox(
                "Sélectionnez une fonctionnalité augmentant le risque",
                [""] + top_positive_shap["Feature"].tolist(),
                key="positive_selectbox"  # Clé unique
            )
        # Appelez `plot_distribution` uniquement si une valeur est sélectionnée
        if selected_feature_positive:
            plot_distribution(selected_feature_positive, col1)

    # Mettez la deuxième liste déroulante dans col2
    with col2:
        if top_negative_shap.empty:
            st.warning("Aucune fonctionnalité réduisant le risque n'est disponible.")
            selected_feature_negative = None
        else:
            selected_feature_negative = st.selectbox(
                "Sélectionnez une fonctionnalité réduisant le risque",
                [""] + top_negative_shap["Feature"].tolist(),
                key="negative_selectbox"  # Clé unique
            )
        # Appelez `plot_distribution` uniquement si une valeur est sélectionnée
        if selected_feature_negative:
            plot_distribution(selected_feature_negative, col2)
