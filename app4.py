import os
import warnings
# Configuration pour éviter l'avertissement sur les cœurs physiques
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # Ajustez selon votre configuration
# Ignorer l'avertissement de dépréciation pour choropleth_mapbox
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*choropleth_mapbox.*")

import tempfile
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.platypus import PageBreak, ListFlowable, ListItem
from reportlab.lib.units import inch, cm
from io import BytesIO
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import base64
import kaleido
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import json

# Configuration de la page
st.set_page_config(
    page_title="Medical'IA - Analyse des Déserts Médicaux",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonctions utilitaires
@st.cache_data
def load_data():
    """Chargement des données"""
    try:
        df = pd.read_csv('medical_desert_data_with_coords.csv', low_memory=False)
        # Assurer que CODGEO est de type string
        df['CODGEO'] = df['CODGEO'].astype(str)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return pd.DataFrame()

@st.cache_data
def calculate_stats(df):
    """Calcul des statistiques générales"""
    stats = {
        "communes_count": len(df),
        "population_total": df["P16_POP"].sum(),
        "avg_apl": df["APL"].mean(),
        "weighted_avg_apl": (df["APL"] * df["P16_POP"]).sum() / df["P16_POP"].sum(),
        "median_apl": df["APL"].median(),
        "desert_count": len(df[df["APL"] < 2.5]),
        "desert_percent": (len(df[df["APL"] < 2.5]) / len(df)) * 100,
    }
    return stats

import json

@st.cache_data
def load_geojson(filepath):
    """Chargement d'un fichier GeoJSON"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier GeoJSON: {e}")
        return None

@st.cache_data
def create_choropleth_map(df_stats, geojson, level="departement", value_col="apl_pondere", title="Carte choroplèthe"):
    """Création d'une carte choroplèthe par territoire"""
    # Déterminer la clé d'identification dans le GeoJSON selon le niveau
    if level == "commune":
        # Nous désactivons complètement la détection des métropoles pour éviter les erreurs
        # metropoles = df_stats[df_stats.get('is_metropole', False) == True]
        
        id_key = "properties.code"
        location_col = "territoire"
        hover_data = ["nom_commune", "population", "desert_percent"]
        # Pour les communes, ajuster le zoom pour une meilleure visualisation
        zoom = 5.5
    elif level == "departement":
        id_key = "properties.code"
        location_col = "territoire"
        hover_data = ["population", "desert_percent"]
        zoom = 5
    else:  # region
        id_key = "properties.nom"
        location_col = "territoire"
        hover_data = ["population", "desert_percent"]
        zoom = 4.5
    
    # Créer la carte choroplèthe
    fig = px.choropleth_mapbox(
        df_stats, 
        geojson=geojson, 
        locations=location_col,
        featureidkey=id_key,
        color=value_col,
        color_continuous_scale="RdYlGn",
        range_color=[1, 5],
        mapbox_style="carto-positron",
        zoom=zoom,
        center={"lat": 46.603354, "lon": 1.888334},
        opacity=0.8,
        labels={value_col: 'APL pondéré'},
        hover_data=hover_data
    )
    
    # Adapter le template de hover selon le niveau
    if level == "commune":
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>APL: %{z:.2f}<br>Population: %{customdata[1]}<br>Désert médical: %{customdata[2]:.0f}%"
        )
        
        # Ajouter des marqueurs pour les métropoles s'il y en a - avec vérification des colonnes
        required_columns = ['nom_commune', 'latitude_mairie', 'longitude_mairie']
        if 'metropoles' in locals() and not metropoles.empty and all(col in metropoles.columns for col in required_columns):
            try:
                fig.add_trace(go.Scatter(
                    lat=metropoles['latitude_mairie'],
                    lon=metropoles['longitude_mairie'],
                    mode='markers+text',
                    marker=dict(size=10, color='black', symbol='circle'),
                    text=metropoles['nom_commune'],
                    textposition="top center",
                    name="Métropoles"
                ))
            except Exception as e:
                # Si une erreur se produit, on continue sans les marqueurs de métropoles
                pass
    
    fig.update_layout(
        title=title,
        margin={"r":0,"t":50,"l":0,"b":0},
        height=700
    )
    
    return fig

@st.cache_data
def calculate_territorial_apl(df, geo_level):
    """Calcul de l'APL pondéré par territoire (commune, département ou région)
    
    Args:
        df: DataFrame contenant les données
        geo_level: Niveau géographique ('commune', 'departement' ou 'region')
    
    Returns:
        DataFrame avec les statistiques par territoire
    """
    # Pour les communes, retourner directement les données
    if geo_level == "commune":
        result = []
        # Identifiez les métropoles (codes INSEE des grandes villes)
        metropoles = ['75056', '13055', '69123', '31555', '59350', '33063', '44109', '67482', '06088', '76540']  # Paris, Marseille, Lyon, Toulouse, Lille, Bordeaux, Nantes, Strasbourg, Nice, Rouen
        metropole_names = {'75056': 'Paris', '13055': 'Marseille', '69123': 'Lyon', '31555': 'Toulouse', 
                        '59350': 'Lille', '33063': 'Bordeaux', '44109': 'Nantes', '67482': 'Strasbourg', 
                        '06088': 'Nice', '76540': 'Rouen'}
        
        # S'assurer que les métropoles sont toujours incluses
        metros_included = set()
        
        for _, row in df.iterrows():
            is_metro = row['CODGEO'] in metropoles
            
            # Si c'est une métropole, marquer comme incluse
            if is_metro:
                metros_included.add(row['CODGEO'])
                
            result.append({
                'territoire': row['CODGEO'],
                'nom_commune': row['Communes'],
                'population': row['P16_POP'],
                'apl_pondere': row['APL'],
                'communes_count': 1,
                'desert_percent': 100 if row['APL'] < 2.5 else 0,
                'desert_count': 1 if row['APL'] < 2.5 else 0,
                'min_apl': row['APL'],
                'max_apl': row['APL'],
                'is_metropole': is_metro
            })
        
        # Vérifier si le résultat est vide avant de créer le DataFrame
        if result:
            result_df = pd.DataFrame(result).sort_values('apl_pondere')
        else:
            result_df = pd.DataFrame(columns=['territoire', 'nom_commune', 'population', 'apl_pondere', 
                                             'communes_count', 'desert_percent', 'desert_count', 
                                             'min_apl', 'max_apl', 'is_metropole'])
        return result_df
    
    # Créer les colonnes de département et région si nécessaire
    df_copy = df.copy()
    df_copy['departement'] = df_copy['CODGEO'].str[:2]
    
    # Pour les régions, utiliser une table de correspondance département-région
    # Correspondance simplifiée (à remplacer par une table plus complète si nécessaire)
    region_map = {
        '01': 'Auvergne-Rhône-Alpes', '03': 'Auvergne-Rhône-Alpes', '07': 'Auvergne-Rhône-Alpes', 
        '15': 'Auvergne-Rhône-Alpes', '26': 'Auvergne-Rhône-Alpes', '38': 'Auvergne-Rhône-Alpes',
        '42': 'Auvergne-Rhône-Alpes', '43': 'Auvergne-Rhône-Alpes', '63': 'Auvergne-Rhône-Alpes',
        '69': 'Auvergne-Rhône-Alpes', '73': 'Auvergne-Rhône-Alpes', '74': 'Auvergne-Rhône-Alpes',
        '21': 'Bourgogne-Franche-Comté', '25': 'Bourgogne-Franche-Comté', '39': 'Bourgogne-Franche-Comté',
        '58': 'Bourgogne-Franche-Comté', '70': 'Bourgogne-Franche-Comté', '71': 'Bourgogne-Franche-Comté',
        '89': 'Bourgogne-Franche-Comté', '90': 'Bourgogne-Franche-Comté',
        '22': 'Bretagne', '29': 'Bretagne', '35': 'Bretagne', '56': 'Bretagne',
        '18': 'Centre-Val de Loire', '28': 'Centre-Val de Loire', '36': 'Centre-Val de Loire',
        '37': 'Centre-Val de Loire', '41': 'Centre-Val de Loire', '45': 'Centre-Val de Loire',
        '2A': 'Corse', '2B': 'Corse',
        '08': 'Grand Est', '10': 'Grand Est', '51': 'Grand Est', '52': 'Grand Est',
        '54': 'Grand Est', '55': 'Grand Est', '57': 'Grand Est', '67': 'Grand Est',
        '68': 'Grand Est', '88': 'Grand Est',
        '02': 'Hauts-de-France', '59': 'Hauts-de-France', '60': 'Hauts-de-France',
        '62': 'Hauts-de-France', '80': 'Hauts-de-France',
        '75': 'Île-de-France', '77': 'Île-de-France', '78': 'Île-de-France', '91': 'Île-de-France',
        '92': 'Île-de-France', '93': 'Île-de-France', '94': 'Île-de-France', '95': 'Île-de-France',
        '14': 'Normandie', '27': 'Normandie', '50': 'Normandie', '61': 'Normandie', '76': 'Normandie',
        '16': 'Nouvelle-Aquitaine', '17': 'Nouvelle-Aquitaine', '19': 'Nouvelle-Aquitaine',
        '23': 'Nouvelle-Aquitaine', '24': 'Nouvelle-Aquitaine', '33': 'Nouvelle-Aquitaine',
        '40': 'Nouvelle-Aquitaine', '47': 'Nouvelle-Aquitaine', '64': 'Nouvelle-Aquitaine',
        '79': 'Nouvelle-Aquitaine', '86': 'Nouvelle-Aquitaine', '87': 'Nouvelle-Aquitaine',
        '09': 'Occitanie', '11': 'Occitanie', '12': 'Occitanie', '30': 'Occitanie',
        '31': 'Occitanie', '32': 'Occitanie', '34': 'Occitanie', '46': 'Occitanie',
        '48': 'Occitanie', '65': 'Occitanie', '66': 'Occitanie', '81': 'Occitanie', '82': 'Occitanie',
        '44': 'Pays de la Loire', '49': 'Pays de la Loire', '53': 'Pays de la Loire',
        '72': 'Pays de la Loire', '85': 'Pays de la Loire',
        '04': 'Provence-Alpes-Côte d\'Azur', '05': 'Provence-Alpes-Côte d\'Azur',
        '06': 'Provence-Alpes-Côte d\'Azur', '13': 'Provence-Alpes-Côte d\'Azur',
        '83': 'Provence-Alpes-Côte d\'Azur', '84': 'Provence-Alpes-Côte d\'Azur',
        '971': 'Outre-Mer', '972': 'Outre-Mer', '973': 'Outre-Mer', '974': 'Outre-Mer', '976': 'Outre-Mer',
        '975': 'Outre-Mer', '977': 'Outre-Mer', '978': 'Outre-Mer', '986': 'Outre-Mer', '987': 'Outre-Mer',
        '988': 'Outre-Mer'
    }
    
    df_copy['region'] = df_copy['departement'].map(region_map)
    
    # Sélectionner le niveau géographique
    group_col = 'departement' if geo_level == 'departement' else 'region'
    
    # Calcul de l'APL pondéré et des statistiques associées
    result = []
    
    for territory, group in df_copy.groupby(group_col):
        # Ignorer les territoires non identifiés
        if pd.isna(territory):
            continue
            
        total_pop = group['P16_POP'].sum()
        weighted_apl = (group['APL'] * group['P16_POP']).sum() / total_pop if total_pop > 0 else 0
        desert_pop = group[group['APL'] < 2.5]['P16_POP'].sum()
        desert_percent = (desert_pop / total_pop) * 100 if total_pop > 0 else 0
        
        result.append({
            'territoire': territory,
            'population': total_pop,
            'apl_pondere': weighted_apl,
            'communes_count': len(group),
            'desert_percent': desert_percent,
            'desert_count': len(group[group['APL'] < 2.5]),
            'min_apl': group['APL'].min(),
            'max_apl': group['APL'].max()
        })
    
    # Convertir en DataFrame et trier par APL pondéré
    if result:  # Vérifier si result n'est pas vide
        result_df = pd.DataFrame(result).sort_values('apl_pondere')
    else:
        # Créer un DataFrame vide avec les colonnes appropriées
        result_df = pd.DataFrame(columns=['territoire', 'population', 'apl_pondere', 'communes_count', 
                                         'desert_percent', 'desert_count', 'min_apl', 'max_apl'])
    
    return result_df
    
    # Créer les colonnes de département et région si nécessaire
    df_copy = df.copy()
    df_copy['departement'] = df_copy['CODGEO'].str[:2]
    
    # Pour les régions, utiliser une table de correspondance département-région
    # Correspondance simplifiée (à remplacer par une table plus complète si nécessaire)
    region_map = {
        '01': 'Auvergne-Rhône-Alpes', '03': 'Auvergne-Rhône-Alpes', '07': 'Auvergne-Rhône-Alpes', 
        '15': 'Auvergne-Rhône-Alpes', '26': 'Auvergne-Rhône-Alpes', '38': 'Auvergne-Rhône-Alpes',
        '42': 'Auvergne-Rhône-Alpes', '43': 'Auvergne-Rhône-Alpes', '63': 'Auvergne-Rhône-Alpes',
        '69': 'Auvergne-Rhône-Alpes', '73': 'Auvergne-Rhône-Alpes', '74': 'Auvergne-Rhône-Alpes',
        '21': 'Bourgogne-Franche-Comté', '25': 'Bourgogne-Franche-Comté', '39': 'Bourgogne-Franche-Comté',
        '58': 'Bourgogne-Franche-Comté', '70': 'Bourgogne-Franche-Comté', '71': 'Bourgogne-Franche-Comté',
        '89': 'Bourgogne-Franche-Comté', '90': 'Bourgogne-Franche-Comté',
        '22': 'Bretagne', '29': 'Bretagne', '35': 'Bretagne', '56': 'Bretagne',
        '18': 'Centre-Val de Loire', '28': 'Centre-Val de Loire', '36': 'Centre-Val de Loire',
        '37': 'Centre-Val de Loire', '41': 'Centre-Val de Loire', '45': 'Centre-Val de Loire',
        '2A': 'Corse', '2B': 'Corse',
        '08': 'Grand Est', '10': 'Grand Est', '51': 'Grand Est', '52': 'Grand Est',
        '54': 'Grand Est', '55': 'Grand Est', '57': 'Grand Est', '67': 'Grand Est',
        '68': 'Grand Est', '88': 'Grand Est',
        '02': 'Hauts-de-France', '59': 'Hauts-de-France', '60': 'Hauts-de-France',
        '62': 'Hauts-de-France', '80': 'Hauts-de-France',
        '75': 'Île-de-France', '77': 'Île-de-France', '78': 'Île-de-France', '91': 'Île-de-France',
        '92': 'Île-de-France', '93': 'Île-de-France', '94': 'Île-de-France', '95': 'Île-de-France',
        '14': 'Normandie', '27': 'Normandie', '50': 'Normandie', '61': 'Normandie', '76': 'Normandie',
        '16': 'Nouvelle-Aquitaine', '17': 'Nouvelle-Aquitaine', '19': 'Nouvelle-Aquitaine',
        '23': 'Nouvelle-Aquitaine', '24': 'Nouvelle-Aquitaine', '33': 'Nouvelle-Aquitaine',
        '40': 'Nouvelle-Aquitaine', '47': 'Nouvelle-Aquitaine', '64': 'Nouvelle-Aquitaine',
        '79': 'Nouvelle-Aquitaine', '86': 'Nouvelle-Aquitaine', '87': 'Nouvelle-Aquitaine',
        '09': 'Occitanie', '11': 'Occitanie', '12': 'Occitanie', '30': 'Occitanie',
        '31': 'Occitanie', '32': 'Occitanie', '34': 'Occitanie', '46': 'Occitanie',
        '48': 'Occitanie', '65': 'Occitanie', '66': 'Occitanie', '81': 'Occitanie', '82': 'Occitanie',
        '44': 'Pays de la Loire', '49': 'Pays de la Loire', '53': 'Pays de la Loire',
        '72': 'Pays de la Loire', '85': 'Pays de la Loire',
        '04': 'Provence-Alpes-Côte d\'Azur', '05': 'Provence-Alpes-Côte d\'Azur',
        '06': 'Provence-Alpes-Côte d\'Azur', '13': 'Provence-Alpes-Côte d\'Azur',
        '83': 'Provence-Alpes-Côte d\'Azur', '84': 'Provence-Alpes-Côte d\'Azur',
        '971': 'Outre-Mer', '972': 'Outre-Mer', '973': 'Outre-Mer', '974': 'Outre-Mer', '976': 'Outre-Mer',
        '975': 'Outre-Mer', '977': 'Outre-Mer', '978': 'Outre-Mer', '986': 'Outre-Mer', '987': 'Outre-Mer',
        '988': 'Outre-Mer'
    }
    
    df_copy['region'] = df_copy['departement'].map(region_map)
    
    # Sélectionner le niveau géographique
    group_col = 'departement' if geo_level == 'departement' else 'region'
    
    # Calcul de l'APL pondéré et des statistiques associées
    result = []
    
    for territory, group in df_copy.groupby(group_col):
        # Ignorer les territoires non identifiés
        if pd.isna(territory):
            continue
            
        total_pop = group['P16_POP'].sum()
        weighted_apl = (group['APL'] * group['P16_POP']).sum() / total_pop if total_pop > 0 else 0
        desert_pop = group[group['APL'] < 2.5]['P16_POP'].sum()
        desert_percent = (desert_pop / total_pop) * 100 if total_pop > 0 else 0
        
        result.append({
            'territoire': territory,
            'population': total_pop,
            'apl_pondere': weighted_apl,
            'communes_count': len(group),
            'desert_percent': desert_percent,
            'desert_count': len(group[group['APL'] < 2.5]),
            'min_apl': group['APL'].min(),
            'max_apl': group['APL'].max()
        })
    
    # Convertir en DataFrame et trier par APL pondéré
    if result:  # Vérifiez si result n'est pas vide
        result_df = pd.DataFrame(result).sort_values('apl_pondere')
    else:
        # Créez un DataFrame vide avec les colonnes appropriées
        result_df = pd.DataFrame(columns=['territoire', 'population', 'apl_pondere', 'communes_count', 
                                      'desert_percent', 'desert_count', 'min_apl', 'max_apl'])
    return result_df

@st.cache_data
def create_apl_categories(df):
    """Création de catégories pour l'indice APL"""
    df_with_cat = df.copy()
    
    # Définition des seuils selon la littérature sur les déserts médicaux
    conditions = [
        (df_with_cat["APL"] < 1.5),
        (df_with_cat["APL"] >= 1.5) & (df_with_cat["APL"] < 2.5),
        (df_with_cat["APL"] >= 2.5) & (df_with_cat["APL"] < 3.5),
        (df_with_cat["APL"] >= 3.5) & (df_with_cat["APL"] < 4.5),
        (df_with_cat["APL"] >= 4.5)
    ]
    
    categories = [
        "Désert médical critique",
        "Désert médical",
        "Sous-équipement médical",
        "Équipement médical suffisant",
        "Bon équipement médical"
    ]
    
    colors = [
        "darkred",
        "red",
        "orange",
        "lightgreen",
        "green"
    ]
    
    df_with_cat["APL_category"] = np.select(conditions, categories, default="Non catégorisé")
    df_with_cat["APL_color"] = np.select(conditions, colors, default="gray")
    
    return df_with_cat

@st.cache_data
def create_map(df, column="APL", title="Carte des déserts médicaux"):
    """Création d'une carte avec Plotly"""
    # Nettoyer les données (suppression des valeurs manquantes)
    df_clean = df.dropna(subset=['latitude_mairie', 'longitude_mairie', column])
    
    # Déterminer la palette de couleurs en fonction de la colonne
    if column == "APL":
        # Pour l'APL, on veut une échelle inversée (rouge = faible valeur = désert médical)
        color_scale = "RdYlGn"
        hover_template = "<b>%{customdata[0]}</b><br>APL: %{z:.2f}<br>Population: %{customdata[1]:.0f}<br>Catégorie: %{customdata[2]}"
        custom_data = np.column_stack((df_clean['Communes'], df_clean['P16_POP'], df_clean['APL_category']))
    elif column == "desert_risk_score":
        # Pour le score de risque, valeurs élevées = risque élevé = rouge
        color_scale = "YlOrRd"
        hover_template = "<b>%{customdata[0]}</b><br>APL actuel: %{customdata[1]:.2f}<br>Score de risque: %{z:.1f}/100<br>Catégorie: %{customdata[2]}"
        custom_data = np.column_stack((df_clean['Communes'], df_clean['APL'], df_clean['risk_category']))
    else:
        color_scale = "Viridis"
        hover_template = "<b>%{customdata[0]}</b><br>Valeur: %{z:.2f}"
        custom_data = np.column_stack((df_clean['Communes'],))
        
    # Créer la carte
    fig = go.Figure(data=go.Densitymap(
        lat=df_clean['latitude_mairie'],
        lon=df_clean['longitude_mairie'],
        z=df_clean[column],
        radius=10,
        colorscale=color_scale,
        colorbar=dict(title=column),
        customdata=custom_data,
        hovertemplate=hover_template,
        opacity=0.8
    ))
    
    # Configuration de la carte
    fig.update_layout(
        title=title,
        mapbox_style="carto-positron",
        mapbox=dict(
            center=dict(lat=46.603354, lon=1.888334),
            zoom=5
        ),
        height=700,
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    
    return fig

@st.cache_data
def create_territorial_map(data, geo_level, column="apl_pondere", title="Carte par territoire"):
    """Création d'une carte de chaleur par département ou région"""
    
    # Préparer les données territoriales
    if geo_level == "departement":
        # Calculer l'APL pondéré et les coordonnées moyennes pour chaque département
        dept_data = {}
        
        for dept_code, dept_group in data.groupby(data['CODGEO'].str[:2]):
            # Filtrer les données valides pour ce département
            valid_data = dept_group.dropna(subset=['latitude_mairie', 'longitude_mairie', 'APL', 'P16_POP'])
            
            if len(valid_data) > 0:
                # Calculer l'APL pondéré
                total_pop = valid_data['P16_POP'].sum()
                apl_pondere = (valid_data['APL'] * valid_data['P16_POP']).sum() / total_pop if total_pop > 0 else 0
                
                # Calculer les coordonnées moyennes (pondérées par population)
                lat_moy = (valid_data['latitude_mairie'] * valid_data['P16_POP']).sum() / total_pop
                lon_moy = (valid_data['longitude_mairie'] * valid_data['P16_POP']).sum() / total_pop
                
                # Autres statistiques
                desert_pop = valid_data[valid_data['APL'] < 2.5]['P16_POP'].sum()
                desert_percent = (desert_pop / total_pop) * 100 if total_pop > 0 else 0
                
                dept_data[dept_code] = {
                    'code': dept_code,
                    'nom': f"Département {dept_code}",
                    'lat': lat_moy,
                    'lon': lon_moy,
                    'apl_pondere': apl_pondere,
                    'population': total_pop,
                    'desert_percent': desert_percent
                }
        
        # Convertir en DataFrame
        map_df = pd.DataFrame(list(dept_data.values()))
        
    elif geo_level == "commune":
        # Pour les communes, utiliser directement les coordonnées disponibles
        commune_data = []
        
        for _, row in data.iterrows():
            if not pd.isna(row['latitude_mairie']) and not pd.isna(row['longitude_mairie']):
                commune_data.append({
                    'code': row['CODGEO'],
                    'nom': row['Communes'],
                    'lat': row['latitude_mairie'],
                    'lon': row['longitude_mairie'],
                    'apl_pondere': row['APL'],
                    'population': row['P16_POP'],
                    'desert_percent': 100 if row['APL'] < 2.5 else 0
                })
                
        # Convertir en DataFrame
        map_df = pd.DataFrame(commune_data)
        
    else:  # region
        # Table de correspondance département-région
        region_map = {
            '01': 'Auvergne-Rhône-Alpes', '03': 'Auvergne-Rhône-Alpes', '07': 'Auvergne-Rhône-Alpes', 
            '15': 'Auvergne-Rhône-Alpes', '26': 'Auvergne-Rhône-Alpes', '38': 'Auvergne-Rhône-Alpes',
            '42': 'Auvergne-Rhône-Alpes', '43': 'Auvergne-Rhône-Alpes', '63': 'Auvergne-Rhône-Alpes',
            '69': 'Auvergne-Rhône-Alpes', '73': 'Auvergne-Rhône-Alpes', '74': 'Auvergne-Rhône-Alpes',
            '21': 'Bourgogne-Franche-Comté', '25': 'Bourgogne-Franche-Comté', '39': 'Bourgogne-Franche-Comté',
            '58': 'Bourgogne-Franche-Comté', '70': 'Bourgogne-Franche-Comté', '71': 'Bourgogne-Franche-Comté',
            '89': 'Bourgogne-Franche-Comté', '90': 'Bourgogne-Franche-Comté',
            '22': 'Bretagne', '29': 'Bretagne', '35': 'Bretagne', '56': 'Bretagne',
            '18': 'Centre-Val de Loire', '28': 'Centre-Val de Loire', '36': 'Centre-Val de Loire',
            '37': 'Centre-Val de Loire', '41': 'Centre-Val de Loire', '45': 'Centre-Val de Loire',
            '2A': 'Corse', '2B': 'Corse',
            '08': 'Grand Est', '10': 'Grand Est', '51': 'Grand Est', '52': 'Grand Est',
            '54': 'Grand Est', '55': 'Grand Est', '57': 'Grand Est', '67': 'Grand Est',
            '68': 'Grand Est', '88': 'Grand Est',
            '02': 'Hauts-de-France', '59': 'Hauts-de-France', '60': 'Hauts-de-France',
            '62': 'Hauts-de-France', '80': 'Hauts-de-France',
            '75': 'Île-de-France', '77': 'Île-de-France', '78': 'Île-de-France', '91': 'Île-de-France',
            '92': 'Île-de-France', '93': 'Île-de-France', '94': 'Île-de-France', '95': 'Île-de-France',
            '14': 'Normandie', '27': 'Normandie', '50': 'Normandie', '61': 'Normandie', '76': 'Normandie',
            '16': 'Nouvelle-Aquitaine', '17': 'Nouvelle-Aquitaine', '19': 'Nouvelle-Aquitaine',
            '23': 'Nouvelle-Aquitaine', '24': 'Nouvelle-Aquitaine', '33': 'Nouvelle-Aquitaine',
            '40': 'Nouvelle-Aquitaine', '47': 'Nouvelle-Aquitaine', '64': 'Nouvelle-Aquitaine',
            '79': 'Nouvelle-Aquitaine', '86': 'Nouvelle-Aquitaine', '87': 'Nouvelle-Aquitaine',
            '09': 'Occitanie', '11': 'Occitanie', '12': 'Occitanie', '30': 'Occitanie',
            '31': 'Occitanie', '32': 'Occitanie', '34': 'Occitanie', '46': 'Occitanie',
            '48': 'Occitanie', '65': 'Occitanie', '66': 'Occitanie', '81': 'Occitanie', '82': 'Occitanie',
            '44': 'Pays de la Loire', '49': 'Pays de la Loire', '53': 'Pays de la Loire',
            '72': 'Pays de la Loire', '85': 'Pays de la Loire',
            '04': 'Provence-Alpes-Côte d\'Azur', '05': 'Provence-Alpes-Côte d\'Azur',
            '06': 'Provence-Alpes-Côte d\'Azur', '13': 'Provence-Alpes-Côte d\'Azur',
            '83': 'Provence-Alpes-Côte d\'Azur', '84': 'Provence-Alpes-Côte d\'Azur',
            '971': 'Outre-Mer', '972': 'Outre-Mer', '973': 'Outre-Mer', '974': 'Outre-Mer', '976': 'Outre-Mer',
            '975': 'Outre-Mer', '977': 'Outre-Mer', '978': 'Outre-Mer', '986': 'Outre-Mer', '987': 'Outre-Mer',
            '988': 'Outre-Mer'
        }
        
        # Ajouter la colonne région
        data_copy = data.copy()
        data_copy['region'] = data_copy['CODGEO'].str[:2].map(region_map)
        
        # Regrouper par région
        region_data = {}
        
        for region_name, region_group in data_copy.groupby('region'):
            if pd.isna(region_name):
                continue
                
            # Filtrer les données valides pour cette région
            valid_data = region_group.dropna(subset=['latitude_mairie', 'longitude_mairie', 'APL', 'P16_POP'])
            
            if len(valid_data) > 0:
                # Calculer l'APL pondéré
                total_pop = valid_data['P16_POP'].sum()
                apl_pondere = (valid_data['APL'] * valid_data['P16_POP']).sum() / total_pop if total_pop > 0 else 0
                
                # Calculer les coordonnées moyennes (pondérées par population)
                lat_moy = (valid_data['latitude_mairie'] * valid_data['P16_POP']).sum() / total_pop
                lon_moy = (valid_data['longitude_mairie'] * valid_data['P16_POP']).sum() / total_pop
                
                # Autres statistiques
                desert_pop = valid_data[valid_data['APL'] < 2.5]['P16_POP'].sum()
                desert_percent = (desert_pop / total_pop) * 100 if total_pop > 0 else 0
                
                region_data[region_name] = {
                    'code': region_name,
                    'nom': region_name,
                    'lat': lat_moy,
                    'lon': lon_moy,
                    'apl_pondere': apl_pondere,
                    'population': total_pop,
                    'desert_percent': desert_percent
                }
        
        # Convertir en DataFrame
        map_df = pd.DataFrame(list(region_data.values()))
    
    # Vérifier que des données sont disponibles
    if len(map_df) == 0:
        return None
    
    # Créer une heatmap basée sur les territoires
    fig = go.Figure()
    
    # Adapter le marker_size selon le niveau géographique
    if geo_level == 'commune':
        size_divisor = 100  # Plus petit pour les communes car plus nombreuses
        size_min = 5
    else:
        size_divisor = 20
        size_min = 10
    
    # Ajouter des cercles proportionnels à la population, colorés selon l'APL
    fig.add_trace(go.Scattermap(
        lat=map_df['lat'],
        lon=map_df['lon'],
        mode='markers',
        marker=dict(
            size=np.sqrt(map_df['population']) / size_divisor,  # Taille proportionnelle à la racine carrée de la population
            sizemin=size_min,
            sizemode='area',
            color=map_df['apl_pondere'],
            colorscale='RdYlGn',
            colorbar=dict(title="APL pondéré"),
            cmin=1,  # Minimum de l'échelle de couleur
            cmax=5,  # Maximum de l'échelle de couleur
            opacity=0.8
        ),
        text=map_df['nom'],
        hovertemplate="<b>%{text}</b><br>" +
                      "APL pondéré: %{marker.color:.2f}<br>" +
                      "Population: %{customdata[0]:,.0f}<br>" +
                      "Communes en désert médical: %{customdata[1]:.1f}%<br>",
        customdata=np.column_stack((
            map_df['population'],
            map_df['desert_percent']
        ))
    ))
    
    # Configuration de la carte
    fig.update_layout(
        title=title,
        mapbox_style="carto-positron",
        mapbox=dict(
            center=dict(lat=46.603354, lon=1.888334),
            zoom=5
        ),
        height=700,
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    
    return fig

@st.cache_data
def analyze_by_population(df):
    """Analyse des déserts médicaux par taille de population"""
    # Créer des catégories de taille de communes
    population_bins = [0, 500, 1000, 5000, 10000, 50000, float('inf')]
    population_labels = ["<500", "500-1 000", "1 000-5 000", "5 000-10 000", "10 000-50 000", ">50 000"]
    
    df_pop = df.copy()
    df_pop['population_category'] = pd.cut(df_pop['P16_POP'], bins=population_bins, labels=population_labels)
    
    # Calculer l'APL moyen par catégorie de population
    pop_analysis = df_pop.groupby('population_category', observed=True)['APL'].agg(['mean', 'count']).reset_index()
    pop_analysis['desert_count'] = df_pop[df_pop['APL'] < 2.5].groupby('population_category', observed=True).size().values
    pop_analysis['desert_percent'] = (pop_analysis['desert_count'] / pop_analysis['count']) * 100
    
    return pop_analysis

@st.cache_data
def analyze_by_age(df):
    """Analyse des déserts médicaux par structure d'âge"""
    # Calculer la corrélation entre l'APL et les catégories d'âge
    age_corr = pd.DataFrame({
        'Catégorie': ['0-14 ans', '15-59 ans', '60+ ans'],
        'Corrélation avec APL': [
            df['APL'].corr(df['0_14_pop_rate']),
            df['APL'].corr(df['15_59_pop_rate']),
            df['APL'].corr(df['60+_pop_rate'])
        ]
    })
    
    # Regrouper les communes en fonction de la prédominance des catégories d'âge
    def age_group_predominant(row):
        ages = {
            '0-14 ans': row['0_14_pop_rate'],
            '15-59 ans': row['15_59_pop_rate'],
            '60+ ans': row['60+_pop_rate']
        }
        return max(ages, key=ages.get)
    
    df_age = df.copy()
    df_age['predominant_age'] = df_age.apply(age_group_predominant, axis=1)
    
    age_analysis = df_age.groupby('predominant_age', observed=True)['APL'].agg(['mean', 'count']).reset_index()
    age_analysis['desert_count'] = df_age[df_age['APL'] < 2.5].groupby('predominant_age', observed=True).size().values
    age_analysis['desert_percent'] = (age_analysis['desert_count'] / age_analysis['count']) * 100
    
    return age_corr, age_analysis

@st.cache_data
def create_clusters(df, n_clusters=5):
    """Création de clusters de communes similaires basés sur plusieurs variables"""
    # Sélectionner les variables les plus pertinentes pour le clustering
    cluster_vars = [
        'APL',                  # Accessibilité médicale actuelle
        'P16_POP',              # Population
        'median_living_standard',  # Niveau de vie
        'density_area',         # Densité de population
        '60+_pop_rate'          # Population âgée
    ]
    
    # Ajouter des variables optionnelles si elles existent dans le df
    optional_vars = [
        'healthcare_education_establishments',  # Infrastructures de santé
        'city_social_amenities_rate'           # Équipements sociaux
    ]
    
    for var in optional_vars:
        if var in df.columns:
            cluster_vars.append(var)
    
    # Création d'un sous-ensemble avec les variables sélectionnées
    # Suppression des communes avec trop de valeurs manquantes
    df_cluster = df[cluster_vars].copy().dropna(thresh=len(cluster_vars)-2)
    
    # Gestion des valeurs manquantes restantes
    df_cluster = df_cluster.fillna(df_cluster.median())
    
    # Normalisation des données pour éviter les biais d'échelle
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_cluster),
        columns=df_cluster.columns,
        index=df_cluster.index
    )
    
    # Clustering K-means avec nombre de clusters fixé
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_scaled['cluster'] = kmeans.fit_predict(df_scaled)
    
    # Ajouter les clusters au dataframe original en préservant l'index
    df_with_clusters = df.copy()
    cluster_mapping = pd.Series(df_scaled['cluster'].values, index=df_scaled.index)
    df_with_clusters = df_with_clusters.join(cluster_mapping.rename('cluster'), how='left')
    
    # Analyse des clusters
    cluster_analysis = df_with_clusters.groupby('cluster')[cluster_vars].mean().reset_index()
    
    # Caractériser les clusters de manière plus détaillée
    cluster_profiles = []
    for i, row in cluster_analysis.iterrows():
        # Déterminer les caractéristiques dominantes de chaque cluster
        if row['APL'] < 2.5:
            if row['60+_pop_rate'] > 30:
                if row['density_area'] < 50:
                    profile = "Désert médical rural vieillissant"
                else:
                    profile = "Désert médical urbain vieillissant"
            elif row['density_area'] < 50:
                profile = "Désert médical rural"
            else:
                profile = "Désert médical urbain"
        else:
            if row['P16_POP'] > 10000:
                if 'healthcare_education_establishments' in row and row['healthcare_education_establishments'] > cluster_analysis['healthcare_education_establishments'].median():
                    profile = "Grande ville bien équipée"
                else:
                    profile = "Grande ville avec accès médical moyen"
            else:
                if row['APL'] > 4:
                    profile = "Petite commune très bien desservie"
                else:
                    profile = "Petite commune moyennement desservie"
        
        cluster_profiles.append(profile)
    
    cluster_analysis['profile'] = cluster_profiles
    
    # Calculer des statistiques supplémentaires par cluster
    cluster_stats = df_with_clusters.groupby('cluster').agg({
        'CODGEO': 'count',
        'P16_POP': 'sum',
        'APL': ['mean', 'min', 'max', 'std']
    }).reset_index()
    
    # Aplatir les colonnes multiindex
    cluster_stats.columns = ['_'.join(col).strip('_') for col in cluster_stats.columns.values]
    
    # Renommer les colonnes pour plus de clarté
    cluster_stats = cluster_stats.rename(columns={
        'cluster_': 'cluster',
        'CODGEO_count': 'nb_communes',
        'P16_POP_sum': 'population_totale',
        'APL_mean': 'apl_moyen',
        'APL_min': 'apl_min',
        'APL_max': 'apl_max',
        'APL_std': 'apl_std'
    })
    
    # Fusionner les analyses de clusters
    cluster_full_analysis = pd.merge(
        cluster_analysis,
        cluster_stats,
        on='cluster'
    )
    
    # Joindre le profil au dataframe avec clusters
    df_with_clusters = df_with_clusters.merge(
        cluster_analysis[['cluster', 'profile']], 
        on='cluster', 
        how='left'
    )
    
    return df_with_clusters, cluster_full_analysis

@st.cache_data
def predict_future_desert_risk(df):
    """Prédiction du risque futur de désert médical avec distribution améliorée sur 0-100"""
    # Définition des facteurs de risque
    risk_factors = {
        'APL': -0.5,
        '60+_pop_rate': 0.2,
        'density_area': -0.15,
        'median_living_standard': -0.15,
        'healthcare_education_establishments': -0.1,
        'city_social_amenities_rate': -0.1,
        'mobility_rate': -0.1
    }
    
    # Normalisation des facteurs pour le calcul du score
    df_risk = df.copy()
    
    # Gestion des valeurs manquantes
    for factor in risk_factors.keys():
        if factor in df_risk.columns:
            df_risk[factor] = df_risk[factor].fillna(df_risk[factor].median())
    
    # Calcul du score de risque
    df_risk['desert_risk_score'] = 0
    for factor, weight in risk_factors.items():
        if factor in df_risk.columns:
            # Normalisation Min-Max pour chaque facteur
            factor_min = df_risk[factor].min()
            factor_max = df_risk[factor].max()
            factor_range = factor_max - factor_min
            
            if factor_range > 0:  # Éviter la division par zéro
                if weight < 0:  # Si facteur négatif, les petites valeurs = haut risque
                    df_risk[f"{factor}_norm"] = 1 - ((df_risk[factor] - factor_min) / factor_range)
                else:  # Si facteur positif, les grandes valeurs = haut risque
                    df_risk[f"{factor}_norm"] = (df_risk[factor] - factor_min) / factor_range
                
                # Ajout au score avec le poids approprié
                df_risk['desert_risk_score'] += abs(weight) * df_risk[f"{factor}_norm"]
    
    # Assurer une meilleure distribution sur l'échelle 0-100
    # 1. Normalisation simple pour ramener entre 0 et 1
    score_min = df_risk['desert_risk_score'].min()
    score_max = df_risk['desert_risk_score'].max()
    
    if score_max > score_min:
        df_risk['desert_risk_score'] = (df_risk['desert_risk_score'] - score_min) / (score_max - score_min)
    
    # 2. Appliquer une transformation pour étaler la distribution
    # Utilisation de la fonction quantile pour répartir uniformément les scores
    quantiles = np.linspace(0, 1, 101)  # 101 points pour obtenir 100 intervalles
    score_values = df_risk['desert_risk_score'].quantile(quantiles).values
    
    # Créer un mappeur de score
    from scipy.interpolate import interp1d
    score_mapper = interp1d(
        score_values, 
        np.linspace(0, 100, 101),
        bounds_error=False, 
        fill_value=(0, 100)
    )
    
    # Appliquer le mappeur pour obtenir une distribution uniforme
    df_risk['desert_risk_score'] = score_mapper(df_risk['desert_risk_score'])
    
    # Catégorisation du risque
    risk_conditions = [
        (df_risk['desert_risk_score'] >= 80),
        (df_risk['desert_risk_score'] >= 60) & (df_risk['desert_risk_score'] < 80),
        (df_risk['desert_risk_score'] >= 40) & (df_risk['desert_risk_score'] < 60),
        (df_risk['desert_risk_score'] >= 20) & (df_risk['desert_risk_score'] < 40),
        (df_risk['desert_risk_score'] < 20)
    ]
    
    risk_categories = [
        "Risque très élevé",
        "Risque élevé",
        "Risque modéré",
        "Risque faible",
        "Risque très faible"
    ]
    
    risk_colors = [
        "darkred",
        "red",
        "orange",
        "yellow",
        "green"
    ]
    
    df_risk['risk_category'] = np.select(risk_conditions, risk_categories, default="Non évalué")
    df_risk['risk_color'] = np.select(risk_conditions, risk_colors, default="gray")
    
    # Ajout d'une prédiction simple d'APL à 5 ans (limitée entre 0 et 5)
    max_decrease = 0.5
    df_risk['projected_APL_5y'] = df_risk['APL'] * (1 - ((df_risk['desert_risk_score'] / 100) * max_decrease))
    df_risk['projected_APL_5y'] = df_risk['projected_APL_5y'].clip(0, 5)
    
    # Facteurs principaux de risque
    factor_names = {
        'APL': 'Faible accessibilité actuelle',
        '60+_pop_rate': 'Population âgée importante',
        'density_area': 'Faible densité de population',
        'median_living_standard': 'Niveau de vie modeste',
        'healthcare_education_establishments': 'Manque d\'infrastructures de santé',
        'city_social_amenities_rate': 'Peu d\'équipements sociaux',
        'mobility_rate': 'Faible mobilité'
    }
    
    # Identifier les principaux facteurs de risque pour chaque commune
    df_risk['main_factor1'] = ''
    df_risk['main_factor2'] = ''
    
    for idx in df_risk.index:
        factor_scores = {}
        for factor, weight in risk_factors.items():
            if f"{factor}_norm" in df_risk.columns:
                factor_scores[factor] = df_risk.loc[idx, f"{factor}_norm"] * abs(weight)
        
        # Trier les facteurs par leur contribution au score
        sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Stocker les deux principaux facteurs
        if len(sorted_factors) > 0:
            df_risk.at[idx, 'main_factor1'] = factor_names.get(sorted_factors[0][0], sorted_factors[0][0])
        if len(sorted_factors) > 1:
            df_risk.at[idx, 'main_factor2'] = factor_names.get(sorted_factors[1][0], sorted_factors[1][0])
    
    return df_risk

@st.cache_data
def prepare_zone_clusters(data_zoned, min_communes=10):
    """Préparation des clusters territoriaux"""
    # Catégoriser les communes
    data_zoned['departement'] = data_zoned['CODGEO'].str[:2]
    
    # Grouper par département et type de zone pour identifier les "clusters"
    dept_zones = data_zoned.groupby(['departement', 'zone_type']).agg({
        'CODGEO': 'count',
        'P16_POP': 'sum',
        'APL': 'mean',
        'latitude_mairie': 'mean',
        'longitude_mairie': 'mean',
        'zone_color': 'first'
    }).reset_index()
    
    # Renommer les colonnes
    dept_zones.columns = ['Département', 'Type de zone', 'Nombre de communes', 
                         'Population', 'APL moyen', 'Latitude', 'Longitude', 'Couleur']
    
    # Filtrer pour garder uniquement les zones significatives
    significant_zones = dept_zones[dept_zones['Nombre de communes'] >= min_communes]
    
    # Trier par population décroissante
    significant_zones = significant_zones.sort_values('Population', ascending=False)
    
    return significant_zones



def cluster_view(data_zoned, significant_zones):
    tab1, tab2, tab3 = st.tabs(["Vue d'ensemble", "Explorer par zone", "Analyse détaillée"])
    
    with tab1:
        display_overview(significant_zones)
    
    with tab2:
        display_zone_explorer(data_zoned, significant_zones)
    
    with tab3:
        display_detailed_analysis(data_zoned)


def create_optimized_map(significant_zones, max_zones=50):
    """Crée une carte optimisée avec moins de points"""
    # Limiter le nombre de zones affichées si nécessaire
    if len(significant_zones) > max_zones:
        display_zones = significant_zones.head(max_zones)
        st.info(f"Affichage des {max_zones} zones les plus importantes par population.")
    else:
        display_zones = significant_zones
    
    fig = go.Figure()
    
    # Ajouter chaque zone comme un cercle proportionnel à sa population
    for _, zone in display_zones.iterrows():
        population_scale = np.sqrt(zone['Population']) / 100
        radius = max(5, min(30, population_scale))
        
        fig.add_trace(go.Scattermapbox(
            lat=[zone['Latitude']],
            lon=[zone['Longitude']],
            mode='markers',
            marker=dict(
                size=radius,
                color=zone['Couleur'],
                opacity=0.7
            ),
            text=[f"{zone['Type de zone']} - Dept {zone['Département']}"],
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "Département: " + zone['Département'] + "<br>" +
                "Communes: " + str(zone['Nombre de communes']) + "<br>" +
                "Population: " + f"{int(zone['Population']):,}".replace(',', ' ') + "<br>" +
                "APL moyen: " + f"{zone['APL moyen']:.2f}"
            ),
            name=f"{zone['Type de zone']} - Dept {zone['Département']}"
        ))
    
    # Configuration de la carte
    fig.update_layout(
        title="Principales zones d'accessibilité médicale similaire",
        mapbox_style="carto-positron",
        mapbox=dict(
            center=dict(lat=46.603354, lon=1.888334),
            zoom=5
        ),
        height=700,
        margin={"r":0,"t":50,"l":0,"b":0},
        showlegend=False
    )
    
    return fig


def display_zone_communes(zone_communes, max_points=100):
    """Affiche la carte d'une zone spécifique avec limitation des points"""
    if len(zone_communes) > max_points:
        # Échantillonnage stratifié pour garder une représentation correcte
        # Assurer que les communes importantes sont incluses
        top_communes = zone_communes.nlargest(max_points//5, 'P16_POP')
        rest_sample = zone_communes[~zone_communes.index.isin(top_communes.index)].sample(
            min(max_points - len(top_communes), len(zone_communes) - len(top_communes))
        )
        display_communes = pd.concat([top_communes, rest_sample])
        st.info(f"Affichage d'un échantillon de {len(display_communes)} communes sur {len(zone_communes)}.")
    else:
        display_communes = zone_communes
    
    # Créer une carte focalisée sur la zone échantillonnée
    zone_map = go.Figure()
    
    zone_map.add_trace(go.Scattermapbox(
        lat=display_communes['latitude_mairie'],
        lon=display_communes['longitude_mairie'],
        mode='markers',
        marker=dict(
            size=8,
            color=display_communes['APL'],
            colorscale='RdYlGn',
            colorbar=dict(title="APL"),
            cmin=1,
            cmax=5,
            opacity=0.8
        ),
        text=display_communes['Communes'],
        hovertemplate="<b>%{text}</b><br>APL: %{marker.color:.2f}<br>Population: %{customdata:,.0f}",
        customdata=display_communes['P16_POP']
    ))
    
    # Configuration de la carte
    lat_center = display_communes['latitude_mairie'].mean()
    lon_center = display_communes['longitude_mairie'].mean()
    
    zone_map.update_layout(
        title=f"Communes échantillonnées de la zone",
        mapbox_style="carto-positron",
        mapbox=dict(
            center=dict(lat=lat_center, lon=lon_center),
            zoom=8
        ),
        height=600,
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    
    return zone_map

@st.cache_data
def load_optimized_geojson(filepath, simplify_tolerance=None):
    """
    Chargement optimisé d'un fichier GeoJSON avec option de simplification
    
    Args:
        filepath: chemin vers le fichier GeoJSON
        simplify_tolerance: tolérance pour la simplification des géométries (None = pas de simplification)
    
    Returns:
        GeoJSON simplifié si demandé
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            geojson = json.load(f)
            
        # Simplification optionnelle des géométries pour améliorer les performances
        if simplify_tolerance is not None and simplify_tolerance > 0:
            # Ne pas simplifier pour les petits jeux de données (régions, départements)
            if len(geojson['features']) > 100:  # Principalement pour les communes
                for feature in geojson['features']:
                    if 'geometry' in feature and feature['geometry']:
                        # Simplification pour les polygones et multipolygones
                        if feature['geometry']['type'] == 'Polygon':
                            for i, ring in enumerate(feature['geometry']['coordinates']):
                                # Ne garder qu'1 point sur N, où N dépend de la tolérance
                                step = max(1, int(simplify_tolerance * 10))
                                feature['geometry']['coordinates'][i] = ring[::step]
                        elif feature['geometry']['type'] == 'MultiPolygon':
                            for i, polygon in enumerate(feature['geometry']['coordinates']):
                                for j, ring in enumerate(polygon):
                                    step = max(1, int(simplify_tolerance * 10))
                                    feature['geometry']['coordinates'][i][j] = ring[::step]
                
                st.info(f"GeoJSON simplifié pour améliorer les performances (tolérance: {simplify_tolerance})")
        
        return geojson
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier GeoJSON: {e}")
        return None

@st.cache_data
def create_enhanced_choropleth_map(df_stats, geojson, level="departement", value_col="apl_pondere", 
                                 title="Carte choroplèthe", color_scale="RdYlGn", range_color=[1, 5],
                                 simplify_tolerance=None):
    """
    Création d'une carte choroplèthe améliorée et optimisée
    
    Args:
        df_stats: DataFrame avec les statistiques territoriales
        geojson: données GeoJSON pour la carte
        level: niveau territorial ("commune", "departement" ou "region")
        value_col: colonne contenant les valeurs à afficher
        title: titre de la carte
        color_scale: échelle de couleurs Plotly
        range_color: plage de valeurs pour la colorisation
        simplify_tolerance: tolérance de simplification pour les grandes cartes
    
    Returns:
        Figure Plotly avec la carte choroplèthe
    """
    # Déterminer la clé d'identification dans le GeoJSON selon le niveau
    if level == "commune":
        id_key = "properties.code"
        location_col = "territoire"
        zoom = 5.5
        center = {"lat": 46.603354, "lon": 1.888334}
    elif level == "departement":
        id_key = "properties.code"
        location_col = "territoire"
        zoom = 5
        center = {"lat": 46.603354, "lon": 1.888334}
    else:  # region
        id_key = "properties.nom"
        location_col = "territoire"
        zoom = 4.5
        center = {"lat": 46.603354, "lon": 1.888334}
    
    # Créer la carte choroplèthe - sans custom_data pour l'instant
    fig = px.choropleth_mapbox(
        df_stats, 
        geojson=geojson, 
        locations=location_col,
        featureidkey=id_key,
        color=value_col,
        color_continuous_scale=color_scale,
        range_color=range_color,
        mapbox_style="carto-positron",
        zoom=zoom,
        center=center,
        opacity=0.8,
        labels={value_col: 'APL pondéré'}
    )
    
    # Appliquer un template de hover personnalisé selon le niveau
    if level == "commune":
        hover_template = "<b>%{location}</b>"
        if 'nom_commune' in df_stats.columns:
            hover_template = "<b>%{customdata[0]}</b>"
        hover_template += "<br>APL: %{z:.2f}"
        
        if 'population' in df_stats.columns:
            hover_template += "<br>Population: %{customdata[1]:,.0f}"
        
        if 'desert_percent' in df_stats.columns:
            hover_template += "<br>Désert médical: %{customdata[2]:.1f}%"
        
        # Préparer les données custom pour la mise à jour
        custom_data_cols = []
        if 'nom_commune' in df_stats.columns:
            custom_data_cols.append(df_stats['nom_commune'])
        if 'population' in df_stats.columns:
            custom_data_cols.append(df_stats['population'])
        if 'desert_percent' in df_stats.columns:
            custom_data_cols.append(df_stats['desert_percent'])
            
        # Ne mettre à jour le customdata que si les colonnes existent
        if custom_data_cols:
            custom_data = np.column_stack(custom_data_cols)
            fig.update_traces(customdata=custom_data)
    else:
        hover_template = "<b>%{location}</b><br>APL pondéré: %{z:.2f}"
        
        custom_data_cols = []
        if 'population' in df_stats.columns:
            hover_template += "<br>Population: %{customdata[0]:,.0f}"
            custom_data_cols.append(df_stats['population'])
            
        if 'desert_percent' in df_stats.columns:
            hover_template += "<br>Désert médical: %{customdata[1]:.1f}%"
            custom_data_cols.append(df_stats['desert_percent'])
            
        if 'desert_count' in df_stats.columns and 'communes_count' in df_stats.columns:
            hover_template += "<br>Communes en désert: %{customdata[2]} / %{customdata[3]}"
            custom_data_cols.append(df_stats['desert_count'])
            custom_data_cols.append(df_stats['communes_count'])
        
        # Ne mettre à jour le customdata que si les colonnes existent
        if custom_data_cols:
            custom_data = np.column_stack(custom_data_cols)
            fig.update_traces(customdata=custom_data)
    
    # Appliquer le template de hover
    fig.update_traces(hovertemplate=hover_template)
    
    # Ajuster le layout de la figure
    fig.update_layout(
        title=title,
        margin={"r":0,"t":50,"l":0,"b":0},
        height=700
    )
    
    return fig

# Cette fonction permet de créer rapidement une nouvelle carte choroplèthe départementale ou régionale
@st.cache_data
def create_thematic_choropleth(filtered_data, geo_level, value_col, title, color_scale="RdYlGn", range_color=None):
    """
    Crée une carte choroplèthe thématique pour une variable spécifique
    
    Args:
        filtered_data: DataFrame avec les données à visualiser
        geo_level: niveau géographique ("departement" ou "region")
        value_col: colonne à visualiser
        title: titre de la carte
        color_scale: échelle de couleurs
        range_color: plage de valeurs pour la colorisation
    
    Returns:
        Figure Plotly avec la carte choroplèthe
    """
    # Préparer les données agrégées
    if geo_level == "departement":
        # Créer une colonne département
        df_temp = filtered_data.copy()
        df_temp['departement'] = df_temp['CODGEO'].str[:2]
        
        # Agréger les données
        agg_data = df_temp.groupby('departement').agg({
            'P16_POP': 'sum',
            value_col: 'mean' if value_col != 'APL' else lambda x: (df_temp['APL'] * df_temp['P16_POP']).sum() / df_temp['P16_POP'].sum()
        }).reset_index()
        
        # Renommer les colonnes
        agg_data.columns = ['territoire', 'population', value_col]
        
        # Charger le GeoJSON
        geojson_file = "departements.geojson"
    else:  # region
        # Créer une colonne département
        df_temp = filtered_data.copy()
        df_temp['departement'] = df_temp['CODGEO'].str[:2]
        
        # Table de correspondance département-région
        region_map = {
            '01': 'Auvergne-Rhône-Alpes', '03': 'Auvergne-Rhône-Alpes', '07': 'Auvergne-Rhône-Alpes', 
            '15': 'Auvergne-Rhône-Alpes', '26': 'Auvergne-Rhône-Alpes', '38': 'Auvergne-Rhône-Alpes',
            '42': 'Auvergne-Rhône-Alpes', '43': 'Auvergne-Rhône-Alpes', '63': 'Auvergne-Rhône-Alpes',
            '69': 'Auvergne-Rhône-Alpes', '73': 'Auvergne-Rhône-Alpes', '74': 'Auvergne-Rhône-Alpes',
            # ... [reste de la table de correspondance] ...
        }
        
        # Ajouter la colonne région
        df_temp['region'] = df_temp['departement'].map(region_map)
        
        # Agréger les données
        agg_data = df_temp.groupby('region').agg({
            'P16_POP': 'sum',
            value_col: 'mean' if value_col != 'APL' else lambda x: (df_temp['APL'] * df_temp['P16_POP']).sum() / df_temp['P16_POP'].sum()
        }).reset_index()
        
        # Renommer les colonnes
        agg_data.columns = ['territoire', 'population', value_col]
        
        # Charger le GeoJSON
        geojson_file = "regions-avec-outre-mer.geojson"
    
    # Charger le GeoJSON
    geojson_data = load_geojson(geojson_file)
    
    # Déterminer la plage de valeurs si non spécifiée
    if range_color is None:
        range_color = [agg_data[value_col].min(), agg_data[value_col].max()]
    
    # Créer la carte
    fig = px.choropleth_mapbox(
        agg_data, 
        geojson=geojson_data, 
        locations='territoire',
        featureidkey=f"properties.{'code' if geo_level == 'departement' else 'nom'}",
        color=value_col,
        color_continuous_scale=color_scale,
        range_color=range_color,
        mapbox_style="carto-positron",
        zoom=5 if geo_level == 'departement' else 4.5,
        center={"lat": 46.603354, "lon": 1.888334},
        opacity=0.8,
        labels={value_col: value_col},
        hover_data=["population"]
    )
    
    # Adapter le template de hover
    hover_template = f"<b>%{{location}}</b><br>{value_col}: %{{z:.2f}}<br>Population: %{{customdata[0]:,.0f}}"
    fig.update_traces(hovertemplate=hover_template)
    
    # Configuration de la figure
    fig.update_layout(
        title=title,
        margin={"r":0,"t":50,"l":0,"b":0},
        height=600
    )
    
    return fig

def view_mode_predictions(filtered_data):
    """
    Module optimisé pour la section Prévisions & Risques du dashboard Medical'IA
    """
    st.header("Prévisions & Risques de désertification médicale")
    
    st.markdown("""
    ### Modèle prédictif d'évolution de l'accès aux soins
    
    Notre modèle intègre des facteurs démographiques, économiques et territoriaux 
    pour identifier les communes à risque de devenir des déserts médicaux dans 
    les prochaines années.
    """)
    
    # Utiliser les données prétraitées avec prédictions
    data_risk = predict_future_desert_risk(filtered_data)
    
    # Structure en deux colonnes pour la section principale
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Carte principale du risque
        st.subheader("Carte du risque de désertification médicale")
        
        # Sélection du type de visualisation
        map_type = st.radio(
            "Type de visualisation",
            ["Score de risque", "APL projeté à 5 ans"],
            horizontal=True
        )
        
        # Création de la carte selon le type choisi
        with st.spinner("Génération de la carte d'analyse des risques..."):
            if map_type == "Score de risque":
                risk_map = create_risk_map(data_risk, "desert_risk_score", 
                                          "Risque de désertification médicale",
                                          color_scale="YlOrRd", 
                                          range_color=[0, 100])
            else:  # APL projeté
                risk_map = create_risk_map(data_risk, "projected_APL_5y", 
                                          "APL projeté à 5 ans",
                                          color_scale="RdYlGn", 
                                          range_color=[0, 5])
            
            st.plotly_chart(risk_map, use_container_width=True)
    
    with col2:
        # Métriques clés des prédictions
        st.subheader("Indicateurs de risque")
        
        # Calcul des métriques principales
        metrics = calculate_risk_metrics(data_risk)
        
        # Affichage des KPIs
        st.metric("Score de risque moyen", f"{metrics['avg_risk_score']:.1f}/100")
        st.metric("Communes à risque élevé", f"{metrics['high_risk_count']:,}".replace(',', ' '))
        st.metric("Population en zone à risque", f"{metrics['pop_at_risk']:,}".replace(',', ' '))
        st.metric("% de la population à risque", f"{metrics['pop_risk_percent']:.1f}%")
        
        # Distribution des scores de risque
        fig_dist = px.histogram(
            data_risk,
            x="desert_risk_score",
            nbins=20,
            title="Distribution des scores de risque",
            color_discrete_sequence=["#FF5533"],
            labels={"desert_risk_score": "Score de risque (0-100)"}
        )
        
        # Ajouter des lignes de référence
        fig_dist.add_vline(x=70, line_dash="dash", line_color="red", 
                          annotation_text="Risque élevé", annotation_position="top right")
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Section des facteurs de risque
    st.subheader("Facteurs de risque prédominants")
    
    # Analyse des facteurs principaux de risque
    risk_factors_data = analyze_risk_factors(data_risk)
    
    col3, col4 = st.columns([3, 2])
    
    with col3:
        # Graphique d'importance des facteurs
        st.plotly_chart(risk_factors_data["importance_chart"], use_container_width=True)
    
    with col4:
        # Explication des facteurs
        st.markdown("""
        ### Impact des facteurs sur le risque
        
        Les principaux facteurs influençant le risque de désertification médicale:
        
        **Facteurs à fort impact négatif:**
        - **Faible APL actuel** : Les zones déjà sous-dotées risquent de se dégrader davantage
        - **Population âgée** : Augmente les besoins médicaux tout en réduisant l'attractivité
        - **Faible densité** : Les zones moins peuplées attirent moins de professionnels de santé
        
        **Facteurs d'amélioration:**
        - **Niveau de vie élevé** : Les territoires plus aisés sont plus attractifs
        - **Équipements de santé** : L'infrastructure existante favorise l'installation
        - **Mobilité** : Un bon réseau de transport compense partiellement l'éloignement
        """)
    
    # Section des communes à surveiller
    st.subheader("Zones prioritaires à surveiller")
    
    # Trouver les communes à risque imminent de désertification
    high_risk_communes = get_high_risk_communes(data_risk)
    
    if not high_risk_communes.empty:
        # Carte des communes à risque avec tableau
        col5, col6 = st.columns([2, 3])
        
        with col5:
            # Tableau des communes à risque les plus peuplées
            st.markdown("### Top communes à risque")
            st.dataframe(high_risk_communes[['Commune', 'Département', 'Population', 
                                            'APL actuel', 'Score de risque', 'Facteur principal']])
        
        with col6:
            # Carte focalisée sur les communes à risque
            high_risk_map = create_focused_risk_map(high_risk_communes, data_risk)
            st.plotly_chart(high_risk_map, use_container_width=True)
    else:
        st.info("Aucune commune ne correspond aux critères de risque élevé dans les filtres actuels.")
    
    # Téléchargement des données de prédiction
    st.download_button(
        label="Télécharger les prédictions complètes (CSV)",
        data=convert_df_to_csv(data_risk[['CODGEO', 'Communes', 'APL', 'projected_APL_5y', 
                                         'desert_risk_score', 'risk_category', 
                                         'main_factor1', 'main_factor2']]),
        file_name='predictions_risque_desertification.csv',
        mime='text/csv'
    )


# Fonctions utilitaires pour le module
@st.cache_data
def predict_future_desert_risk(df):
    """
    Prédiction optimisée du risque futur de désert médical (échelle 0-100)
    """
    # Définition des facteurs de risque avec leurs poids
    risk_factors = {
        'APL': -0.5,                                     # Poids négatif: plus l'APL est bas, plus le risque est élevé
        '60+_pop_rate': 0.2,                             # Poids positif: plus la pop. âgée est élevée, plus le risque est élevé
        'density_area': -0.15,                           # Poids négatif: plus la densité est basse, plus le risque est élevé
        'median_living_standard': -0.15,                 # Poids négatif: plus le niveau de vie est bas, plus le risque est élevé
        'healthcare_education_establishments': -0.1,     # Poids négatif: moins d'établissements = plus de risque
        'city_social_amenities_rate': -0.1,              # Poids négatif: moins d'équipements sociaux = plus de risque
        'mobility_rate': -0.1                            # Poids négatif: moins de mobilité = plus de risque
    }
    
    # Copie du DataFrame pour éviter les modifications sur l'original
    df_risk = df.copy()
    
    # Gestion des valeurs manquantes pour chaque facteur
    for factor in risk_factors.keys():
        if factor in df_risk.columns:
            df_risk[factor] = df_risk[factor].fillna(df_risk[factor].median())
    
    # Initialisation du score de risque
    df_risk['desert_risk_score'] = 0
    
    # Calcul du score pour chaque facteur
    for factor, weight in risk_factors.items():
        if factor in df_risk.columns:
            # Normalisation Min-Max pour chaque facteur
            factor_min = df_risk[factor].min()
            factor_max = df_risk[factor].max()
            factor_range = factor_max - factor_min
            
            if factor_range > 0:  # Éviter la division par zéro
                # Les facteurs à poids négatifs sont inversés (valeurs faibles = risque élevé)
                if weight < 0:
                    df_risk[f"{factor}_norm"] = 1 - ((df_risk[factor] - factor_min) / factor_range)
                else:  # Facteurs à poids positif (valeurs élevées = risque élevé)
                    df_risk[f"{factor}_norm"] = (df_risk[factor] - factor_min) / factor_range
                
                # Contribution de chaque facteur au score total (valeur absolue du poids)
                df_risk['desert_risk_score'] += abs(weight) * df_risk[f"{factor}_norm"]
    
    # Normalisation du score final sur l'échelle 0-100
    score_min = df_risk['desert_risk_score'].min()
    score_max = df_risk['desert_risk_score'].max()
    if score_max > score_min:
        df_risk['desert_risk_score'] = (df_risk['desert_risk_score'] - score_min) / (score_max - score_min) * 100
    
    # Catégorisation du risque
    risk_conditions = [
        (df_risk['desert_risk_score'] >= 80),
        (df_risk['desert_risk_score'] >= 60) & (df_risk['desert_risk_score'] < 80),
        (df_risk['desert_risk_score'] >= 40) & (df_risk['desert_risk_score'] < 60),
        (df_risk['desert_risk_score'] >= 20) & (df_risk['desert_risk_score'] < 40),
        (df_risk['desert_risk_score'] < 20)
    ]
    
    risk_categories = [
        "Risque très élevé",
        "Risque élevé",
        "Risque modéré",
        "Risque faible",
        "Risque très faible"
    ]
    
    df_risk['risk_category'] = np.select(risk_conditions, risk_categories, default="Non évalué")
    
    # Prédiction simple de l'APL à 5 ans (diminution proportionnelle au risque)
    max_decrease = 0.5  # Diminution maximale de 50%
    df_risk['projected_APL_5y'] = df_risk['APL'] * (1 - ((df_risk['desert_risk_score'] / 100) * max_decrease))
    df_risk['projected_APL_5y'] = df_risk['projected_APL_5y'].clip(0, 5)  # Limiter entre 0 et 5
    
    # Déterminer les facteurs principaux de risque pour chaque commune
    factor_names = {
        'APL': 'Faible accessibilité actuelle',
        '60+_pop_rate': 'Population âgée importante',
        'density_area': 'Faible densité de population',
        'median_living_standard': 'Niveau de vie modeste',
        'healthcare_education_establishments': 'Manque d\'infrastructures de santé',
        'city_social_amenities_rate': 'Peu d\'équipements sociaux',
        'mobility_rate': 'Faible mobilité'
    }
    
    # Identifier les deux principaux facteurs pour chaque commune
    df_risk['main_factor1'] = ''
    df_risk['main_factor2'] = ''
    
    for idx in df_risk.index:
        factor_scores = {}
        for factor, weight in risk_factors.items():
            if f"{factor}_norm" in df_risk.columns:
                factor_scores[factor] = df_risk.loc[idx, f"{factor}_norm"] * abs(weight)
        
        sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_factors) > 0:
            df_risk.at[idx, 'main_factor1'] = factor_names.get(sorted_factors[0][0], sorted_factors[0][0])
        if len(sorted_factors) > 1:
            df_risk.at[idx, 'main_factor2'] = factor_names.get(sorted_factors[1][0], sorted_factors[1][0])
    
    # Ajouter la région et le département pour faciliter les analyses
    df_risk['Département'] = df_risk['CODGEO'].str[:2]
    
    return df_risk


@st.cache_data
def calculate_risk_metrics(df_risk):
    """Calcule les métriques clés des prédictions de risque"""
    metrics = {
        'high_risk_count': len(df_risk[df_risk['desert_risk_score'] >= 70]),
        'risk_percent': len(df_risk[df_risk['desert_risk_score'] >= 70]) / len(df_risk) * 100 if len(df_risk) > 0 else 0,
        'avg_risk_score': df_risk['desert_risk_score'].mean(),
        'pop_at_risk': df_risk[df_risk['desert_risk_score'] >= 70]['P16_POP'].sum(),
        'pop_risk_percent': df_risk[df_risk['desert_risk_score'] >= 70]['P16_POP'].sum() / df_risk['P16_POP'].sum() * 100 if df_risk['P16_POP'].sum() > 0 else 0,
        'new_deserts_count': len(df_risk[(df_risk['APL'] >= 2.5) & (df_risk['projected_APL_5y'] < 2.5)]),
        'deterioration_percent': len(df_risk[df_risk['projected_APL_5y'] < df_risk['APL']]) / len(df_risk) * 100 if len(df_risk) > 0 else 0
    }
    return metrics


@st.cache_data
def analyze_risk_factors(df_risk):
    """Analyse des facteurs de risque les plus impactants"""
    # Préparation des données
    risk_factor_importance = pd.DataFrame({
        'Facteur': [
            'APL actuel', 
            'Population âgée (60+)', 
            'Densité de population',
            'Niveau de vie médian',
            'Équipements de santé',
            'Équipements sociaux',
            'Mobilité'
        ],
        'Importance': [50, 20, 15, 15, 10, 10, 10],
        'Direction': ['négatif', 'positif', 'négatif', 'négatif', 'négatif', 'négatif', 'négatif']
    })
    
    # Créer le graphique d'importance des facteurs
    importance_chart = px.bar(
        risk_factor_importance,
        x='Importance',
        y='Facteur',
        orientation='h',
        color='Direction',
        color_discrete_map={'positif': 'indianred', 'négatif': 'steelblue'},
        labels={'Importance': 'Poids dans le modèle (%)', 'Facteur': ''},
        title="Importance relative des facteurs de risque",
        height=400
    )
    
    # Analyse des facteurs principaux par commune
    factor_counts = df_risk['main_factor1'].value_counts().reset_index()
    factor_counts.columns = ['Facteur', 'Nombre de communes']
    factor_counts['Pourcentage'] = (factor_counts['Nombre de communes'] / factor_counts['Nombre de communes'].sum() * 100).round(1)
    
    # Créer le graphique des facteurs principaux
    factors_chart = px.pie(
        factor_counts,
        values='Pourcentage',
        names='Facteur',
        title="Distribution des facteurs principaux de risque",
        hole=0.4
    )
    
    return {
        "importance_chart": importance_chart,
        "factors_chart": factors_chart,
        "factor_counts": factor_counts
    }


@st.cache_data
def get_high_risk_communes(df_risk, risk_threshold=70, min_population=500):
    """Identifie les communes à risque élevé pour le suivi prioritaire"""
    # Filtrer les communes à risque élevé avec une population minimale
    high_risk_df = df_risk[
        (df_risk['desert_risk_score'] >= risk_threshold) & 
        (df_risk['P16_POP'] >= min_population)
    ].copy()
    
    if high_risk_df.empty:
        return pd.DataFrame()
    
    # Prioriser les communes bien desservies actuellement mais à risque dans le futur
    potential_new_deserts = high_risk_df[high_risk_df['APL'] >= 2.5].copy()
    
    # Si aucune commune ne répond à ce critère, utiliser toutes les communes à risque
    if potential_new_deserts.empty:
        potential_new_deserts = high_risk_df.copy()
    
    # Trier par score de risque décroissant
    potential_new_deserts = potential_new_deserts.sort_values('desert_risk_score', ascending=False)
    
    # Limiter à 10 communes pour la clarté
    top_communes = potential_new_deserts.head(10)
    
    # Préparer le tableau final
    result_df = top_communes[['Communes', 'CODGEO', 'P16_POP', 'APL', 'desert_risk_score', 'main_factor1',
                             'projected_APL_5y', 'latitude_mairie', 'longitude_mairie']].copy()
    
    # Renommer et formater pour l'affichage
    result_df.columns = ['Commune', 'Code INSEE', 'Population', 'APL actuel', 'Score de risque', 
                         'Facteur principal', 'APL projeté', 'Latitude', 'Longitude']
    
    # Ajouter une colonne département
    result_df['Département'] = result_df['Code INSEE'].str[:2]
    
    # Arrondir les valeurs numériques
    for col in ['APL actuel', 'APL projeté', 'Score de risque']:
        if col in result_df.columns:
            result_df[col] = result_df[col].round(2)
    
    return result_df


@st.cache_data
def create_risk_map(df_risk, column, title, color_scale="YlOrRd", range_color=None):
    """Crée une carte optimisée des risques"""
    # Nettoyer les données (suppression des valeurs manquantes)
    df_clean = df_risk.dropna(subset=['latitude_mairie', 'longitude_mairie', column]).copy()
    
    # Limiter le nombre de points si nécessaire pour la performance
    max_points = 3000
    if len(df_clean) > max_points:
        # Échantillonnage stratifié
        if column == 'desert_risk_score':
            # Pour la carte de risque, prioriser les communes à haut risque
            high_risk = df_clean[df_clean['desert_risk_score'] >= 70]
            other = df_clean[df_clean['desert_risk_score'] < 70]
            
            # Garder toutes les communes à haut risque si possible
            if len(high_risk) <= max_points * 0.4:
                remaining_points = max_points - len(high_risk)
                sample_other = other.sample(min(remaining_points, len(other)), random_state=42)
                df_clean = pd.concat([high_risk, sample_other])
            else:
                # Échantillonner proportionnellement
                high_risk_sample = high_risk.sample(int(max_points * 0.4), random_state=42)
                other_sample = other.sample(int(max_points * 0.6), random_state=42)
                df_clean = pd.concat([high_risk_sample, other_sample])
        else:
            # Pour les autres cartes, échantillonnage basé sur d'autres critères
            if column == 'projected_APL_5y':
                # Prioriser les communes avec un APL projeté bas
                low_apl = df_clean[df_clean['projected_APL_5y'] < 2.5]
                other = df_clean[df_clean['projected_APL_5y'] >= 2.5]
                
                if len(low_apl) <= max_points * 0.4:
                    remaining_points = max_points - len(low_apl)
                    sample_other = other.sample(min(remaining_points, len(other)), random_state=42)
                    df_clean = pd.concat([low_apl, sample_other])
                else:
                    low_apl_sample = low_apl.sample(int(max_points * 0.4), random_state=42)
                    other_sample = other.sample(int(max_points * 0.6), random_state=42)
                    df_clean = pd.concat([low_apl_sample, other_sample])
            else:
                # Échantillonnage aléatoire stratifié par population
                df_clean = pd.concat([
                    df_clean.nlargest(max_points//5, 'P16_POP'),
                    df_clean.sample(min(max_points - max_points//5, len(df_clean) - max_points//5), random_state=42)
                ])
    
    # Adapter les paramètres en fonction de la colonne
    if column == 'desert_risk_score':
        colorbar_title = "Score de risque"
        if range_color is None:
            range_color = [0, 100]
        hover_template = "<b>%{text}</b><br>Score de risque: %{marker.color:.1f}/100<br>APL actuel: %{customdata[0]:.2f}<br>Catégorie: %{customdata[1]}"
        customdata = np.column_stack((df_clean['APL'], df_clean['risk_category']))
    elif column == 'projected_APL_5y':
        colorbar_title = "APL projeté"
        if range_color is None:
            range_color = [0, 5]
        hover_template = "<b>%{text}</b><br>APL actuel: %{customdata[0]:.2f}<br>APL projeté: %{marker.color:.2f}<br>Évolution: %{customdata[1]:.1f}%"
        customdata = np.column_stack((
            df_clean['APL'], 
            (df_clean['projected_APL_5y'] / df_clean['APL'] - 1) * 100
        ))
    else:
        colorbar_title = column
        if range_color is None:
            range_color = [df_clean[column].min(), df_clean[column].max()]
        hover_template = "<b>%{text}</b><br>Valeur: %{marker.color:.2f}"
        customdata = None
    
    # Créer la carte
    fig = go.Figure(data=go.Scattermapbox(
        lat=df_clean['latitude_mairie'],
        lon=df_clean['longitude_mairie'],
        mode='markers',
        marker=dict(
            size=8,
            color=df_clean[column],
            colorscale=color_scale,
            colorbar=dict(title=colorbar_title),
            cmin=range_color[0],
            cmax=range_color[1],
            opacity=0.8
        ),
        text=df_clean['Communes'],
        hovertemplate=hover_template,
        customdata=customdata
    ))
    
    # Configuration de la carte
    fig.update_layout(
        title=title,
        mapbox_style="carto-positron",
        mapbox=dict(
            center=dict(lat=46.603354, lon=1.888334),
            zoom=5
        ),
        height=600,
        margin={"r":0,"t":50,"l":0,"b":0},
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    return fig


# Correction de la fonction create_focused_risk_map
@st.cache_data
def create_focused_risk_map(high_risk_communes, data_risk):
    """Crée une carte focalisée sur les communes à risque élevé"""
    # Créer une carte avec toutes les communes en arrière-plan
    fig = go.Figure()
    
    # Correction ici: filtrer d'abord puis calculer la taille d'échantillon correcte
    filtered_data = data_risk.dropna(subset=['latitude_mairie', 'longitude_mairie'])
    sample_size = min(2000, len(filtered_data))  # Utiliser la longueur des données filtrées
    
    # Assurez-vous que sample_size est au moins 1 pour éviter les erreurs
    sample_size = max(1, sample_size)
    
    # Maintenant échantillonner les données filtrées
    background_data = filtered_data.sample(sample_size)
    
    fig.add_trace(go.Scattermapbox(
        lat=background_data['latitude_mairie'],
        lon=background_data['longitude_mairie'],
        mode='markers',
        marker=dict(
            size=5,
            color='lightgray',
            opacity=0.3
        ),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Reste du code inchangé...
    # Ajouter les communes à risque élevé
    if not high_risk_communes.empty:  # Vérifier si high_risk_communes n'est pas vide
        fig.add_trace(go.Scattermapbox(
            lat=high_risk_communes['Latitude'],
            lon=high_risk_communes['Longitude'],
            mode='markers',
            marker=dict(
                size=12,
                color=high_risk_communes['Score de risque'],
                colorscale='YlOrRd',
                colorbar=dict(title="Score de risque"),
                cmin=70,
                cmax=100,
                opacity=0.9
            ),
            text=high_risk_communes['Commune'],
            hovertemplate="<b>%{text}</b><br>APL actuel: %{customdata[0]:.2f}<br>Score de risque: %{marker.color:.1f}<br>Facteur principal: %{customdata[1]}",
            customdata=np.column_stack((high_risk_communes['APL actuel'], high_risk_communes['Facteur principal'])),
            name="Communes à surveiller"
        ))
    
    # Déterminer le centre de la carte
    if len(high_risk_communes) > 0:
        center_lat = high_risk_communes['Latitude'].mean()
        center_lon = high_risk_communes['Longitude'].mean()
        zoom = 5
    else:
        center_lat = 46.603354
        center_lon = 1.888334
        zoom = 5
    
    # Configuration de la carte
    fig.update_layout(
        title="Communes prioritaires à surveiller",
        mapbox_style="carto-positron",
        mapbox=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        height=400,
        margin={"r":0,"t":50,"l":0,"b":0},
        showlegend=False
    )
    
    return fig


def convert_df_to_csv(df):
    """Convertit un DataFrame en CSV pour téléchargement"""
    return df.to_csv(index=False).encode('utf-8')


# Fonction principale pour intégrer au dashboard principal
def select_view_mode():
    if view_mode == "Prévisions & Risques":
        view_mode_predictions(filtered_data)

def view_mode_clusters(filtered_data):
    """
    Module complet pour les clusters de communes avec visualisations avancées
    """
    st.header("Zones d'accessibilité médicale similaire")
    
    st.markdown("""
    Cette analyse identifie les principaux regroupements de communes ayant des caractéristiques similaires d'accès aux soins.
    Nous utilisons une méthodologie avancée qui combine classification géographique et indicateurs socio-démographiques.
    """)
    
    # Définir les catégories d'APL avec des seuils précis basés sur les recommandations médicales
    apl_categories = [
        {"name": "Déserts médicaux critiques", "min": 0, "max": 1.5, "color": "darkred", "description": "Accès aux soins très difficile, situation urgente"},
        {"name": "Déserts médicaux", "min": 1.5, "max": 2.5, "color": "red", "description": "Accès aux soins insuffisant, actions nécessaires"},
        {"name": "Zones sous-équipées", "min": 2.5, "max": 3.5, "color": "orange", "description": "Accès limité, vigilance requise"},
        {"name": "Zones bien équipées", "min": 3.5, "max": 4.5, "color": "lightgreen", "description": "Accès satisfaisant aux soins médicaux"},
        {"name": "Zones très bien équipées", "min": 4.5, "max": 10, "color": "green", "description": "Excellent accès aux soins médicaux"}
    ]
    
    # Fonction améliorée pour catégoriser les communes
    @st.cache_data
    def prepare_zoned_data(data):
        """
        Catégorisation des communes avec métriques avancées
        """
        data_zoned = data.copy()
        
        # Ajout des catégories de zone
        data_zoned['zone_type'] = None
        data_zoned['zone_color'] = None
        data_zoned['zone_description'] = None
        
        for cat in apl_categories:
            mask = (data_zoned['APL'] >= cat['min']) & (data_zoned['APL'] < cat['max'])
            data_zoned.loc[mask, 'zone_type'] = cat['name']
            data_zoned.loc[mask, 'zone_color'] = cat['color']
            data_zoned.loc[mask, 'zone_description'] = cat['description']
        
        # Identifier le département et la région
        data_zoned['departement'] = data_zoned['CODGEO'].str[:2]
        
        # Table de correspondance département-région
        region_map = {
            '01': 'Auvergne-Rhône-Alpes', '03': 'Auvergne-Rhône-Alpes', '07': 'Auvergne-Rhône-Alpes', 
            '15': 'Auvergne-Rhône-Alpes', '26': 'Auvergne-Rhône-Alpes', '38': 'Auvergne-Rhône-Alpes',
            '42': 'Auvergne-Rhône-Alpes', '43': 'Auvergne-Rhône-Alpes', '63': 'Auvergne-Rhône-Alpes',
            '69': 'Auvergne-Rhône-Alpes', '73': 'Auvergne-Rhône-Alpes', '74': 'Auvergne-Rhône-Alpes',
            '21': 'Bourgogne-Franche-Comté', '25': 'Bourgogne-Franche-Comté', '39': 'Bourgogne-Franche-Comté',
            '58': 'Bourgogne-Franche-Comté', '70': 'Bourgogne-Franche-Comté', '71': 'Bourgogne-Franche-Comté',
            '89': 'Bourgogne-Franche-Comté', '90': 'Bourgogne-Franche-Comté',
            '22': 'Bretagne', '29': 'Bretagne', '35': 'Bretagne', '56': 'Bretagne',
            '18': 'Centre-Val de Loire', '28': 'Centre-Val de Loire', '36': 'Centre-Val de Loire',
            '37': 'Centre-Val de Loire', '41': 'Centre-Val de Loire', '45': 'Centre-Val de Loire',
            '2A': 'Corse', '2B': 'Corse',
            '08': 'Grand Est', '10': 'Grand Est', '51': 'Grand Est', '52': 'Grand Est',
            '54': 'Grand Est', '55': 'Grand Est', '57': 'Grand Est', '67': 'Grand Est',
            '68': 'Grand Est', '88': 'Grand Est',
            '02': 'Hauts-de-France', '59': 'Hauts-de-France', '60': 'Hauts-de-France',
            '62': 'Hauts-de-France', '80': 'Hauts-de-France',
            '75': 'Île-de-France', '77': 'Île-de-France', '78': 'Île-de-France', '91': 'Île-de-France',
            '92': 'Île-de-France', '93': 'Île-de-France', '94': 'Île-de-France', '95': 'Île-de-France',
            '14': 'Normandie', '27': 'Normandie', '50': 'Normandie', '61': 'Normandie', '76': 'Normandie',
            '16': 'Nouvelle-Aquitaine', '17': 'Nouvelle-Aquitaine', '19': 'Nouvelle-Aquitaine',
            '23': 'Nouvelle-Aquitaine', '24': 'Nouvelle-Aquitaine', '33': 'Nouvelle-Aquitaine',
            '40': 'Nouvelle-Aquitaine', '47': 'Nouvelle-Aquitaine', '64': 'Nouvelle-Aquitaine',
            '79': 'Nouvelle-Aquitaine', '86': 'Nouvelle-Aquitaine', '87': 'Nouvelle-Aquitaine',
            '09': 'Occitanie', '11': 'Occitanie', '12': 'Occitanie', '30': 'Occitanie',
            '31': 'Occitanie', '32': 'Occitanie', '34': 'Occitanie', '46': 'Occitanie',
            '48': 'Occitanie', '65': 'Occitanie', '66': 'Occitanie', '81': 'Occitanie', '82': 'Occitanie',
            '44': 'Pays de la Loire', '49': 'Pays de la Loire', '53': 'Pays de la Loire',
            '72': 'Pays de la Loire', '85': 'Pays de la Loire',
            '04': 'Provence-Alpes-Côte d\'Azur', '05': 'Provence-Alpes-Côte d\'Azur',
            '06': 'Provence-Alpes-Côte d\'Azur', '13': 'Provence-Alpes-Côte d\'Azur',
            '83': 'Provence-Alpes-Côte d\'Azur', '84': 'Provence-Alpes-Côte d\'Azur',
            '971': 'Outre-Mer', '972': 'Outre-Mer', '973': 'Outre-Mer', '974': 'Outre-Mer', '976': 'Outre-Mer',
            '975': 'Outre-Mer', '977': 'Outre-Mer', '978': 'Outre-Mer', '986': 'Outre-Mer', '987': 'Outre-Mer',
            '988': 'Outre-Mer'
        }
        
        data_zoned['region'] = data_zoned['departement'].map(region_map)
        
        # Ajout de métriques pour l'analyse avancée
        if '60+_pop_rate' in data_zoned.columns:
            # Catégorisation démographique
            data_zoned['demographic_profile'] = 'Équilibrée'
            data_zoned.loc[data_zoned['60+_pop_rate'] > 30, 'demographic_profile'] = 'Vieillissante'
            data_zoned.loc[data_zoned['60+_pop_rate'] < 15, 'demographic_profile'] = 'Jeune'
        
        if 'density_area' in data_zoned.columns:
            # Catégorisation urbain/rural
            data_zoned['urban_rural'] = 'Périurbain'
            data_zoned.loc[data_zoned['density_area'] < 50, 'urban_rural'] = 'Rural'
            data_zoned.loc[data_zoned['density_area'] > 500, 'urban_rural'] = 'Urbain'
            data_zoned.loc[data_zoned['density_area'] > 2000, 'urban_rural'] = 'Très urbain'
        
        if 'median_living_standard' in data_zoned.columns:
            # Niveau économique
            median_income = data_zoned['median_living_standard'].median()
            data_zoned['economic_level'] = 'Moyen'
            data_zoned.loc[data_zoned['median_living_standard'] < median_income * 0.8, 'economic_level'] = 'Modeste'
            data_zoned.loc[data_zoned['median_living_standard'] > median_income * 1.2, 'economic_level'] = 'Aisé'
        
        # Ajouter une étiquette composée pour classification avancée
        if 'urban_rural' in data_zoned.columns and 'demographic_profile' in data_zoned.columns:
            data_zoned['composite_label'] = data_zoned['urban_rural'] + ' ' + data_zoned['demographic_profile']
            if 'economic_level' in data_zoned.columns:
                data_zoned['composite_label'] += ' ' + data_zoned['economic_level']
        
        return data_zoned
    
    @st.cache_data
    def prepare_advanced_clusters(data_zoned, min_communes=10):
        """
        Préparation des clusters territoriaux avancés
        """
        # Grouper par département et type de zone
        dept_zones = data_zoned.groupby(['departement', 'zone_type']).agg({
            'CODGEO': 'count',
            'P16_POP': 'sum',
            'APL': 'mean',
            'latitude_mairie': 'mean',
            'longitude_mairie': 'mean',
            'zone_color': 'first',
            'zone_description': 'first'
        }).reset_index()
        
        # Enrichir avec des informations supplémentaires
        if '60+_pop_rate' in data_zoned.columns:
            pop_60_agg = data_zoned.groupby(['departement', 'zone_type'])['60+_pop_rate'].mean().reset_index()
            dept_zones = dept_zones.merge(pop_60_agg, on=['departement', 'zone_type'])
        
        if 'density_area' in data_zoned.columns:
            density_agg = data_zoned.groupby(['departement', 'zone_type'])['density_area'].mean().reset_index()
            dept_zones = dept_zones.merge(density_agg, on=['departement', 'zone_type'])
        
        if 'demographic_profile' in data_zoned.columns:
            # Obtenir le profil démographique le plus fréquent pour chaque groupe
            demo_agg = data_zoned.groupby(['departement', 'zone_type'])['demographic_profile'].agg(
                lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else None
            ).reset_index()
            dept_zones = dept_zones.merge(demo_agg, on=['departement', 'zone_type'])
        
        if 'urban_rural' in data_zoned.columns:
            # Obtenir le type urbain/rural le plus fréquent pour chaque groupe
            urban_agg = data_zoned.groupby(['departement', 'zone_type'])['urban_rural'].agg(
                lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else None
            ).reset_index()
            dept_zones = dept_zones.merge(urban_agg, on=['departement', 'zone_type'])
        
        # Créer une typologie plus détaillée des zones
        if 'demographic_profile' in dept_zones.columns and 'urban_rural' in dept_zones.columns:
            dept_zones['zone_typography'] = dept_zones['zone_type'] + ' - ' + dept_zones['urban_rural'] + ' ' + dept_zones['demographic_profile']
        else:
            dept_zones['zone_typography'] = dept_zones['zone_type']
        
        # Renommer les colonnes pour plus de clarté
        dept_zones.columns = ['Département', 'Type de zone', 'Nombre de communes', 
                             'Population', 'APL moyen', 'Latitude', 'Longitude', 'Couleur',
                             'Description'] + list(dept_zones.columns[9:])
        
        # Filtrer les zones significatives et trier
        significant_zones = dept_zones[dept_zones['Nombre de communes'] >= min_communes]
        significant_zones = significant_zones.sort_values('Population', ascending=False)
        
        return significant_zones
    
    # Création des onglets pour améliorer l'organisation de l'interface
    tab1, tab2 = st.tabs(["Vue d'ensemble", "Explorer par zone"])
    
    # Préparer les données
    data_zoned = prepare_zoned_data(filtered_data)
    significant_zones = prepare_advanced_clusters(data_zoned)
    
    with tab1:
        st.subheader("Principales zones d'accessibilité médicale identifiées")
        
        # Statistiques générales sur les types de zones
        @st.cache_data
        def calculate_zone_stats(zones):
            stats = zones.groupby('Type de zone').agg({
                'Nombre de communes': 'sum',
                'Population': 'sum',
                'APL moyen': 'mean'
            }).reset_index()
            
            # Formater la population
            stats['Population'] = stats['Population'].apply(lambda x: f"{int(x):,}".replace(',', ' '))
            stats['APL moyen'] = stats['APL moyen'].round(2)
            
            # Réordonner selon la sévérité
            order = [cat["name"] for cat in apl_categories]
            stats['sort_order'] = stats['Type de zone'].map({zone: i for i, zone in enumerate(order)})
            stats = stats.sort_values('sort_order').drop('sort_order', axis=1)
            
            # Calcul du pourcentage de la population par type
            total_pop = sum([int(pop.replace(' ', '')) for pop in stats['Population']])
            stats['% de la population'] = stats['Population'].apply(
                lambda x: f"{int(int(x.replace(' ', '')) / total_pop * 100)}%"
            )
            
            return stats
        
        zone_stats = calculate_zone_stats(significant_zones)
        
        # Affichage avec une mise en forme améliorée
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown("### Répartition par type de zone")
            st.table(zone_stats)
            
            # Explication des types de zones
            with st.expander("Comprendre les types de zones"):
                for cat in apl_categories:
                    st.markdown(f"**{cat['name']}** ({cat['min']}-{cat['max']} APL): {cat['description']}")
        
        with col2:
            st.markdown("### Carte des principales zones homogènes")
            
            # Contrôles avancés pour la carte
            col_controls1, col_controls2 = st.columns(2)
            with col_controls1:
                # Ajouter un filtre pour le type de zone
                zone_type_filter = st.multiselect(
                    "Filtrer par type de zone",
                    options=significant_zones['Type de zone'].unique(),
                    default=significant_zones['Type de zone'].unique()
                )
            
            with col_controls2:
                # Contrôle du nombre de zones à afficher
                max_zones_to_display = st.slider(
                    "Nombre de zones à afficher", 
                    min_value=10, 
                    max_value=100, 
                    value=30,
                    help="Ajuster pour équilibrer détail et performance"
                )
            
            # Filtrer les zones selon les sélections
            display_zones = significant_zones[significant_zones['Type de zone'].isin(zone_type_filter)]
            display_zones = display_zones.head(max_zones_to_display)
            
            # Fonction améliorée pour créer la carte
            @st.cache_data
            def create_enhanced_zones_map(zones):
                fig = go.Figure()
                
                # Ajouter chaque zone comme un cercle proportionnel à sa population
                for _, zone in zones.iterrows():
                    # Taille proportionnelle à la racine carrée de la population, mais avec limites
                    population_scale = np.sqrt(zone['Population']) / 100
                    radius = max(5, min(30, population_scale))
                    
                    # Texte enrichi pour le hover
                    hover_text = f"""
                    <b>{zone['Type de zone']} - Dept {zone['Département']}</b><br>
                    <i>{zone['Description']}</i><br>
                    Communes: {zone['Nombre de communes']}<br>
                    Population: {int(zone['Population']):,}<br>
                    APL moyen: {zone['APL moyen']:.2f}
                    """
                    
                    # Ajouter des informations supplémentaires si disponibles
                    if 'demographic_profile' in zone and not pd.isna(zone['demographic_profile']):
                        hover_text += f"<br>Profil: {zone['demographic_profile']}"
                    
                    if 'urban_rural' in zone and not pd.isna(zone['urban_rural']):
                        hover_text += f"<br>Type: {zone['urban_rural']}"
                    
                    # Remplacer les virgules dans les nombres formatés
                    hover_text = hover_text.replace(',', ' ')
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=[zone['Latitude']],
                        lon=[zone['Longitude']],
                        mode='markers',
                        marker=dict(
                            size=radius,
                            color=zone['Couleur'],
                            opacity=0.7
                        ),
                        text=[hover_text],
                        hoverinfo='text',
                        name=f"{zone['Type de zone']} - Dept {zone['Département']}"
                    ))
                
                # Configuration améliorée de la carte
                fig.update_layout(
                    title="Principales zones d'accessibilité médicale similaire",
                    mapbox_style="carto-positron",
                    mapbox=dict(
                        center=dict(lat=46.603354, lon=1.888334),
                        zoom=5
                    ),
                    height=700,
                    margin={"r":0,"t":50,"l":0,"b":0},
                    showlegend=False,
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_family="Arial"
                    )
                )
                
                return fig
            
            # Afficher la carte
            with st.spinner("Génération de la carte des zones..."):
                zone_map = create_enhanced_zones_map(display_zones)
                st.plotly_chart(zone_map, use_container_width=True)
        
        # Analyse des regroupements territoriaux
        st.subheader("Distribution des zones par région")
        
        @st.cache_data
        def analyze_region_distribution(zones, data_zoned):
            # Ajouter la région à chaque zone départementale
            zones_with_region = zones.copy()
            
            # Map des départements aux régions
            dept_to_region = data_zoned[['departement', 'region']].drop_duplicates()
            dept_to_region_map = dict(zip(dept_to_region['departement'], dept_to_region['region']))
            
            zones_with_region['Région'] = zones_with_region['Département'].map(dept_to_region_map)
            
            # Compter les types de zones par région
            region_analysis = zones_with_region.groupby(['Région', 'Type de zone']).agg({
                'Nombre de communes': 'sum',
                'Population': 'sum'
            }).reset_index()
            
            # Pivoter pour avoir les types de zones en colonnes
            region_pivot = region_analysis.pivot_table(
                index='Région',
                columns='Type de zone',
                values='Nombre de communes',
                fill_value=0
            ).reset_index()
            
            return region_analysis, region_pivot
        
        region_analysis, region_pivot = analyze_region_distribution(significant_zones, data_zoned)
        
        # Heatmap des zones par région avec palette rouge-vert
        fig = px.imshow(
            region_pivot.iloc[:, 1:],
            x=region_pivot.columns[1:],
            y=region_pivot['Région'],
            color_continuous_scale='RdYlGn',  # Rouge vers vert
            labels=dict(x="Type de zone", y="Région", color="Nombre de communes"),
            title="Répartition des types de zones par région",
            height=500
        )
        
        fig.update_layout(
            xaxis_title="Type de zone",
            yaxis_title="Région",
            margin={"r":0,"t":50,"l":0,"b":0}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Explorer une zone spécifique")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sélection du type de zone
            zone_type_options = significant_zones['Type de zone'].unique()
            selected_zone_type = st.selectbox(
                "Sélectionner un type de zone:",
                options=zone_type_options
            )
        
        with col2:
            # Filtrer les zones par type sélectionné
            zones_of_type = significant_zones[significant_zones['Type de zone'] == selected_zone_type]
            
            # Sélection du département
            dept_options = zones_of_type['Département'].unique()
            selected_dept = st.selectbox(
                f"Sélectionner un département avec des {selected_zone_type.lower()}:",
                options=dept_options
            )
        
        # Filtrer pour le département et type de zone sélectionnés
        selected_zone = zones_of_type[zones_of_type['Département'] == selected_dept].iloc[0]
        
        # Obtenir toutes les communes de cette zone
        zone_communes = data_zoned[
            (data_zoned['departement'] == selected_dept) & 
            (data_zoned['zone_type'] == selected_zone_type)
        ]
        
        # Afficher les détails de la zone de manière plus attrayante
        st.markdown(f"## {selected_zone_type} du département {selected_dept}")
        st.markdown(f"*{selected_zone['Description']}*")
        
        # Métriques clés
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Communes", f"{int(selected_zone['Nombre de communes']):,}".replace(',', ' '))
        
        with col2:
            st.metric("Population", f"{int(selected_zone['Population']):,}".replace(',', ' '))
        
        with col3:
            st.metric("APL moyen", f"{selected_zone['APL moyen']:.2f}")
        
        with col4:
            # Vérifier si la colonne existe avant d'afficher
            if 'demographic_profile' in selected_zone and not pd.isna(selected_zone['demographic_profile']):
                st.metric("Profil", selected_zone['demographic_profile'])
            elif 'urban_rural' in selected_zone and not pd.isna(selected_zone['urban_rural']):
                st.metric("Type", selected_zone['urban_rural'])
        
        # Analyse avancée des communes de la zone
        st.markdown("### Caractéristiques détaillées des communes")
        
        # Analyses démographiques et socio-économiques si les données sont disponibles
        if '60+_pop_rate' in zone_communes.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Répartition par âge
                age_data = pd.DataFrame({
                    'Tranche d\'âge': ['0-14 ans', '15-59 ans', '60+ ans'],
                    'Pourcentage': [
                        zone_communes['0_14_pop_rate'].mean(),
                        zone_communes['15_59_pop_rate'].mean(),
                        zone_communes['60+_pop_rate'].mean()
                    ]
                })
                
                fig = px.pie(
                    age_data,
                    values='Pourcentage',
                    names='Tranche d\'âge',
                    title=f"Répartition par âge - {selected_zone_type} (Dept {selected_dept})",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Distribution démographique si catégorisation disponible
                if 'demographic_profile' in zone_communes.columns:
                    demo_counts = zone_communes['demographic_profile'].value_counts().reset_index()
                    demo_counts.columns = ['Profil démographique', 'Nombre de communes']
                    
                    fig = px.bar(
                        demo_counts,
                        x='Profil démographique',
                        y='Nombre de communes',
                        title=f"Profils démographiques - {selected_zone_type} (Dept {selected_dept})",
                        color='Profil démographique',
                        color_discrete_sequence=px.colors.qualitative.Safe
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Caractéristiques territoriales
        if 'urban_rural' in zone_communes.columns and 'density_area' in zone_communes.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution urbain/rural
                urban_counts = zone_communes['urban_rural'].value_counts().reset_index()
                urban_counts.columns = ['Type territorial', 'Nombre de communes']
                
                fig = px.bar(
                    urban_counts,
                    x='Type territorial',
                    y='Nombre de communes',
                    title=f"Types territoriaux - {selected_zone_type} (Dept {selected_dept})",
                    color='Type territorial',
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Distribution de la densité
                fig = px.histogram(
                    zone_communes,
                    x='density_area',
                    nbins=20,
                    title=f"Distribution des densités - {selected_zone_type} (Dept {selected_dept})",
                    color_discrete_sequence=['blue']
                )
                
                fig.update_layout(
                    xaxis_title="Densité (hab/km²)",
                    yaxis_title="Nombre de communes"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Carte des communes avec limitation du nombre de points
        st.markdown("### Carte des communes de la zone")
        
        # Options avancées pour la carte
        col1, col2 = st.columns(2)
        
        with col1:
            max_communes_to_display = st.slider(
                "Nombre maximum de communes à afficher", 
                min_value=10, 
                max_value=300, 
                value=100,
                help="Ajuster pour équilibrer détail et performance"
            )
        
        with col2:
            color_var = st.selectbox(
                "Colorer par variable",
                options=["APL", "Population", "Densité"] if 'density_area' in zone_communes.columns else ["APL", "Population"],
                index=0
            )
        
        @st.cache_data
        def create_enhanced_commune_map(communes, max_points, zone_type, dept, color_var):
            # Échantillonnage stratifié pour garder une représentation correcte
            if len(communes) > max_points:
                # Assurer que les communes importantes sont incluses
                top_communes = communes.nlargest(max_points//5, 'P16_POP')
                rest_sample = communes[~communes.index.isin(top_communes.index)].sample(
                    min(max_points - len(top_communes), len(communes) - len(top_communes)),
                    random_state=42
                )
                display_communes = pd.concat([top_communes, rest_sample])
            else:
                display_communes = communes
            
            # Déterminer la variable de coloration
            if color_var == "APL":
                color_col = 'APL'
                colorscale = 'RdYlGn'
                colorbar_title = "APL"
                cmin = 1
                cmax = 5
            elif color_var == "Population":
                color_col = 'P16_POP'
                colorscale = 'Viridis'
                colorbar_title = "Population"
                cmin = None
                cmax = None
            elif color_var == "Densité" and 'density_area' in display_communes.columns:
                color_col = 'density_area'
                colorscale = 'Blues'
                colorbar_title = "Densité (hab/km²)"
                cmin = None
                cmax = None
            else:
                color_col = 'APL'
                colorscale = 'RdYlGn'
                colorbar_title = "APL"
                cmin = 1
                cmax = 5
            
            # Créer une carte focalisée sur la zone échantillonnée
            zone_map = go.Figure()
            
            # Adapter le template de hover selon les données disponibles
            hover_template = "<b>%{text}</b><br>"
            hover_template += f"{colorbar_title}: %{{marker.color}}"
            
            if color_col != 'P16_POP':
                hover_template += "<br>Population: %{customdata[0]:,.0f}"
                custom_data = [display_communes['P16_POP']]
            else:
                custom_data = []
            
            # Ajouter des données supplémentaires au hover si disponibles
            if 'demographic_profile' in display_communes.columns:
                hover_template += "<br>Profil: %{customdata[" + str(len(custom_data)) + "]}"
                custom_data.append(display_communes['demographic_profile'])
            
            if 'urban_rural' in display_communes.columns:
                hover_template += "<br>Type: %{customdata[" + str(len(custom_data)) + "]}"
                custom_data.append(display_communes['urban_rural'])
            
            # Combiner les données personnalisées
            if custom_data:
                customdata = np.column_stack(custom_data)
            else:
                customdata = None
            
            # Ajouter les marqueurs des communes à la carte
            zone_map.add_trace(go.Scattermapbox(
                lat=display_communes['latitude_mairie'],
                lon=display_communes['longitude_mairie'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=display_communes[color_col],
                    colorscale=colorscale,
                    colorbar=dict(title=colorbar_title),
                    cmin=cmin,
                    cmax=cmax,
                    opacity=0.8
                ),
                text=display_communes['Communes'],
                hovertemplate=hover_template,
                customdata=customdata,
                name=f"{zone_type} - Département {dept}"
            ))
            
            # Déterminer les coordonnées du centre en utilisant la médiane (plus robuste aux outliers)
            lat_center = display_communes['latitude_mairie'].median()
            lon_center = display_communes['longitude_mairie'].median()
            
            # Configuration avancée de la carte
            zone_map.update_layout(
                title=f"{zone_type} - Département {dept}",
                mapbox_style="carto-positron",
                mapbox=dict(
                    center=dict(lat=lat_center, lon=lon_center),
                    zoom=8
                ),
                height=600,
                margin={"r":0,"t":50,"l":0,"b":0},
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial"
                )
            )
            
            return zone_map, len(display_communes)
        
        # Créer et afficher la carte
        with st.spinner("Génération de la carte des communes..."):
            commune_map, displayed_count = create_enhanced_commune_map(
                zone_communes, 
                max_communes_to_display,
                selected_zone_type,
                selected_dept,
                color_var
            )
            
            if len(zone_communes) > displayed_count:
                st.info(f"Affichage d'un échantillon représentatif de {displayed_count} communes sur {len(zone_communes)} pour améliorer les performances.")
            
            st.plotly_chart(commune_map, use_container_width=True)
        
        # Principales communes de la zone (10 plus grandes)
        st.markdown("### Principales communes de la zone")
        
        # Trier par population décroissante
        top_communes = zone_communes.sort_values('P16_POP', ascending=False).head(10)
        
        # Tableau des principales communes avec plus d'informations
        columns_to_display = ['Communes', 'P16_POP', 'APL']
        display_columns = ['Commune', 'Population', 'APL']
        
        # Ajouter des colonnes conditionnellement si elles existent
        if 'density_area' in zone_communes.columns:
            columns_to_display.append('density_area')
            display_columns.append('Densité')
        
        if '60+_pop_rate' in zone_communes.columns:
            columns_to_display.append('60+_pop_rate')
            display_columns.append('% 60+ ans')
        
        if 'median_living_standard' in zone_communes.columns:
            columns_to_display.append('median_living_standard')
            display_columns.append('Niveau de vie')
        
        if 'healthcare_education_establishments' in zone_communes.columns:
            columns_to_display.append('healthcare_education_establishments')
            display_columns.append('Éts santé/éducation')
        
        communes_display = top_communes[columns_to_display].reset_index(drop=True)
        communes_display.columns = display_columns
        
        # Formater les valeurs numériques
        for col in communes_display.select_dtypes(include=['float']).columns:
            if col in ['APL', '% 60+ ans']:
                communes_display[col] = communes_display[col].round(2)
            elif col == 'Niveau de vie':
                communes_display[col] = communes_display[col].round(0).astype(int)
            elif col == 'Densité':
                communes_display[col] = communes_display[col].round(1)
        
        st.dataframe(communes_display)
        
        # Option pour télécharger les données complètes de la zone
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv = convert_df(zone_communes[columns_to_display])
        
        st.download_button(
            label=f"Télécharger les données de la zone (CSV)",
            data=csv,
            file_name=f'{selected_zone_type.replace(" ", "_").lower()}_{selected_dept}.csv',
            mime='text/csv',
        )
        
        # Recommandations spécifiques basées sur le type de zone
        st.markdown("### Recommandations stratégiques")
        
        # Définir des recommandations par type de zone
        recommendations = {
            "Déserts médicaux critiques": [
                "Développer des centres de soins d'urgence mobiles",
                "Mettre en place des incitations financières exceptionnelles pour l'installation",
                "Déployer des solutions de télémédecine d'urgence",
                "Élaborer un plan d'action territorial prioritaire"
            ],
            "Déserts médicaux": [
                "Créer des maisons de santé pluridisciplinaires",
                "Proposer des aides à l'installation pour les nouveaux praticiens",
                "Établir des partenariats avec les facultés de médecine",
                "Développer le transport médical à la demande"
            ],
            "Zones sous-équipées": [
                "Anticiper les départs en retraite des médecins actuels",
                "Diversifier l'offre de soins (spécialistes, paramédicaux)",
                "Améliorer l'attractivité du territoire pour les professionnels",
                "Intégrer la planification médicale dans les projets urbains"
            ],
            "Zones bien équipées": [
                "Maintenir le niveau d'équipement actuel",
                "Favoriser une répartition équilibrée des spécialités",
                "Développer des pôles d'excellence médicale",
                "Optimiser la coordination entre professionnels"
            ],
            "Zones très bien équipées": [
                "Promouvoir l'innovation médicale",
                "Étendre la couverture vers les zones périphériques moins bien desservies",
                "Servir de centre de référence et de formation",
                "Anticiper l'évolution des besoins démographiques futurs"
            ]
        }
        
        # Recommandations spécifiques selon le profil démographique
        demographic_recommendations = {
            "Vieillissante": [
                "Développer des services de maintien à domicile",
                "Renforcer la présence de gériatres et spécialistes des maladies chroniques",
                "Mettre en place des navettes médicales dédiées aux seniors",
                "Créer des programmes de prévention ciblés pour les seniors"
            ],
            "Équilibrée": [
                "Assurer une offre de soins diversifiée pour tous les âges",
                "Développer des centres de santé familiaux",
                "Promouvoir l'éducation à la santé dans les écoles et les entreprises",
                "Équilibrer les services de pédiatrie et de gériatrie"
            ],
            "Jeune": [
                "Renforcer l'offre pédiatrique et obstétrique",
                "Développer des services de planification familiale",
                "Mettre en place des programmes de santé scolaire renforcés",
                "Créer des centres de soins adaptés aux jeunes familles"
            ]
        }
        
        # Recommandations spécifiques selon le profil territorial
        territorial_recommendations = {
            "Rural": [
                "Déployer des cabinets médicaux mobiles",
                "Développer les solutions de télémédecine",
                "Mettre en place des incitations spécifiques pour zones rurales",
                "Créer des maisons de santé inter-communales"
            ],
            "Périurbain": [
                "Renforcer les connexions avec les centres médicaux urbains",
                "Développer des centres de santé de proximité",
                "Optimiser les transports en commun vers les pôles médicaux",
                "Créer des antennes de spécialistes à temps partiel"
            ],
            "Urbain": [
                "Assurer une répartition équilibrée dans tous les quartiers",
                "Développer des pôles de spécialités complémentaires",
                "Renforcer la coordination hôpital-ville",
                "Adapter l'offre aux spécificités socio-démographiques des quartiers"
            ],
            "Très urbain": [
                "Optimiser l'accessibilité des centres de soins existants",
                "Développer des centres de soins non programmés pour désengorger les urgences",
                "Améliorer la coordination des acteurs médicaux nombreux",
                "Adapter l'offre aux populations précaires et aux disparités intra-urbaines"
            ]
        }
        
        # Afficher les recommandations adaptées au type de zone
        if selected_zone_type in recommendations:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### Recommandations pour {selected_zone_type}")
                for rec in recommendations[selected_zone_type]:
                    st.markdown(f"• {rec}")
            
            with col2:
                # Recommandations démographiques si le profil est disponible
                if 'demographic_profile' in selected_zone and not pd.isna(selected_zone['demographic_profile']):
                    profile = selected_zone['demographic_profile']
                    if profile in demographic_recommendations:
                        st.markdown(f"#### Recommandations pour profil {profile}")
                        for rec in demographic_recommendations[profile]:
                            st.markdown(f"• {rec}")
                
                # Ou recommandations territoriales si le type est disponible
                elif 'urban_rural' in selected_zone and not pd.isna(selected_zone['urban_rural']):
                    territory = selected_zone['urban_rural']
                    if territory in territorial_recommendations:
                        st.markdown(f"#### Recommandations pour zone {territory}")
                        for rec in territorial_recommendations[territory]:
                            st.markdown(f"• {rec}")


def generate_report_pdf(data, territory_level, territory_name, include_sections, include_recommendations=True):
    """
    Génère un rapport PDF pour un territoire donné
    
    Args:
        data: DataFrame contenant les données
        territory_level: Niveau territorial ('region', 'departement', 'commune')
        territory_name: Nom du territoire sélectionné
        include_sections: Dict avec les sections à inclure
        include_recommendations: Booléen pour inclure des recommandations
    
    Returns:
        BytesIO contenant le PDF généré
    """
    # Imports explicites pour s'assurer que tout est disponible
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from io import BytesIO
    from datetime import datetime
    import tempfile
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.platypus import PageBreak, ListFlowable, ListItem
    from reportlab.lib.units import inch, cm
    
    # Créer un buffer pour stocker le PDF
    buffer = BytesIO()
    
    # Configuration du document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
        title=f"Rapport Medical'IA - {territory_name}",
        author="Medical'IA - KESK'IA"
    )
    
    # Styles
    styles = getSampleStyleSheet()
    styles['Title'].fontSize = 24
    styles['Title'].spaceAfter = 30
    styles['Heading1'].fontSize = 16
    styles['Heading1'].spaceAfter = 12
    styles['Heading1'].spaceBefore = 20

    styles['Heading2'].fontSize = 14
    styles['Heading2'].spaceAfter = 10
    styles['Heading2'].spaceBefore = 15

    styles['Normal'].fontSize = 10
    styles['Normal'].spaceAfter = 8
    
    # Liste des éléments du document
    elements = []
    
    # Titre et en-tête
    current_date = datetime.now().strftime("%d/%m/%Y")
    
    elements.append(Paragraph(f"Medical'IA - Analyse des Déserts Médicaux", styles['Title']))
    elements.append(Paragraph(f"Rapport pour : {territory_name}", styles['Heading1']))
    elements.append(Paragraph(f"Date de génération : {current_date}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Logo et introduction
    intro_text = f"""
    Ce rapport fournit une analyse détaillée de l'accessibilité aux soins médicaux pour 
    {territory_name}. L'indice APL (Accessibilité Potentielle Localisée) est utilisé comme 
    indicateur principal, mesurant le nombre de consultations/visites accessibles par habitant par an.
    
    Un territoire est considéré comme un désert médical lorsque l'APL est inférieur à 2,5, 
    et la situation est critique lorsqu'il est inférieur à 1,5.
    """
    elements.append(Paragraph(intro_text, styles['Normal']))
    elements.append(Spacer(1, 10))
    
    # Préparer les données spécifiques au territoire
    if territory_level == 'region':
        # Assurons-nous que nous travaillons avec des chaînes
        data_copy = data.copy()
        if 'region' not in data_copy.columns:
            # Si la colonne region n'existe pas, créer une mappage à partir des codes département
            region_map = {
                '01': 'Auvergne-Rhône-Alpes', '03': 'Auvergne-Rhône-Alpes', '07': 'Auvergne-Rhône-Alpes', 
                '15': 'Auvergne-Rhône-Alpes', '26': 'Auvergne-Rhône-Alpes', '38': 'Auvergne-Rhône-Alpes',
                '42': 'Auvergne-Rhône-Alpes', '43': 'Auvergne-Rhône-Alpes', '63': 'Auvergne-Rhône-Alpes',
                '69': 'Auvergne-Rhône-Alpes', '73': 'Auvergne-Rhône-Alpes', '74': 'Auvergne-Rhône-Alpes',
                '21': 'Bourgogne-Franche-Comté', '25': 'Bourgogne-Franche-Comté', '39': 'Bourgogne-Franche-Comté',
                '58': 'Bourgogne-Franche-Comté', '70': 'Bourgogne-Franche-Comté', '71': 'Bourgogne-Franche-Comté',
                '89': 'Bourgogne-Franche-Comté', '90': 'Bourgogne-Franche-Comté',
                '22': 'Bretagne', '29': 'Bretagne', '35': 'Bretagne', '56': 'Bretagne',
                '18': 'Centre-Val de Loire', '28': 'Centre-Val de Loire', '36': 'Centre-Val de Loire',
                '37': 'Centre-Val de Loire', '41': 'Centre-Val de Loire', '45': 'Centre-Val de Loire',
                '2A': 'Corse', '2B': 'Corse',
                '08': 'Grand Est', '10': 'Grand Est', '51': 'Grand Est', '52': 'Grand Est',
                '54': 'Grand Est', '55': 'Grand Est', '57': 'Grand Est', '67': 'Grand Est',
                '68': 'Grand Est', '88': 'Grand Est',
                '02': 'Hauts-de-France', '59': 'Hauts-de-France', '60': 'Hauts-de-France',
                '62': 'Hauts-de-France', '80': 'Hauts-de-France',
                '75': 'Île-de-France', '77': 'Île-de-France', '78': 'Île-de-France', '91': 'Île-de-France',
                '92': 'Île-de-France', '93': 'Île-de-France', '94': 'Île-de-France', '95': 'Île-de-France',
                '14': 'Normandie', '27': 'Normandie', '50': 'Normandie', '61': 'Normandie', '76': 'Normandie',
                '16': 'Nouvelle-Aquitaine', '17': 'Nouvelle-Aquitaine', '19': 'Nouvelle-Aquitaine',
                '23': 'Nouvelle-Aquitaine', '24': 'Nouvelle-Aquitaine', '33': 'Nouvelle-Aquitaine',
                '40': 'Nouvelle-Aquitaine', '47': 'Nouvelle-Aquitaine', '64': 'Nouvelle-Aquitaine',
                '79': 'Nouvelle-Aquitaine', '86': 'Nouvelle-Aquitaine', '87': 'Nouvelle-Aquitaine',
                '09': 'Occitanie', '11': 'Occitanie', '12': 'Occitanie', '30': 'Occitanie',
                '31': 'Occitanie', '32': 'Occitanie', '34': 'Occitanie', '46': 'Occitanie',
                '48': 'Occitanie', '65': 'Occitanie', '66': 'Occitanie', '81': 'Occitanie', '82': 'Occitanie',
                '44': 'Pays de la Loire', '49': 'Pays de la Loire', '53': 'Pays de la Loire',
                '72': 'Pays de la Loire', '85': 'Pays de la Loire',
                '04': 'Provence-Alpes-Côte d\'Azur', '05': 'Provence-Alpes-Côte d\'Azur',
                '06': 'Provence-Alpes-Côte d\'Azur', '13': 'Provence-Alpes-Côte d\'Azur',
                '83': 'Provence-Alpes-Côte d\'Azur', '84': 'Provence-Alpes-Côte d\'Azur',
                '971': 'Outre-Mer', '972': 'Outre-Mer', '973': 'Outre-Mer', '974': 'Outre-Mer', '976': 'Outre-Mer',
                '975': 'Outre-Mer', '977': 'Outre-Mer', '978': 'Outre-Mer', '986': 'Outre-Mer', '987': 'Outre-Mer',
                '988': 'Outre-Mer'
            }
            data_copy['departement'] = data_copy['CODGEO'].apply(lambda x: str(x)[:2])
            data_copy['region'] = data_copy['departement'].map(region_map)
        
        # Filtrer pour la région spécifiée
        filtered_data = data_copy[data_copy['region'] == territory_name]
        
        # Extraire les départements uniques (en tant que chaînes)
        departments = sorted(filtered_data['CODGEO'].apply(lambda x: str(x)[:2]).unique())
        
        # Tableau d'information sur la région
        if not filtered_data.empty:
            region_data = filtered_data.groupby('region').agg({
                'P16_POP': 'sum',
                'APL': lambda x: np.average(x, weights=filtered_data.loc[x.index, 'P16_POP']),
                'CODGEO': 'count'
            }).reset_index()
            
            desert_count = len(filtered_data[filtered_data['APL'] < 2.5])
            desert_percent = (desert_count / len(filtered_data)) * 100 if len(filtered_data) > 0 else 0
            
            if not region_data.empty:
                region_info = [
                    ["Population totale", f"{int(region_data['P16_POP'].iloc[0]):,}".replace(',', ' ')],
                    ["Nombre de communes", f"{int(region_data['CODGEO'].iloc[0]):,}".replace(',', ' ')],
                    ["APL moyen pondéré", f"{region_data['APL'].iloc[0]:.2f}"],
                    ["Communes en désert médical", f"{desert_count} ({desert_percent:.1f}%)"]
                ]
                
                # Ajouter le tableau d'information
                elements.append(Paragraph("Informations générales sur la région", styles['Heading2']))
                region_table = Table(region_info, colWidths=[4*cm, 4*cm])
                region_table.setStyle(TableStyle([
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9)
                ]))
                elements.append(region_table)
        
    elif territory_level == 'departement':
        # Filtrer pour le département spécifié
        data_copy = data.copy()
        # Assurons-nous que CODGEO est une chaîne pour l'extraction du département
        data_copy['dept_code'] = data_copy['CODGEO'].apply(lambda x: str(x)[:2])
        filtered_data = data_copy[data_copy['dept_code'] == territory_name]
        
        # Tableau d'information sur le département
        if not filtered_data.empty:
            total_pop = filtered_data['P16_POP'].sum()
            weighted_apl = (filtered_data['APL'] * filtered_data['P16_POP']).sum() / total_pop if total_pop > 0 else 0
            desert_count = len(filtered_data[filtered_data['APL'] < 2.5])
            desert_percent = (desert_count / len(filtered_data)) * 100 if len(filtered_data) > 0 else 0
            
            dept_info = [
                ["Population totale", f"{int(total_pop):,}".replace(',', ' ')],
                ["Nombre de communes", f"{len(filtered_data)}"],
                ["APL moyen pondéré", f"{weighted_apl:.2f}"],
                ["Communes en désert médical", f"{desert_count} ({desert_percent:.1f}%)"]
            ]
            
            # Ajouter le tableau d'information
            elements.append(Paragraph(f"Informations générales sur le département {territory_name}", styles['Heading2']))
            dept_table = Table(dept_info, colWidths=[4*cm, 4*cm])
            dept_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9)
            ]))
            elements.append(dept_table)
        
    else:  # commune
        # Filtrer pour la commune spécifiée
        commune_data = data[data['CODGEO'] == territory_name]
        
        if not commune_data.empty:
            commune_name = commune_data['Communes'].iloc[0]
            
            # Tableau d'information sur la commune
            commune_info = [
                ["Nom de la commune", commune_name],
                ["Code INSEE", territory_name],
                ["Population", f"{int(commune_data['P16_POP'].iloc[0]):,}".replace(',', ' ')],
                ["APL", f"{commune_data['APL'].iloc[0]:.2f}"],
                ["Statut", "Désert médical" if commune_data['APL'].iloc[0] < 2.5 else "Accès médical suffisant"]
            ]
            
            # Ajouter le tableau d'information
            elements.append(Paragraph(f"Informations générales sur la commune {commune_name}", styles['Heading2']))
            commune_table = Table(commune_info, colWidths=[4*cm, 4*cm])
            commune_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9)
            ]))
            elements.append(commune_table)
    
    elements.append(Spacer(1, 20))
    
    # Sections configurables
    if include_sections.get('carte_apl', False):
        elements.append(Paragraph("Carte de l'accessibilité aux soins", styles['Heading1']))
        elements.append(Paragraph("Une visualisation cartographique est disponible dans l'interface interactive de l'application web Medical'IA.", styles['Normal']))
        elements.append(Spacer(1, 10))
        
        # Créer un graphique temporaire
        elements.append(Spacer(1, 10))
    
    if include_sections.get('statistiques_detaillees', False):
        elements.append(Paragraph("Statistiques détaillées", styles['Heading1']))
        
        if territory_level in ['region', 'departement']:
            # Analyser la distribution des communes par catégorie d'APL
            apl_categories = [
                "Désert médical critique (APL < 1.5)",
                "Désert médical (APL 1.5-2.5)",
                "Sous-équipement médical (APL 2.5-3.5)",
                "Équipement médical suffisant (APL 3.5-4.5)",
                "Bon équipement médical (APL > 4.5)"
            ]
            
            # Créer les conditions pour la catégorisation
            filtered_copy = filtered_data.copy()  # Pour éviter les avertissements SettingWithCopyWarning
            conditions = [
                (filtered_copy["APL"] < 1.5),
                (filtered_copy["APL"] >= 1.5) & (filtered_copy["APL"] < 2.5),
                (filtered_copy["APL"] >= 2.5) & (filtered_copy["APL"] < 3.5),
                (filtered_copy["APL"] >= 3.5) & (filtered_copy["APL"] < 4.5),
                (filtered_copy["APL"] >= 4.5)
            ]
            
            # Assigner les catégories
            filtered_copy['APL_category'] = np.select(conditions, apl_categories, default="Non catégorisé")
            
            # Compter les communes par catégorie
            apl_counts = filtered_copy['APL_category'].value_counts().reset_index()
            apl_counts.columns = ['Catégorie', 'Nombre de communes']
            
            # Calculer le pourcentage
            total_communes = apl_counts['Nombre de communes'].sum()
            apl_counts['Pourcentage'] = (apl_counts['Nombre de communes'] / total_communes * 100).round(1)
                        
            # Ajouter un tableau avec les chiffres
            elements.append(Paragraph("Répartition des communes par catégorie d'APL", styles['Heading2']))
            
            # Préparer les données pour le tableau
            table_data = [['Catégorie', 'Nombre de communes', 'Pourcentage (%)']]
            for _, row in apl_counts.iterrows():
                table_data.append([
                    row['Catégorie'],
                    str(row['Nombre de communes']),
                    f"{row['Pourcentage']}%"
                ])
            
            # Créer le tableau
            table = Table(table_data, colWidths=[7*cm, 3*cm, 3*cm])
            table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9)
            ]))
            elements.append(table)
        
        # Pour les communes, montrer des statistiques supplémentaires si disponibles
        if territory_level == 'commune' and not commune_data.empty:
            elements.append(Paragraph("Contexte socio-démographique", styles['Heading2']))
            
            # Préparer les données démographiques
            demo_data = []
            if '0_14_pop_rate' in commune_data.columns:
                demo_data.append(["Population 0-14 ans", f"{commune_data['0_14_pop_rate'].iloc[0]:.1f}%"])
            if '15_59_pop_rate' in commune_data.columns:
                demo_data.append(["Population 15-59 ans", f"{commune_data['15_59_pop_rate'].iloc[0]:.1f}%"])
            if '60+_pop_rate' in commune_data.columns:
                demo_data.append(["Population 60+ ans", f"{commune_data['60+_pop_rate'].iloc[0]:.1f}%"])
            
            # Ajouter d'autres indicateurs
            if 'median_living_standard' in commune_data.columns:
                demo_data.append(["Niveau de vie médian", f"{commune_data['median_living_standard'].iloc[0]:.0f}€"])
            if 'density_area' in commune_data.columns:
                demo_data.append(["Densité de population", f"{commune_data['density_area'].iloc[0]:.1f} hab/km²"])
            
            # Créer un tableau
            if demo_data:
                demo_table = Table(demo_data, colWidths=[4*cm, 4*cm])
                demo_table.setStyle(TableStyle([
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9)
                ]))
                elements.append(demo_table)
            
            # Ajouter un commentaire sur le contexte
            elements.append(Spacer(1, 10))
            context_comment = f"""
            La commune de {commune_name} présente un indice APL de {commune_data['APL'].iloc[0]:.2f}, 
            ce qui la classe dans la catégorie {"des déserts médicaux" if commune_data['APL'].iloc[0] < 2.5 else "des zones correctement desservies"}.
            """
            elements.append(Paragraph(context_comment, styles['Normal']))

    # Saut de page avant la section analyse comparative
    elements.append(PageBreak())
        
    if include_sections.get('analyse_comparative', False):
        elements.append(Paragraph("Analyse comparative", styles['Heading1']))
        
        # Créer un graphique de comparaison simple
        try:
            # Récupérer les données nationales et locales pour la comparaison
            national_apl = data['APL'].mean()
            weighted_national_apl = (data['APL'] * data['P16_POP']).sum() / data['P16_POP'].sum() if data['P16_POP'].sum() > 0 else 0
            national_desert_percent = len(data[data['APL'] < 2.5]) / len(data) * 100 if len(data) > 0 else 0
            
            if territory_level == 'region' and not filtered_data.empty:
                territory_apl = filtered_data['APL'].mean()
                weighted_territory_apl = (filtered_data['APL'] * filtered_data['P16_POP']).sum() / filtered_data['P16_POP'].sum() if filtered_data['P16_POP'].sum() > 0 else 0
                territory_desert_percent = len(filtered_data[filtered_data['APL'] < 2.5]) / len(filtered_data) * 100 if len(filtered_data) > 0 else 0
                
                territory_label = territory_name
                
            elif territory_level == 'departement' and not filtered_data.empty:
                territory_apl = filtered_data['APL'].mean()
                weighted_territory_apl = (filtered_data['APL'] * filtered_data['P16_POP']).sum() / filtered_data['P16_POP'].sum() if filtered_data['P16_POP'].sum() > 0 else 0
                territory_desert_percent = len(filtered_data[filtered_data['APL'] < 2.5]) / len(filtered_data) * 100 if len(filtered_data) > 0 else 0
                
                territory_label = f"Département {territory_name}"
                
            elif territory_level == 'commune' and not commune_data.empty:
                territory_apl = commune_data['APL'].iloc[0]
                weighted_territory_apl = territory_apl  # Pour une commune, l'APL brut et pondéré sont identiques
                territory_desert_percent = 100 if territory_apl < 2.5 else 0
                
                territory_label = commune_data['Communes'].iloc[0]
                
            else:
                # Si pas de données, créer des valeurs par défaut
                territory_apl = 0
                weighted_territory_apl = 0
                territory_desert_percent = 0
                territory_label = "Non disponible"
                        
            # Ajouter un tableau comparatif
            elements.append(Paragraph("Comparaison des indicateurs clés", styles['Heading2']))
            
            comparative_data = [
                ["Indicateur", territory_label, "Niveau national", "Différence"],
                ["APL moyen", f"{territory_apl:.2f}", f"{national_apl:.2f}", f"{territory_apl - national_apl:+.2f}"],
                ["APL pondéré", f"{weighted_territory_apl:.2f}", f"{weighted_national_apl:.2f}", f"{weighted_territory_apl - weighted_national_apl:+.2f}"],
                ["% en désert médical", f"{territory_desert_percent:.1f}%", f"{national_desert_percent:.1f}%", f"{territory_desert_percent - national_desert_percent:+.1f}%"]
            ]
            
            comp_table = Table(comparative_data, colWidths=[3*cm, 3*cm, 3*cm, 3*cm])
            comp_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9)
            ]))
            elements.append(comp_table)
            
            # Ajouter une analyse textuelle
            elements.append(Spacer(1, 10))
            
            # Texte d'analyse
            if territory_apl < national_apl:
                analysis_text = f"""
                {territory_label} présente un APL moyen inférieur à la moyenne nationale ({territory_apl:.2f} contre {national_apl:.2f}). 
                La proportion de {"communes" if territory_level != 'commune' else "la population"} en situation de désert médical y est {"plus élevée" if territory_desert_percent > national_desert_percent else "moins élevée"} 
                que la moyenne nationale ({territory_desert_percent:.1f}% contre {national_desert_percent:.1f}%).
                """
            else:
                analysis_text = f"""
                {territory_label} présente un APL moyen supérieur à la moyenne nationale ({territory_apl:.2f} contre {national_apl:.2f}). 
                La proportion de {"communes" if territory_level != 'commune' else "la population"} en situation de désert médical y est {"plus élevée" if territory_desert_percent > national_desert_percent else "moins élevée"} 
                que la moyenne nationale ({territory_desert_percent:.1f}% contre {national_desert_percent:.1f}%).
                """
            
            elements.append(Paragraph(analysis_text, styles['Normal']))
                
        except Exception as e:
            # En cas d'erreur dans cette section, ajouter un message d'erreur au rapport
            elements.append(Paragraph(f"Impossible de générer l'analyse comparative : données insuffisantes.", styles['Normal']))
    
    if include_sections.get('facteurs_influents', False):
        elements.append(PageBreak())
        elements.append(Paragraph("Facteurs influençant l'accès aux soins", styles['Heading1']))
        
        # Liste des facteurs corrélés avec l'APL
        try:
            correlation_vars = [
                'median_living_standard', 'healthcare_education_establishments',
                'density_area', 'unemployment_rate', 'active_local_business_rate',
                'city_social_amenities_rate', '0_14_pop_rate', '15_59_pop_rate', '60+_pop_rate'
            ]
            
            # Filtrer pour n'inclure que les variables disponibles
            available_vars = [var for var in correlation_vars if var in filtered_data.columns]
            
            if available_vars:
                # Calculer les corrélations avec l'APL
                corr_data = []
                for var in available_vars:
                    try:
                        corr = filtered_data['APL'].corr(filtered_data[var])
                        if not pd.isna(corr):  # Ignorer les corrélations NaN
                            corr_data.append((var, corr))
                    except:
                        # Ignorer les erreurs potentielles lors du calcul des corrélations
                        pass
                
                # Créer un DataFrame pour le graphique
                if corr_data:
                    corr_df = pd.DataFrame(corr_data, columns=['Variable', 'Corrélation'])
                    
                    # Remplacer les noms des variables par des étiquettes plus lisibles
                    factor_names = {
                        'median_living_standard': 'Niveau de vie médian',
                        'healthcare_education_establishments': 'Établissements de santé/éducation',
                        'density_area': 'Densité de population',
                        'unemployment_rate': 'Taux de chômage',
                        'active_local_business_rate': 'Taux d\'entreprises actives',
                        'city_social_amenities_rate': 'Équipements sociaux',
                        '0_14_pop_rate': 'Population 0-14 ans',
                        '15_59_pop_rate': 'Population 15-59 ans',
                        '60+_pop_rate': 'Population 60+ ans'
                    }
                    
                    corr_df['Variable'] = corr_df['Variable'].map(lambda x: factor_names.get(x, x))
                    
                    # Trier par valeur absolue de corrélation
                    corr_df['Abs_Corr'] = corr_df['Corrélation'].abs()
                    corr_df = corr_df.sort_values('Abs_Corr', ascending=False).drop('Abs_Corr', axis=1)
                                        
                    # Ajouter une analyse textuelle
                    elements.append(Spacer(1, 10))
                    
                    # Identifier les facteurs les plus importants
                    positive_factors = corr_df[corr_df['Corrélation'] > 0].head(3)
                    negative_factors = corr_df[corr_df['Corrélation'] < 0].head(3)
                    
                    # Texte d'analyse
                    factors_text = """
                    Les facteurs ayant la plus forte influence sur l'accessibilité aux soins sont :
                    """
                    elements.append(Paragraph(factors_text, styles['Normal']))
                    
                    # Liste des facteurs positifs
                    if not positive_factors.empty:
                        elements.append(Paragraph("Facteurs favorisant un meilleur accès aux soins :", styles['Heading2']))
                        pos_list = []
                        for _, row in positive_factors.iterrows():
                            factor_item = ListItem(Paragraph(f"{row['Variable']} (corrélation: {row['Corrélation']:.2f})", styles['Normal']))
                            pos_list.append(factor_item)
                        elements.append(ListFlowable(pos_list, bulletType='bullet'))
                    
                    # Liste des facteurs négatifs
                    if not negative_factors.empty:
                        elements.append(Paragraph("Facteurs associés à un accès plus limité aux soins :", styles['Heading2']))
                        neg_list = []
                        for _, row in negative_factors.iterrows():
                            factor_item = ListItem(Paragraph(f"{row['Variable']} (corrélation: {row['Corrélation']:.2f})", styles['Normal']))
                            neg_list.append(factor_item)
                        elements.append(ListFlowable(neg_list, bulletType='bullet'))
                    
                    # Ajouter une explication d'interprétation
                    elements.append(Spacer(1, 10))
                    interpretation = """
                    Note d'interprétation : Une corrélation positive signifie que l'augmentation du facteur est 
                    associée à une meilleure accessibilité aux soins (APL plus élevé). Une corrélation négative 
                    signifie que l'augmentation du facteur est associée à une moins bonne accessibilité aux soins.
                    """
                    elements.append(Paragraph(interpretation, styles['Normal']))
                else:
                    elements.append(Paragraph("Données insuffisantes pour analyser les facteurs d'influence.", styles['Normal']))
            else:
                elements.append(Paragraph("Données insuffisantes pour analyser les facteurs d'influence.", styles['Normal']))
        except Exception as e:
            # Gérer les erreurs potentielles
            elements.append(Paragraph("Impossible d'analyser les facteurs d'influence en raison de données insuffisantes ou incomplètes.", styles['Normal']))
    
    # Recommandations
    if include_recommendations:
        elements.append(PageBreak())
        elements.append(Paragraph("Recommandations", styles['Heading1']))
        
        # Déterminer la situation globale
        situation = "non évaluée"  # Par défaut
        recommendations = []
        
        try:
            if territory_level == 'region' and not filtered_data.empty:
                # Calculer le pourcentage de déserts médicaux
                desert_percent = len(filtered_data[filtered_data['APL'] < 2.5]) / len(filtered_data) * 100 if len(filtered_data) > 0 else 0
                
                if desert_percent > 40:
                    situation = "critique"
                    intro_text = f"""
                    La situation de l'accès aux soins dans {territory_name} est particulièrement préoccupante, 
                    avec {desert_percent:.1f}% des communes en situation de désert médical. Les recommandations 
                    suivantes visent à améliorer rapidement cette situation :
                    """
                    
                    recommendations = [
                        "Mise en place d'un plan d'urgence pour l'attraction et la rétention des professionnels de santé",
                        "Développement prioritaire de centres de santé pluridisciplinaires dans les zones les plus touchées",
                        "Déploiement de solutions de télémédecine avec des points d'accès dans chaque commune",
                        "Création d'incitations financières exceptionnelles pour l'installation dans les zones critiques",
                        "Mise en place d'un système de transport médical pour les populations vulnérables",
                        "Coordination avec les facultés de médecine pour favoriser les stages en zone sous-dotée"
                    ]
                elif desert_percent > 20:
                    situation = "préoccupante"
                    intro_text = f"""
                    La situation de l'accès aux soins dans {territory_name} est préoccupante, 
                    avec {desert_percent:.1f}% des communes en situation de désert médical. Les recommandations 
                    suivantes peuvent contribuer à améliorer cette situation :
                    """
                    
                    recommendations = [
                        "Développement de maisons de santé pluridisciplinaires dans les zones prioritaires",
                        "Mise en place d'incitations à l'installation pour les nouveaux praticiens",
                        "Renforcement de l'attractivité du territoire pour les professionnels de santé",
                        "Amélioration des infrastructures de transport vers les pôles de santé",
                        "Développement de solutions de télémédecine complémentaires"
                    ]
                else:
                    situation = "relativement favorable"
                    intro_text = f"""
                    La situation de l'accès aux soins dans {territory_name} est relativement favorable, 
                    avec seulement {desert_percent:.1f}% des communes en situation de désert médical. Les recommandations 
                    suivantes visent à maintenir et améliorer cette situation :
                    """
                    
                    recommendations = [
                        "Mise en place d'un observatoire de l'accès aux soins pour anticiper les évolutions",
                        "Planification des remplacements des départs en retraite des médecins",
                        "Renforcement de l'offre de spécialistes dans les zones les moins bien pourvues",
                        "Développement d'une stratégie d'attraction des professionnels de santé sur le long terme",
                        "Optimisation de la coordination entre professionnels de santé"
                    ]
                
                elements.append(Paragraph(intro_text, styles['Normal']))
                elements.append(Spacer(1, 10))
                
                # Créer une liste à puces pour les recommandations
                bullet_list = []
                for recommendation in recommendations:
                    bullet_list.append(ListItem(Paragraph(recommendation, styles['Normal'])))
                elements.append(ListFlowable(bullet_list, bulletType='bullet'))
                
                # Ajouter des recommandations spécifiques pour les zones critiques
                critical_desert_count = len(filtered_data[filtered_data['APL'] < 1.5])
                critical_desert_percent = critical_desert_count / len(filtered_data) * 100 if len(filtered_data) > 0 else 0
                
                if critical_desert_percent > 10:
                    elements.append(Spacer(1, 10))
                    elements.append(Paragraph("Recommandations spécifiques pour les zones critiques (APL < 1,5)", styles['Heading2']))
                    
                    critical_recs = [
                        "Déploiement de cabinets médicaux mobiles pour assurer une présence médicale régulière",
                        f"Priorisation des {critical_desert_count} communes en situation critique dans les plans d'action",
                        "Mise en place d'aides financières exceptionnelles pour l'installation de médecins",
                        "Développement de solutions de télémédecine d'urgence"
                    ]
                    
                    # Créer une liste à puces pour les recommandations critiques
                    critical_list = []
                    for recommendation in critical_recs:
                        critical_list.append(ListItem(Paragraph(recommendation, styles['Normal'])))
                    elements.append(ListFlowable(critical_list, bulletType='bullet'))
            
            elif territory_level == 'departement' and not filtered_data.empty:
                # Calculer le pourcentage de déserts médicaux
                desert_percent = len(filtered_data[filtered_data['APL'] < 2.5]) / len(filtered_data) * 100 if len(filtered_data) > 0 else 0
                
                if desert_percent > 40:
                    situation = "critique"
                    intro_text = f"""
                    La situation de l'accès aux soins dans le département {territory_name} est particulièrement préoccupante, 
                    avec {desert_percent:.1f}% des communes en situation de désert médical. Les recommandations 
                    suivantes visent à améliorer rapidement cette situation :
                    """
                    
                    recommendations = [
                        "Mise en place d'un plan d'urgence départemental pour l'attraction des professionnels de santé",
                        "Développement de centres de santé pluridisciplinaires dans les zones les plus touchées",
                        "Déploiement de solutions de télémédecine avec points d'accès dans les communes isolées",
                        "Création d'incitations financières exceptionnelles pour l'installation",
                        "Mise en place d'un système de transport médical départemental"
                    ]
                elif desert_percent > 20:
                    situation = "préoccupante"
                    intro_text = f"""
                    La situation de l'accès aux soins dans le département {territory_name} est préoccupante, 
                    avec {desert_percent:.1f}% des communes en situation de désert médical. Les recommandations 
                    suivantes peuvent contribuer à améliorer cette situation :
                    """
                    
                    recommendations = [
                        "Développement de maisons de santé pluridisciplinaires dans les zones prioritaires",
                        "Mise en place d'incitations à l'installation ciblées",
                        "Renforcement de l'attractivité du département pour les professionnels de santé",
                        "Amélioration des transports vers les pôles de santé",
                        "Développement de solutions de télémédecine complémentaires"
                    ]
                else:
                    situation = "relativement favorable"
                    intro_text = f"""
                    La situation de l'accès aux soins dans le département {territory_name} est relativement favorable, 
                    avec seulement {desert_percent:.1f}% des communes en situation de désert médical. Les recommandations 
                    suivantes visent à maintenir et améliorer cette situation :
                    """
                    
                    recommendations = [
                        "Mise en place d'un observatoire départemental de l'accès aux soins",
                        "Planification anticipée des remplacements des départs en retraite",
                        "Renforcement de l'offre de spécialistes",
                        "Développement d'une stratégie de long terme",
                        "Optimisation de la coordination entre professionnels de santé"
                    ]
                
                elements.append(Paragraph(intro_text, styles['Normal']))
                elements.append(Spacer(1, 10))
                
                # Créer une liste à puces pour les recommandations
                bullet_list = []
                for recommendation in recommendations:
                    bullet_list.append(ListItem(Paragraph(recommendation, styles['Normal'])))
                elements.append(ListFlowable(bullet_list, bulletType='bullet'))
                
            elif territory_level == 'commune' and not commune_data.empty:
                # Recommandation pour une commune unique
                commune_apl = commune_data['APL'].iloc[0]
                commune_name = commune_data['Communes'].iloc[0]
                
                if commune_apl < 1.5:
                    situation = "critique"
                    intro_text = f"""
                    La situation de l'accès aux soins dans la commune de {commune_name} est critique (APL = {commune_apl:.2f}). 
                    Les recommandations suivantes visent à améliorer rapidement cette situation :
                    """
                    
                    recommendations = [
                        "Mise en place d'un cabinet médical avec des aides à l'installation exceptionnelles",
                        "Développement de solutions de télémédecine avec un point d'accès dans la commune",
                        "Organisation de consultations régulières de médecins itinérants",
                        "Mise en place d'un service de transport médical pour les habitants",
                        "Collaboration avec les communes environnantes pour mutualiser les ressources médicales"
                    ]
                elif commune_apl < 2.5:
                    situation = "préoccupante"
                    intro_text = f"""
                    La situation de l'accès aux soins dans la commune de {commune_name} est préoccupante (APL = {commune_apl:.2f}). 
                    Les recommandations suivantes peuvent contribuer à améliorer cette situation :
                    """
                    
                    recommendations = [
                        "Développement d'incitations à l'installation pour les professionnels de santé",
                        "Création d'un cabinet médical partagé avec plusieurs professionnels",
                        "Mise en place de consultations régulières de spécialistes",
                        "Amélioration des infrastructures de transport vers les pôles de santé",
                        "Développement de solutions de télémédecine complémentaires"
                    ]
                else:
                    situation = "satisfaisante"
                    intro_text = f"""
                    La situation de l'accès aux soins dans la commune de {commune_name} est satisfaisante (APL = {commune_apl:.2f}). 
                    Les recommandations suivantes visent à maintenir et améliorer cette situation :
                    """
                    
                    recommendations = [
                        "Maintien de l'attractivité pour les professionnels de santé",
                        "Anticipation des départs en retraite des médecins",
                        "Diversification de l'offre de soins spécialisés",
                        "Optimisation de la coordination entre professionnels de santé",
                        "Promotion de la prévention et de l'éducation à la santé"
                    ]
                
                elements.append(Paragraph(intro_text, styles['Normal']))
                elements.append(Spacer(1, 10))
                
                # Créer une liste à puces pour les recommandations
                bullet_list = []
                for recommendation in recommendations:
                    bullet_list.append(ListItem(Paragraph(recommendation, styles['Normal'])))
                elements.append(ListFlowable(bullet_list, bulletType='bullet'))
        except Exception as e:
            # En cas d'erreur, ajouter un message générique
            elements.append(Paragraph("Des recommandations personnalisées n'ont pas pu être générées en raison de données insuffisantes.", styles['Normal']))
    
    # Pied de page et conclusion
    elements.append(PageBreak())
    elements.append(Paragraph("Conclusion", styles['Heading1']))
    
    # Adapter la conclusion au territoire et à sa situation
    try:
        if territory_level == 'region':
            # Calculer le pourcentage de déserts médicaux pour la conclusion
            desert_percent = len(filtered_data[filtered_data['APL'] < 2.5]) / len(filtered_data) * 100 if len(filtered_data) > 0 else 0
            
            conclusion_text = f"""
            Ce rapport présente une analyse approfondie de la situation de l'accès aux soins dans la région {territory_name}. 
            Les données montrent une situation globalement {"critique" if desert_percent > 40 else "préoccupante" if desert_percent > 20 else "relativement favorable"}, 
            avec {desert_percent:.1f}% des communes en situation de désert médical.
            
            Les recommandations proposées visent à améliorer l'accès aux soins dans la région en tenant compte 
            des spécificités territoriales et des facteurs influençant l'accessibilité médicale.
            
            Ce rapport a été généré automatiquement par Medical'IA, outil d'analyse des déserts médicaux développé 
            par l'équipe KESK'IA.
            """
        elif territory_level == 'departement':
            # Calculer le pourcentage de déserts médicaux pour la conclusion
            desert_percent = len(filtered_data[filtered_data['APL'] < 2.5]) / len(filtered_data) * 100 if len(filtered_data) > 0 else 0
            
            conclusion_text = f"""
            Ce rapport présente une analyse approfondie de la situation de l'accès aux soins dans le département {territory_name}. 
            Les données montrent une situation globalement {"critique" if desert_percent > 40 else "préoccupante" if desert_percent > 20 else "relativement favorable"}, 
            avec {desert_percent:.1f}% des communes en situation de désert médical.
            
            Les recommandations proposées visent à améliorer l'accès aux soins dans le département en tenant compte 
            des spécificités territoriales et des facteurs influençant l'accessibilité médicale.
            
            Ce rapport a été généré automatiquement par Medical'IA, outil d'analyse des déserts médicaux développé 
            par l'équipe KESK'IA.
            """
        elif territory_level == 'commune' and not commune_data.empty:
            # Récupérer l'APL de la commune pour la conclusion
            commune_name = commune_data['Communes'].iloc[0]
            commune_apl = commune_data['APL'].iloc[0]
            
            conclusion_text = f"""
            Ce rapport présente une analyse de la situation de l'accès aux soins dans la commune de {commune_name}. 
            Les données montrent une situation {"critique" if commune_apl < 1.5 else "préoccupante" if commune_apl < 2.5 else "satisfaisante"}, 
            avec un APL de {commune_apl:.2f}.
            
            Les recommandations proposées visent à {"améliorer rapidement" if commune_apl < 2.5 else "maintenir et optimiser"} 
            l'accès aux soins dans la commune en tenant compte des facteurs influençant l'accessibilité médicale.
            
            Ce rapport a été généré automatiquement par Medical'IA, outil d'analyse des déserts médicaux développé 
            par l'équipe KESK'IA.
            """
        else:
            # Texte générique si les données sont insuffisantes
            conclusion_text = """
            Ce rapport présente une analyse de la situation de l'accès aux soins pour le territoire sélectionné.
            
            Les recommandations proposées visent à améliorer l'accès aux soins en tenant compte 
            des spécificités territoriales et des facteurs influençant l'accessibilité médicale.
            
            Ce rapport a été généré automatiquement par Medical'IA, outil d'analyse des déserts médicaux développé 
            par l'équipe KESK'IA.
            """
    except Exception as e:
        # Texte en cas d'erreur
        conclusion_text = """
        Ce rapport présente une analyse de l'accessibilité aux soins médicaux pour le territoire sélectionné.
        
        Ce rapport a été généré automatiquement par Medical'IA, outil d'analyse des déserts médicaux développé 
        par l'équipe KESK'IA.
        """
    
    elements.append(Paragraph(conclusion_text, styles['Normal']))
    
    # Ajouter contact et informations supplémentaires
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Pour plus d'informations ou une analyse personnalisée, contactez l'équipe KESK'IA.", styles['Normal']))
    
    # Construction finale du document
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Ajout de l'interface utilisateur pour la génération de rapports dans Streamlit
def add_report_generator_ui(data, filtered_data):
    st.header("📊 Générateur de rapports PDF")
    
    st.markdown("""
    Cette fonctionnalité permet de générer des rapports PDF détaillés sur la situation des déserts médicaux 
    pour différents niveaux territoriaux. Ces rapports sont destinés aux collectivités administratives et 
    aux décideurs pour faciliter la compréhension et la prise de décision.
    """)
    
    # Sélection du niveau territorial
    territory_level = st.selectbox(
        "Niveau territorial",
        ["Région", "Département", "Commune"],
        help="Sélectionnez le niveau territorial pour lequel vous souhaitez générer un rapport"
    )
    
    # Options en fonction du niveau territorial
    # Modification de la partie de sélection de région dans add_report_generator_ui
    if territory_level == "Région":
        # Vérifier si la colonne region existe
        if 'region' in filtered_data.columns:
            # Convertir tous les éléments en chaînes de caractères avant de trier
            regions = sorted([str(r) for r in filtered_data['region'].dropna().unique() if r is not None])
            if regions:  # Vérifier que la liste n'est pas vide
                territory_name = st.selectbox("Sélectionnez une région", regions)
                selected_level = "region"
            else:
                st.error("Aucune région trouvée dans les données filtrées.")
                return
        else:
            # Créer une correspondance départements -> régions
            region_map = {
                '01': 'Auvergne-Rhône-Alpes', '03': 'Auvergne-Rhône-Alpes', '07': 'Auvergne-Rhône-Alpes', 
                '15': 'Auvergne-Rhône-Alpes', '26': 'Auvergne-Rhône-Alpes', '38': 'Auvergne-Rhône-Alpes',
                '42': 'Auvergne-Rhône-Alpes', '43': 'Auvergne-Rhône-Alpes', '63': 'Auvergne-Rhône-Alpes',
                '69': 'Auvergne-Rhône-Alpes', '73': 'Auvergne-Rhône-Alpes', '74': 'Auvergne-Rhône-Alpes',
                '21': 'Bourgogne-Franche-Comté', '25': 'Bourgogne-Franche-Comté', '39': 'Bourgogne-Franche-Comté',
                '58': 'Bourgogne-Franche-Comté', '70': 'Bourgogne-Franche-Comté', '71': 'Bourgogne-Franche-Comté',
                '89': 'Bourgogne-Franche-Comté', '90': 'Bourgogne-Franche-Comté',
                '22': 'Bretagne', '29': 'Bretagne', '35': 'Bretagne', '56': 'Bretagne',
                '18': 'Centre-Val de Loire', '28': 'Centre-Val de Loire', '36': 'Centre-Val de Loire',
                '37': 'Centre-Val de Loire', '41': 'Centre-Val de Loire', '45': 'Centre-Val de Loire',
                '2A': 'Corse', '2B': 'Corse',
                '08': 'Grand Est', '10': 'Grand Est', '51': 'Grand Est', '52': 'Grand Est',
                '54': 'Grand Est', '55': 'Grand Est', '57': 'Grand Est', '67': 'Grand Est',
                '68': 'Grand Est', '88': 'Grand Est',
                '02': 'Hauts-de-France', '59': 'Hauts-de-France', '60': 'Hauts-de-France',
                '62': 'Hauts-de-France', '80': 'Hauts-de-France',
                '75': 'Île-de-France', '77': 'Île-de-France', '78': 'Île-de-France', '91': 'Île-de-France',
                '92': 'Île-de-France', '93': 'Île-de-France', '94': 'Île-de-France', '95': 'Île-de-France',
                '14': 'Normandie', '27': 'Normandie', '50': 'Normandie', '61': 'Normandie', '76': 'Normandie',
                '16': 'Nouvelle-Aquitaine', '17': 'Nouvelle-Aquitaine', '19': 'Nouvelle-Aquitaine',
                '23': 'Nouvelle-Aquitaine', '24': 'Nouvelle-Aquitaine', '33': 'Nouvelle-Aquitaine',
                '40': 'Nouvelle-Aquitaine', '47': 'Nouvelle-Aquitaine', '64': 'Nouvelle-Aquitaine',
                '79': 'Nouvelle-Aquitaine', '86': 'Nouvelle-Aquitaine', '87': 'Nouvelle-Aquitaine',
                '09': 'Occitanie', '11': 'Occitanie', '12': 'Occitanie', '30': 'Occitanie',
                '31': 'Occitanie', '32': 'Occitanie', '34': 'Occitanie', '46': 'Occitanie',
                '48': 'Occitanie', '65': 'Occitanie', '66': 'Occitanie', '81': 'Occitanie', '82': 'Occitanie',
                '44': 'Pays de la Loire', '49': 'Pays de la Loire', '53': 'Pays de la Loire',
                '72': 'Pays de la Loire', '85': 'Pays de la Loire',
                '04': 'Provence-Alpes-Côte d\'Azur', '05': 'Provence-Alpes-Côte d\'Azur',
                '06': 'Provence-Alpes-Côte d\'Azur', '13': 'Provence-Alpes-Côte d\'Azur',
                '83': 'Provence-Alpes-Côte d\'Azur', '84': 'Provence-Alpes-Côte d\'Azur',
                '971': 'Outre-Mer', '972': 'Outre-Mer', '973': 'Outre-Mer', '974': 'Outre-Mer', '976': 'Outre-Mer',
                '975': 'Outre-Mer', '977': 'Outre-Mer', '978': 'Outre-Mer', '986': 'Outre-Mer', '987': 'Outre-Mer',
                '988': 'Outre-Mer'
            }
            
            # Ajouter la colonne région aux données filtrées
            filtered_data['region'] = filtered_data['CODGEO'].str[:2].map(region_map)
            regions = sorted([r for r in filtered_data['region'].dropna().unique() if r is not None])
            if regions:  # Vérifier que la liste n'est pas vide
                territory_name = st.selectbox("Sélectionnez une région", regions)
                selected_level = "region"
            else:
                st.error("Aucune région trouvée dans les données filtrées.")
                return
    elif territory_level == "Département":
        departments = sorted(filtered_data['CODGEO'].str[:2].unique())
        territory_name = st.selectbox("Sélectionnez un département", departments)
        selected_level = "departement"
    else:  # Commune
        if len(filtered_data) > 1000:
            # Pour faciliter la sélection, demander d'abord le département
            departments = sorted(filtered_data['CODGEO'].str[:2].unique())
            selected_dept = st.selectbox("Sélectionnez d'abord un département", departments)
            
            # Filtrer les communes du département sélectionné
            communes_in_dept = filtered_data[filtered_data['CODGEO'].str[:2] == selected_dept]
            
            # Créer une liste des communes avec leur nom et code INSEE
            commune_list = communes_in_dept[['CODGEO', 'Communes']].copy()
            commune_list['selection'] = commune_list['Communes'] + ' (' + commune_list['CODGEO'] + ')'
            
            # Trier par nom de commune
            commune_list = commune_list.sort_values('Communes')
            
            # Sélection de la commune
            selected_commune = st.selectbox(
                "Sélectionnez une commune",
                commune_list['selection'].tolist()
            )
            
            # Extraire le code INSEE de la sélection
            territory_name = selected_commune.split('(')[-1].split(')')[0].strip()
        else:
            # Si peu de communes, on peut toutes les afficher directement
            commune_list = filtered_data[['CODGEO', 'Communes']].copy()
            commune_list['selection'] = commune_list['Communes'] + ' (' + commune_list['CODGEO'] + ')'
            commune_list = commune_list.sort_values('Communes')
            
            selected_commune = st.selectbox(
                "Sélectionnez une commune",
                commune_list['selection'].tolist()
            )
            
            territory_name = selected_commune.split('(')[-1].split(')')[0].strip()
        
        selected_level = "commune"
    
    # Configuration des sections du rapport
    st.subheader("Contenu du rapport")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_sections = {
            "carte_apl": st.checkbox("Carte de l'accessibilité aux soins", value=True),
            "statistiques_detaillees": st.checkbox("Statistiques détaillées", value=True),
            "analyse_comparative": st.checkbox("Analyse comparative", value=True),
            "facteurs_influents": st.checkbox("Facteurs influençant l'accès aux soins", value=True)
        }
    
    with col2:
        include_recommendations = st.checkbox("Inclure des recommandations", value=True)
        
        # Personnalisation du rapport (options avancées)
        with st.expander("Options avancées"):
            custom_title = st.text_input("Titre personnalisé du rapport (optionnel)")
            include_logo = st.checkbox("Inclure le logo Medical'IA", value=True)
            include_contact = st.checkbox("Inclure les informations de contact", value=True)
    
    # Bouton pour générer le rapport
    if st.button("Générer le rapport PDF"):
        with st.spinner("Génération du rapport en cours..."):
            try:
                # Générer le rapport
                pdf_buffer = generate_report_pdf(
                    data=filtered_data,
                    territory_level=selected_level,
                    territory_name=territory_name,
                    include_sections=include_sections,
                    include_recommendations=include_recommendations
                )
                
                # Convertir le PDF en base64 pour le téléchargement
                b64_pdf = base64.b64encode(pdf_buffer.read()).decode('utf-8')
                
                # Créer un bouton de téléchargement
                if selected_level == "region":
                    filename = f"rapport_medical_ia_{territory_name.replace(' ', '_').lower()}.pdf"
                elif selected_level == "departement":
                    filename = f"rapport_medical_ia_dept_{territory_name}.pdf"
                else:  # commune
                    commune_name = filtered_data[filtered_data['CODGEO'] == territory_name]['Communes'].iloc[0]
                    filename = f"rapport_medical_ia_{commune_name.replace(' ', '_').lower()}_{territory_name}.pdf"
                
                # Afficher le lien de téléchargement
                href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}">Télécharger le rapport PDF</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # Afficher un aperçu du PDF
                st.success("Le rapport a été généré avec succès. Cliquez sur le lien ci-dessus pour le télécharger.")
                
                # Ajouter un aperçu des premières pages du PDF
                st.markdown("### Aperçu du rapport (première page)")
                st.markdown(f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="700" height="500"></iframe>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Une erreur s'est produite lors de la génération du rapport : {e}")
                st.error("Veuillez vérifier les données et réessayer.")


# Cette fonction permet d'intégrer le module à l'application principale
def select_view_mode():
    if view_mode == "Clusters de communes":
        view_mode_clusters(filtered_data)

    """
    Module complet pour les clusters de communes avec visualisations avancées
    """
    st.header("Zones d'accessibilité médicale similaire")
    
    st.markdown("""
    Cette analyse identifie les principaux regroupements de communes ayant des caractéristiques similaires d'accès aux soins.
    Nous utilisons une méthodologie avancée qui combine classification géographique et indicateurs socio-démographiques.
    """)
    
    # Définir les catégories d'APL avec des seuils précis basés sur les recommandations médicales
    apl_categories = [
        {"name": "Déserts médicaux critiques", "min": 0, "max": 1.5, "color": "darkred", "description": "Accès aux soins très difficile, situation urgente"},
        {"name": "Déserts médicaux", "min": 1.5, "max": 2.5, "color": "red", "description": "Accès aux soins insuffisant, actions nécessaires"},
        {"name": "Zones sous-équipées", "min": 2.5, "max": 3.5, "color": "orange", "description": "Accès limité, vigilance requise"},
        {"name": "Zones bien équipées", "min": 3.5, "max": 4.5, "color": "lightgreen", "description": "Accès satisfaisant aux soins médicaux"},
        {"name": "Zones très bien équipées", "min": 4.5, "max": 10, "color": "green", "description": "Excellent accès aux soins médicaux"}
    ]
    
    # Fonction améliorée pour catégoriser les communes
    @st.cache_data
    def prepare_zoned_data(data):
        """
        Catégorisation des communes avec métriques avancées
        """
        data_zoned = data.copy()
        
        # Ajout des catégories de zone
        data_zoned['zone_type'] = None
        data_zoned['zone_color'] = None
        data_zoned['zone_description'] = None
        
        for cat in apl_categories:
            mask = (data_zoned['APL'] >= cat['min']) & (data_zoned['APL'] < cat['max'])
            data_zoned.loc[mask, 'zone_type'] = cat['name']
            data_zoned.loc[mask, 'zone_color'] = cat['color']
            data_zoned.loc[mask, 'zone_description'] = cat['description']
        
        # Identifier le département et la région
        data_zoned['departement'] = data_zoned['CODGEO'].str[:2]
        
        # Table de correspondance département-région
        region_map = {
            '01': 'Auvergne-Rhône-Alpes', '03': 'Auvergne-Rhône-Alpes', '07': 'Auvergne-Rhône-Alpes', 
            '15': 'Auvergne-Rhône-Alpes', '26': 'Auvergne-Rhône-Alpes', '38': 'Auvergne-Rhône-Alpes',
            '42': 'Auvergne-Rhône-Alpes', '43': 'Auvergne-Rhône-Alpes', '63': 'Auvergne-Rhône-Alpes',
            '69': 'Auvergne-Rhône-Alpes', '73': 'Auvergne-Rhône-Alpes', '74': 'Auvergne-Rhône-Alpes',
            '21': 'Bourgogne-Franche-Comté', '25': 'Bourgogne-Franche-Comté', '39': 'Bourgogne-Franche-Comté',
            '58': 'Bourgogne-Franche-Comté', '70': 'Bourgogne-Franche-Comté', '71': 'Bourgogne-Franche-Comté',
            '89': 'Bourgogne-Franche-Comté', '90': 'Bourgogne-Franche-Comté',
            '22': 'Bretagne', '29': 'Bretagne', '35': 'Bretagne', '56': 'Bretagne',
            '18': 'Centre-Val de Loire', '28': 'Centre-Val de Loire', '36': 'Centre-Val de Loire',
            '37': 'Centre-Val de Loire', '41': 'Centre-Val de Loire', '45': 'Centre-Val de Loire',
            '2A': 'Corse', '2B': 'Corse',
            '08': 'Grand Est', '10': 'Grand Est', '51': 'Grand Est', '52': 'Grand Est',
            '54': 'Grand Est', '55': 'Grand Est', '57': 'Grand Est', '67': 'Grand Est',
            '68': 'Grand Est', '88': 'Grand Est',
            '02': 'Hauts-de-France', '59': 'Hauts-de-France', '60': 'Hauts-de-France',
            '62': 'Hauts-de-France', '80': 'Hauts-de-France',
            '75': 'Île-de-France', '77': 'Île-de-France', '78': 'Île-de-France', '91': 'Île-de-France',
            '92': 'Île-de-France', '93': 'Île-de-France', '94': 'Île-de-France', '95': 'Île-de-France',
            '14': 'Normandie', '27': 'Normandie', '50': 'Normandie', '61': 'Normandie', '76': 'Normandie',
            '16': 'Nouvelle-Aquitaine', '17': 'Nouvelle-Aquitaine', '19': 'Nouvelle-Aquitaine',
            '23': 'Nouvelle-Aquitaine', '24': 'Nouvelle-Aquitaine', '33': 'Nouvelle-Aquitaine',
            '40': 'Nouvelle-Aquitaine', '47': 'Nouvelle-Aquitaine', '64': 'Nouvelle-Aquitaine',
            '79': 'Nouvelle-Aquitaine', '86': 'Nouvelle-Aquitaine', '87': 'Nouvelle-Aquitaine',
            '09': 'Occitanie', '11': 'Occitanie', '12': 'Occitanie', '30': 'Occitanie',
            '31': 'Occitanie', '32': 'Occitanie', '34': 'Occitanie', '46': 'Occitanie',
            '48': 'Occitanie', '65': 'Occitanie', '66': 'Occitanie', '81': 'Occitanie', '82': 'Occitanie',
            '44': 'Pays de la Loire', '49': 'Pays de la Loire', '53': 'Pays de la Loire',
            '72': 'Pays de la Loire', '85': 'Pays de la Loire',
            '04': 'Provence-Alpes-Côte d\'Azur', '05': 'Provence-Alpes-Côte d\'Azur',
            '06': 'Provence-Alpes-Côte d\'Azur', '13': 'Provence-Alpes-Côte d\'Azur',
            '83': 'Provence-Alpes-Côte d\'Azur', '84': 'Provence-Alpes-Côte d\'Azur',
            '971': 'Outre-Mer', '972': 'Outre-Mer', '973': 'Outre-Mer', '974': 'Outre-Mer', '976': 'Outre-Mer',
            '975': 'Outre-Mer', '977': 'Outre-Mer', '978': 'Outre-Mer', '986': 'Outre-Mer', '987': 'Outre-Mer',
            '988': 'Outre-Mer'
        }
        
        data_zoned['region'] = data_zoned['departement'].map(region_map)
        
        # Ajout de métriques pour l'analyse avancée
        if '60+_pop_rate' in data_zoned.columns:
            # Catégorisation démographique
            data_zoned['demographic_profile'] = 'Équilibrée'
            data_zoned.loc[data_zoned['60+_pop_rate'] > 30, 'demographic_profile'] = 'Vieillissante'
            data_zoned.loc[data_zoned['60+_pop_rate'] < 15, 'demographic_profile'] = 'Jeune'
        
        if 'density_area' in data_zoned.columns:
            # Catégorisation urbain/rural
            data_zoned['urban_rural'] = 'Périurbain'
            data_zoned.loc[data_zoned['density_area'] < 50, 'urban_rural'] = 'Rural'
            data_zoned.loc[data_zoned['density_area'] > 500, 'urban_rural'] = 'Urbain'
            data_zoned.loc[data_zoned['density_area'] > 2000, 'urban_rural'] = 'Très urbain'
        
        if 'median_living_standard' in data_zoned.columns:
            # Niveau économique
            median_income = data_zoned['median_living_standard'].median()
            data_zoned['economic_level'] = 'Moyen'
            data_zoned.loc[data_zoned['median_living_standard'] < median_income * 0.8, 'economic_level'] = 'Modeste'
            data_zoned.loc[data_zoned['median_living_standard'] > median_income * 1.2, 'economic_level'] = 'Aisé'
        
        # Ajouter une étiquette composée pour classification avancée
        if 'urban_rural' in data_zoned.columns and 'demographic_profile' in data_zoned.columns:
            data_zoned['composite_label'] = data_zoned['urban_rural'] + ' ' + data_zoned['demographic_profile']
            if 'economic_level' in data_zoned.columns:
                data_zoned['composite_label'] += ' ' + data_zoned['economic_level']
        
        return data_zoned
    
    @st.cache_data
    def prepare_advanced_clusters(data_zoned, min_communes=10):
        """
        Préparation des clusters territoriaux avancés
        """
        # Grouper par département et type de zone
        dept_zones = data_zoned.groupby(['departement', 'zone_type']).agg({
            'CODGEO': 'count',
            'P16_POP': 'sum',
            'APL': 'mean',
            'latitude_mairie': 'mean',
            'longitude_mairie': 'mean',
            'zone_color': 'first',
            'zone_description': 'first'
        }).reset_index()
        
        # Enrichir avec des informations supplémentaires
        if '60+_pop_rate' in data_zoned.columns:
            pop_60_agg = data_zoned.groupby(['departement', 'zone_type'])['60+_pop_rate'].mean().reset_index()
            dept_zones = dept_zones.merge(pop_60_agg, on=['departement', 'zone_type'])
        
        if 'density_area' in data_zoned.columns:
            density_agg = data_zoned.groupby(['departement', 'zone_type'])['density_area'].mean().reset_index()
            dept_zones = dept_zones.merge(density_agg, on=['departement', 'zone_type'])
        
        if 'demographic_profile' in data_zoned.columns:
            # Obtenir le profil démographique le plus fréquent pour chaque groupe
            demo_agg = data_zoned.groupby(['departement', 'zone_type'])['demographic_profile'].agg(
                lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else None
            ).reset_index()
            dept_zones = dept_zones.merge(demo_agg, on=['departement', 'zone_type'])
        
        if 'urban_rural' in data_zoned.columns:
            # Obtenir le type urbain/rural le plus fréquent pour chaque groupe
            urban_agg = data_zoned.groupby(['departement', 'zone_type'])['urban_rural'].agg(
                lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else None
            ).reset_index()
            dept_zones = dept_zones.merge(urban_agg, on=['departement', 'zone_type'])
        
        # Créer une typologie plus détaillée des zones
        if 'demographic_profile' in dept_zones.columns and 'urban_rural' in dept_zones.columns:
            dept_zones['zone_typography'] = dept_zones['zone_type'] + ' - ' + dept_zones['urban_rural'] + ' ' + dept_zones['demographic_profile']
        else:
            dept_zones['zone_typography'] = dept_zones['zone_type']
        
        # Renommer les colonnes pour plus de clarté
        dept_zones.columns = ['Département', 'Type de zone', 'Nombre de communes', 
                             'Population', 'APL moyen', 'Latitude', 'Longitude', 'Couleur',
                             'Description'] + list(dept_zones.columns[9:])
        
        # Filtrer les zones significatives et trier
        significant_zones = dept_zones[dept_zones['Nombre de communes'] >= min_communes]
        significant_zones = significant_zones.sort_values('Population', ascending=False)
        
        return significant_zones
    
    # Création des onglets pour améliorer l'organisation de l'interface
    tab1, tab2, tab3 = st.tabs(["Vue d'ensemble", "Explorer par zone", "Analyse croisée"])
    
    # Préparer les données
    data_zoned = prepare_zoned_data(filtered_data)
    significant_zones = prepare_advanced_clusters(data_zoned)
    
    with tab1:
        st.subheader("Principales zones d'accessibilité médicale identifiées")
        
        # Statistiques générales sur les types de zones
        @st.cache_data
        def calculate_zone_stats(zones):
            stats = zones.groupby('Type de zone').agg({
                'Nombre de communes': 'sum',
                'Population': 'sum',
                'APL moyen': 'mean'
            }).reset_index()
            
            # Formater la population
            stats['Population'] = stats['Population'].apply(lambda x: f"{int(x):,}".replace(',', ' '))
            stats['APL moyen'] = stats['APL moyen'].round(2)
            
            # Réordonner selon la sévérité
            order = [cat["name"] for cat in apl_categories]
            stats['sort_order'] = stats['Type de zone'].map({zone: i for i, zone in enumerate(order)})
            stats = stats.sort_values('sort_order').drop('sort_order', axis=1)
            
            # Calcul du pourcentage de la population par type
            total_pop = sum([int(pop.replace(' ', '')) for pop in stats['Population']])
            stats['% de la population'] = stats['Population'].apply(
                lambda x: f"{int(int(x.replace(' ', '')) / total_pop * 100)}%"
            )
            
            return stats
        
        zone_stats = calculate_zone_stats(significant_zones)
        
        # Affichage avec une mise en forme améliorée
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown("### Répartition par type de zone")
            st.table(zone_stats)
            
            # Explication des types de zones
            with st.expander("Comprendre les types de zones"):
                for cat in apl_categories:
                    st.markdown(f"**{cat['name']}** ({cat['min']}-{cat['max']} APL): {cat['description']}")
        
        with col2:
            st.markdown("### Carte des principales zones homogènes")
            
            # Contrôles avancés pour la carte
            col_controls1, col_controls2 = st.columns(2)
            with col_controls1:
                # Ajouter un filtre pour le type de zone
                zone_type_filter = st.multiselect(
                    "Filtrer par type de zone",
                    options=significant_zones['Type de zone'].unique(),
                    default=significant_zones['Type de zone'].unique()
                )
            
            with col_controls2:
                # Contrôle du nombre de zones à afficher
                max_zones_to_display = st.slider(
                    "Nombre de zones à afficher", 
                    min_value=10, 
                    max_value=100, 
                    value=30,
                    help="Ajuster pour équilibrer détail et performance"
                )
            
            # Filtrer les zones selon les sélections
            display_zones = significant_zones[significant_zones['Type de zone'].isin(zone_type_filter)]
            display_zones = display_zones.head(max_zones_to_display)
            
            # Fonction améliorée pour créer la carte
            @st.cache_data
            def create_enhanced_zones_map(zones):
                fig = go.Figure()
                
                # Ajouter chaque zone comme un cercle proportionnel à sa population
                for _, zone in zones.iterrows():
                    # Taille proportionnelle à la racine carrée de la population, mais avec limites
                    population_scale = np.sqrt(zone['Population']) / 100
                    radius = max(5, min(30, population_scale))
                    
                    # Texte enrichi pour le hover
                    hover_text = f"""
                    <b>{zone['Type de zone']} - Dept {zone['Département']}</b><br>
                    <i>{zone['Description']}</i><br>
                    Communes: {zone['Nombre de communes']}<br>
                    Population: {int(zone['Population']):,}<br>
                    APL moyen: {zone['APL moyen']:.2f}
                    """
                    
                    # Ajouter des informations supplémentaires si disponibles
                    if 'demographic_profile' in zone and not pd.isna(zone['demographic_profile']):
                        hover_text += f"<br>Profil: {zone['demographic_profile']}"
                    
                    if 'urban_rural' in zone and not pd.isna(zone['urban_rural']):
                        hover_text += f"<br>Type: {zone['urban_rural']}"
                    
                    # Remplacer les virgules dans les nombres formatés
                    hover_text = hover_text.replace(',', ' ')
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=[zone['Latitude']],
                        lon=[zone['Longitude']],
                        mode='markers',
                        marker=dict(
                            size=radius,
                            color=zone['Couleur'],
                            opacity=0.7
                        ),
                        text=[hover_text],
                        hoverinfo='text',
                        name=f"{zone['Type de zone']} - Dept {zone['Département']}"
                    ))
                
                # Configuration améliorée de la carte
                fig.update_layout(
                    title="Principales zones d'accessibilité médicale similaire",
                    mapbox_style="carto-positron",
                    mapbox=dict(
                        center=dict(lat=46.603354, lon=1.888334),
                        zoom=5
                    ),
                    height=700,
                    margin={"r":0,"t":50,"l":0,"b":0},
                    showlegend=False,
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_family="Arial"
                    )
                )
                
                return fig
            
            # Afficher la carte
            with st.spinner("Génération de la carte des zones..."):
                zone_map = create_enhanced_zones_map(display_zones)
                st.plotly_chart(zone_map, use_container_width=True)
        
        # Analyse des regroupements territoriaux
        st.subheader("Distribution des zones par région")
        
        @st.cache_data
        def analyze_region_distribution(zones, data_zoned):
            # Ajouter la région à chaque zone départementale
            zones_with_region = zones.copy()
            
            # Map des départements aux régions
            dept_to_region = data_zoned[['departement', 'region']].drop_duplicates()
            dept_to_region_map = dict(zip(dept_to_region['departement'], dept_to_region['region']))
            
            zones_with_region['Région'] = zones_with_region['Département'].map(dept_to_region_map)
            
            # Compter les types de zones par région
            region_analysis = zones_with_region.groupby(['Région', 'Type de zone']).agg({
                'Nombre de communes': 'sum',
                'Population': 'sum'
            }).reset_index()
            
            # Pivoter pour avoir les types de zones en colonnes
            region_pivot = region_analysis.pivot_table(
                index='Région',
                columns='Type de zone',
                values='Nombre de communes',
                fill_value=0
            ).reset_index()
            
            return region_analysis, region_pivot
        
        region_analysis, region_pivot = analyze_region_distribution(significant_zones, data_zoned)
        
        # Heatmap des zones par région
        fig = px.imshow(
            region_pivot.iloc[:, 1:],
            x=region_pivot.columns[1:],
            y=region_pivot['Région'],
            color_continuous_scale='YlOrRd_r',
            labels=dict(x="Type de zone", y="Région", color="Nombre de communes"),
            title="Répartition des types de zones par région",
            height=500
        )
        
        fig.update_layout(
            xaxis_title="Type de zone",
            yaxis_title="Région",
            margin={"r":0,"t":50,"l":0,"b":0}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Explorer une zone spécifique")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sélection du type de zone
            zone_type_options = significant_zones['Type de zone'].unique()
            selected_zone_type = st.selectbox(
                "Sélectionner un type de zone:",
                options=zone_type_options
            )
        
        with col2:
            # Filtrer les zones par type sélectionné
            zones_of_type = significant_zones[significant_zones['Type de zone'] == selected_zone_type]
            
            # Sélection du département
            dept_options = zones_of_type['Département'].unique()
            selected_dept = st.selectbox(
                f"Sélectionner un département avec des {selected_zone_type.lower()}:",
                options=dept_options
            )
        
        # Filtrer pour le département et type de zone sélectionnés
        selected_zone = zones_of_type[zones_of_type['Département'] == selected_dept].iloc[0]
        
        # Obtenir toutes les communes de cette zone
        zone_communes = data_zoned[
            (data_zoned['departement'] == selected_dept) & 
            (data_zoned['zone_type'] == selected_zone_type)
        ]
        
        # Afficher les détails de la zone de manière plus attrayante
        st.markdown(f"## {selected_zone_type} du département {selected_dept}")
        st.markdown(f"*{selected_zone['Description']}*")
        
        # Métriques clés
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Communes", f"{int(selected_zone['Nombre de communes']):,}".replace(',', ' '))
        
        with col2:
            st.metric("Population", f"{int(selected_zone['Population']):,}".replace(',', ' '))
        
        with col3:
            st.metric("APL moyen", f"{selected_zone['APL moyen']:.2f}")
        
        with col4:
            # Vérifier si la colonne existe avant d'afficher
            if 'demographic_profile' in selected_zone and not pd.isna(selected_zone['demographic_profile']):
                st.metric("Profil", selected_zone['demographic_profile'])
            elif 'urban_rural' in selected_zone and not pd.isna(selected_zone['urban_rural']):
                st.metric("Type", selected_zone['urban_rural'])
        
        # Analyse avancée des communes de la zone
        st.markdown("### Caractéristiques détaillées des communes")
        
        # Analyses démographiques et socio-économiques si les données sont disponibles
        if '60+_pop_rate' in zone_communes.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Répartition par âge
                age_data = pd.DataFrame({
                    'Tranche d\'âge': ['0-14 ans', '15-59 ans', '60+ ans'],
                    'Pourcentage': [
                        zone_communes['0_14_pop_rate'].mean(),
                        zone_communes['15_59_pop_rate'].mean(),
                        zone_communes['60+_pop_rate'].mean()
                    ]
                })
                
                fig = px.pie(
                    age_data,
                    values='Pourcentage',
                    names='Tranche d\'âge',
                    title=f"Répartition par âge - {selected_zone_type} (Dept {selected_dept})",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Distribution démographique si catégorisation disponible
                if 'demographic_profile' in zone_communes.columns:
                    demo_counts = zone_communes['demographic_profile'].value_counts().reset_index()
                    demo_counts.columns = ['Profil démographique', 'Nombre de communes']
                    
                    fig = px.bar(
                        demo_counts,
                        x='Profil démographique',
                        y='Nombre de communes',
                        title=f"Profils démographiques - {selected_zone_type} (Dept {selected_dept})",
                        color='Profil démographique',
                        color_discrete_sequence=px.colors.qualitative.Safe
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Caractéristiques territoriales
        if 'urban_rural' in zone_communes.columns and 'density_area' in zone_communes.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution urbain/rural
                urban_counts = zone_communes['urban_rural'].value_counts().reset_index()
                urban_counts.columns = ['Type territorial', 'Nombre de communes']
                
                fig = px.bar(
                    urban_counts,
                    x='Type territorial',
                    y='Nombre de communes',
                    title=f"Types territoriaux - {selected_zone_type} (Dept {selected_dept})",
                    color='Type territorial',
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Distribution de la densité
                fig = px.histogram(
                    zone_communes,
                    x='density_area',
                    nbins=20,
                    title=f"Distribution des densités - {selected_zone_type} (Dept {selected_dept})",
                    color_discrete_sequence=['blue']
                )
                
                fig.update_layout(
                    xaxis_title="Densité (hab/km²)",
                    yaxis_title="Nombre de communes"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Carte des communes avec limitation du nombre de points
        st.markdown("### Carte des communes de la zone")
        
        # Options avancées pour la carte
        col1, col2 = st.columns(2)
        
        with col1:
            max_communes_to_display = st.slider(
                "Nombre maximum de communes à afficher", 
                min_value=10, 
                max_value=300, 
                value=100,
                help="Ajuster pour équilibrer détail et performance"
            )
        
        with col2:
            color_var = st.selectbox(
                "Colorer par variable",
                options=["APL", "Population", "Densité"] if 'density_area' in zone_communes.columns else ["APL", "Population"],
                index=0
            )
        
        @st.cache_data
        def create_enhanced_commune_map(communes, max_points, zone_type, dept, color_var):
            # Échantillonnage stratifié pour garder une représentation correcte
            if len(communes) > max_points:
                # Assurer que les communes importantes sont incluses
                top_communes = communes.nlargest(max_points//5, 'P16_POP')
                rest_sample = communes[~communes.index.isin(top_communes.index)].sample(
                    min(max_points - len(top_communes), len(communes) - len(top_communes)),
                    random_state=42
                )
                display_communes = pd.concat([top_communes, rest_sample])
            else:
                display_communes = communes
            
            # Déterminer la variable de coloration
            if color_var == "APL":
                color_col = 'APL'
                colorscale = 'RdYlGn'
                colorbar_title = "APL"
                cmin = 1
                cmax = 5
            elif color_var == "Population":
                color_col = 'P16_POP'
                colorscale = 'Viridis'
                colorbar_title = "Population"
                cmin = None
                cmax = None
            elif color_var == "Densité" and 'density_area' in display_communes.columns:
                color_col = 'density_area'
                colorscale = 'Blues'
                colorbar_title = "Densité (hab/km²)"
                cmin = None
                cmax = None
            else:
                color_col = 'APL'
                colorscale = 'RdYlGn'
                colorbar_title = "APL"
                cmin = 1
                cmax = 5
            
            # Créer une carte focalisée sur la zone échantillonnée
            zone_map = go.Figure()
            
            # Adapter le template de hover selon les données disponibles
            hover_template = "<b>%{text}</b><br>"
            hover_template += f"{colorbar_title}: %{{marker.color}}"
            
            if color_col != 'P16_POP':
                hover_template += "<br>Population: %{customdata[0]:,.0f}"
                custom_data = [display_communes['P16_POP']]
            else:
                custom_data = []
            
            # Ajouter des données supplémentaires au hover si disponibles
            if 'demographic_profile' in display_communes.columns:
                hover_template += "<br>Profil: %{customdata[" + str(len(custom_data)) + "]}"
                custom_data.append(display_communes['demographic_profile'])
            
            if 'urban_rural' in display_communes.columns:
                hover_template += "<br>Type: %{customdata[" + str(len(custom_data)) + "]}"
                custom_data.append(display_communes['urban_rural'])
            
            # Combiner les données personnalisées
            if custom_data:
                customdata = np.column_stack(custom_data)
            else:
                customdata = None
            
            # Ajouter les marqueurs des communes à la carte
            zone_map.add_trace(go.Scattermapbox(
                lat=display_communes['latitude_mairie'],
                lon=display_communes['longitude_mairie'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=display_communes[color_col],
                    colorscale=colorscale,
                    colorbar=dict(title=colorbar_title),
                    cmin=cmin,
                    cmax=cmax,
                    opacity=0.8
                ),
                text=display_communes['Communes'],
                hovertemplate=hover_template,
                customdata=customdata,
                name=f"{zone_type} - Département {dept}"
            ))
            
            # Déterminer les coordonnées du centre en utilisant la médiane (plus robuste aux outliers)
            lat_center = display_communes['latitude_mairie'].median()
            lon_center = display_communes['longitude_mairie'].median()
            
            # Configuration avancée de la carte
            zone_map.update_layout(
                title=f"{zone_type} - Département {dept}",
                mapbox_style="carto-positron",
                mapbox=dict(
                    center=dict(lat=lat_center, lon=lon_center),
                    zoom=8
                ),
                height=600,
                margin={"r":0,"t":50,"l":0,"b":0},
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial"
                )
            )
            
            return zone_map, len(display_communes)
        
        # Créer et afficher la carte
        with st.spinner("Génération de la carte des communes..."):
            commune_map, displayed_count = create_enhanced_commune_map(
                zone_communes, 
                max_communes_to_display,
                selected_zone_type,
                selected_dept,
                color_var
            )
            
            if len(zone_communes) > displayed_count:
                st.info(f"Affichage d'un échantillon représentatif de {displayed_count} communes sur {len(zone_communes)} pour améliorer les performances.")
            
            st.plotly_chart(commune_map, use_container_width=True)
        
        # Principales communes de la zone (10 plus grandes)
        st.markdown("### Principales communes de la zone")
        
        # Trier par population décroissante
        top_communes = zone_communes.sort_values('P16_POP', ascending=False).head(10)
        
        # Tableau des principales communes avec plus d'informations
        columns_to_display = ['Communes', 'P16_POP', 'APL']
        display_columns = ['Commune', 'Population', 'APL']
        
        # Ajouter des colonnes conditionnellement si elles existent
        if 'density_area' in zone_communes.columns:
            columns_to_display.append('density_area')
            display_columns.append('Densité')
        
        if '60+_pop_rate' in zone_communes.columns:
            columns_to_display.append('60+_pop_rate')
            display_columns.append('% 60+ ans')
        
        if 'median_living_standard' in zone_communes.columns:
            columns_to_display.append('median_living_standard')
            display_columns.append('Niveau de vie')
        
        if 'healthcare_education_establishments' in zone_communes.columns:
            columns_to_display.append('healthcare_education_establishments')
            display_columns.append('Éts santé/éducation')
        
        communes_display = top_communes[columns_to_display].reset_index(drop=True)
        communes_display.columns = display_columns
        
        # Formater les valeurs numériques
        for col in communes_display.select_dtypes(include=['float']).columns:
            if col in ['APL', '% 60+ ans']:
                communes_display[col] = communes_display[col].round(2)
            elif col == 'Niveau de vie':
                communes_display[col] = communes_display[col].round(0).astype(int)
            elif col == 'Densité':
                communes_display[col] = communes_display[col].round(1)
        
        st.dataframe(communes_display)
        
        # Option pour télécharger les données complètes de la zone
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv = convert_df(zone_communes[columns_to_display])
        
        st.download_button(
            label=f"Télécharger les données de la zone (CSV)",
            data=csv,
            file_name=f'{selected_zone_type.replace(" ", "_").lower()}_{selected_dept}.csv',
            mime='text/csv',
        )
        
        # Recommandations spécifiques basées sur le type de zone
        st.markdown("### Recommandations stratégiques")
        
        # Définir des recommandations par type de zone
        recommendations = {
            "Déserts médicaux critiques": [
                "Développer des centres de soins d'urgence mobiles",
                "Mettre en place des incitations financières exceptionnelles pour l'installation",
                "Déployer des solutions de télémédecine d'urgence",
                "Élaborer un plan d'action territorial prioritaire"
            ],
            "Déserts médicaux": [
                "Créer des maisons de santé pluridisciplinaires",
                "Proposer des aides à l'installation pour les nouveaux praticiens",
                "Établir des partenariats avec les facultés de médecine",
                "Développer le transport médical à la demande"
            ],
            "Zones sous-équipées": [
                "Anticiper les départs en retraite des médecins actuels",
                "Diversifier l'offre de soins (spécialistes, paramédicaux)",
                "Améliorer l'attractivité du territoire pour les professionnels",
                "Intégrer la planification médicale dans les projets urbains"
            ],
            "Zones bien équipées": [
                "Maintenir le niveau d'équipement actuel",
                "Favoriser une répartition équilibrée des spécialités",
                "Développer des pôles d'excellence médicale",
                "Optimiser la coordination entre professionnels"
            ],
            "Zones très bien équipées": [
                "Promouvoir l'innovation médicale",
                "Étendre la couverture vers les zones périphériques moins bien desservies",
                "Servir de centre de référence et de formation",
                "Anticiper l'évolution des besoins démographiques futurs"
            ]
        }
        
        # Recommandations spécifiques selon le profil démographique
        demographic_recommendations = {
            "Vieillissante": [
                "Développer des services de maintien à domicile",
                "Renforcer la présence de gériatres et spécialistes des maladies chroniques",
                "Mettre en place des navettes médicales dédiées aux seniors",
                "Créer des programmes de prévention ciblés pour les seniors"
            ],
            "Équilibrée": [
                "Assurer une offre de soins diversifiée pour tous les âges",
                "Développer des centres de santé familiaux",
                "Promouvoir l'éducation à la santé dans les écoles et les entreprises",
                "Équilibrer les services de pédiatrie et de gériatrie"
            ],
            "Jeune": [
                "Renforcer l'offre pédiatrique et obstétrique",
                "Développer des services de planification familiale",
                "Mettre en place des programmes de santé scolaire renforcés",
                "Créer des centres de soins adaptés aux jeunes familles"
            ]
        }
        
        # Recommandations spécifiques selon le profil territorial
        territorial_recommendations = {
            "Rural": [
                "Déployer des cabinets médicaux mobiles",
                "Développer les solutions de télémédecine",
                "Mettre en place des incitations spécifiques pour zones rurales",
                "Créer des maisons de santé inter-communales"
            ],
            "Périurbain": [
                "Renforcer les connexions avec les centres médicaux urbains",
                "Développer des centres de santé de proximité",
                "Optimiser les transports en commun vers les pôles médicaux",
                "Créer des antennes de spécialistes à temps partiel"
            ],
            "Urbain": [
                "Assurer une répartition équilibrée dans tous les quartiers",
                "Développer des pôles de spécialités complémentaires",
                "Renforcer la coordination hôpital-ville",
                "Adapter l'offre aux spécificités socio-démographiques des quartiers"
            ],
            "Très urbain": [
                "Optimiser l'accessibilité des centres de soins existants",
                "Développer des centres de soins non programmés pour désengorger les urgences",
                "Améliorer la coordination des acteurs médicaux nombreux",
                "Adapter l'offre aux populations précaires et aux disparités intra-urbaines"
            ]
        }
        
        # Afficher les recommandations adaptées au type de zone
        if selected_zone_type in recommendations:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### Recommandations pour {selected_zone_type}")
                for rec in recommendations[selected_zone_type]:
                    st.markdown(f"• {rec}")
            
            with col2:
                # Recommandations démographiques si le profil est disponible
                if 'demographic_profile' in selected_zone and not pd.isna(selected_zone['demographic_profile']):
                    profile = selected_zone['demographic_profile']
                    if profile in demographic_recommendations:
                        st.markdown(f"#### Recommandations pour profil {profile}")
                        for rec in demographic_recommendations[profile]:
                            st.markdown(f"• {rec}")
                
                # Ou recommandations territoriales si le type est disponible
                elif 'urban_rural' in selected_zone and not pd.isna(selected_zone['urban_rural']):
                    territory = selected_zone['urban_rural']
                    if territory in territorial_recommendations:
                        st.markdown(f"#### Recommandations pour zone {territory}")
                        for rec in territorial_recommendations[territory]:
                            st.markdown(f"• {rec}")
    
    with tab3:
        st.subheader("Analyse croisée des facteurs")
        
        # Sélection des variables à analyser
        st.markdown("""
        Cette section permet d'analyser les relations entre différentes variables 
        et l'accessibilité aux soins à travers l'ensemble des zones.
        """)
        
        # Vérifier quelles variables sont disponibles
        available_vars = ['APL', 'P16_POP']
        available_names = ['APL', 'Population']
        
        optional_vars = [
            ('density_area', 'Densité de population'),
            ('median_living_standard', 'Niveau de vie médian'),
            ('60+_pop_rate', 'Population âgée (60+)'),
            ('healthcare_education_establishments', 'Équipements de santé/éducation'),
            ('city_social_amenities_rate', 'Équipements sociaux')
        ]
        
        for var, name in optional_vars:
            if var in data_zoned.columns:
                available_vars.append(var)
                available_names.append(name)
        
        # Créer un dictionnaire de correspondance
        var_name_map = dict(zip(available_names, available_vars))
        
        # Interface de sélection des variables
        col1, col2 = st.columns(2)
        
        with col1:
            x_var_name = st.selectbox(
                "Variable X (axe horizontal)",
                options=[name for name in available_names if name != 'APL'],
                index=0
            )
            x_var = var_name_map[x_var_name]
        
        with col2:
            color_by = st.radio(
                "Colorer par",
                options=["Type de zone", "Région", "Profil territorial"],
                horizontal=True
            )
        
        # Préparation des données pour la visualisation
        @st.cache_data
        def prepare_cross_analysis_data(data, x_var, color_by):
            # Agréger les données au niveau approprié
            if color_by == "Type de zone":
                # Agrégation par département et type de zone
                grouped = data.groupby(['departement', 'zone_type']).agg({
                    'CODGEO': 'count',
                    'P16_POP': 'sum',
                    'APL': 'mean',
                    x_var: 'mean' if x_var != 'P16_POP' else 'sum'
                }).reset_index()
                
                plot_data = grouped.rename(columns={
                    'departement': 'Département',
                    'zone_type': 'Type de zone',
                    'CODGEO': 'Nombre de communes',
                    'APL': 'APL moyen',
                    x_var: x_var_name
                })
                
                color_col = 'Type de zone'
                
            elif color_by == "Région":
                # Ajouter la région si elle n'est pas déjà présente
                if 'region' not in data.columns:
                    # La logique pour ajouter la région devrait être ici
                    pass
                
                # Agrégation par région et type de zone
                grouped = data.groupby(['region', 'zone_type']).agg({
                    'CODGEO': 'count',
                    'P16_POP': 'sum',
                    'APL': 'mean',
                    x_var: 'mean' if x_var != 'P16_POP' else 'sum'
                }).reset_index()
                
                plot_data = grouped.rename(columns={
                    'region': 'Région',
                    'zone_type': 'Type de zone',
                    'CODGEO': 'Nombre de communes',
                    'APL': 'APL moyen',
                    x_var: x_var_name
                })
                
                color_col = 'Région'
                
            else:  # "Profil territorial"
                if 'urban_rural' in data.columns:
                    # Agrégation par type territorial et type de zone
                    grouped = data.groupby(['urban_rural', 'zone_type']).agg({
                        'CODGEO': 'count',
                        'P16_POP': 'sum',
                        'APL': 'mean',
                        x_var: 'mean' if x_var != 'P16_POP' else 'sum'
                    }).reset_index()
                    
                    plot_data = grouped.rename(columns={
                        'urban_rural': 'Type territorial',
                        'zone_type': 'Type de zone',
                        'CODGEO': 'Nombre de communes',
                        'APL': 'APL moyen',
                        x_var: x_var_name
                    })
                    
                    color_col = 'Type territorial'
                else:
                    # Fallback si urban_rural n'est pas disponible
                    grouped = data.groupby(['zone_type']).agg({
                        'CODGEO': 'count',
                        'P16_POP': 'sum',
                        'APL': 'mean',
                        x_var: 'mean' if x_var != 'P16_POP' else 'sum'
                    }).reset_index()
                    
                    plot_data = grouped.rename(columns={
                        'zone_type': 'Type de zone',
                        'CODGEO': 'Nombre de communes',
                        'APL': 'APL moyen',
                        x_var: x_var_name
                    })
                    
                    color_col = 'Type de zone'
            
            return plot_data, color_col
        
        # Préparer les données
        plot_data, color_col = prepare_cross_analysis_data(data_zoned, x_var, color_by)
        
        # Créer un nuage de points interactif
        fig = px.scatter(
            plot_data,
            x=x_var_name,
            y='APL moyen',
            color=color_col,
            size='Nombre de communes',
            hover_name='Type de zone' if color_col != 'Type de zone' else None,
            hover_data=['Nombre de communes', 'APL moyen'],
            title=f"Relation entre {x_var_name} et APL moyen",
            height=600,
            opacity=0.8
        )
        
        # Améliorer l'apparence du graphique
        fig.update_layout(
            xaxis_title=x_var_name,
            yaxis_title="APL moyen",
            legend_title=color_col,
            margin={"r":0,"t":50,"l":0,"b":0}
        )
        
        # Ajouter une ligne horizontale pour le seuil de désert médical
        fig.add_hline(
            y=2.5,
            line_dash="dash",
            line_color="red",
            annotation_text="Seuil désert médical (2.5)",
            annotation_position="right"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Analyse de corrélation
        st.subheader("Analyse de corrélation")
        
        # Variables pour l'analyse de corrélation
        @st.cache_data
        def analyze_correlations(data):
            # Sélectionner les variables numériques pertinentes
            numeric_vars = ['APL']
            
            for var in available_vars:
                if var != 'APL' and var in data.columns:
                    numeric_vars.append(var)
            
            # Calculer la matrice de corrélation
            corr_matrix = data[numeric_vars].corr()
            
            # Créer un DataFrame formaté pour l'affichage
            corr_with_apl = pd.DataFrame({
                'Variable': corr_matrix.index[1:],  # Exclure APL de la liste
                'Corrélation avec APL': corr_matrix.iloc[0, 1:]  # Prendre la première ligne (APL) sans la diagonale
            })
            
            # Traduire les noms de variables
            var_translation = dict(zip(available_vars, available_names))
            corr_with_apl['Variable'] = corr_with_apl['Variable'].map(lambda x: var_translation.get(x, x))
            
            # Trier par valeur absolue de corrélation
            corr_with_apl['Abs_Corr'] = corr_with_apl['Corrélation avec APL'].abs()
            corr_with_apl = corr_with_apl.sort_values('Abs_Corr', ascending=False).drop('Abs_Corr', axis=1)
            
            return corr_with_apl, corr_matrix
        
        corr_with_apl, corr_matrix = analyze_correlations(data_zoned)
        
        # Afficher les corrélations avec l'APL
        col1, col2 = st.columns([3, 2])
        
        with col1:
            fig = px.bar(
                corr_with_apl,
                x='Corrélation avec APL',
                y='Variable',
                orientation='h',
                title="Facteurs influençant l'accessibilité aux soins (APL)",
                color='Corrélation avec APL',
                color_continuous_scale='RdBu_r',
                height=400
            )
            
            fig.update_layout(
                xaxis_title="Coefficient de corrélation avec l'APL",
                yaxis_title="",
                margin={"r":0,"t":50,"l":0,"b":0}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Interprétation des corrélations")
            st.markdown("""
            Les barres bleues indiquent une corrélation positive avec l'APL (meilleur accès aux soins),
            tandis que les barres rouges montrent une corrélation négative (moins bon accès aux soins).
            
            L'intensité de la couleur et la longueur de la barre indiquent la force de la relation.
            
            **Quelques observations clés:**
            - Les zones denses sont généralement mieux desservies médicalement
            - Le niveau de vie influence positivement l'accès aux soins
            - Les équipements de santé et sociaux sont associés à un meilleur APL
            """)
            
            # Recommandations basées sur les corrélations
            strongest_corr = corr_with_apl.iloc[0]['Variable']
            corr_value = corr_with_apl.iloc[0]['Corrélation avec APL']
            
            st.markdown(f"**Facteur le plus influent: {strongest_corr}** (corrélation: {corr_value:.2f})")
            
            # Recommandations selon le facteur principal
            if strongest_corr == "Densité de population" and corr_value > 0:
                st.markdown("""
                **Recommandation prioritaire:**
                Développer des solutions adaptées aux zones peu denses, comme les cabinets mobiles
                et la télémédecine, pour compenser la relation entre densité et accès aux soins.
                """)
            elif strongest_corr == "Population âgée (60+)" and corr_value < 0:
                st.markdown("""
                **Recommandation prioritaire:**
                Déployer des programmes spécifiques pour les territoires vieillissants,
                incluant transport médical adapté et services médicaux à domicile.
                """)
            elif strongest_corr == "Niveau de vie médian" and corr_value > 0:
                st.markdown("""
                **Recommandation prioritaire:**
                Mettre en place des incitations financières ciblées pour l'installation
                dans les zones à niveau de vie modeste.
                """)
        
        # Synthèse finale
        st.subheader("Synthèse et recommandations globales")
        
        # Calculer quelques statistiques clés
        @st.cache_data
        def calculate_final_metrics(data):
            metrics = {
                'desert_count': len(data[data['APL'] < 2.5]),
                'total_count': len(data),
                'desert_percent': len(data[data['APL'] < 2.5]) / len(data) * 100 if len(data) > 0 else 0,
                'pop_in_desert': data[data['APL'] < 2.5]['P16_POP'].sum(),
                'total_pop': data['P16_POP'].sum(),
                'pop_desert_percent': data[data['APL'] < 2.5]['P16_POP'].sum() / data['P16_POP'].sum() * 100 if data['P16_POP'].sum() > 0 else 0
            }
            return metrics
        
        metrics = calculate_final_metrics(data_zoned)
        
        # Affichage du résumé
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Communes en désert médical",
                f"{metrics['desert_count']:,} / {metrics['total_count']:,}".replace(',', ' '),
                f"{metrics['desert_percent']:.1f}%"
            )
        
        with col2:
            st.metric(
                "Population en désert médical",
                f"{int(metrics['pop_in_desert']):,} / {int(metrics['total_pop']):,}".replace(',', ' '),
                f"{metrics['pop_desert_percent']:.1f}%"
            )
        
        with col3:
            # Trouver la région ou le département le plus touché
            if 'region' in data_zoned.columns:
                region_stats = data_zoned.groupby('region').apply(
                    lambda x: (x[x['APL'] < 2.5]['P16_POP'].sum() / x['P16_POP'].sum() * 100) if x['P16_POP'].sum() > 0 else 0
                ).sort_values(ascending=False)
                
                if not region_stats.empty:
                    worst_region = region_stats.index[0]
                    worst_pct = region_stats.iloc[0]
                    st.metric("Région la plus touchée", worst_region, f"{worst_pct:.1f}%")
                else:
                    st.metric("Analyse régionale", "Données insuffisantes", "")
            else:
                dept_stats = data_zoned.groupby('departement').apply(
                    lambda x: (x[x['APL'] < 2.5]['P16_POP'].sum() / x['P16_POP'].sum() * 100) if x['P16_POP'].sum() > 0 else 0
                ).sort_values(ascending=False)
                
                if not dept_stats.empty:
                    worst_dept = dept_stats.index[0]
                    worst_pct = dept_stats.iloc[0]
                    st.metric("Département le plus touché", worst_dept, f"{worst_pct:.1f}%")
                else:
                    st.metric("Analyse départementale", "Données insuffisantes", "")
        
        # Recommandations finales
        st.markdown("""
        ### Stratégies recommandées pour lutter contre les déserts médicaux
        
        Ces recommandations sont basées sur l'analyse des clusters et des facteurs corrélés à l'accessibilité aux soins :
        
        **1. Adaptation territoriale différenciée**
        - Déployer des stratégies spécifiques à chaque type de territoire (urbain, rural, périurbain)
        - Prioriser les interventions dans les zones à haut risque identifiées
        
        **2. Solutions innovantes pour zones peu denses**
        - Cabinets médicaux mobiles pour les zones rurales isolées
        - Développement ciblé de la télémédecine avec points d'accès dans chaque commune
        - Système de navettes médicales pour les populations à mobilité réduite
        
        **3. Incitations économiques**
        - Aides financières et fiscales adaptées au niveau de vie du territoire
        - Prise en charge des coûts d'installation dans les zones prioritaires
        - Bonus de rémunération pour exercice en zone sous-dotée
        
        **4. Planification coordonnée**
        - Intégration de l'accessibilité médicale dans tous les projets d'aménagement
        - Coopération intercommunale pour mutualiser les ressources
        - Anticipation des départs en retraite des médecins avec recrutement préventif
        """)

def select_view_mode():
    if view_mode == "Clusters de communes":
        view_mode_clusters(filtered_data)


# Chargement des données
data = load_data()

if not data.empty:
    # Préparation des données
    data = create_apl_categories(data)
    data_risk = predict_future_desert_risk(data)
    data_clusters, cluster_analysis = create_clusters(data)
    
    # Statistiques générales
    stats = calculate_stats(data)
    
    # Analyses spécifiques
    pop_analysis = analyze_by_population(data)
    age_corr, age_analysis = analyze_by_age(data)
    
    # Interface utilisateur
    st.title("🏥 Medical'IA - Analyse des Déserts Médicaux")
    
    # Sidebar
    st.sidebar.title("Filtres & Navigation")
    
    # Filtres principaux
    view_mode = st.sidebar.radio(
        "Mode de visualisation",
        ["Vue d'ensemble", "Cartographie détaillée", "Analyses territoriales", "Analyses socio-démographiques", 
        "Clusters de communes", "Prévisions & Risques", "Générateur de rapports"]
    )

    analytics_data = {
        "Vue actuelle": view_mode, # Indiquer la vue active
        # Ajoutez ici les stats globales si nécessaire
        "Statistiques globales": {
            "Communes analysées": f"{stats['communes_count']:,}".replace(",", " "),
            "APL moyen (pondéré)": f"{stats['weighted_avg_apl']:.2f}",
            "% Communes en désert": f"{stats['desert_percent']:.1f}%"
        }
    }


    
    # Filtres secondaires
    region_filter = st.sidebar.multiselect(
        "Filtrer par départements (codes)",
        options=sorted(data['CODGEO'].astype(str).str[:2].unique().tolist()),
        default=[]
    )
    
    population_min = st.sidebar.slider(
        "Population minimale",
        min_value=int(data['P16_POP'].min()),
        max_value=int(data['P16_POP'].max()),
        value=0
    )
    
    apl_range = st.sidebar.slider(
        "Plage d'APL",
        min_value=float(data['APL'].min()),
        max_value=float(data['APL'].max()),
        value=(float(data['APL'].min()), float(data['APL'].max()))
    )
    
    # Application des filtres
    filtered_data = data.copy()
    
    if region_filter:
        filtered_data = filtered_data[filtered_data['CODGEO'].str[:2].isin(region_filter)]
    
    filtered_data = filtered_data[
        (filtered_data['P16_POP'] >= population_min) &
        (filtered_data['APL'] >= apl_range[0]) &
        (filtered_data['APL'] <= apl_range[1])
    ]
    
    # Adaptation des analyses aux données filtrées
    filtered_stats = calculate_stats(filtered_data)
    filtered_data_risk = data_risk[data_risk['CODGEO'].isin(filtered_data['CODGEO'])]
    
    if view_mode == "Générateur de rapports":
        add_report_generator_ui(data, filtered_data)

    # Affichage en fonction du mode choisi
    elif view_mode == "Vue d'ensemble":
        # En-tête et présentation

        analytics_data["Détails Vue d'ensemble"] = {
            "Corrélations notables": "L'APL est négativement corrélé à la part des 60+ ans et positivement au niveau de vie.",
            # Ajoutez d'autres résumés pertinents
        }

        st.header("État des lieux des déserts médicaux en France")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Communes analysées", f"{filtered_stats['communes_count']:,}".replace(",", " "))
        with col2:
            st.metric("APL moyen", f"{filtered_stats['avg_apl']:.2f}")
        with col3:
            st.metric("APL moyen pondéré", f"{filtered_stats['weighted_avg_apl']:.2f}", 
                    help="Pondéré par la population: Σ(APL × Population) / Σ(Population)")
        with col4:
            st.metric("Communes en désert médical", f"{filtered_stats['desert_percent']:.1f}%")
        
        st.markdown("""
        ### Qu'est-ce que l'indice APL ?
        L'**Accessibilité Potentielle Localisée (APL)** est un indicateur développé par la DREES et l'IRDES 
        qui mesure l'accès aux médecins généralistes. Il s'exprime en nombre de consultations/visites 
        accessibles par habitant et par an.
        
        - **APL < 2,5** : Zone considérée comme désert médical
        - **APL < 1,5** : Situation critique
        """)
        
        # Carte résumée de la France
        st.subheader("Cartographie des déserts médicaux par département")

        # Préparation des données pour la carte départementale
        @st.cache_data
        def prepare_department_data(data):
            """Prépare les données agrégées par département pour la carte choroplèthe"""
            # Extraire le code département des codes INSEE
            data['departement'] = data['CODGEO'].str[:2]
            
            # Grouper les données par département
            dept_data = data.groupby('departement').agg({
                'P16_POP': 'sum',
                'APL': lambda x: np.average(x, weights=data.loc[x.index, 'P16_POP']),
                'CODGEO': 'count'
            }).reset_index()
            
            # Calculer le pourcentage de communes en désert médical
            desert_by_dept = data[data['APL'] < 2.5].groupby('departement').size()
            dept_data['desert_count'] = dept_data['departement'].map(desert_by_dept).fillna(0)
            dept_data['desert_percent'] = (dept_data['desert_count'] / dept_data['CODGEO'] * 100).round(1)
            
            # Renommer les colonnes
            dept_data.columns = ['departement', 'population', 'apl_pondere', 'communes_count', 'desert_count', 'desert_percent']
            
            return dept_data

        # Chargement du fichier GeoJSON des départements
        @st.cache_data
        def load_dept_geojson():
            """Charge le fichier GeoJSON des départements français"""
            try:
                with open("departements.geojson", 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier GeoJSON des départements: {e}")
                return None

        # Préparer les données
        dept_data = prepare_department_data(filtered_data)
        dept_geojson = load_dept_geojson()

        if dept_geojson:
            # Options pour la carte
            map_metric = st.radio(
                "Afficher sur la carte:",
                ["APL moyen pondéré", "% communes en désert médical"],
                horizontal=True
            )
            
            if map_metric == "APL moyen pondéré":
                # Carte de l'APL
                fig = px.choropleth_mapbox(
                    dept_data, 
                    geojson=dept_geojson, 
                    locations='departement',
                    featureidkey="properties.code",
                    color='apl_pondere',
                    color_continuous_scale="RdYlGn",
                    range_color=[1, 5],
                    mapbox_style="carto-positron",
                    zoom=4.5,
                    center={"lat": 46.603354, "lon": 1.888334},
                    opacity=0.8,
                    labels={'apl_pondere': 'APL pondéré'},
                    hover_data=['population', 'communes_count', 'desert_percent']
                )
                
                # Améliorer le hover
                fig.update_traces(
                    hovertemplate="<b>Département %{location}</b><br>" +
                                "APL moyen pondéré: %{z:.2f}<br>" +
                                "Population: %{customdata[0]:,.0f}<br>" +
                                "Communes: %{customdata[1]}<br>" +
                                "% en désert médical: %{customdata[2]:.1f}%"
                )
                
                # Titre
                fig.update_layout(
                    title="Accessibilité potentielle localisée (APL) par département",
                    coloraxis_colorbar=dict(title="APL"),
                    height=700,
                    margin={"r":0,"t":50,"l":0,"b":0}
                )
                
            else:
                # Carte du pourcentage de déserts médicaux
                fig = px.choropleth_mapbox(
                    dept_data, 
                    geojson=dept_geojson, 
                    locations='departement',
                    featureidkey="properties.code",
                    color='desert_percent',
                    color_continuous_scale="RdYlGn_r",  # Inversé pour que le rouge = mauvais
                    range_color=[0, 50],  # Max à 50% pour mieux voir les différences
                    mapbox_style="carto-positron",
                    zoom=4.5,
                    center={"lat": 46.603354, "lon": 1.888334},
                    opacity=0.8,
                    labels={'desert_percent': '% en désert médical'},
                    hover_data=['population', 'communes_count', 'apl_pondere']
                )
                
                # Améliorer le hover
                fig.update_traces(
                    hovertemplate="<b>Département %{location}</b><br>" +
                                "% en désert médical: %{z:.1f}%<br>" +
                                "Population: %{customdata[0]:,.0f}<br>" +
                                "Communes: %{customdata[1]}<br>" +
                                "APL moyen: %{customdata[2]:.2f}"
                )
                
                # Titre
                fig.update_layout(
                    title="Pourcentage de communes en désert médical par département",
                    coloraxis_colorbar=dict(title="%"),
                    height=700,
                    margin={"r":0,"t":50,"l":0,"b":0}
                )
            
            # Afficher la carte
            st.plotly_chart(fig, use_container_width=True)
            
            # Afficher les départements les plus touchés
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 5 départements les plus touchés")
                worst_depts = dept_data.sort_values('desert_percent', ascending=False).head(5)
                
                # Créer un tableau lisible
                worst_display = worst_depts[['departement', 'desert_percent', 'apl_pondere']].copy()
                worst_display.columns = ['Département', '% en désert médical', 'APL moyen']
                worst_display['APL moyen'] = worst_display['APL moyen'].round(2)
                
                st.table(worst_display)
            
            with col2:
                st.subheader("Top 5 départements les mieux desservis")
                best_depts = dept_data.sort_values('desert_percent').head(5)
                
                # Créer un tableau lisible
                best_display = best_depts[['departement', 'desert_percent', 'apl_pondere']].copy()
                best_display.columns = ['Département', '% en désert médical', 'APL moyen']
                best_display['APL moyen'] = best_display['APL moyen'].round(2)
                
                st.table(best_display)
                
        else:
            st.error("Impossible de charger la carte des départements. Vérifiez que le fichier 'departements.geojson' est disponible.")
            
            # Affichage de secours (carte basique)
            st.warning("Affichage d'une carte simplifiée en attendant.")
            map_overview = create_map(filtered_data, column="APL", title="Carte des déserts médicaux en France (intensité inversement proportionnelle à l'APL)")
            st.plotly_chart(map_overview, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Répartition des communes par catégorie d'APL
        st.subheader("Répartition des communes par niveau d'accès aux soins")
        apl_counts = filtered_data['APL_category'].value_counts().reset_index()
        apl_counts.columns = ['Catégorie', 'Nombre de communes']
        
        ordre_categories_acces = [
            "Désert médical critique",    # 1er (le pire)
            "Désert médical",             # 2ème
            "Sous-équipement médical",    # 3ème
            "Équipement médical suffisant",# 4ème
            "Bon équipement médical"      # 5ème (le meilleur)
        ]

        fig = px.bar(
            apl_counts,
            x='Catégorie',                 # L'axe à ordonner
            y='Nombre de communes',
            color='Catégorie',
            color_discrete_map={           # Garder le mapping des couleurs
                'Désert médical critique': 'darkred',
                'Désert médical': 'red',
                'Sous-équipement médical': 'orange',
                'Équipement médical suffisant': 'lightgreen',
                'Bon équipement médical': 'green'
            },
            labels={
                'Nombre de communes': 'Nombre de communes',
                'Catégorie': "Niveau d'accès" # Label X mis à jour (optionnel)
                },
            height=400,
            category_orders={
                "Catégorie": ordre_categories_acces  # <-- Ajout de cette ligne pour forcer l'ordre
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Facteurs corrélés avec les déserts médicaux
        st.subheader("Principaux facteurs liés aux déserts médicaux")
        
        # Calcul des corrélations
        correlation_vars = [
            'P16_POP', 'median_living_standard', 'healthcare_education_establishments',
            'density_area', 'unemployment_rate', 'active_local_business_rate',
            'city_social_amenities_rate', '0_14_pop_rate', '15_59_pop_rate', '60+_pop_rate'
        ]
        
        correlations = filtered_data[correlation_vars + ['APL']].corr()['APL'].sort_values().drop('APL')
        
        # Afficher les corrélations
        corr_data = pd.DataFrame({
            'Facteur': correlations.index,
            'Corrélation': correlations.values
        })
        
        # Renommage des facteurs pour meilleure lisibilité
        factor_names = {
            'P16_POP': 'Population',
            'median_living_standard': 'Niveau de vie médian',
            'healthcare_education_establishments': 'Établissements de santé/éducation',
            'density_area': 'Densité de population',
            'unemployment_rate': 'Taux de chômage',
            'active_local_business_rate': 'Taux d\'entreprises actives',
            'city_social_amenities_rate': 'Équipements sociaux',
            '0_14_pop_rate': 'Population 0-14 ans',
            '15_59_pop_rate': 'Population 15-59 ans',
            '60+_pop_rate': 'Population 60+ ans'
        }
        
        corr_data['Facteur'] = corr_data['Facteur'].map(factor_names).fillna(corr_data['Facteur'])
        
        fig = px.bar(
            corr_data,
            x='Corrélation',
            y='Facteur',
            orientation='h',
            color='Corrélation',
            color_continuous_scale='RdBu_r',
            labels={'Corrélation': 'Corrélation avec l\'indice APL'},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
    elif view_mode == "Cartographie détaillée":
        st.header("Cartographie détaillée des déserts médicaux")
        
        # Afficher uniquement les cartes choroplèthes
        st.subheader("Cartes choroplèthes par territoire")
        
        # Choix du niveau territorial
        geo_level = st.radio(
            "Niveau territorial", 
            ["Régions", "Départements", "Communes"], 
            horizontal=True
        )
        
        # Définir le niveau et le fichier GeoJSON approprié
        if geo_level == "Communes":
            geojson_file = "communes.geojson"
            level = "commune"
            map_title = "Accessibilité aux soins par commune en France"
        elif geo_level == "Départements":
            geojson_file = "departements.geojson"
            level = "departement"
            map_title = "Accessibilité aux soins par département"
        else:  # Régions
            geojson_file = "regions-avec-outre-mer.geojson"
            level = "region"
            map_title = "Accessibilité aux soins par région"
        
        # Préparation des données territoriales
        territory_stats = calculate_territorial_apl(filtered_data, level)
        
        # Chargement du GeoJSON - utiliser la version optimisée
        geojson_data = load_optimized_geojson(geojson_file)
        
        if geojson_data:
            # Modifier les options pour le cas des communes
            if level == "commune":
                with st.expander("Options avancées pour l'affichage des communes"):
                    # Ajout de contrôles pour limiter ou non l'affichage
                    limit_display = st.checkbox("Limiter le nombre de communes affichées", value=False,
                                            help="Cochez pour limiter le nombre de communes affichées et améliorer les performances")
                    
                    max_communes = None
                    if limit_display:
                        max_communes = st.slider("Nombre maximum de communes à afficher", 
                                            min_value=100, 
                                            max_value=10000, 
                                            value=3000)
                        st.warning(f"L'affichage est limité à {max_communes} communes pour des raisons de performance.")
                    
                    # Si trop de communes et l'utilisateur a choisi de ne pas limiter
                    if not limit_display and len(territory_stats) > 10000:
                        st.warning(f"Attention: l'affichage de {len(territory_stats)} communes peut ralentir votre navigateur. "
                                f"Considérez d'utiliser les filtres pour réduire le nombre de communes ou activer la limitation.")
                
                # Limiter si demandé ou si trop de communes
                if limit_display and max_communes is not None:
                    # Si limitation activée, prendre un échantillon stratifié
                    if len(territory_stats) > max_communes:
                        # Assurer la présence des communes importantes et des extrêmes
                        desert_communes = territory_stats[territory_stats['apl_pondere'] < 2.5].nsmallest(max_communes//4, 'apl_pondere')
                        high_apl_communes = territory_stats.nlargest(max_communes//4, 'apl_pondere')
                        large_communes = territory_stats.nlargest(max_communes//4, 'population')
                        
                        # Prendre le reste aléatoirement
                        selected_indices = set(desert_communes.index) | set(high_apl_communes.index) | set(large_communes.index)
                        remaining = max_communes - len(selected_indices)
                        if remaining > 0:
                            other_communes = territory_stats[~territory_stats.index.isin(selected_indices)]
                            if len(other_communes) > 0:
                                random_communes = other_communes.sample(min(remaining, len(other_communes)))
                                selected_indices = selected_indices | set(random_communes.index)
                        
                        # Combiner les sélections
                        territory_stats_display = territory_stats.loc[list(selected_indices)]
                        st.info(f"Affichage de {len(territory_stats_display)} communes sur {len(territory_stats)} au total.")
                    else:
                        territory_stats_display = territory_stats
                else:
                    territory_stats_display = territory_stats
            else:
                territory_stats_display = territory_stats
                
            # Options d'affichage avancées
            with st.expander("Options d'affichage de la carte"):
                col1, col2 = st.columns(2)
                with col1:
                    color_scale = st.selectbox(
                        "Échelle de couleurs",
                        options=["RdYlGn", "RdYlGn_r", "YlOrRd", "YlGnBu", "Viridis", "Plasma"],
                        index=0
                    )
                with col2:
                    if level == "commune":
                        simplify_tolerance = st.slider(
                            "Simplification de la carte (performance)",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.0,
                            step=0.1,
                            help="Augmenter pour améliorer la performance (au détriment de la précision)"
                        )
                    else:
                        simplify_tolerance = 0
            
            # Créer la carte choroplèthe avec la fonction améliorée
            choropleth_map = create_enhanced_choropleth_map(
                territory_stats_display,
                geojson_data,
                level=level,
                value_col="apl_pondere",
                title=map_title,
                color_scale=color_scale,
                range_color=[1, 5],
                simplify_tolerance=simplify_tolerance
            )
            
            # Afficher la carte
            st.plotly_chart(choropleth_map, use_container_width=True)
            
            # Afficher le tableau des statistiques
            st.subheader(f"Statistiques par {geo_level.lower()}")
            
            # Formater les données pour l'affichage
            if level == "commune":
                display_df = territory_stats[['territoire', 'nom_commune', 'population', 'apl_pondere', 'desert_percent']].copy()
                display_df.columns = ['Code INSEE', 'Commune', 'Population', 'APL', 'Désert médical (%)']
            elif level == "departement":
                display_df = territory_stats.rename(columns={
                    'territoire': 'Département',
                    'population': 'Population',
                    'apl_pondere': 'APL pondéré',
                    'communes_count': 'Nombre de communes',
                    'desert_percent': '% en désert médical',
                    'desert_count': 'Communes en désert',
                    'min_apl': 'APL minimum',
                    'max_apl': 'APL maximum'
                })
            else:  # région
                display_df = territory_stats.rename(columns={
                    'territoire': 'Région',
                    'population': 'Population',
                    'apl_pondere': 'APL pondéré',
                    'communes_count': 'Nombre de communes',
                    'desert_percent': '% en désert médical',
                    'desert_count': 'Communes en désert',
                    'min_apl': 'APL minimum',
                    'max_apl': 'APL maximum'
                })
            
            # Arrondir les colonnes numériques
            for col in display_df.select_dtypes(include=['float']).columns:
                display_df[col] = display_df[col].round(2)
            
            # Afficher le tableau avec options de tri et filtrage
            st.dataframe(display_df, height=500)
            
            # Option pour télécharger les données
            @st.cache_data
            def generate_csv(df):
                return df.to_csv(index=False).encode('utf-8')
            
            csv = generate_csv(display_df)
            st.download_button(
                label=f"Télécharger les données ({geo_level.lower()})",
                data=csv,
                file_name=f"{geo_level.lower()}_stats.csv",
                mime='text/csv'
            )
            
            # Ajouter des métriques clés
            if level != "commune":  # Pour régions et départements uniquement
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "APL moyen pondéré", 
                        f"{(territory_stats['apl_pondere'] * territory_stats['population']).sum() / territory_stats['population'].sum():.2f}"
                    )
                with col2:
                    st.metric(
                        "Population en désert médical", 
                        f"{sum(territory_stats['desert_percent'] * territory_stats['population']) / 100 / sum(territory_stats['population']) * 100:.1f}%"
                    )
                with col3:
                    highest_desert = display_df.nlargest(1, '% en désert médical' if level != 'commune' else 'Désert médical (%)')
                    highest_name = highest_desert.iloc[0]['Région' if level == 'region' else 'Département' if level == 'departement' else 'Commune']
                    highest_value = highest_desert.iloc[0]['% en désert médical' if level != 'commune' else 'Désert médical (%)']
                    st.metric(
                        f"{geo_level[:-1] if geo_level.endswith('s') else geo_level} le plus touché", 
                        f"{highest_name} ({highest_value:.1f}%)"
                    )
        else:
            st.error(f"Impossible de charger le fichier GeoJSON pour les {geo_level.lower()}.")

    elif view_mode == "Analyses territoriales":
        st.header("Analyse territoriale des déserts médicaux")
        
        analysis_mode = st.radio(
            "Mode d'analyse territoriale",
            ["Recherche ciblée", "Vue complète du territoire"],
            horizontal=True
        )

        if analysis_mode == "Recherche ciblée":
            # Initialiser les variables de session pour le drill-down territorial
            if 'territorial_level' not in st.session_state:
                st.session_state.territorial_level = "region"
            if 'selected_region' not in st.session_state:
                st.session_state.selected_region = None
            if 'selected_department' not in st.session_state:
                st.session_state.selected_department = None

            # Bouton pour remonter d'un niveau
            if st.session_state.territorial_level != "region":
                if st.button("↩️ Remonter d'un niveau"):
                    if st.session_state.territorial_level == "departement":
                        st.session_state.territorial_level = "region"
                        st.session_state.selected_region = None
                    elif st.session_state.territorial_level == "commune":
                        st.session_state.territorial_level = "departement"
                        if hasattr(st.session_state, 'selected_department'):
                            st.session_state.selected_department = None
                        if hasattr(st.session_state, 'selected_departments'):
                            delattr(st.session_state, 'selected_departments')
                    st.rerun()

            # Afficher le chemin de navigation actuel
            nav_text = "🌍 France"
            if st.session_state.selected_region:
                nav_text += f" > 🏞️ {st.session_state.selected_region}"
            if st.session_state.selected_department:
                nav_text += f" > 🏙️ Département {st.session_state.selected_department}"
            st.write(nav_text)

            # Déterminer le niveau territorial actuel
            geo_level = st.session_state.territorial_level

            # Table de correspondance département-région pour filtrer les données
            region_map = {
                '01': 'Auvergne-Rhône-Alpes', '03': 'Auvergne-Rhône-Alpes', '07': 'Auvergne-Rhône-Alpes', 
                '15': 'Auvergne-Rhône-Alpes', '26': 'Auvergne-Rhône-Alpes', '38': 'Auvergne-Rhône-Alpes',
                '42': 'Auvergne-Rhône-Alpes', '43': 'Auvergne-Rhône-Alpes', '63': 'Auvergne-Rhône-Alpes',
                '69': 'Auvergne-Rhône-Alpes', '73': 'Auvergne-Rhône-Alpes', '74': 'Auvergne-Rhône-Alpes',
                '21': 'Bourgogne-Franche-Comté', '25': 'Bourgogne-Franche-Comté', '39': 'Bourgogne-Franche-Comté',
                '58': 'Bourgogne-Franche-Comté', '70': 'Bourgogne-Franche-Comté', '71': 'Bourgogne-Franche-Comté',
                '89': 'Bourgogne-Franche-Comté', '90': 'Bourgogne-Franche-Comté',
                '22': 'Bretagne', '29': 'Bretagne', '35': 'Bretagne', '56': 'Bretagne',
                '18': 'Centre-Val de Loire', '28': 'Centre-Val de Loire', '36': 'Centre-Val de Loire',
                '37': 'Centre-Val de Loire', '41': 'Centre-Val de Loire', '45': 'Centre-Val de Loire',
                '2A': 'Corse', '2B': 'Corse',
                '08': 'Grand Est', '10': 'Grand Est', '51': 'Grand Est', '52': 'Grand Est',
                '54': 'Grand Est', '55': 'Grand Est', '57': 'Grand Est', '67': 'Grand Est',
                '68': 'Grand Est', '88': 'Grand Est',
                '02': 'Hauts-de-France', '59': 'Hauts-de-France', '60': 'Hauts-de-France',
                '62': 'Hauts-de-France', '80': 'Hauts-de-France',
                '75': 'Île-de-France', '77': 'Île-de-France', '78': 'Île-de-France', '91': 'Île-de-France',
                '92': 'Île-de-France', '93': 'Île-de-France', '94': 'Île-de-France', '95': 'Île-de-France',
                '14': 'Normandie', '27': 'Normandie', '50': 'Normandie', '61': 'Normandie', '76': 'Normandie',
                '16': 'Nouvelle-Aquitaine', '17': 'Nouvelle-Aquitaine', '19': 'Nouvelle-Aquitaine',
                '23': 'Nouvelle-Aquitaine', '24': 'Nouvelle-Aquitaine', '33': 'Nouvelle-Aquitaine',
                '40': 'Nouvelle-Aquitaine', '47': 'Nouvelle-Aquitaine', '64': 'Nouvelle-Aquitaine',
                '79': 'Nouvelle-Aquitaine', '86': 'Nouvelle-Aquitaine', '87': 'Nouvelle-Aquitaine',
                '09': 'Occitanie', '11': 'Occitanie', '12': 'Occitanie', '30': 'Occitanie',
                '31': 'Occitanie', '32': 'Occitanie', '34': 'Occitanie', '46': 'Occitanie',
                '48': 'Occitanie', '65': 'Occitanie', '66': 'Occitanie', '81': 'Occitanie', '82': 'Occitanie',
                '44': 'Pays de la Loire', '49': 'Pays de la Loire', '53': 'Pays de la Loire',
                '72': 'Pays de la Loire', '85': 'Pays de la Loire',
                '04': 'Provence-Alpes-Côte d\'Azur', '05': 'Provence-Alpes-Côte d\'Azur',
                '06': 'Provence-Alpes-Côte d\'Azur', '13': 'Provence-Alpes-Côte d\'Azur',
                '83': 'Provence-Alpes-Côte d\'Azur', '84': 'Provence-Alpes-Côte d\'Azur',
                '971': 'Outre-Mer', '972': 'Outre-Mer', '973': 'Outre-Mer', '974': 'Outre-Mer', '976': 'Outre-Mer',
                '975': 'Outre-Mer', '977': 'Outre-Mer', '978': 'Outre-Mer', '986': 'Outre-Mer', '987': 'Outre-Mer',
                '988': 'Outre-Mer'
            }

            # Si au niveau commune, vérifier le nombre pour éviter les problèmes de performance
            if geo_level == "commune":
                if hasattr(st.session_state, 'selected_departments') and len(st.session_state.selected_departments) > 0:
                    # Pour Paris (75), nous voulons inclure tous les arrondissements
                    if '75' in st.session_state.selected_departments:
                        # Créer une condition spéciale pour Paris qui capture les arrondissements
                        paris_condition = filtered_data['CODGEO'].str.startswith('751')
                        other_depts = [d for d in st.session_state.selected_departments if d != '75']
                        if other_depts:
                            # Combiner avec les autres départements sélectionnés
                            dept_condition = filtered_data['CODGEO'].str[:2].isin(other_depts)
                            dept_data = filtered_data[paris_condition | dept_condition]
                        else:
                            # Uniquement Paris
                            dept_data = filtered_data[paris_condition]
                    else:
                        # Filtrage standard pour les autres départements
                        dept_data = filtered_data[filtered_data['CODGEO'].str[:2].isin(st.session_state.selected_departments)]
                elif hasattr(st.session_state, 'selected_department'):
                    # Cas spécial pour Paris
                    if st.session_state.selected_department == '75':
                        dept_data = filtered_data[filtered_data['CODGEO'].str.startswith('751')]
                    else:
                        # Compatibilité avec l'ancienne logique pour les autres départements
                        dept_data = filtered_data[filtered_data['CODGEO'].str[:2] == st.session_state.selected_department]
                else:
                    dept_data = pd.DataFrame()  # Fallback au cas où
                
                # Limiter le nombre de communes si nécessaire pour des raisons de performance
                if len(dept_data) > 3000:
                    st.warning("Le nombre de communes sélectionnées dépasse la limite d'affichage. Seules les 3000 premières communes sont affichées.")
                    dept_data = dept_data.head(3000)
                
                filtered_data_for_map = dept_data

                # Afficher un message informatif si Paris est sélectionné
                if (hasattr(st.session_state, 'selected_departments') and '75' in st.session_state.selected_departments) or \
                (hasattr(st.session_state, 'selected_department') and st.session_state.selected_department == '75'):
                    st.info("Pour Paris, les données affichées correspondent aux 20 arrondissements (codes INSEE 75101 à 75120).")

            elif geo_level == "departement" and st.session_state.selected_region:
                # Filtrer pour les départements de la région sélectionnée
                region_departments = [dept for dept, region in region_map.items() if region == st.session_state.selected_region]
                filtered_data_for_map = filtered_data[filtered_data['CODGEO'].str[:2].isin(region_departments)]
            else:
                filtered_data_for_map = filtered_data

            # Calcul des statistiques pour le niveau territorial actuel
            territory_stats = calculate_territorial_apl(filtered_data_for_map, geo_level)

            # Charger le GeoJSON approprié
            if geo_level == "commune":
                # Vérifier si Paris est sélectionné
                paris_selected = (hasattr(st.session_state, 'selected_departments') and '75' in st.session_state.selected_departments) or \
                                    (hasattr(st.session_state, 'selected_department') and st.session_state.selected_department == '75')
                
                # Si Paris est la seule sélection, utiliser le fichier paris.geojson
                if paris_selected and not (hasattr(st.session_state, 'selected_departments') and 
                                        len(st.session_state.selected_departments) > 1 and 
                                        any(d != '75' for d in st.session_state.selected_departments)):
                    geojson_file = "paris.geojson"
                    map_title = "Arrondissements de Paris"  # Titre spécifique pour Paris
                    st.info("Utilisation de la cartographie détaillée de Paris avec les arrondissements.")
                else:
                    geojson_file = "communes.geojson"
                    # Définir le titre de la carte comme avant
                    if hasattr(st.session_state, 'selected_departments') and len(st.session_state.selected_departments) > 1:
                        map_title = f"Communes des départements {', '.join(st.session_state.selected_departments)}"
                    else:
                        map_title = f"Communes du département {st.session_state.selected_department}"
            elif geo_level == "departement":
                geojson_file = "departements.geojson"
                map_title = f"Départements de {st.session_state.selected_region if st.session_state.selected_region else 'France'}"
            else:  # region
                geojson_file = "regions-avec-outre-mer.geojson"
                map_title = "Régions de France"

            geojson_data = load_geojson(geojson_file)

            if geojson_data:
                choropleth_map = create_choropleth_map(
                    territory_stats,
                    geojson_data,
                    level=geo_level,
                    title=map_title
                )
                
                # Afficher la carte
                st.plotly_chart(choropleth_map, use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Simuler la fonctionnalité de clic en permettant à l'utilisateur de sélectionner
                if geo_level == "region":
                    selected_region = st.selectbox(
                        "Sélectionnez une région pour voir ses départements",
                        options=sorted(territory_stats['territoire'].unique()),
                        format_func=lambda x: x  # Affiche directement le nom de la région
                    )
                    
                    if st.button("Voir les départements de cette région"):
                        st.session_state.selected_region = selected_region
                        st.session_state.territorial_level = "departement"
                        st.rerun()
                        
                elif geo_level == "departement":
                    dept_options = territory_stats['territoire'].unique()
                    selected_depts = st.multiselect(
                        "Sélectionnez un ou plusieurs départements pour voir leurs communes",
                        options=sorted(dept_options),
                        format_func=lambda x: f"Département {x}"
                    )
                    
                    if selected_depts and st.button("Voir les communes des départements sélectionnés"):
                        # Stocker les départements sélectionnés dans la session
                        st.session_state.selected_departments = selected_depts
                        # Pour la compatibilité avec le code existant
                        if len(selected_depts) == 1:
                            st.session_state.selected_department = selected_depts[0]
                        st.session_state.territorial_level = "commune"
                        st.rerun()
            else:
                st.error(f"Impossible de charger le fichier GeoJSON pour le niveau {geo_level}.")

            # Afficher des statistiques supplémentaires pour le niveau actuel
            st.subheader(f"Statistiques pour le niveau {geo_level}")

            # Formater le tableau selon le niveau territorial
            table_data = territory_stats.copy()

            if geo_level == "commune":
                table_data = table_data.rename(columns={
                    'territoire': 'Code INSEE',
                    'nom_commune': 'Commune',
                    'population': 'Population',
                    'apl_pondere': 'APL',
                    'desert_percent': 'Désert médical'
                })
                table_display = table_data[['Code INSEE', 'Commune', 'Population', 'APL', 'Désert médical']]
            else:
                table_data = table_data.rename(columns={
                    'territoire': 'Région' if geo_level == 'region' else 'Département',
                    'population': 'Population',
                    'apl_pondere': 'APL pondéré',
                    'communes_count': 'Nombre de communes',
                    'desert_percent': '% en désert médical',
                    'desert_count': 'Communes en désert',
                    'min_apl': 'APL minimum',
                    'max_apl': 'APL maximum'
                })
                table_display = table_data

            # Trier par APL (croissant - les territoires les plus en difficulté d'abord)
            if geo_level == "commune":
                table_display = table_display.sort_values('APL')
            else:
                table_display = table_display.sort_values('APL pondéré')

            # Formater les colonnes numériques
            for col in table_display.select_dtypes(include=['float']).columns:
                table_display[col] = table_display[col].round(2)

            # Afficher le tableau complet avec une interface interactive
            if geo_level == "commune":
                st.write(f"Affichage de toutes les {len(table_display)} communes - utilisez les fonctionnalités natives du tableau pour filtrer et trier")
                
                # Afficher directement le tableau complet
                st.dataframe(table_display, height=600)
                
                # Générer le CSV une seule fois et le mettre en cache pour éviter de le recalculer
                @st.cache_data
                def generate_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv = generate_csv(table_display)

                # Déterminer le nom du fichier en fonction de la sélection
                if geo_level == "commune":
                    if hasattr(st.session_state, 'selected_departments') and len(st.session_state.selected_departments) > 1:
                        file_name = f'communes_multi_dept_{"-".join(st.session_state.selected_departments)}.csv'
                    else:
                        file_name = f'communes_{st.session_state.selected_department}.csv'
                else:
                    file_name = f'{geo_level}_data.csv'

                st.download_button(
                    label="Télécharger les données en CSV",
                    data=csv,
                    file_name=file_name,
                    mime='text/csv',
                )
            else:
                # Pour les autres niveaux (département, région), conserver l'affichage simple
                st.dataframe(table_display)
        




        else:  # Vue complète du territoire
            # Code pour la vue complète du territoire - garder cette partie du fichier app4.py
            st.subheader("Carte choroplèthe du territoire français")
            
            # Chargement du GeoJSON approprié selon le niveau territorial
            level_options = ["Régions", "Départements", "Communes"]
            geo_level = st.radio("Niveau territorial", level_options, horizontal=True)
            
            if geo_level == "Communes":
                geojson_file = "communes.geojson"
                level = "commune"
                map_title = "Accessibilité aux soins par commune en France"
                # Préparer les données communales
                territory_stats = calculate_territorial_apl(filtered_data, "commune")
                
            elif geo_level == "Départements":
                geojson_file = "departements.geojson"
                level = "departement"
                map_title = "Accessibilité aux soins par département"
                # Préparer les données départementales
                territory_stats = calculate_territorial_apl(filtered_data, "departement")
                
            else:  # Régions
                geojson_file = "regions-avec-outre-mer.geojson"
                level = "region"
                map_title = "Accessibilité aux soins par région"
                # Préparer les données régionales
                territory_stats = calculate_territorial_apl(filtered_data, "region")
            
            # Charger le GeoJSON
            geojson_data = load_geojson(geojson_file)
            
            if geojson_data:
                # Créer la carte choroplèthe
                choropleth_map = create_choropleth_map(
                    territory_stats,
                    geojson_data,
                    level=level,
                    title=map_title
                )
                
                # Afficher la carte
                st.plotly_chart(choropleth_map, use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Afficher le tableau des statistiques
                st.subheader(f"Statistiques par {geo_level.lower()}")
                
                # Formater les données selon le niveau territorial
                if level == "commune":
                    display_df = territory_stats[['territoire', 'nom_commune', 'population', 'apl_pondere', 'desert_percent']].copy()
                    display_df.columns = ['Code INSEE', 'Commune', 'Population', 'APL', 'Désert médical (%)']
                elif level == "departement":
                    display_df = territory_stats.rename(columns={
                        'territoire': 'Département',
                        'population': 'Population',
                        'apl_pondere': 'APL pondéré',
                        'communes_count': 'Nombre de communes',
                        'desert_percent': '% en désert médical',
                        'desert_count': 'Communes en désert',
                        'min_apl': 'APL minimum',
                        'max_apl': 'APL maximum'
                    })
                else:  # région
                    display_df = territory_stats.rename(columns={
                        'territoire': 'Région',
                        'population': 'Population',
                        'apl_pondere': 'APL pondéré',
                        'communes_count': 'Nombre de communes',
                        'desert_percent': '% en désert médical',
                        'desert_count': 'Communes en désert',
                        'min_apl': 'APL minimum',
                        'max_apl': 'APL maximum'
                    })
                
                # Arrondir les colonnes numériques
                for col in display_df.select_dtypes(include=['float']).columns:
                    display_df[col] = display_df[col].round(2)
                
                # Afficher le tableau de données
                st.dataframe(display_df, height=500)
                
                # Option pour télécharger les données
                @st.cache_data
                def generate_csv(df):
                    return df.to_csv(index=False).encode('utf-8')
                
                csv = generate_csv(display_df)
                st.download_button(
                    label=f"Télécharger les données ({geo_level.lower()})",
                    data=csv,
                    file_name=f"{geo_level.lower()}_stats.csv",
                    mime='text/csv'
                )
            else:
                st.error(f"Impossible de charger le fichier GeoJSON pour les {geo_level.lower()}.")

    elif view_mode == "Analyses socio-démographiques":
        st.header("Analyses socio-démographiques des déserts médicaux")
        
        analysis_type = st.selectbox(
            "Type d'analyse",
            ["Démographie", "Catégories socio-professionnelles", "Équipements et services"]
        )
        
        if analysis_type == "Démographie":
            st.subheader("Impact de la taille des communes sur l'accès aux soins")
            
            # Graphique APL moyen par taille de commune
            fig = px.bar(
                pop_analysis,
                x='population_category',
                y='mean',
                labels={'population_category': 'Taille de la commune (habitants)', 'mean': 'APL moyen'},
                title="APL moyen par taille de commune",
                height=400,
                color='mean',
                color_continuous_scale='RdYlGn'
            )
            
            # Ligne de seuil désert médical
            fig.add_hline(
                y=2.5,
                line_dash="dash",
                line_color="red",
                annotation_text="Seuil désert médical (2.5)",
                annotation_position="bottom right"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Graphique pourcentage de déserts médicaux par taille de commune
            fig2 = px.bar(
                pop_analysis,
                x='population_category',
                y='desert_percent',
                labels={'population_category': 'Taille de la commune (habitants)', 'desert_percent': '% communes en désert médical'},
                title="Pourcentage de communes en désert médical par taille",
                height=400,
                color='desert_percent',
                color_continuous_scale='Reds'
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            #### Observations clés :
            - Les petites communes sont généralement plus touchées par les déserts médicaux
            - La proportion de communes en situation de désert médical diminue avec la taille de la population
            - Les communes de plus de 10 000 habitants ont en moyenne un meilleur accès aux soins
            """)
            
            st.subheader("Structure par âge et accès aux soins")
            
            # Graphique de corrélation entre structure d'âge et APL
            fig3 = px.bar(
                age_corr,
                x='Catégorie',
                y='Corrélation avec APL',
                labels={'Corrélation avec APL': 'Corrélation'},
                title="Corrélation entre catégories d'âge et APL",
                height=400,
                color='Corrélation avec APL',
                color_continuous_scale='RdBu_r'
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Graphique APL moyen selon la catégorie d'âge prédominante
            fig4 = px.bar(
                age_analysis,
                x='predominant_age',
                y='mean',
                labels={'predominant_age': 'Catégorie d\'âge prédominante', 'mean': 'APL moyen'},
                title="APL moyen selon la catégorie d'âge prédominante",
                height=400,
                color='mean',
                color_continuous_scale='RdYlGn'
            )
            
            # Ligne de seuil désert médical
            fig4.add_hline(
                y=2.5,
                line_dash="dash",
                line_color="red",
                annotation_text="Seuil désert médical (2.5)",
                annotation_position="bottom right"
            )
            
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
        elif analysis_type == "Catégories socio-professionnelles":
            st.subheader("Impact des catégories socio-professionnelles sur l'accès aux soins")
            
            # Corrélation entre APL et CSP
            csp_columns = ['CSP1_rate', 'CSP2_rate', 'CSP3_rate', 'CSP4_rate', 'CSP5_rate', 'CSP6_rate', 'CSP7_rate', 'CSP8_rate']
            csp_names = [
                'Agriculteurs exploitants',
                'Artisans, commerçants, chefs d\'entreprise',
                'Cadres et professions intellectuelles',
                'Professions intermédiaires',
                'Employés',
                'Ouvriers',
                'Retraités',
                'Sans activité professionnelle'
            ]
            
            csp_corr = pd.DataFrame({
                'CSP': csp_names,
                'Corrélation': [filtered_data['APL'].corr(filtered_data[col]) for col in csp_columns]
            })
            
            fig5 = px.bar(
                csp_corr,
                x='CSP',
                y='Corrélation',
                labels={'Corrélation': 'Corrélation avec APL'},
                title="Corrélation entre catégories socio-professionnelles et APL",
                height=500,
                color='Corrélation',
                color_continuous_scale='RdBu_r'
            )
            
            st.plotly_chart(fig5, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Distribution des CSP selon les zones (désert médical vs non désert)
            desert_data = filtered_data[filtered_data['APL'] < 2.5]
            non_desert_data = filtered_data[filtered_data['APL'] >= 2.5]
            
            csp_desert = desert_data[csp_columns].mean()
            csp_non_desert = non_desert_data[csp_columns].mean()
            
            csp_comparison = pd.DataFrame({
                'CSP': csp_names,
                'Déserts médicaux': csp_desert.values,
                'Zones bien desservies': csp_non_desert.values
            })
            
            csp_comparison_long = pd.melt(
                csp_comparison,
                id_vars=['CSP'],
                value_vars=['Déserts médicaux', 'Zones bien desservies'],
                var_name='Type de zone',
                value_name='Pourcentage moyen'
            )
            
            fig6 = px.bar(
                csp_comparison_long,
                x='CSP',
                y='Pourcentage moyen',
                color='Type de zone',
                barmode='group',
                title="Composition socio-professionnelle moyenne selon le type de zone",
                height=500
            )
            
            st.plotly_chart(fig6, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Niveau de vie et accès aux soins
            st.subheader("Niveau de vie et accès aux soins")
            
            # Créer des bins de niveau de vie
            income_bins = [0, 15000, 18000, 21000, 24000, 100000]
            income_labels = ["<15 000€", "15 000-18 000€", "18 000-21 000€", "21 000-24 000€", ">24 000€"]
            
            filtered_data['income_category'] = pd.cut(
                filtered_data['median_living_standard'],
                bins=income_bins,
                labels=income_labels
            )
            
            income_analysis = filtered_data.groupby('income_category', observed=True)['APL'].agg(['mean', 'count']).reset_index()
            income_analysis['desert_count'] = filtered_data[filtered_data['APL'] < 2.5].groupby('income_category', observed=True).size().values
            income_analysis['desert_percent'] = (income_analysis['desert_count'] / income_analysis['count']) * 100
            
            fig7 = px.bar(
                income_analysis,
                x='income_category',
                y='mean',
                labels={'income_category': 'Niveau de vie médian', 'mean': 'APL moyen'},
                title="APL moyen par niveau de vie",
                height=400,
                color='mean',
                color_continuous_scale='RdYlGn'
            )
            
            # Ligne de seuil désert médical
            fig7.add_hline(
                y=2.5,
                line_dash="dash",
                line_color="red",
                annotation_text="Seuil désert médical (2.5)",
                annotation_position="bottom right"
            )
            
            st.plotly_chart(fig7, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
        else:  # Équipements et services
            st.subheader("Impact des équipements et services sur l'accès aux soins")
            
            # Relation entre équipements de santé/éducation et APL
            fig9 = px.scatter(
                filtered_data,
                x='healthcare_education_establishments',
                y='APL',
                labels={'healthcare_education_establishments': 'Établissements de santé et d\'éducation', 'APL': 'APL'},
                title="Relation entre équipements de santé/éducation et APL",
                height=600,
                color='APL_category',
                color_discrete_map={
                    'Désert médical critique': 'darkred',
                    'Désert médical': 'red',
                    'Sous-équipement médical': 'orange',
                    'Équipement médical suffisant': 'lightgreen',
                    'Bon équipement médical': 'green'
                },
                opacity=0.7,
                size='P16_POP',
                size_max=50,
                hover_name='Communes'
            )
            
            # Ligne de seuil désert médical
            fig9.add_hline(
                y=2.5,
                line_dash="dash",
                line_color="red",
                annotation_text="Seuil désert médical (2.5)",
                annotation_position="bottom right"
            )
            
            st.plotly_chart(fig9, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Relation entre équipements sociaux et APL
            fig10 = px.scatter(
                filtered_data,
                x='city_social_amenities_rate',
                y='APL',
                labels={'city_social_amenities_rate': 'Taux d\'équipements sociaux', 'APL': 'APL'},
                title="Relation entre équipements sociaux et APL",
                height=600,
                color='APL_category',
                color_discrete_map={
                    'Désert médical critique': 'darkred',
                    'Désert médical': 'red',
                    'Sous-équipement médical': 'orange',
                    'Équipement médical suffisant': 'lightgreen',
                    'Bon équipement médical': 'green'
                },
                opacity=0.7,
                size='P16_POP',
                size_max=50,
                hover_name='Communes'
            )
            
            st.plotly_chart(fig10, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
    

    elif view_mode == "Clusters de communes":
        view_mode_clusters(filtered_data)
    

    elif view_mode == "Prévisions & Risques":
        view_mode_predictions(filtered_data)



# Message d'erreur si les données ne sont pas chargées
else:
    st.error("Erreur : Impossible de charger les données. Veuillez vérifier que le fichier 'medical_desert_data_with_coords.csv' est disponible.")

# Pied de page
st.markdown("---")
st.markdown("Dashboard créé par l'équipe Medical'IA - KESK'IA 2025 | Medical'IA : Combattre les déserts médicaux grâce à l'IA")
