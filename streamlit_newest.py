#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Streamlit - Analyse de Fiabilité des Relais de Protection
Simulation Monte Carlo avec Modèle Weibull à Risques Concurrents

Auteur: [Votre Nom]
Date: 2025
Version: 1.0.0
"""

import subprocess
import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# VÉRIFICATION ET INSTALLATION AUTOMATIQUE DES DÉPENDANCES
# ====================================================================

def install_requirements():
    """Installation automatique des packages depuis requirements.txt"""
    requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    
    if not os.path.exists(requirements_file):
        print("⚠️ Fichier requirements.txt non trouvé. Installation manuelle requise.")
        return True  # Continue l'exécution
    
    try:
        print("📦 Vérification des dépendances...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', requirements_file, '--quiet'
        ])
        print("✅ Dépendances installées avec succès!")
        return True
    except Exception as e:
        print(f"❌ Erreur installation: {str(e)}")
        return False

# Installation des dépendances au démarrage
if __name__ == "__main__":
    print("🚀 Initialisation du Dashboard Fiabilité des Relais...")
    if not install_requirements():
        sys.exit(1)

# ====================================================================
# IMPORTS PRINCIPAUX
# ====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from lifelines import KaplanMeierFitter, WeibullFitter
from lifelines.statistics import logrank_test
import joblib
from concurrent.futures import ThreadPoolExecutor

# ====================================================================
# CONFIGURATION STREAMLIT
# ====================================================================

st.set_page_config(
    page_title="Dashboard Fiabilité des Relais",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================================
# STYLES CSS PERSONNALISÉS
# ====================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(145deg, #f0f2f6, #ffffff);
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 0.4rem solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# CLASSE WEIBULL À RISQUES CONCURRENTS (VOTRE IMPLÉMENTATION)
# ====================================================================

class WeibullCompetingRisksModel:
    """
    Modèle Weibull à Risques Concurrents (Mécanique vs Électrique)
    Basé sur votre implémentation réelle
    """
    
    def __init__(self):
        self.fitter_meca = None
        self.fitter_elec = None
        self.fitted = False
        self.df_processed = None
    
    def prepare_data(self, df):
        """
        Nettoyage et préparation des données selon votre méthode
        """
        print("🔄 Nettoyage des données...")
        
        # 1) Nettoyage selon votre code
        df_real = df.copy()
        df_real = df_real[df_real["ACTIF"].notna() & (df_real["ACTIF"] > 0)]
        df_real["time"] = df_real["ACTIF"].astype(float)
        df_real["event"] = df_real["censure"].astype(int)  # 1=panne, 0=censuré
        
        # Attribution aléatoire des causes (selon votre code)
        np.random.seed(42)  # Pour reproductibilité
        df_real["cause"] = np.random.choice(
            ["mecanique", "electrique"], 
            size=len(df_real)
        )
        
        self.df_processed = df_real
        return df_real
    
    def fit(self, df):
        """
        Ajustement par cause selon votre méthode exacte
        """
        print("⚙️ Ajustement du modèle Weibull CR...")
        
        df_real = self.prepare_data(df)
        
        # 2) Filtrer par cause observée (selon votre code)
        df_meca = df_real[
            (df_real["cause"] == "mecanique") | (df_real["event"] == 0)
        ].copy()
        
        df_elec = df_real[
            (df_real["cause"] == "electrique") | (df_real["event"] == 0)
        ].copy()
        
        # 3) Ajustement Weibull par cause avec censure à droite
        self.fitter_meca = WeibullFitter().fit(
            durations=df_meca["time"].values,
            event_observed=df_meca["event"].values,
            label="Weibull - Mécanique",
        )
        
        self.fitter_elec = WeibullFitter().fit(
            durations=df_elec["time"].values,
            event_observed=df_elec["event"].values,
            label="Weibull - Électrique",
        )
        
        self.fitted = True
        print(f"✅ Modèle ajusté - λ_meca={self.fitter_meca.lambda_:.1f}, "
              f"ρ_meca={self.fitter_meca.rho_:.2f}")
        print(f"✅ Modèle ajusté - λ_elec={self.fitter_elec.lambda_:.1f}, "
              f"ρ_elec={self.fitter_elec.rho_:.2f}")
        
        return self
    
    def generate_remaining_lifetime_weibull(self, age, rho, lambd):
        """
        Tirage conditionnel exact selon votre fonction
        """
        u = np.random.uniform(0, 1)
        # Inversion de la survie conditionnelle
        inside = (-np.log(u) + (age / lambd) ** rho)
        if inside <= 0:
            return 0.0
        t_total = (lambd ** rho * inside) ** (1 / rho)
        return max(0, t_total - age)
    
    def generate_remaining_lifetime_competing(self, age):
        """
        Génère la durée de vie restante avec risques concurrents
        selon votre implémentation exacte
        """
        if not self.fitted:
            raise ValueError("Modèle non ajusté!")
        
        rm = self.generate_remaining_lifetime_weibull(
            age, self.fitter_meca.rho_, self.fitter_meca.lambda_
        )
        re = self.generate_remaining_lifetime_weibull(
            age, self.fitter_elec.rho_, self.fitter_elec.lambda_
        )
        return min(rm, re)
    
    def get_initial_fleet_ages(self, df):
        """
        Extraction des âges du parc vivant selon votre méthode
        """
        df_vivants = df[(df["censure"] == 0) & (~df["ACTIF"].isna())]
        ages_actuels = df_vivants["ACTIF"].values
        return np.array(ages_actuels)
    
    def simulation_mc_conditional(self, parc_initial, n_iterations=100, N_years=26):
        """
        Simulation Monte Carlo conditionnelle - VOTRE CODE EXACT
        """
        print(f"🎯 Lancement simulation MC: {n_iterations} itérations, {N_years} années")
        start = time.time()
        
        consommation_annuelle = []
        
        for sim in range(n_iterations):
            parc = list(parc_initial)
            conso_annuelle = []
            
            for year in range(N_years):
                if sim < 5:  # Affichage pour les 5 premières simulations seulement
                    print(f"   Simulation {sim+1} - Année {year+1}")
                
                replacements = 0
                new_parc = []
                
                for age in parc:
                    vie_restante = self.generate_remaining_lifetime_competing(age)
                    
                    # Gestion des valeurs négatives selon votre code
                    retry_count = 0
                    while vie_restante < 0 and retry_count < 10:  # Limite les retirages
                        vie_restante = self.generate_remaining_lifetime_competing(age)
                        retry_count += 1
                        if retry_count == 1 and sim < 3:  # Log limité
                            print(f"      Retirage vie restante {vie_restante:.2f} sur composant âge {age}")
                    
                    if vie_restante <= 1:
                        replacements += 1
                        new_parc.append(0)  # Remplacé -> âge 0
                    else:
                        new_parc.append(age + 1)  # Vieillissement
                
                parc = new_parc
                conso_annuelle.append(replacements)
            
            consommation_annuelle.append(conso_annuelle)
            if sim % 10 == 0 or sim < 5:
                print(f"### Simulation {sim+1} terminée")
        
        elapsed = time.time() - start
        print(f"✅ Simulation terminée en {elapsed:.2f} secondes.")
        
        return np.array(consommation_annuelle)

# ====================================================================
# FONCTIONS UTILITAIRES
# ====================================================================

@st.cache_data
def load_and_validate_data(uploaded_file):
    """Charge et valide les données"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Validation des colonnes requises pour votre modèle
        required_cols = ['ACTIF', 'censure']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Colonnes manquantes : {missing_cols}")
            st.info("Format attendu : 'ACTIF' (âge en années), 'censure' (0=vivant, 1=défaillant)")
            return None
            
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement : {str(e)}")
        return None

def generate_synthetic_data():
    """Génère des données synthétiques réalistes pour la démo"""
    np.random.seed(42)
    n_samples = 2500
    
    # Génération Weibull avec paramètres réalistes
    ages_meca = np.random.weibull(2.3, n_samples//2) * 45 + 5
    ages_elec = np.random.weibull(1.9, n_samples//2) * 50 + 8
    ages = np.concatenate([ages_meca, ages_elec])
    
    # Censure réaliste (70% d'événements observés)
    censure = np.random.binomial(1, 0.7, n_samples)
    
    return pd.DataFrame({
        'ACTIF': ages,
        'censure': censure
    })

def perform_kaplan_meier_analysis(df_processed):
    """Analyse Kaplan-Meier sur données nettoyées"""
    kmf = KaplanMeierFitter()
    kmf.fit(df_processed['time'], df_processed['event'], label='Kaplan-Meier Global')
    
    # Analyse par cause
    kmf_results = {}
    for cause in df_processed['cause'].unique():
        mask = df_processed['cause'] == cause
        kmf_cause = KaplanMeierFitter()
        kmf_cause.fit(
            df_processed[mask]['time'], 
            df_processed[mask]['event'], 
            label=f'KM - {cause.title()}'
        )
        kmf_results[cause] = kmf_cause
    
    return kmf, kmf_results

# ====================================================================
# INTERFACE UTILISATEUR PRINCIPALE
# ====================================================================

def main():
    # Titre principal
    st.markdown('<h1 class="main-header">⚡ Dashboard Analyse de Fiabilité des Relais</h1>', 
                unsafe_allow_html=True)
    
    # ====================================================================
    # SIDEBAR - CONFIGURATION
    # ====================================================================
    
    st.sidebar.header("📋 Configuration")
    
    # Upload de fichier
    uploaded_file = st.sidebar.file_uploader(
        "📁 Charger la base de données",
        type=['csv', 'xlsx'],
        help="Format: colonnes 'ACTIF' (âge), 'censure' (0=vivant, 1=défaillant)"
    )
    
    # Chargement des données
    if uploaded_file is None:
        st.sidebar.info("💡 Utilisation des données synthétiques")
        df = generate_synthetic_data()
        st.info("📝 **Données synthétiques chargées** (2500 relais) pour démonstration")
    else:
        df = load_and_validate_data(uploaded_file)
        if df is None:
            return
        st.success(f"✅ **Données chargées** : {len(df)} relais")
    
    # Paramètres de simulation
    st.sidebar.subheader("🎯 Paramètres Simulation MC")
    n_iterations = st.sidebar.slider("Itérations Monte Carlo", 50, 500, 100)
    N_years = st.sidebar.slider("Horizon (années)", 10, 35, 26)
    start_year = st.sidebar.number_input("Année de début", 2025, 2030, 2025)
    
    # Options avancées
    with st.sidebar.expander("⚙️ Options Avancées"):
        show_debug = st.checkbox("Affichage debug", False)
        use_parallel = st.checkbox("Calcul parallèle", True)
        random_seed = st.number_input("Seed aléatoire", 0, 9999, 42)
    
    # Bouton principal
    run_full_analysis = st.sidebar.button("🚀 Lancer l'Analyse Complète", type="primary")
    
    # ====================================================================
    # MÉTRIQUES GÉNÉRALES
    # ====================================================================
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 Total Relais", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        n_events = int(df['censure'].sum()) if 'censure' in df.columns else 0
        st.metric("⚡ Défaillances", f"{n_events:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_age = df['ACTIF'].mean() if 'ACTIF' in df.columns else 0
        st.metric("📈 Âge Moyen", f"{avg_age:.1f} ans")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        n_alive = len(df[df['censure'] == 0]) if 'censure' in df.columns else len(df)
        st.metric("🔋 Actifs", f"{n_alive:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ====================================================================
    # ONGLETS PRINCIPAUX
    # ====================================================================
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Exploration", "📈 Kaplan-Meier", "⚙️ Modèle Weibull CR", 
        "🎯 Simulation Monte Carlo", "📋 Résultats & Export"
    ])
    
    # ====================================================================
    # TAB 1 : EXPLORATION DES DONNÉES
    # ====================================================================
    
    with tab1:
        st.subheader("🔍 Exploration des Données")
        
        if 'ACTIF' in df.columns and 'censure' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution des âges
                fig_age = px.histogram(
                    df, x='ACTIF', nbins=40,
                    title="Distribution des Âges (Années de Service)",
                    labels={'ACTIF': 'Âge (années)', 'count': 'Nombre de relais'},
                    color_discrete_sequence=['#1f77b4']
                )
                fig_age.update_layout(showlegend=False)
                st.plotly_chart(fig_age, use_container_width=True)
            
            with col2:
                # Statut (censure)
                status_counts = df['censure'].value_counts()
                fig_status = px.pie(
                    values=status_counts.values,
                    names=['Actifs (Censurés)', 'Défaillants'],
                    title="Répartition Statut des Relais",
                    color_discrete_sequence=['#2ca02c', '#d62728']
                )
                st.plotly_chart(fig_status, use_container_width=True)
            
            # Statistiques détaillées
            st.subheader("📊 Statistiques Descriptives")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Âges (ACTIF)**")
                st.write(df['ACTIF'].describe())
            
            with col2:
                st.write("**Distribution par Statut**")
                status_df = pd.DataFrame({
                    'Statut': ['Actifs', 'Défaillants'],
                    'Nombre': [len(df[df['censure']==0]), len(df[df['censure']==1])],
                    'Pourcentage': [
                        len(df[df['censure']==0])/len(df)*100,
                        len(df[df['censure']==1])/len(df)*100
                    ]
                })
                st.dataframe(status_df, hide_index=True)
            
            # Aperçu des données
            st.subheader("👀 Aperçu des Données")
            st.dataframe(df.head(10), use_container_width=True)
        
        else:
            st.error("❌ Colonnes 'ACTIF' et 'censure' requises pour l'exploration")
    
    # ====================================================================
    # TAB 2 : ANALYSE KAPLAN-MEIER
    # ====================================================================
    
    with tab2:
        st.subheader("📈 Analyse de Survie Kaplan-Meier")
        
        if st.button("🔄 Calculer Kaplan-Meier", key="km_button"):
            if 'ACTIF' in df.columns and 'censure' in df.columns:
                
                with st.spinner("Calcul des estimateurs Kaplan-Meier..."):
                    # Préparation des données pour KM
                    model = WeibullCompetingRisksModel()
                    df_processed = model.prepare_data(df)
                    
                    # Analyse KM
                    kmf, kmf_results = perform_kaplan_meier_analysis(df_processed)
                    
                    # Sauvegarde en session
                    st.session_state['kmf'] = kmf
                    st.session_state['kmf_results'] = kmf_results
                    st.session_state['df_processed'] = df_processed
                
                st.success("✅ Analyse Kaplan-Meier terminée!")
                
                # Graphique KM
                ages_plot = np.linspace(0, df_processed['time'].max(), 100)
                
                fig_km = go.Figure()
                
                # Courbe globale
                survival_global = kmf.survival_function_at_times(ages_plot)
                fig_km.add_trace(go.Scatter(
                    x=ages_plot, y=survival_global,
                    name='Global', line=dict(color='blue', width=3)
                ))
                
                # Courbes par cause
                colors = {'mecanique': 'red', 'electrique': 'green'}
                for cause, kmf_cause in kmf_results.items():
                    survival_cause = kmf_cause.survival_function_at_times(ages_plot)
                    fig_km.add_trace(go.Scatter(
                        x=ages_plot, y=survival_cause,
                        name=f'{cause.title()}',
                        line=dict(color=colors[cause], width=2, dash='dash')
                    ))
                
                fig_km.update_layout(
                    title="Courbes de Survie Kaplan-Meier par Cause de Défaillance",
                    xaxis_title="Âge (années)",
                    yaxis_title="Probabilité de Survie",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_km, use_container_width=True)
                
                # Métriques de survie
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    median_survival = kmf.median_survival_time_
                    st.metric("📊 Survie Médiane", 
                             f"{median_survival:.1f} ans" if median_survival else "Non atteinte")
                
                with col2:
                    try:
                        s_30 = kmf.survival_function_at_times([30]).iloc[0]
                        st.metric("📈 Survie à 30 ans", f"{s_30:.1%}")
                    except:
                        st.metric("📈 Survie à 30 ans", "N/A")
                
                with col3:
                    try:
                        s_50 = kmf.survival_function_at_times([50]).iloc[0]
                        st.metric("📉 Survie à 50 ans", f"{s_50:.1%}")
                    except:
                        st.metric("📉 Survie à 50 ans", "N/A")
            
            else:
                st.error("❌ Colonnes requises manquantes")
    
    # ====================================================================
    # TAB 3 : MODÉLISATION WEIBULL CR
    # ====================================================================
    
    with tab3:
        st.subheader("⚙️ Modélisation Weibull à Risques Concurrents")
        
        if st.button("🔧 Ajuster le Modèle Weibull CR", key="weibull_button"):
            if 'ACTIF' in df.columns and 'censure' in df.columns:
                
                with st.spinner("Ajustement du modèle Weibull CR en cours..."):
                    np.random.seed(random_seed)
                    
                    # Création et ajustement du modèle
                    model = WeibullCompetingRisksModel()
                    model.fit(df)
                    
                    # Sauvegarde en session
                    st.session_state['weibull_model'] = model
                
                st.markdown('<div class="success-box">✅ <b>Modèle Weibull CR ajusté avec succès!</b></div>', 
                           unsafe_allow_html=True)
                
                # Affichage des paramètres
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔩 Défaillances Mécaniques")
                    st.metric("λ (échelle)", f"{model.fitter_meca.lambda_:.1f} ans")
                    st.metric("ρ (forme)", f"{model.fitter_meca.rho_:.2f}")
                    st.info("ρ > 1 : Usure progressive (vieillissement)")
                
                with col2:
                    st.subheader("⚡ Défaillances Électriques")
                    st.metric("λ (échelle)", f"{model.fitter_elec.lambda_:.1f} ans")
                    st.metric("ρ (forme)", f"{model.fitter_elec.rho_:.2f}")
                    st.info("ρ < 2 : Défaillances aléatoires + usure")
                
                # Graphique comparatif des modèles
                ages_plot = np.linspace(0, 70, 100)
                
                fig_weibull = go.Figure()
                
                # Survie mécanique
                survival_meca = np.exp(-(ages_plot/model.fitter_meca.lambda_)**model.fitter_meca.rho_)
                fig_weibull.add_trace(go.Scatter(
                    x=ages_plot, y=survival_meca,
                    name='Mécanique seul',
                    line=dict(color='red', width=2, dash='dot')
                ))
                
                # Survie électrique
                survival_elec = np.exp(-(ages_plot/model.fitter_elec.lambda_)**model.fitter_elec.rho_)
                fig_weibull.add_trace(go.Scatter(
                    x=ages_plot, y=survival_elec,
                    name='Électrique seul',
                    line=dict(color='green', width=2, dash='dot')
                ))
                
                # Survie combinée (risques concurrents)
                survival_combined = survival_meca * survival_elec
                fig_weibull.add_trace(go.Scatter(
                    x=ages_plot, y=survival_combined,
                    name='Risques Concurrents',
                    line=dict(color='blue', width=3)
                ))
                
                # Comparaison avec KM si disponible
                if 'kmf' in st.session_state:
                    kmf = st.session_state['kmf']
                    survival_km = kmf.survival_function_at_times(ages_plot)
                    fig_weibull.add_trace(go.Scatter(
                        x=ages_plot, y=survival_km,
                        name='Kaplan-Meier',
                        line=dict(color='orange', width=2, dash='dash')
                    ))
                
                fig_weibull.update_layout(
                    title="Comparaison Modèles de Survie",
                    xaxis_title="Âge (années)",
                    yaxis_title="Probabilité de Survie",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_weibull, use_container_width=True)
                
                # Interprétation
                st.subheader("📋 Interprétation des Résultats")
                
                st.write(f"""
                **Analyse des paramètres Weibull:**
                
                - **Mécanique** (λ={model.fitter_meca.lambda_:.1f}, ρ={model.fitter_meca.rho_:.2f}):
                  - Durée de vie caractéristique: {model.fitter_meca.lambda_:.1f} ans
                  - Mode de défaillance: {"Usure" if model.fitter_meca.rho_ > 1 else "Aléatoire"}
                
                - **Électrique** (λ={model.fitter_elec.lambda_:.1f}, ρ={model.fitter_elec.rho_:.2f}):
                  - Durée de vie caractéristique: {model.fitter_elec.lambda_:.1f} ans  
                  - Mode de défaillance: {"Usure" if model.fitter_elec.rho_ > 1 else "Aléatoire"}
                
                - **Risques Concurrents**: Le premier des deux modes détermine la défaillance
                """)
            
            else:
                st.error("❌ Colonnes requises manquantes pour l'ajustement")
    
    # ====================================================================
    # TAB 4 : SIMULATION MONTE CARLO
    # ====================================================================
    
    with tab4:
        st.subheader("🎯 Simulation Monte Carlo - Prévisions de Remplacements")
        
        if run_full_analysis:
            if 'weibull_model' not in st.session_state:
                # Ajustement automatique du modèle si pas encore fait
                with st.spinner("Ajustement automatique du modèle..."):
                    np.random.seed(random_seed)
                    model = WeibullCompetingRisksModel()
                    model.fit(df)
                    st.session_state['weibull_model'] = model
                    st.success("✅ Modèle Weibull CR ajusté automatiquement")
            
            model = st.session_state['weibull_model']
            
            # Extraction du parc initial selon votre méthode
            parc_initial = model.get_initial_fleet_ages(df)
            st.info(f"📊 **Parc initial**: {len(parc_initial):,} relais actifs")
            
            # Lancement de la simulation MC
            with st.spinner(f"🔄 Simulation Monte Carlo en cours ({n_iterations} itérations, {N_years} années)..."):
                
                # Conteneur pour les logs en temps réel
                log_container = st.empty()
                
                # Simulation avec votre méthode exacte
                np.random.seed(random_seed)
                consommation_annuelle = model.simulation_mc_conditional(
                    parc_initial, n_iterations, N_years
                )
                
                # Sauvegarde des résultats
                st.session_state['mc_results'] = consommation_annuelle
                st.session_state['simulation_params'] = {
                    'n_iterations': n_iterations,
                    'N_years': N_years,
                    'start_year': start_year,
                    'parc_size': len(parc_initial)
                }
            
            # Calcul des statistiques
            years_list = list(range(start_year, start_year + N_years))
            means = np.mean(consommation_annuelle, axis=0)
            stds = np.std(consommation_annuelle, axis=0)
            p5 = np.percentile(consommation_annuelle, 5, axis=0)
            p25 = np.percentile(consommation_annuelle, 25, axis=0)
            p50 = np.percentile(consommation_annuelle, 50, axis=0)
            p75 = np.percentile(consommation_annuelle, 75, axis=0)
            p95 = np.percentile(consommation_annuelle, 95, axis=0)
            
            st.success(f"✅ **Simulation terminée!** {n_iterations} itérations sur {N_years} années")
            
            # ====================================================================
            # GRAPHIQUE PRINCIPAL DES RÉSULTATS
            # ====================================================================
            
            fig_mc = go.Figure()
            
            # Bande de confiance P5-P95
            fig_mc.add_trace(go.Scatter(
                x=years_list + years_list[::-1],
                y=list(p95) + list(p5[::-1]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name='P5-P95',
                showlegend=True
            ))
            
            # Bande de confiance P25-P75
            fig_mc.add_trace(go.Scatter(
                x=years_list + years_list[::-1],
                y=list(p75) + list(p25[::-1]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                name='P25-P75',
                showlegend=True
            ))
            
            # Courbe moyenne
            fig_mc.add_trace(go.Scatter(
                x=years_list, y=means,
                name='Moyenne',
                line=dict(color='blue', width=4),
                mode='lines+markers'
            ))
            
            # Médiane
            fig_mc.add_trace(go.Scatter(
                x=years_list, y=p50,
                name='Médiane (P50)',
                line=dict(color='red', width=2, dash='dash'),
                mode='lines'
            ))
            
            fig_mc.update_layout(
                title=f"📈 Projection Monte Carlo - Remplacements Annuels<br>"
                      f"<sub>{n_iterations} simulations • Parc initial: {len(parc_initial):,} relais</sub>",
                xaxis_title="Année",
                yaxis_title="Nombre de Remplacements",
                hovermode='x unified',
                height=600
            )
            
            st.plotly_chart(fig_mc, use_container_width=True)
            
            # ====================================================================
            # MÉTRIQUES CLÉS
            # ====================================================================
            
            st.subheader("🎯 Indicateurs Clés de la Projection")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    f"📅 Remplacements {start_year}",
                    f"{int(means[0]):,}",
                    f"±{int(stds[0])}"
                )
            
            with col2:
                final_year = start_year + N_years - 1
                st.metric(
                    f"📅 Remplacements {final_year}",
                    f"{int(means[-1]):,}",
                    f"±{int(stds[-1])}"
                )
            
            with col3:
                total_replacements = int(np.sum(means))
                st.metric(
                    "📊 Total Période",
                    f"{total_replacements:,}",
                    f"sur {N_years} ans"
                )
            
            with col4:
                croissance = ((means[-1] - means[0]) / means[0]) * 100
                st.metric(
                    "📈 Évolution",
                    f"{croissance:+.1f}%",
                    f"sur {N_years} ans"
                )
            
            # ====================================================================
            # ANALYSE DES TENDANCES
            # ====================================================================
            
            st.subheader("📊 Analyse des Tendances")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Identification des phases
                peak_year_idx = np.argmax(means)
                peak_year = years_list[peak_year_idx]
                peak_value = int(means[peak_year_idx])
                
                st.write("**🎯 Points Clés de la Projection:**")
                st.write(f"• **Pic prévu**: {peak_year} ({peak_value:,} remplacements)")
                st.write(f"• **Croissance moyenne**: {np.mean(np.diff(means)):.1f} remp./an")
                st.write(f"• **Variabilité moyenne**: ±{np.mean(stds):.0f} remplacements")
                
                # Détection des phases
                if peak_year_idx < len(years_list) * 0.3:
                    phase = "📈 **Phase croissante** sur toute la période"
                elif peak_year_idx > len(years_list) * 0.7:
                    phase = "📉 **Phase décroissante** en fin de période"
                else:
                    phase = f"🔄 **Pic intermédiaire** vers {peak_year}"
                
                st.write(f"• **Tendance**: {phase}")
            
            with col2:
                # Recommandations opérationnelles
                st.write("**💡 Recommandations Opérationnelles:**")
                
                if croissance > 10:
                    st.warning("⚠️ **Forte croissance** : Renforcer les stocks")
                elif croissance > 0:
                    st.info("📊 **Croissance modérée** : Ajustement progressif")
                else:
                    st.success("✅ **Tendance stable** : Maintenance du niveau")
                
                st.write(f"""
                • **Stock de sécurité** : Dimensionner sur P95 = {int(np.mean(p95)):,}
                • **Budget annuel moyen** : {int(np.mean(means)):,} remplacements
                • **Période critique** : {peak_year-2}-{peak_year+2}
                • **Incertitude** : ±{int(np.mean(stds))}/{int(np.mean(means))*100:.0f}% en moyenne
                """)
        
        else:
            st.info("🎯 Cliquez sur **'Lancer l'Analyse Complète'** dans la barre latérale pour démarrer la simulation Monte Carlo")
            
            # Affichage des paramètres actuels
            st.write("**Paramètres configurés:**")
            st.write(f"• {n_iterations} itérations Monte Carlo")
            st.write(f"• Horizon: {N_years} années ({start_year}-{start_year+N_years-1})")
            st.write(f"• Seed aléatoire: {random_seed}")
    
    # ====================================================================
    # TAB 5 : RÉSULTATS & EXPORT
    # ====================================================================
    
    with tab5:
        st.subheader("📋 Résultats Détaillés & Export")
        
        if 'mc_results' in st.session_state:
            consommation_annuelle = st.session_state['mc_results']
            params = st.session_state['simulation_params']
            
            # Reconstruction des statistiques
            years_list = list(range(params['start_year'], params['start_year'] + params['N_years']))
            means = np.mean(consommation_annuelle, axis=0)
            stds = np.std(consommation_annuelle, axis=0)
            
            # Tableau détaillé des résultats
            results_df = pd.DataFrame({
                'Année': years_list,
                'Moyenne': [int(x) for x in means],
                'Médiane': [int(np.median(consommation_annuelle[:, i])) for i in range(len(years_list))],
                'Écart-type': [int(x) for x in stds],
                'P5': [int(np.percentile(consommation_annuelle[:, i], 5)) for i in range(len(years_list))],
                'P25': [int(np.percentile(consommation_annuelle[:, i], 25)) for i in range(len(years_list))],
                'P75': [int(np.percentile(consommation_annuelle[:, i], 75)) for i in range(len(years_list))],
                'P95': [int(np.percentile(consommation_annuelle[:, i], 95)) for i in range(len(years_list))],
                'Min': [int(np.min(consommation_annuelle[:, i])) for i in range(len(years_list))],
                'Max': [int(np.max(consommation_annuelle[:, i])) for i in range(len(years_list))]
            })
            
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # ====================================================================
            # OPTIONS D'EXPORT
            # ====================================================================
            
            st.subheader("💾 Options d'Export")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export CSV des résultats
                csv_results = results_df.to_csv(index=False, encoding='utf-8')
                st.download_button(
                    label="📊 Télécharger Résultats (CSV)",
                    data=csv_results,
                    file_name=f"monte_carlo_results_{params['start_year']}_{params['start_year']+params['N_years']-1}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export des données brutes
                if st.button("💿 Préparer Export Données Brutes"):
                    # Création du DataFrame complet avec toutes les simulations
                    raw_data = []
                    for sim in range(params['n_iterations']):
                        for year_idx, year in enumerate(years_list):
                            raw_data.append({
                                'Simulation': sim + 1,
                                'Année': year,
                                'Remplacements': consommation_annuelle[sim, year_idx]
                            })
                    
                    raw_df = pd.DataFrame(raw_data)
                    csv_raw = raw_df.to_csv(index=False, encoding='utf-8')
                    
                    st.download_button(
                        label="📁 Télécharger Données Brutes (CSV)",
                        data=csv_raw,
                        file_name=f"monte_carlo_raw_data_{params['n_iterations']}sim.csv",
                        mime="text/csv"
                    )
            
            with col3:
                # Export rapport synthèse
                if st.button("📄 Générer Rapport PDF"):
                    st.info("🚧 Fonction en développement - Export PDF disponible prochainement")
            
            # ====================================================================
            # INFORMATIONS SUR LA SIMULATION
            # ====================================================================
            
            st.subheader("ℹ️ Informations sur la Simulation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Paramètres de Simulation:**")
                st.write(f"• Nombre d'itérations: {params['n_iterations']:,}")
                st.write(f"• Horizon temporel: {params['N_years']} années")
                st.write(f"• Période: {params['start_year']}-{params['start_year']+params['N_years']-1}")
                st.write(f"• Taille du parc initial: {params['parc_size']:,} relais")
            
            with col2:
                if 'weibull_model' in st.session_state:
                    model = st.session_state['weibull_model']
                    st.write("**Paramètres du Modèle Weibull CR:**")
                    st.write(f"• λ_mécanique: {model.fitter_meca.lambda_:.1f} ans")
                    st.write(f"• ρ_mécanique: {model.fitter_meca.rho_:.2f}")
                    st.write(f"• λ_électrique: {model.fitter_elec.lambda_:.1f} ans")
                    st.write(f"• ρ_électrique: {model.fitter_elec.rho_:.2f}")
            
            # Validation et qualité des résultats
            st.subheader("✅ Validation des Résultats")
            
            # Tests de cohérence
            coherence_checks = []
            
            # Test 1: Monotonie des quantiles
            if all(results_df['P5'] <= results_df['P25']) and all(results_df['P25'] <= results_df['P50']) and all(results_df['P50'] <= results_df['P75']) and all(results_df['P75'] <= results_df['P95']):
                coherence_checks.append("✅ Ordre des quantiles cohérent")
            else:
                coherence_checks.append("⚠️ Problème ordre des quantiles")
            
            # Test 2: Variabilité réaliste
            cv_mean = np.mean(stds / means)  # Coefficient de variation moyen
            if 0.05 <= cv_mean <= 0.3:
                coherence_checks.append(f"✅ Variabilité réaliste (CV={cv_mean:.2f})")
            else:
                coherence_checks.append(f"⚠️ Variabilité atypique (CV={cv_mean:.2f})")
            
            # Test 3: Tendance physiquement plausible
            trend = np.polyfit(range(len(means)), means, 1)[0]
            if abs(trend) < len(means) * 0.1:  # Tendance raisonnable
                coherence_checks.append("✅ Tendance temporelle plausible")
            else:
                coherence_checks.append("⚠️ Tendance temporelle atypique")
            
            for check in coherence_checks:
                st.write(check)
        
        else:
            st.info("🔄 Aucune simulation disponible. Lancez d'abord l'analyse Monte Carlo.")
    
    # ====================================================================
    # FOOTER
    # ====================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666666; font-size: 0.9em;'>
        <p>⚡ Dashboard Fiabilité des Relais | Analyse de Survie & Simulation Monte Carlo</p>
        <p>Modèle Weibull à Risques Concurrents • Version 1.0.0</p>
    </div>
    """, unsafe_allow_html=True)

# ====================================================================
# POINT D'ENTRÉE PRINCIPAL
# ====================================================================

if __name__ == "__main__":
    main()
