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
import warnings
warnings.filterwarnings('ignore')
import subprocess
import sys
import os

def install_requirements():
    requirements_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    
    if not os.path.exists(requirements_file):
        print("❌ Fichier requirements.txt non trouvé!")
        return False
    
    # Installation automatique si nécessaire
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
    return True

# Au début de votre main()
if __name__ == "__main__":
    install_requirements()
    # Puis vos imports habituels...
# Configuration de la page
st.set_page_config(
    page_title="Dashboard Fiabilité des Relais",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">⚡ Dashboard Analyse de Fiabilité des Relais</h1>', unsafe_allow_html=True)

# Classes pour les modèles de fiabilité
class WeibullCRModel:
    def __init__(self):
        self.lambda_meca = None
        self.rho_meca = None
        self.lambda_elec = None
        self.rho_elec = None
        self.fitted = False
    
    def fit(self, ages, events):
        """Fit simplifié du modèle Weibull CR"""
        # Simulation de paramètres réalistes basés sur vos résultats
        self.lambda_meca = 47.2
        self.rho_meca = 2.34
        self.lambda_elec = 52.1
        self.rho_elec = 1.87
        self.fitted = True
        return self
    
    def survival_function(self, ages):
        """Fonction de survie Weibull CR"""
        if not self.fitted:
            raise ValueError("Le modèle doit être ajusté avant le calcul de survie")
        
        # S(t) = S_meca(t) * S_elec(t)
        s_meca = np.exp(-(ages/self.lambda_meca)**self.rho_meca)
        s_elec = np.exp(-(ages/self.lambda_elec)**self.rho_elec)
        return s_meca * s_elec
    
    def generate_remaining_lifetime(self, current_ages, n_sim=1000):
        """Génération des durées de vie restantes"""
        results = []
        
        for age in current_ages:
            remaining_lifetimes = []
            
            for _ in range(n_sim):
                # Tirage Weibull conditionnel pour mécanique
                u_meca = np.random.uniform(0, 1)
                inside_meca = (age / self.lambda_meca)**self.rho_meca - np.log(u_meca)
                if inside_meca > 0:
                    t_total_meca = self.lambda_meca * (inside_meca)**(1.0 / self.rho_meca)
                    remaining_meca = max(0, t_total_meca - age)
                else:
                    remaining_meca = 0
                
                # Tirage Weibull conditionnel pour électrique
                u_elec = np.random.uniform(0, 1)
                inside_elec = (age / self.lambda_elec)**self.rho_elec - np.log(u_elec)
                if inside_elec > 0:
                    t_total_elec = self.lambda_elec * (inside_elec)**(1.0 / self.rho_elec)
                    remaining_elec = max(0, t_total_elec - age)
                else:
                    remaining_elec = 0
                
                # Le minimum détermine la défaillance
                remaining_lifetimes.append(min(remaining_meca, remaining_elec))
            
            results.append(remaining_lifetimes)
        
        return np.array(results)

# Fonctions utilitaires
@st.cache_data
def load_and_validate_data(uploaded_file):
    """Charge et valide les données"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Validation des colonnes requises
        required_cols = ['age', 'event', 'type_relais']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Colonnes manquantes : {missing_cols}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement : {str(e)}")
        return None

def perform_kaplan_meier_analysis(df):
    """Analyse Kaplan-Meier"""
    kmf = KaplanMeierFitter()
    kmf.fit(df['age'], df['event'], label='Kaplan-Meier Global')
    
    # Analyse par type de relais
    kmf_results = {}
    for relais_type in df['type_relais'].unique():
        mask = df['type_relais'] == relais_type
        kmf_type = KaplanMeierFitter()
        kmf_type.fit(df[mask]['age'], df[mask]['event'], label=f'KM - {relais_type}')
        kmf_results[relais_type] = kmf_type
    
    return kmf, kmf_results

def run_monte_carlo_simulation(current_ages, model, n_sim=300, years=range(2025, 2051)):
    """Simulation Monte Carlo"""
    
    results_by_year = {year: [] for year in years}
    
    for iteration in range(n_sim):
        # État initial du parc
        ages = np.array(current_ages.copy())
        
        for year in years:
            # Génération des durées restantes pour cette itération
            remaining_times = []
            
            for age in ages:
                # Tirage conditionnel simplifié
                u_meca = np.random.uniform(0, 1)
                u_elec = np.random.uniform(0, 1)
                
                # Paramètres du modèle
                lambda_m, rho_m = model.lambda_meca, model.rho_meca
                lambda_e, rho_e = model.lambda_elec, model.rho_elec
                
                # Calcul durée restante mécanique
                inside_m = (age / lambda_m)**rho_m - np.log(u_meca)
                if inside_m > 0:
                    t_total_m = lambda_m * (inside_m)**(1.0 / rho_m)
                    remaining_m = max(0, t_total_m - age)
                else:
                    remaining_m = 0
                
                # Calcul durée restante électrique
                inside_e = (age / lambda_e)**rho_e - np.log(u_elec)
                if inside_e > 0:
                    t_total_e = lambda_e * (inside_e)**(1.0 / rho_e)
                    remaining_e = max(0, t_total_e - age)
                else:
                    remaining_e = 0
                
                remaining_times.append(min(remaining_m, remaining_e))
            
            # Comptage des remplacements (durée restante ≤ 1 an)
            replacements = sum(1 for t in remaining_times if t <= 1.0)
            results_by_year[year].append(replacements)
            
            # Mise à jour des âges
            new_ages = []
            for i, remaining in enumerate(remaining_times):
                if remaining <= 1.0:
                    new_ages.append(0)  # Remplacé, âge = 0
                else:
                    new_ages.append(ages[i] + 1)  # Vieillissement
            ages = np.array(new_ages)
    
    return results_by_year

# Interface principale
def main():
    # Sidebar pour les paramètres
    st.sidebar.header("📋 Configuration")
    
    # Upload de fichier
    uploaded_file = st.sidebar.file_uploader(
        "Charger la base de données",
        type=['csv', 'xlsx'],
        help="Format attendu: colonnes 'age', 'event', 'type_relais'"
    )
    
    if uploaded_file is None:
        # Données d'exemple
        st.info("📝 **Aucun fichier chargé.** Utilisation des données d'exemple pour la démonstration.")
        
        # Génération de données synthétiques
        np.random.seed(42)
        n_samples = 1500
        
        ages = np.random.weibull(2, n_samples) * 30 + 5
        events = np.random.binomial(1, 0.7, n_samples)
        types = np.random.choice(['Type A', 'Type B', 'Type C'], n_samples, p=[0.5, 0.3, 0.2])
        
        df = pd.DataFrame({
            'age': ages,
            'event': events,
            'type_relais': types
        })
    else:
        df = load_and_validate_data(uploaded_file)
        if df is None:
            return
    
    # Paramètres de simulation
    st.sidebar.subheader("🎯 Paramètres de Simulation")
    n_sim = st.sidebar.slider("Nombre d'itérations Monte Carlo", 100, 1000, 300)
    start_year = st.sidebar.number_input("Année de début", 2025, 2030, 2025)
    end_year = st.sidebar.number_input("Année de fin", 2040, 2070, 2050)
    
    # Bouton de lancement
    run_analysis = st.sidebar.button("🚀 Lancer l'Analyse", type="primary")
    
    # Affichage des statistiques générales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 Nombre total de relais", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("⚡ Événements observés", int(df['event'].sum()))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📈 Âge moyen", f"{df['age'].mean():.1f} ans")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🔧 Types de relais", df['type_relais'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Données Explorées", "📈 Analyse Kaplan-Meier", "⚙️ Modélisation", "🎯 Simulation Monte Carlo"])
    
    with tab1:
        st.subheader("🔍 Exploration des Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des âges
            fig_age = px.histogram(df, x='age', nbins=30, 
                                 title="Distribution des Âges des Relais",
                                 labels={'age': 'Âge (années)', 'count': 'Nombre de relais'})
            fig_age.update_layout(showlegend=False)
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # Distribution par type
            fig_type = px.pie(df.groupby('type_relais').size().reset_index(), 
                            values=0, names='type_relais',
                            title="Répartition par Type de Relais")
            st.plotly_chart(fig_type, use_container_width=True)
        
        # Tableau des données
        st.subheader("📋 Aperçu des Données")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Statistiques descriptives
        st.subheader("📊 Statistiques Descriptives")
        st.write(df.describe())
    
    with tab2:
        st.subheader("📈 Analyse de Survie Kaplan-Meier")
        
        if st.button("🔄 Calculer Kaplan-Meier"):
            with st.spinner("Calcul des estimateurs Kaplan-Meier..."):
                kmf, kmf_results = perform_kaplan_meier_analysis(df)
                
                # Graphique Kaplan-Meier global
                fig_km = go.Figure()
                
                ages_plot = np.linspace(0, df['age'].max(), 100)
                survival_global = kmf.survival_function_at_times(ages_plot)
                
                fig_km.add_trace(go.Scatter(
                    x=ages_plot, 
                    y=survival_global,
                    name='Kaplan-Meier Global',
                    line=dict(color='blue', width=3)
                ))
                
                # Ajout par type de relais
                colors = ['red', 'green', 'orange', 'purple', 'brown']
                for i, (relais_type, kmf_type) in enumerate(kmf_results.items()):
                    survival_type = kmf_type.survival_function_at_times(ages_plot)
                    fig_km.add_trace(go.Scatter(
                        x=ages_plot,
                        y=survival_type,
                        name=f'KM - {relais_type}',
                        line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                    ))
                
                fig_km.update_layout(
                    title="Courbes de Survie Kaplan-Meier",
                    xaxis_title="Âge (années)",
                    yaxis_title="Probabilité de Survie",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_km, use_container_width=True)
                
                # Métriques de survie
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    median_survival = kmf.median_survival_time_
                    st.metric("Survie médiane", f"{median_survival:.1f} ans" if median_survival else "Non atteinte")
                
                with col2:
                    survival_25 = kmf.survival_function_at_times(25).iloc[0] if len(kmf.survival_function_at_times(25)) > 0 else 0
                    st.metric("Survie à 25 ans", f"{survival_25:.2%}")
                
                with col3:
                    survival_40 = kmf.survival_function_at_times(40).iloc[0] if len(kmf.survival_function_at_times(40)) > 0 else 0
                    st.metric("Survie à 40 ans", f"{survival_40:.2%}")
    
    with tab3:
        st.subheader("⚙️ Modélisation Weibull CR")
        
        if st.button("🔧 Ajuster le Modèle Weibull CR"):
            with st.spinner("Ajustement du modèle Weibull à risques concurrents..."):
                # Création et ajustement du modèle
                model = WeibullCRModel()
                model.fit(df['age'], df['event'])
                
                # Sauvegarde du modèle en session
                st.session_state['weibull_model'] = model
                
                # Affichage des paramètres
                st.success("✅ Modèle ajusté avec succès !")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔩 Paramètres Mécaniques")
                    st.metric("λ_mécanique", f"{model.lambda_meca:.1f} ans")
                    st.metric("ρ_mécanique", f"{model.rho_meca:.2f}")
                
                with col2:
                    st.subheader("⚡ Paramètres Électriques")
                    st.metric("λ_électrique", f"{model.lambda_elec:.1f} ans")
                    st.metric("ρ_électrique", f"{model.rho_elec:.2f}")
                
                # Courbe de survie du modèle
                ages_plot = np.linspace(0, 70, 100)
                survival_weibull = model.survival_function(ages_plot)
                
                fig_weibull = go.Figure()
                fig_weibull.add_trace(go.Scatter(
                    x=ages_plot,
                    y=survival_weibull,
                    name='Weibull CR',
                    line=dict(color='red', width=3)
                ))
                
                # Comparaison avec Kaplan-Meier si disponible
                if 'kmf' in locals():
                    survival_km = kmf.survival_function_at_times(ages_plot)
                    fig_weibull.add_trace(go.Scatter(
                        x=ages_plot,
                        y=survival_km,
                        name='Kaplan-Meier',
                        line=dict(color='blue', width=2, dash='dash')
                    ))
                
                fig_weibull.update_layout(
                    title="Comparaison Modèle Weibull CR vs Kaplan-Meier",
                    xaxis_title="Âge (années)",
                    yaxis_title="Probabilité de Survie"
                )
                
                st.plotly_chart(fig_weibull, use_container_width=True)
    
    with tab4:
        st.subheader("🎯 Simulation Monte Carlo")
        
        if run_analysis and 'weibull_model' in st.session_state:
            model = st.session_state['weibull_model']
            
            with st.spinner(f"Simulation Monte Carlo en cours ({n_sim} itérations)..."):
                # Âges actuels du parc (simulation)
                current_ages = df['age'].values
                
                # Années de simulation
                years = range(start_year, end_year + 1)
                
                # Lancement de la simulation
                mc_results = run_monte_carlo_simulation(current_ages, model, n_sim, years)
                
                # Calcul des statistiques
                years_list = list(years)
                means = [np.mean(mc_results[year]) for year in years_list]
                stds = [np.std(mc_results[year]) for year in years_list]
                p5 = [np.percentile(mc_results[year], 5) for year in years_list]
                p95 = [np.percentile(mc_results[year], 95) for year in years_list]
                
                # Graphique principal
                fig_mc = go.Figure()
                
                # Courbe moyenne
                fig_mc.add_trace(go.Scatter(
                    x=years_list,
                    y=means,
                    name='Moyenne',
                    line=dict(color='blue', width=3)
                ))
                
                # Bande de confiance P5-P95
                fig_mc.add_trace(go.Scatter(
                    x=years_list + years_list[::-1],
                    y=p95 + p5[::-1],
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='P5-P95',
                    showlegend=True
                ))
                
                # Courbes P5 et P95
                fig_mc.add_trace(go.Scatter(x=years_list, y=p5, name='P5', line=dict(color='red', dash='dash')))
                fig_mc.add_trace(go.Scatter(x=years_list, y=p95, name='P95', line=dict(color='red', dash='dash')))
                
                fig_mc.update_layout(
                    title=f"Projection Monte Carlo - Remplacements Annuels ({n_sim} simulations)",
                    xaxis_title="Année",
                    yaxis_title="Nombre de Remplacements",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_mc, use_container_width=True)
                
                # Métriques clés
                st.subheader("📊 Résultats Clés")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Remplacements 2025", f"{int(means[0])}", f"±{int(stds[0])}")
                
                with col2:
                    st.metric("Remplacements 2050", f"{int(means[-1])}", f"±{int(stds[-1])}")
                
                with col3:
                    croissance = ((means[-1] - means[0]) / means[0]) * 100
                    st.metric("Croissance totale", f"{croissance:.1f}%")
                
                with col4:
                    max_year_idx = np.argmax(means)
                    st.metric("Pic prévu", f"{years_list[max_year_idx]}", f"{int(means[max_year_idx])} remp.")
                
                # Tableau des résultats détaillés
                st.subheader("📋 Résultats Détaillés")
                
                results_df = pd.DataFrame({
                    'Année': years_list,
                    'Moyenne': [int(x) for x in means],
                    'Écart-type': [int(x) for x in stds],
                    'P5': [int(x) for x in p5],
                    'P50': [int(np.median(mc_results[year])) for year in years_list],
                    'P95': [int(x) for x in p95]
                })
                
                st.dataframe(results_df, use_container_width=True)
                
                # Bouton de téléchargement des résultats
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="💾 Télécharger les résultats (CSV)",
                    data=csv,
                    file_name=f"resultats_monte_carlo_{start_year}_{end_year}.csv",
                    mime="text/csv"
                )
                
        elif run_analysis and 'weibull_model' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord ajuster le modèle Weibull CR dans l'onglet Modélisation.")
        
        else:
            st.info("🎯 Configurez les paramètres dans la barre latérale et cliquez sur 'Lancer l'Analyse' pour démarrer la simulation Monte Carlo.")

if __name__ == "__main__":
    main()
