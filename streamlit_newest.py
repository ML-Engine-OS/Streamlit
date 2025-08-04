import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math

from scipy.optimize import minimize
from scipy.stats import weibull_min, genextreme
from scipy.special import gamma

import reliability.Fitters as ft
from reliability.Fitters import Fit_Weibull_2P, Fit_Weibull_Mixture
from reliability.Nonparametric import KaplanMeier
from reliability.Distributions import Weibull_Distribution

from lifelines import WeibullFitter, KaplanMeierFitter, LogNormalFitter, CoxPHFitter
from lifelines.utils import k_fold_cross_validation

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Survie ferroviaire avancée")
st.title("Tableau de bord avancé : Fiabilité ferroviaire")

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8', on_bad_lines='skip')
            df.columns = df.columns.str.strip().str.lower()
        except Exception as e:
            st.error(f"Erreur lecture fichier uploadé : {e}")
            return None
    else:
        try:
            df = pd.read_csv("your_data.csv")  # chemin par défaut
        except Exception as e:
            st.error(f"Erreur lecture fichier local : {e}")
            return None

    # Nettoyage commun
    df["dtetat"] = pd.to_datetime(df["dtetat"], errors="coerce", format="%Y-%m-%d", exact=False)
    now = pd.Timestamp.today()
    df["age_etat"] = (now - df["dtetat"]).dt.days / 365.25
    df = df[df["dtetat"].notna() & (df["dtetat"].dt.year >= 1950) & (df["dtetat"].dt.year <= 2050)]
    df = df[df["censure"].isin([0, 1])]
    df["censure"] = df["censure"].astype(int)
    return df

uploaded_file = st.file_uploader("Uploader votre fichier CSV", type=["csv"])

df = load_data(uploaded_file)

if df is not None:
    st.success(f"Data chargée avec {df.shape[0]} lignes")
    st.dataframe(df.head())

    n_relais = st.sidebar.slider("Nombre de relais à afficher", min_value=10, max_value=10000, value=100)
    df = df.head(n_relais)

    # suite du traitement...
else:
    st.warning("Aucune donnée disponible")

st.title("🚀 Futuristic Survival Analysis Dashboard")

model_choice = st.sidebar.selectbox(
    "Choisir le modèle / méthode",
    ["Weibull Double + Monte Carlo",
     "Weibull Competing Risks + Monte Carlo",
     "Random Survival Forest (RSF)",
     "Gradient Boosting Survival Analysis (GBSA)",
     "Cox Proportional Hazards (CoxPH)",
     "Log-Normal Monte Carlo Simulation"]
)

def weibull_double_monte_carlo(df):
    st.header("Weibull Double Fitting + Monte Carlo")
    N_simulations = st.number_input("Nombre de simulations", min_value=10, max_value=1000, value=100)
    N_years = st.number_input("Nombre d'années à simuler", min_value=1, max_value=50, value=25)
    
    # Placeholder: your weibull double fitting + monte carlo simulation logic here
    st.info("Simulation en cours... (intégrer votre fonction ici)")
    # Show plots & results here after simulation

def weibull_competing_risks(df):
    st.header("Weibull Competing Risks + Monte Carlo")
    N_simulations = st.number_input("Nombre de simulations", min_value=10, max_value=500, value=100)
    N_years = st.number_input("Nombre d'années à simuler", min_value=1, max_value=50, value=25)

    st.info("Simulation en cours... (intégrer votre fonction ici)")
    # Add your competing risks code + plots

def random_survival_forest(df):
    st.header("Random Survival Forest (RSF)")
    st.write("Entraînement et évaluation du modèle RSF sur un sous-échantillon.")

    sample_size = st.slider("Taille de l'échantillon", min_value=1000, max_value=50000, value=20000, step=1000)
    subset_df = df.sample(n=sample_size, random_state=42)
    
    # Prepare features and target
    # Encode categoricals
    subset_df = subset_df.dropna(subset=["lib_constr", "lib_lettre", "AGE_ETAT", "ACTIF", "censure"])
    X = pd.get_dummies(subset_df[["lib_constr", "lib_lettre", "AGE_ETAT"]], drop_first=True)
    y = np.array([(bool(e), t) for e, t in zip(subset_df["censure"], subset_df["ACTIF"])], dtype=[("event", bool), ("time", float)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rsf = RandomSurvivalForest(n_estimators=50, min_samples_split=10, min_samples_leaf=15, max_features="sqrt", n_jobs=-1, random_state=42)
    with st.spinner("Entraînement du RSF..."):
        rsf.fit(X_train, y_train)
    c_index = concordance_index_censored(y_test["event"], y_test["time"], rsf.predict(X_test))[0]
    st.success(f"C-index RSF: {c_index:.3f}")

    # Plot survival curves
    st.write("Visualisation des courbes de survie pour un sous-échantillon :")
    nb_to_plot = min(100, len(X_test))
    surv_fns = rsf.predict_survival_function(X_test.iloc[:nb_to_plot])
    
    fig, ax = plt.subplots(figsize=(12,7))
    for fn in surv_fns:
        ax.step(fn.x, fn.y, where="post", alpha=0.3)
    ax.set_xlabel("Temps (années)")
    ax.set_ylabel("Probabilité de survie")
    ax.set_title("Courbes de survie RSF")
    ax.grid(True)
    st.pyplot(fig)

def gradient_boosting_survival(df):
    st.header("Gradient Boosting Survival Analysis (GBSA)")
    df = df.dropna(subset=["ACTIF", "censure"])
    features = ["ACTIF"]  # add others if relevant
    X = df[features]
    y = Surv.from_dataframe("censure", "ACTIF", df)
    y = np.array([(bool(e), t) for e, t in zip(df["censure"], df["ACTIF"])], dtype=[("event", bool), ("time", float)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingSurvivalAnalysis(n_estimators=80, learning_rate=0.2, max_depth=3, random_state=42)
    with st.spinner("Entraînement GBSA..."):
        model.fit(X_train, y_train)

    predicted_risks = model.predict(X_test)
    cindex = concordance_index_censored(y_test["event"], y_test["time"], predicted_risks)[0]
    st.success(f"C-index GBSA : {cindex:.4f}")

    surv_functions = model.predict_survival_function(X_test.iloc[:100])

    fig, ax = plt.subplots(figsize=(15, 8))
    for fn in surv_functions:
        ax.step(fn.x, fn.y, where="post", alpha=0.3)
    ax.set_title("Courbes de survie (Gradient Boosting Survival)")
    ax.set_xlabel("Temps (durée en service)")
    ax.set_ylabel("Probabilité de survie")
    ax.grid(True)
    st.pyplot(fig)

def cox_ph(df):
    st.header("Cox Proportional Hazards (CoxPH)")
    colonnes = ['ACTIF', 'censure', 'AGE_ETAT', 'lib_constr', 'lib_lettre']
    df_clean = df[colonnes].dropna()
    df_encoded = pd.get_dummies(df_clean, columns=['lib_constr', 'lib_lettre'], drop_first=True)

    X = df_encoded.drop(columns=['ACTIF', 'censure'])
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    X_filtered = X.drop(columns=to_drop)

    df_final = pd.concat([df_encoded[['ACTIF', 'censure']], X_filtered], axis=1)

    cph = CoxPHFitter(penalizer=0.1)
    with st.spinner("Entraînement du modèle CoxPH..."):
        cph.fit(df_final, duration_col='ACTIF', event_col='censure')
    st.write(cph.summary)

    n = 30
    sample = df_final.drop(columns=['ACTIF', 'censure']).iloc[:n]
    surv = cph.predict_survival_function(sample)

    fig, ax = plt.subplots(figsize=(14, 8))
    for i in range(n):
        ax.step(surv.index, surv.iloc[:, i], where="post", alpha=0.7)
    ax.set_title("Courbes de survie - Modèle de Cox")
    ax.set_xlabel("Temps (années)")
    ax.set_ylabel("Probabilité de survie")
    ax.grid(True)
    st.pyplot(fig)

def lognormal_monte_carlo(df):
    st.header("Log-Normal Monte Carlo Simulation")

    lognorm_df = df[["ACTIF", "censure"]].dropna()
    lognorm_fitter = LogNormalFitter()
    lognorm_fitter.fit(durations=lognorm_df["ACTIF"], event_observed=lognorm_df["censure"])

    st.write(f"Paramètres Log-Normal : mu = {lognorm_fitter.mu_:.2f}, sigma = {lognorm_fitter.sigma_:.2f}")

    def generate_lifetime_lognormal(current_age):
        mu = lognorm_fitter.mu_
        sigma = lognorm_fitter.sigma_
        t = np.random.lognormal(mean=mu, sigma=sigma)
        t_res = t - current_age
        return max(t_res, 0)

    N_simulations = st.number_input("Nombre de simulations", min_value=10, max_value=1000, value=300)
    N_years = st.number_input("Nombre d'années à simuler", min_value=1, max_value=50, value=25)

    ages_actuels = df[df['censure'] == 0]['ACTIF'].dropna().values
    parc_initial = list(ages_actuels)
    consommation_annuelle = []

    with st.spinner("Simulation Monte Carlo Log-Normal en cours..."):
        for sim in range(N_simulations):
            parc = list(parc_initial)
            consommation = []

            for annee in range(N_years):
                nb_remplacements = 0
                nouveau_parc = []

                for age in parc:
                    t_reste = generate_lifetime_lognormal(age)
                    if t_reste <= 2:
                        nb_remplacements += 1
                        nouveau_parc.append(0)
                    else:
                        nouveau_parc.append(age + 1)
                parc = nouveau_parc
                consommation.append(nb_remplacements)
            consommation_annuelle.append(consommation)

    conso_array = np.array(consommation_annuelle)
    moyenne_annuelle = conso_array.mean(axis=0)
    std_annuelle = conso_array.std(axis=0)
    years = np.arange(2025, 2025 + N_years)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(years, moyenne_annuelle, label="Consommation moyenne", color="navy")
    ax.fill_between(years, moyenne_annuelle - std_annuelle, moyenne_annuelle + std_annuelle,
                    alpha=0.4, color="red", label="± 1 écart-type")
    ax.set_xlabel("Année")
    ax.set_ylabel("Nombre de relais remplacés")
    ax.set_title("Prévision annuelle de consommation moyenne (Log-Normal Monte Carlo)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    df_violin = pd.DataFrame(conso_array, columns=years)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.violinplot(data=df_violin, inner="quartile", cut=0, ax=ax2)
    ax2.set_title("Distribution annuelle de la consommation des relais")
    ax2.set_xlabel("Année")
    ax2.set_ylabel("Relais remplacés")
    ax2.grid(True)
    plt.setp(ax2.get_xticklabels(), rotation=45)
    st.pyplot(fig2)

# Main app logic
if model_choice == "Weibull Double + Monte Carlo":
    weibull_double_monte_carlo(df)
elif model_choice == "Weibull Competing Risks + Monte Carlo":
    weibull_competing_risks(df)
elif model_choice == "Random Survival Forest (RSF)":
    random_survival_forest(df)
elif model_choice == "Gradient Boosting Survival Analysis (GBSA)":
    gradient_boosting_survival(df)
elif model_choice == "Cox Proportional Hazards (CoxPH)":
    cox_ph(df)
elif model_choice == "Log-Normal Monte Carlo Simulation":
    lognormal_monte_carlo(df)
else:
    st.warning("Sélectionnez un modèle dans le menu latéral.")

        
