import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import random
import math

from scipy.optimize import minimize
from scipy.stats import weibull_min, genextreme
from scipy.special import gamma
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import reliability.Fitters as ft
from reliability.Fitters import Fit_Weibull_2P, Fit_Weibull_Mixture
from reliability.Nonparametric import KaplanMeier
from reliability.Distributions import Weibull_Distribution


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


from sqlalchemy import create_engine,text
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Pour la modélisation statistique
from lifelines import WeibullFitter, KaplanMeierFitter
from lifelines.plotting import plot_lifetimes
from lifelines import LogNormalFitter
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sksurv.functions import StepFunction
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.datasets import get_x_y

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Survie ferroviaire avancée")
st.title("Tableau de bord avancé : Fiabilité ferroviaire")

# --- Connexion base PostgreSQL ---
def get_db_engine():
    if "db_url" in st.secrets:
        db_string = st.secrets["db_url"]
    else:
        # Valeur par défaut, à remplacer ou configurer dans secrets.toml
        db_string = "postgresql+psycopg2://user:password@host:port/dbname"
    engine = create_engine(db_string)
    return engine

db_pgsql = get_db_engine()

# --- Chargement fichier CSV ---
uploaded_file = st.file_uploader("Uploader votre fichier CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Fichier chargé avec succès : {df.shape[0]} lignes")
    st.dataframe(df.head())
else:
    st.info("Veuillez uploader un fichier CSV pour commencer.")

# --- Sidebar : filtres et chargement base ---
st.sidebar.subheader("Chargement des données")
symb = st.sidebar.text_input("Symbole", "79540230")
constructeurs = st.sidebar.multiselect("Constructeurs", ['ANSA', 'CSEE', 'HITA'], default=['ANSA', 'CSEE'])
n_relais = st.sidebar.slider("Nombre de relais à afficher", min_value=10, max_value=500, value=100)

def load_relais_data(symb, constructeurs):
    # À adapter selon ta source réelle : requête SQL, API, fichier...
    # Exemple fictif : récupération d'un DataFrame depuis la base PostgreSQL
    query = f"""
    SELECT * FROM relais
    WHERE symbole = '{symb}'
    AND lib_constr IN ({','.join(f"'{c}'" for c in constructeurs)})
    """
    df = pd.read_sql_query(query, db_pgsql)
    return df

if st.sidebar.button("Charger"):
    try:
        df = load_relais_data(symb=symb, constructeurs=constructeurs)
        df = df.head(n_relais)
        st.dataframe(df)

        if "ACTIF" in df.columns and "censure" in df.columns:

            # Weibull mixte par constructeur
            with st.expander("Weibull mixte par constructeur"):
                for constr in df['lib_constr'].dropna().unique():
                    subset = df[(df['lib_constr'] == constr) & (~df['ACTIF'].isna())]
                    failures = subset[subset['censure'] == 1]['ACTIF'].values
                    censored = subset[subset['censure'] == 0]['ACTIF'].values
                    if len(failures) > 20:
                        model = Fit_Weibull_Mixture(failures=failures, right_censored=censored, show_plot=False)
                        x_vals = np.linspace(0.1, 200, 1000)
                        total_pdf = np.zeros_like(x_vals)
                        for beta, eta, prop in [(model.beta_1, model.alpha_1, model.proportion_1),
                                                (model.beta_2, model.alpha_2, 1 - model.proportion_1)]:
                            pdf = (beta / eta) * (x_vals / eta) ** (beta - 1) * np.exp(-(x_vals / eta) ** beta)
                            total_pdf += prop * pdf
                        fig, ax = plt.subplots()
                        ax.plot(x_vals, total_pdf, label=f"Mixture {constr}")
                        ax.set_title(f"Densité de défaillance - {constr}")
                        ax.set_xlabel("Temps (années)")
                        ax.legend()
                        st.pyplot(fig)

            # Risques concurrents - Simulation Weibull
            with st.expander("Risques concurrents - Simulation Weibull"):
                np.random.seed(42)
                n = 5000
                age_meca = np.random.weibull(1.5, size=n) * 30
                age_elec = np.random.weibull(2.5, size=n) * 40
                true_event_time = np.minimum(age_meca, age_elec)
                cause = np.where(age_meca < age_elec, 'mecanique', 'electrique')
                censure = np.random.binomial(1, 0.2, size=n)
                observed_time = np.where(censure == 0, true_event_time, true_event_time - np.random.uniform(0, 10, size=n))
                observed_time = np.clip(observed_time, 0.01, None)
                df_comp = pd.DataFrame({"time": observed_time, "event": censure == 0, "cause": cause})

                fm = WeibullFitter().fit(df_comp[df_comp["cause"] == "mecanique"]["time"],
                                         event_observed=df_comp[df_comp["cause"] == "mecanique"]["event"],
                                         label="Méca")
                fe = WeibullFitter().fit(df_comp[df_comp["cause"] == "electrique"]["time"],
                                         event_observed=df_comp[df_comp["cause"] == "electrique"]["event"],
                                         label="Elec")

                fig, ax = plt.subplots()
                fm.plot_survival_function(ax=ax)
                fe.plot_survival_function(ax=ax)
                ax.set_title("Fonction de survie - Risques concurrents")
                st.pyplot(fig)

            # Log-Normal : Modèle de survie
            with st.expander("Log-Normal : Modèle de survie"):
                failures = df[df['censure'] == 1]['ACTIF']
                if len(failures) > 0:
                    model_ln = LogNormalFitter().fit(failures)
                    fig, ax = plt.subplots()
                    model_ln.plot_survival_function(ax=ax)
                    ax.set_title("Survie Log-Normale")
                    st.pyplot(fig)
                else:
                    st.warning("Pas de données de défaillance pour Log-Normal.")

            # Modèle de Cox Proportionnel
            with st.expander("Modèle de Cox Proportionnel"):
                df_cox = df.copy()
                df_cox["event"] = df_cox["censure"] == 1
                df_cox = df_cox.dropna(subset=["ACTIF"])
                if len(df_cox) > 0:
                    cox = CoxPHFitter()
                    try:
                        cox.fit(df_cox[["ACTIF", "event"]], duration_col="ACTIF", event_col="event")
                        fig, ax = plt.subplots()
                        cox.plot_survival_function(ax=ax)
                        ax.set_title("Fonction de survie - Cox PH")
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Erreur Cox PH : {e}")
                else:
                    st.warning("Données insuffisantes pour le modèle Cox PH.")

    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
