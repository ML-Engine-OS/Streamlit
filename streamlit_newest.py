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

uploaded_file = st.file_uploader("Uploader votre fichier CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8', on_bad_lines='skip')

        # Nettoyage des noms de colonnes : enlever espaces, mettre en minuscules
        df.columns = df.columns.str.strip().str.lower()

        st.success(f"Fichier chargé avec succès : {df.shape[0]} lignes")
        st.dataframe(df.head())

        # Sidebar : filtres basés sur le DataFrame uploadé
        st.sidebar.subheader("Filtres")

        if 'lib_constr' in df.columns:
            constructeurs = st.sidebar.multiselect(
                "Constructeurs", options=df['lib_constr'].dropna().unique(), default=df['lib_constr'].dropna().unique())
        else:
            constructeurs = []

        n_relais = st.sidebar.slider("Nombre de relais à afficher", min_value=10, max_value=10000, value=100)

        # Filtrage des données
        if len(constructeurs) > 0:
            df_filtered = df[df['lib_constr'].isin(constructeurs)]
        else:
            df_filtered = df.copy()

        df_filtered = df_filtered.head(n_relais)
        st.dataframe(df_filtered)

        # === ANALYSES (sans condition) ===

        # Weibull mixte par constructeur
        with st.expander("Weibull mixte par constructeur"):
            for constr in df_filtered['lib_constr'].dropna().unique():
                subset = df_filtered[(df_filtered['lib_constr'] == constr) & (~df_filtered['actif'].isna())]
                failures = subset[subset['censure'] == 1]['actif'].values
                censored = subset[subset['censure'] == 0]['actif'].values
                if len(failures) > 20:
                    model = Fit_Weibull_Mixture(failures=failures, right_censored=censored, show_plot=False)
                    x_vals = np.linspace(0.1, 200, 1000)
                    total_pdf = np.zeros_like(x_vals)
                    for beta, eta, prop in [
                        (model.beta_1, model.alpha_1, model.proportion_1),
                        (model.beta_2, model.alpha_2, 1 - model.proportion_1)
                    ]:
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
            failures = df_filtered[df_filtered['censure'] == 1]['actif']
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
            df_cox = df_filtered.copy()
            df_cox["event"] = df_cox["censure"] == 1
            df_cox = df_cox.dropna(subset=["actif"])
            if len(df_cox) > 0:
                cox = CoxPHFitter()
                try:
                    cox.fit(df_cox[["actif", "event"]], duration_col="actif", event_col="event")
                    fig, ax = plt.subplots()
                    cox.plot_survival_function(ax=ax)
                    ax.set_title("Fonction de survie - Cox PH")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Erreur Cox PH : {e}")
            else:
                st.warning("Données insuffisantes pour le modèle Cox PH.")

    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier CSV : {e}")
else:
    st.info("Veuillez uploader un fichier CSV pour commencer.")
