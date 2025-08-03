import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.gradient_boosting import GradientBoostingSurvivalAnalysis
from lifelines import WeibullFitter, LogNormalFitter, CoxPHFitter

st.set_page_config(layout="wide", page_title="Survie ferroviaire avancée")
st.title("Tableau de bord avancé : Fiabilité ferroviaire")

# ------------------ Connexion Postgres ------------------
db_string = "postgresql+psycopg2://Integ:*In30teG@posqresql:5432/dtiesss"
db_pgsql = create_engine(db_string)

def load_relais_data(symb="79540230", constructeurs=['ANSA', 'CSEE', 'HITA']):
    with db_pgsql.connect() as db_pgconn:
        query = text("""
            SELECT * 
            FROM "AGRSIG_PBI"."ASTOTS" a 
            LEFT JOIN "AGRSIG_PBI"."FAS_DUREE_VIE_4" b 
            ON a."CLE" = b."CLE_TB_AS" 
            WHERE symb = :symb 
            AND lib_constr IN :constructeurs
        """)
        df = pd.read_sql(query, db_pgconn, params={"symb": symb, "constructeurs": tuple(constructeurs)})
    if "ACTIF" in df.columns:
        df["ACTIF"] = df["ACTIF"] / 365.25
    return df

# Chargement des données
st.sidebar.subheader("Chargement des données")
symb = st.sidebar.text_input("Symbole", "79540230")
constructeurs = st.sidebar.multiselect("Constructeurs", ['ANSA', 'CSEE', 'HITA'], default=['ANSA', 'CSEE'])
if st.sidebar.button("Charger"):
    df = load_relais_data(symb=symb, constructeurs=constructeurs)
    st.dataframe(df.head())

    if "ACTIF" in df.columns and "censure" in df.columns:
        with st.expander("Weibull double (mixte) par constructeur"):
            for constr in df['lib_constr'].unique():
                subset = df[(df['lib_constr'] == constr) & (~df['ACTIF'].isna())]
                failures = subset[subset['censure'] == 1]['ACTIF'].values
                censored = subset[subset['censure'] == 0]['ACTIF'].values
                if len(failures) > 20:
                    from reliability.Fitters import Fit_Weibull_Mixture
                    model = Fit_Weibull_Mixture(failures=failures, right_censored=censored, show_plot=False)
                    x_vals = np.linspace(0.1, 200, 1000)
                    total_pdf = np.zeros_like(x_vals)
                    for beta, eta, prop in [(model.beta_1, model.alpha_1, model.proportion_1),
                                            (model.beta_2, model.alpha_2, 1-model.proportion_1)]:
                        pdf = (beta/eta)*(x_vals/eta)**(beta-1)*np.exp(-(x_vals/eta)**beta)
                        total_pdf += prop * pdf
                    fig, ax = plt.subplots()
                    ax.plot(x_vals, total_pdf, label=f"Mixture {constr}")
                    ax.set_title(f"Densité de défaillance - {constr}")
                    ax.set_xlabel("Temps (années)")
                    ax.legend()
                    st.pyplot(fig)

        with st.expander("Competing Risks Weibull - Simulation"):
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
            ax.set_title("Survie par cause - Weibull")
            st.pyplot(fig)

        with st.expander("Modèle log-normal"):
            failures = df[df['censure'] == 1]['ACTIF']
            model_ln = LogNormalFitter().fit(failures)
            fig, ax = plt.subplots()
            model_ln.plot_survival_function(ax=ax)
            ax.set_title("Fonction de survie - Log-Normale")
            st.pyplot(fig)

        with st.expander("Cox Proportional Hazards"):
            df_cox = df.copy()
            df_cox["event"] = df_cox["censure"] == 1
            df_cox = df_cox.dropna(subset=["ACTIF"])
            cox = CoxPHFitter()
            try:
                cox.fit(df_cox[["ACTIF", "event"]], duration_col="ACTIF", event_col="event")
                fig, ax = plt.subplots()
                cox.plot_survival_function(ax=ax)
                ax.set_title("Cox PH - Survie")
                st.pyplot(fig)
            except:
                st.error("Erreur lors de l'ajustement de Cox. Variables manquantes ?")