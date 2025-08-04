import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from lifelines import WeibullFitter, LogNormalFitter, CoxPHFitter
from reliability.Fitters import Fit_Weibull_Mixture

st.set_page_config(layout="wide", page_title="Survie ferroviaire avancée")
st.title("Tableau de bord avancé : Fiabilité ferroviaire")

# ---------------- Connexion base PostgreSQL ----------------
db_string = st.secrets["db_url"] if "db_url" in st.secrets else "postgresql+psycopg2://user:password@host:port/dbname"
db_pgsql = create_engine(db_string)

uploaded_file = st.file_uploader("Uploader votre fichier CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Fichier chargé avec succès : {df.shape[0]} lignes")
    st.dataframe(df.head())
else:
    st.info("Veuillez uploader un fichier pour commencer.")
# ---------------- Interface utilisateur ----------------
st.sidebar.subheader("Chargement des données")
symb = st.sidebar.text_input("Symbole", "79540230")
constructeurs = st.sidebar.multiselect("Constructeurs", ['ANSA', 'CSEE', 'HITA'], default=['ANSA', 'CSEE'])
n_relais = st.sidebar.slider("Nombre de relais à afficher", min_value=10, max_value=500, value=100)

if st.sidebar.button("Charger"):
    df = load_relais_data(symb=symb, constructeurs=constructeurs)
    df = df.head(n_relais)
    st.dataframe(df.head())

    if "ACTIF" in df.columns and "censure" in df.columns:
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
                                            (model.beta_2, model.alpha_2, 1-model.proportion_1)]:
                        pdf = (beta/eta)*(x_vals/eta)**(beta-1)*np.exp(-(x_vals/eta)**beta)
                        total_pdf += prop * pdf
                    fig, ax = plt.subplots()
                    ax.plot(x_vals, total_pdf, label=f"Mixture {constr}")
                    ax.set_title(f"Densité de défaillance - {constr}")
                    ax.set_xlabel("Temps (années)")
                    ax.legend()
                    st.pyplot(fig)

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

        with st.expander("Log-Normal : Modèle de survie"):
            failures = df[df['censure'] == 1]['ACTIF']
            model_ln = LogNormalFitter().fit(failures)
            fig, ax = plt.subplots()
            model_ln.plot_survival_function(ax=ax)
            ax.set_title("Survie Log-Normale")
            st.pyplot(fig)

        with st.expander("Modèle de Cox Proportionnel"):
            df_cox = df.copy()
            df_cox["event"] = df_cox["censure"] == 1
            df_cox = df_cox.dropna(subset=["ACTIF"])
            cox = CoxPHFitter()
            try:
                cox.fit(df_cox[["ACTIF", "event"]], duration_col="ACTIF", event_col="event")
                fig, ax = plt.subplots()
                cox.plot_survival_function(ax=ax)
                ax.set_title("Fonction de survie - Cox PH")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Erreur Cox : {e}")

                st.pyplot(fig)
            except:
                st.error("Erreur lors de l'ajustement de Cox. Variables manquantes ?")
