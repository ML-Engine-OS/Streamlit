import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
import time

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
st.title("Tableau de bord  : Analyse prédictive de la survie des relais de signalisation")


@st.cache_data
def load_data(uploaded_file=None):
    #if uploaded_file is not None:
        #try:
            df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8', on_bad_lines='skip')
            return df
       # except Exception as e:
            #st.error(f"Erreur lecture fichier uploadé : {e}")
            #return None
    #return None


uploaded_file = st.file_uploader("Uploader votre fichier CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8', on_bad_lines='skip')
        if 'DTETAT' in df.columns:
            df["DTETAT"] = pd.to_datetime(df["DTETAT"], errors="coerce", format="%Y-%m-%d", exact=False)
            now = pd.Timestamp.today()
            df["AGE_ETAT"] = (now - df["DTETAT"]).dt.days / 365.25
            df["censure;;"] = pd.to_numeric(df["censure;;"], errors="coerce").fillna(0).astype(int)
            st.success(f"Data chargée avec {df.shape[0]} lignes")
            st.dataframe(df.head())
            n_relais = st.sidebar.slider("Nombre de relais à afficher", min_value=10, max_value=10000, value=100)
            
        else:
            st.error("La colonne 'DTETAT' est absente dans le fichier.")
        
    except Exception as e:
        st.error(f"Erreur lecture fichier uploadé : {e}")
else:
    st.warning("Aucun fichier chargé.")
        
        
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


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def weibull_double_monte_carlo(df):
    st.header("Weibull Double Fitting + Monte Carlo")
    N_simulations = st.number_input("Nombre de simulations", min_value=10, max_value=1000, value=100)
    N_years = st.number_input("Nombre d'années à simuler", min_value=1, max_value=50, value=25)
    
    # --- Données fictives pour Weibull Double Fitting ---
    data_failures = np.random.weibull(a=1.5, size=100) * 50
    data_censured = np.random.weibull(a=1.5, size=20) * 50

    # Fit Weibull 2P
    wb = Fit_Weibull_2P(failures=data_failures, right_censored=data_censured, show_probability_plot=False)
    fitted_weibull = Weibull_Distribution(alpha=wb.alpha, beta=wb.beta)

    # Courbe de survie
    x = np.linspace(0, 65, 500)
    sf = fitted_weibull.SF(x)
    fig, ax = plt.subplots()
    ax.plot(x, sf, '--', label="Fitted Weibull 2P")
    ax.set_xlabel("Temps")
    ax.set_ylabel("Fonction de survie")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.info("Simulation en cours... (intégrer votre fonction ici)")

    # --- Âges initiaux simulés ---
    np.random.seed(42)
    ages_actuels = np.random.uniform(0, 20, size=1000)

    # --- Fit Weibull 2P sur jeu de données simulé ---
    failures = np.random.weibull(1.5, 500) * 20
    censored = np.random.weibull(1.5, 200) * 20

    wb = Fit_Weibull_2P(failures=failures, right_censored=censored, show_probability_plot=False)
    fitted_weibull = Weibull_Distribution(alpha=wb.alpha, beta=wb.beta)

    def generate_remaining_lifetime(age_actuel):
        u = np.random.uniform()
        total_life = fitted_weibull.generate_random(1)[0]
        remaining = total_life - age_actuel
        return remaining

    parc_initial = list(ages_actuels)
    st.write(f"Nombre initial de relais : {len(parc_initial)}")
    st.write(f"Paramètres Weibull estimés : alpha = {wb.alpha:.2f}, beta = {wb.beta:.2f}")

    # --- Simulation Monte Carlo ---
    start = time.time()
    consommation_annuelle = []
    ages_par_annee = []

    for sim in range(N_simulations):
        parc = list(parc_initial)
        consommation = []
        ages_sim = []

        for annee in range(N_years):
            nb_remplacements = 0
            nouveau_parc = []

            for age in parc:
                duree_restante = generate_remaining_lifetime(age)
                if 0 < duree_restante <= 1:
                    nb_remplacements += 1
                    nouveau_parc.append(0)
                elif duree_restante < 0:
                    pass  # composant hors service
                else:
                    nouveau_parc.append(age + 1)

            parc = nouveau_parc
            consommation.append(nb_remplacements)
            ages_sim.append(parc.copy())

        consommation_annuelle.append(consommation)
        ages_par_annee.append(ages_sim)

    st.write(f"Simulation terminée en {time.time() - start:.2f} secondes.")

    # --- Résultats statistiques ---
    conso_array = np.array(consommation_annuelle)
    moyenne_annuelle = conso_array.mean(axis=0)
    std_annuelle = conso_array.std(axis=0)

    # --- Graphique consommation moyenne ---
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    years = np.arange(2025, 2025 + N_years)
    ax1.plot(years, moyenne_annuelle, label="Consommation moyenne", color="navy")
    ax1.fill_between(years, moyenne_annuelle - std_annuelle, moyenne_annuelle + std_annuelle,
                     alpha=0.2, color="red", label="± 1 écart-type")
    ax1.set_xlabel("Année")
    ax1.set_ylabel("Nombre de relais remplacés")
    ax1.set_title("Prévision annuelle de consommation moyenne des relais (Monte Carlo)")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    # --- Violin plot ---
    df_violin = pd.DataFrame(conso_array, columns=years)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.violinplot(data=df_violin, inner="quartile", palette="coolwarm", cut=0, ax=ax2)
    ax2.set_title("Distribution annuelle de la consommation des relais")
    ax2.set_xlabel("Année")
    ax2.set_ylabel("Relais remplacés")
    ax2.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # --- Histogrammes des âges pour la 1ère simulation ---
    st.write("### Histogrammes des âges des relais au fil des années (Simulation 1)")
    for annee_target in range(N_years):
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        tous_ages = ages_par_annee[0][annee_target]
        ax3.hist(tous_ages, bins=range(0, int(max(tous_ages)) + 2), edgecolor='black', alpha=0.7)
        ax3.set_title(f"Distribution des âges des relais en {2025 + annee_target}")
        ax3.set_xlabel("Âge des composants (années)")
        ax3.set_ylabel("Nombre de composants")
        ax3.grid(True)
        st.pyplot(fig3)




#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


st.header("Weibull Competing Risks + Monte Carlo")
def weibull_competing_risks():
    
    N_simulations = st.number_input("Nombre de simulations", min_value=10, max_value=500, value=100)
    N_years = st.number_input("Nombre d'années à simuler", min_value=1, max_value=50, value=25)

    st.info("Simulation en cours... (intégrer votre fonction ici)")

    # --- Simulation des données competing risks ---
    np.random.seed(42)
    n = 50000
    age_mecanique = np.random.weibull(1.5, size=n) * 30
    age_electrique = np.random.weibull(2.5, size=n) * 40

    true_event_time = np.minimum(age_mecanique, age_electrique)
    cause = np.where(age_mecanique < age_electrique, 'mecanique', 'electrique')

    censure = np.random.binomial(1, 0.2, size=n)
    observed = (censure == 0)
    censoring_offsets = np.random.uniform(0, 10, size=n)
    observed_time = np.where(observed, true_event_time, true_event_time - censoring_offsets)
    observed_time = np.clip(observed_time, 0.01, None)

    df_competing = pd.DataFrame({
        "time": observed_time,
        "event": observed,
        "cause": cause
    })

    # --- Fit Weibull par cause ---
    fitter_meca = WeibullFitter().fit(
        df_competing[df_competing["cause"] == "mecanique"]["time"],
        event_observed=df_competing[df_competing["cause"] == "mecanique"]["event"],
        label="Panne mécanique"
    )
    fitter_elec = WeibullFitter().fit(
        df_competing[df_competing["cause"] == "electrique"]["time"],
        event_observed=df_competing[df_competing["cause"] == "electrique"]["event"],
        label="Panne électrique"
    )

    # --- Courbes de survie ---
    st.write("### Fonctions de survie par cause")
    fig, ax = plt.subplots(figsize=(10, 6))
    fitter_meca.plot_survival_function(ax=ax)
    fitter_elec.plot_survival_function(ax=ax)
    ax.set_title("Fonctions de survie - Competing Risks (Weibull)")
    ax.set_xlabel("Temps")
    ax.set_ylabel("Probabilité de survie")
    ax.grid(True)
    st.pyplot(fig)

    # --- Fonction durée de vie restante selon competing risks ---
    def generate_lifetime_competing():
        t_m = np.random.weibull(fitter_meca.rho_) * fitter_meca.lambda_
        t_e = np.random.weibull(fitter_elec.rho_) * fitter_elec.lambda_
        return min(t_m, t_e)

    # --- Âges actuels (fictifs) --
    df_vivants = df[(df["censure;;"] == 0) & (~df["ACTIF"].isna())]
    ages_actuels = df_vivants["ACTIF"].values
    ages_actuels = np.array(df[df["censure;;"] == 0]["ACTIF"].values)
    parc_initial = list(ages_actuels) 
    st.write(f"Nombre initial de relais : {len(parc_initial)}")

    # --- Simulation Monte Carlo ---
    start = time.time()
    consommation_annuelle = []
    ages_par_annee = []

    for sim in range(N_simulations):
        parc_sim = list(parc_initial)
        conso_annuelle = []
        ages_sim = []

        for annee in range(N_years):
            parc_temp = []
            remplacement = 0

            for age in parc_sim:
                vie_restante = generate_lifetime_competing()
                if vie_restante < 0:
                    pass
                elif vie_restante <= 1:
                    remplacement += 1
                    parc_temp.append(0)
                else:
                    parc_temp.append(age + 1)

            parc_sim = parc_temp
            conso_annuelle.append(remplacement)
            ages_sim.append(parc_sim.copy())

        consommation_annuelle.append(conso_annuelle)
        ages_par_annee.append(ages_sim)

    st.write(f"Simulation terminée en {time.time() - start:.2f} secondes.")

    # --- Statistiques de consommation ---
    consommation_annuelle = np.array(consommation_annuelle)
    conso_moy = consommation_annuelle.mean(axis=0)
    conso_min = consommation_annuelle.min(axis=0)
    conso_max = consommation_annuelle.max(axis=0)

    # --- Graphique consommation annuelle ---
    years = range(2025, 2025 + N_years)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(years, conso_moy, label="Consommation moyenne", color="steelblue")
    ax2.fill_between(years, conso_min, conso_max, color="lightblue", alpha=0.5, label="Intervalle min-max")
    ax2.set_title("Projection Monte Carlo - Competing Risks avec relais neufs")
    ax2.set_xlabel("Année")
    ax2.set_ylabel("Nombre de remplacements")
    ax2.ticklabel_format(style='plain', axis='y')
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)

    # --- Violin plot ---
    df_violin = pd.DataFrame(consommation_annuelle, columns=years)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    sns.violinplot(data=df_violin, inner="quartile", palette="coolwarm", cut=0, ax=ax3)
    ax3.set_title("Distribution annuelle de la consommation des relais")
    ax3.set_xlabel("Année")
    ax3.set_ylabel("Relais remplacés")
    ax3.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    # --- Histogrammes des âges des relais (simulation 1) ---
    st.write("### Histogrammes des âges des relais (Simulation 1)")
    for annee_target in range(N_years):
        fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
        tous_ages = ages_par_annee[0][annee_target]
        ax_hist.hist(tous_ages, bins=range(0, int(max(tous_ages)) + 2), edgecolor='black', alpha=0.7)
        ax_hist.set_title(f"Distribution des âges des relais en {2025 + annee_target}")
        ax_hist.set_xlabel("Âge des composants (années)")
        ax_hist.set_ylabel("Nombre de composants")
        ax_hist.grid(True)
        plt.tight_layout()
        st.pyplot(fig_hist)




#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def random_survival_forest():
    st.header("Random Survival Forest (RSF)")
    st.write("Entraînement et évaluation du modèle RSF sur un sous-échantillon.")

    sample_size = st.slider("Taille de l'échantillon", min_value=1000, max_value=50000, value=20000, step=1000)
    
    df_rsf = df[
    df["DTETAT"].notna() &
    (df["DTETAT"].dt.year >= 1950) &
    (df["DTETAT"].dt.year <= 2050) &
    df["censure;;"].isin([0, 1]) ].copy()
    # Étape 2 : Encodage des variables catégorielles + suppression des NaN
    subset_df = df_rsf[["lib_constr", "lib_lettre", "AGE_ETAT", "ACTIF", "censure;;"]].dropna()
    subset_df = subset_df.sample(n=sample_size, random_state=42)
    # Prepare features and target
    X = pd.get_dummies(subset_df[["lib_constr", "lib_lettre", "AGE_ETAT"]], drop_first=True)
    y = np.array([(bool(e), t) for e, t in zip(subset_df["censure;;"], subset_df["ACTIF"])], dtype=[("event", bool), ("time", float)])
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

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def gradient_boosting_survival(df):
    st.header("Gradient Boosting Survival Analysis (GBSA)")
    df = df.dropna(subset=["ACTIF", "censure;;"])
    features = ["ACTIF"]  # add others if relevant
    X = df[features]
    y = Surv.from_dataframe("censure;;", "ACTIF", df)
    y = np.array([(bool(e), t) for e, t in zip(df["censure;;"], df["ACTIF"])], dtype=[("event", bool), ("time", float)])
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


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def cox_ph(df):
    st.header("Cox Proportional Hazards (CoxPH)")
    colonnes = ['ACTIF', 'censure;;', 'AGE_ETAT', 'lib_constr', 'lib_lettre']
    df_clean = df[colonnes].dropna()
    df_encoded = pd.get_dummies(df_clean, columns=['lib_constr', 'lib_lettre'], drop_first=True)

    X = df_encoded.drop(columns=['ACTIF', 'censure;;'])
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    X_filtered = X.drop(columns=to_drop)

    df_final = pd.concat([df_encoded[['ACTIF', 'censure;;']], X_filtered], axis=1)

    cph = CoxPHFitter(penalizer=0.1)
    with st.spinner("Entraînement du modèle CoxPH..."):
        cph.fit(df_final, duration_col='ACTIF', event_col='censure;;')
    st.write(cph.summary)

    n = 30
    sample = df_final.drop(columns=['ACTIF', 'censure;;']).iloc[:n]
    surv = cph.predict_survival_function(sample)

    fig, ax = plt.subplots(figsize=(14, 8))
    for i in range(n):
        ax.step(surv.index, surv.iloc[:, i], where="post", alpha=0.7)
    ax.set_title("Courbes de survie - Modèle de Cox")
    ax.set_xlabel("Temps (années)")
    ax.set_ylabel("Probabilité de survie")
    ax.grid(True)
    st.pyplot(fig)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def lognormal_monte_carlo(df):
    st.header("Log-Normal Monte Carlo Simulation")

    lognorm_df = df[["ACTIF", "censure;;"]].dropna()
    lognorm_fitter = LogNormalFitter()
    lognorm_fitter.fit(durations=lognorm_df["ACTIF"], event_observed=lognorm_df["censure;;"])

    st.write(f"Paramètres Log-Normal : mu = {lognorm_fitter.mu_:.2f}, sigma = {lognorm_fitter.sigma_:.2f}")

    def generate_lifetime_lognormal(current_age):
        mu = lognorm_fitter.mu_
        sigma = lognorm_fitter.sigma_
        t = np.random.lognormal(mean=mu, sigma=sigma)
        t_res = t - current_age
        return max(t_res, 0)

    N_simulations = st.number_input("Nombre de simulations", min_value=10, max_value=1000, value=300)
    N_years = st.number_input("Nombre d'années à simuler", min_value=1, max_value=50, value=25)

    ages_actuels = df[df['censure;;'] == 0]['ACTIF'].dropna().values
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

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



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

        
