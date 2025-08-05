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
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest as RandomSurvivalForestModel
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

# Import conditionnel pour GBSA
try:
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    GBSA_AVAILABLE = True
except ImportError:
    GBSA_AVAILABLE = False

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
st.title("Tableau de bord : Analyse prédictive de la survie des relais de signalisation")

@st.cache_data
def load_data(uploaded_file):
    """Charge les données depuis un fichier uploadé"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8', on_bad_lines='skip')
            return df
        except Exception as e:
            st.error(f"Erreur lors du chargement : {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def detect_column_names(df):
    """Détecte automatiquement les noms des colonnes importantes"""
    detected = {}
    
    # Colonnes possibles pour la durée
    duration_candidates = ['ACTIF', 'duree', 'temps', 'time', 'duration']
    for col in df.columns:
        if any(candidate.lower() in col.lower() for candidate in duration_candidates):
            detected['duration'] = col
            break
    
    # Colonnes possibles pour la censure
    censure_candidates = ['censure', 'event', 'evenement', 'status']
    for col in df.columns:
        if any(candidate.lower() in col.lower() for candidate in censure_candidates):
            detected['censure'] = col
            break
    
    # Colonnes possibles pour la date
    date_candidates = ['DTETAT', 'date', 'mise_en_service']
    for col in df.columns:
        if any(candidate.lower() in col.lower() for candidate in date_candidates):
            detected['date'] = col
            break
            
    return detected

def validate_dataframe(df):
    """Valide que le DataFrame contient les colonnes nécessaires"""
    if df.empty:
        return False, "DataFrame vide", {}
    
    # Détection automatique des colonnes
    detected_cols = detect_column_names(df)
    
    missing = []
    if 'duration' not in detected_cols:
        missing.append("colonne de durée (ex: ACTIF, duree, temps)")
    if 'censure' not in detected_cols:
        missing.append("colonne de censure (ex: censure, event, status)")
    
    if missing:
        available_cols = list(df.columns)
        return False, f"Colonnes manquantes : {missing}. Colonnes disponibles : {available_cols}", detected_cols
    
    return True, "OK", detected_cols

def preprocess_data(df, detected_cols):
    """Préprocessing des données avec colonnes détectées"""
    df_processed = df.copy()
    
    # Standardisation des noms de colonnes
    if 'duration' in detected_cols:
        df_processed['ACTIF'] = df_processed[detected_cols['duration']]
    if 'censure' in detected_cols:
        df_processed['censure'] = pd.to_numeric(df_processed[detected_cols['censure']], errors='coerce').fillna(0).astype(int)
    
    # Traitement de la colonne date si elle existe
    if 'date' in detected_cols:
        try:
            df_processed["DTETAT"] = pd.to_datetime(df_processed[detected_cols['date']], errors="coerce")
            now = pd.Timestamp.today()
            df_processed["AGE_ETAT"] = (now - df_processed["DTETAT"]).dt.days / 365.25
        except Exception as e:
            st.warning(f"Erreur traitement date : {e}")
    
    # Nettoyage des données numériques
    if 'ACTIF' in df_processed.columns:
        df_processed['ACTIF'] = pd.to_numeric(df_processed['ACTIF'], errors='coerce')
        # Suppression des valeurs aberrantes
        df_processed = df_processed[df_processed['ACTIF'] > 0]
        df_processed = df_processed[df_processed['ACTIF'] < 1000]  # Limite raisonnable
    
    return df_processed

# Interface utilisateur
uploaded_file = st.file_uploader("Uploader votre fichier CSV", type=["csv"])

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
    """Analyse Weibull Double avec Monte Carlo"""
    st.header("Weibull Double Fitting + Monte Carlo")
    
    N_simulations = st.number_input("Nombre de simulations", min_value=10, max_value=1000, value=100)
    N_years = st.number_input("Nombre d'années à simuler", min_value=1, max_value=50, value=25)
    
    try:
        # Vérification des données
        if 'ACTIF' not in df.columns or 'censure' not in df.columns:
            st.error("Colonnes ACTIF et censure requises après préprocessing")
            return
            
        # Données pour l'ajustement Weibull
        df_clean = df[['ACTIF', 'censure']].dropna()
        failures = df_clean[df_clean['censure'] == 1]['ACTIF'].values
        censored = df_clean[df_clean['censure'] == 0]['ACTIF'].values
        
        if len(failures) == 0:
            st.warning("Aucune défaillance observée, utilisation de données simulées")
            np.random.seed(42)
            failures = np.random.weibull(a=1.5, size=100) * 50
            censored = np.random.weibull(a=1.5, size=20) * 50
        
        st.write(f"Nombre de défaillances : {len(failures)}")
        st.write(f"Nombre d'observations censurées : {len(censored)}")

        # Fit Weibull 2P
        wb = Fit_Weibull_2P(failures=failures, right_censored=censored, show_probability_plot=False)
        fitted_weibull = Weibull_Distribution(alpha=wb.alpha, beta=wb.beta)

        # Courbe de survie
        x = np.linspace(0, max(max(failures), max(censored) if len(censored) > 0 else max(failures)) * 1.2, 500)
        sf = fitted_weibull.SF(x)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, sf, '--', label="Fitted Weibull 2P", color='blue', linewidth=2)
        ax.set_xlabel("Temps")
        ax.set_ylabel("Fonction de survie")
        ax.set_title("Fonction de survie Weibull")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.info("Simulation Monte Carlo en cours...")

        # Âges initiaux (relais encore en service)
        ages_vivants = df[df['censure'] == 0]['ACTIF'].dropna()
        if len(ages_vivants) > 0:
            ages_actuels = ages_vivants.values
        else:
            st.warning("Aucun relais en service trouvé, génération d'âges aléatoires")
            ages_actuels = np.random.uniform(0, 20, size=1000)

        def generate_remaining_lifetime(current_age):
            """Génère la durée de vie restante"""
            u = random.random()
            inside_log = u * math.exp(- (current_age / wb.alpha) ** wb.beta)
            if inside_log <= 0:
                return wb.alpha
            durée_totale = wb.alpha * (-math.log(inside_log)) ** (1 / wb.beta)
            return max(0, durée_totale - current_age)

        parc_initial = list(ages_actuels)
        st.write(f"Nombre initial de relais : {len(parc_initial)}")
        st.write(f"Paramètres Weibull estimés : alpha = {wb.alpha:.2f}, beta = {wb.beta:.2f}")

        # Simulation Monte Carlo
        start_time = time.time()
        consommation_annuelle = []

        progress_bar = st.progress(0)
        for sim in range(N_simulations):
            parc = list(parc_initial)
            consommation = []

            for annee in range(N_years):
                nb_remplacements = 0
                nouveau_parc = []

                for age in parc:
                    duree_restante = generate_remaining_lifetime(age)
                    if 0 < duree_restante <= 1:
                        nb_remplacements += 1
                        nouveau_parc.append(0)  # Nouveau relais
                    elif duree_restante <= 0:
                        pass  # Composant hors service
                    else:
                        nouveau_parc.append(age + 1)

                parc = nouveau_parc
                consommation.append(nb_remplacements)

            consommation_annuelle.append(consommation)
            progress_bar.progress((sim + 1) / N_simulations)

        st.success(f"Simulation terminée en {time.time() - start_time:.2f} secondes.")

        # Résultats statistiques
        conso_array = np.array(consommation_annuelle)
        moyenne_annuelle = conso_array.mean(axis=0)
        std_annuelle = conso_array.std(axis=0)

        # Graphique consommation moyenne
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        years = np.arange(2025, 2025 + N_years)
        ax1.plot(years, moyenne_annuelle, label="Consommation moyenne", color="navy", linewidth=2)
        ax1.fill_between(years, moyenne_annuelle - std_annuelle, moyenne_annuelle + std_annuelle,
                         alpha=0.3, color="red", label="± 1 écart-type")
        ax1.set_xlabel("Année")
        ax1.set_ylabel("Nombre de relais remplacés")
        ax1.set_title("Prévision annuelle de consommation moyenne des relais (Monte Carlo)")
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)

        # Tableau de résultats
        results_df = pd.DataFrame({
            'Année': years,
            'Moyenne': np.round(moyenne_annuelle, 1),
            'Écart-type': np.round(std_annuelle, 1),
            'Min': conso_array.min(axis=0),
            'Max': conso_array.max(axis=0)
        })
        st.write("### Résultats détaillés")
        st.dataframe(results_df)

    except Exception as e:
        st.error(f"Erreur dans l'analyse Weibull : {e}")
        st.exception(e)

def weibull_competing_risks(df):
    """Analyse Weibull Competing Risks avec Monte Carlo"""
    st.header("Weibull Competing Risks + Monte Carlo")
    
    N_simulations = st.number_input("Nombre de simulations", min_value=10, max_value=500, value=100, key="cr_sim")
    N_years = st.number_input("Nombre d'années à simuler", min_value=1, max_value=50, value=25, key="cr_years")

    try:
        st.info("Simulation Competing Risks en cours...")

        # Simulation des données competing risks (car difficile d'identifier les causes dans les données réelles)
        np.random.seed(42)
        n = 5000
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

        # Fit Weibull par cause
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

        # Courbes de survie
        st.write("### Fonctions de survie par cause")
        fig, ax = plt.subplots(figsize=(10, 6))
        fitter_meca.plot_survival_function(ax=ax)
        fitter_elec.plot_survival_function(ax=ax)
        ax.set_title("Fonctions de survie - Competing Risks (Weibull)")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Probabilité de survie")
        ax.grid(True)
        st.pyplot(fig)

        def generate_lifetime_competing():
            """Génère durée de vie selon competing risks"""
            t_m = np.random.weibull(fitter_meca.rho_) * fitter_meca.lambda_
            t_e = np.random.weibull(fitter_elec.rho_) * fitter_elec.lambda_
            return min(t_m, t_e)

        # Âges actuels
        if 'ACTIF' in df.columns and 'censure' in df.columns:
            ages_actuels = df[df["censure"] == 0]["ACTIF"].dropna().values
            if len(ages_actuels) == 0:
                ages_actuels = np.random.uniform(0, 20, size=1000)
        else:
            ages_actuels = np.random.uniform(0, 20, size=1000)

        parc_initial = list(ages_actuels)
        st.write(f"Nombre initial de relais : {len(parc_initial)}")

        # Simulation Monte Carlo
        start_time = time.time()
        consommation_annuelle = []

        progress_bar = st.progress(0)
        for sim in range(N_simulations):
            parc_sim = list(parc_initial)
            conso_annuelle = []

            for annee in range(N_years):
                parc_temp = []
                remplacement = 0

                for age in parc_sim:
                    vie_restante = generate_lifetime_competing()
                    if vie_restante <= 0:
                        pass
                    elif vie_restante <= 1:
                        remplacement += 1
                        parc_temp.append(0)
                    else:
                        parc_temp.append(age + 1)

                parc_sim = parc_temp
                conso_annuelle.append(remplacement)

            consommation_annuelle.append(conso_annuelle)
            progress_bar.progress((sim + 1) / N_simulations)

        st.success(f"Simulation terminée en {time.time() - start_time:.2f} secondes.")

        # Statistiques et graphiques
        consommation_annuelle = np.array(consommation_annuelle)
        conso_moy = consommation_annuelle.mean(axis=0)
        conso_std = consommation_annuelle.std(axis=0)

        years = range(2025, 2025 + N_years)
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(years, conso_moy, label="Consommation moyenne", color="steelblue", linewidth=2)
        ax2.fill_between(years, conso_moy - conso_std, conso_moy + conso_std, 
                         color="lightblue", alpha=0.5, label="± 1 écart-type")
        ax2.set_title("Projection Monte Carlo - Competing Risks")
        ax2.set_xlabel("Année")
        ax2.set_ylabel("Nombre de remplacements")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Erreur dans l'analyse Competing Risks : {e}")

def random_survival_forest(df):
    """Random Survival Forest Analysis"""
    st.header("Random Survival Forest (RSF)")
    
    try:
        # Vérification des colonnes requises
        required_cols = ['ACTIF', 'censure']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Colonnes requises manquantes : {required_cols}")
            return

        sample_size = st.slider("Taille de l'échantillon", min_value=1000, max_value=min(50000, len(df)), 
                               value=min(20000, len(df)), step=1000)

        # Préparation des données
        df_rsf = df.dropna(subset=['ACTIF', 'censure']).copy()
        
        if sample_size > len(df_rsf):
            sample_size = len(df_rsf)
            st.warning(f"Échantillon réduit à {sample_size}")

        df_sample = df_rsf.sample(n=sample_size, random_state=42)

        # Features
        feature_cols = ['ACTIF']
        if 'AGE_ETAT' in df_sample.columns:
            feature_cols.append('AGE_ETAT')
        if 'lib_constr' in df_sample.columns:
            feature_cols.append('lib_constr')
        if 'lib_lettre' in df_sample.columns:
            feature_cols.append('lib_lettre')

        # Encodage des variables catégorielles
        X = pd.get_dummies(df_sample[feature_cols], drop_first=True)
        
        # Target
        y = np.array([(bool(e), t) for e, t in zip(df_sample["censure"], df_sample["ACTIF"])],
                     dtype=[("event", bool), ("time", float)])

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.write("## Entraînement du modèle RSF")
        st.write(f"Taille d'entraînement : {len(X_train)}")
        st.write(f"Nombre de features : {X.shape[1]}")

        # Modèle RSF
        rsf = RandomSurvivalForestModel(
            n_estimators=50,
            min_samples_split=10,
            min_samples_leaf=15,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        )

        with st.spinner("Entraînement en cours..."):
            rsf.fit(X_train, y_train)
        st.success("Modèle entraîné!")

        # Évaluation
        predictions = rsf.predict(X_test)
        c_index = concordance_index_censored(y_test["event"], y_test["time"], predictions)[0]
        st.write(f"**C-index (qualité du modèle) :** {c_index:.3f}")

        # Importance des variables
        if len(X.columns) > 1:
            def score_fn(model, X, y):
                preds = model.predict(X)
                return concordance_index_censored(y["event"], y["time"], preds)[0]

            with st.spinner("Calcul des importances..."):
                result = permutation_importance(rsf, X_test, y_test, n_repeats=3, 
                                              random_state=42, scoring=score_fn)

            importances = pd.Series(result.importances_mean, index=X.columns).sort_values()
            
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            importances.plot(kind="barh", ax=ax1)
            ax1.set_xlabel("Impact moyen sur la performance")
            ax1.set_title("Importance des variables")
            ax1.grid(True)
            st.pyplot(fig1)

        # Courbes de survie
        st.write("### Courbes de survie prédictives")
        nb_to_plot = st.slider("Nombre de courbes à afficher", min_value=10, max_value=100, value=50)
        
        subset_indices = np.random.choice(len(X_test), size=min(nb_to_plot, len(X_test)), replace=False)
        surv_fns = rsf.predict_survival_function(X_test.iloc[subset_indices], return_array=False)

        fig2, ax2 = plt.subplots(figsize=(12, 7))
        for i, fn in enumerate(surv_fns[:min(20, len(surv_fns))]):  # Limite pour la lisibilité
            ax2.step(fn.x, fn.y, where="post", alpha=0.5, color="gray")

        # Courbe moyenne
        if len(surv_fns) > 0:
            common_times = np.linspace(0, 60, 500)
            all_surv_probs = []
            for fn in surv_fns:
                surv_interp = np.interp(common_times, fn.x, fn.y, left=1.0, right=0.0)
                all_surv_probs.append(surv_interp)
            
            if all_surv_probs:
                mean_surv = np.mean(all_surv_probs, axis=0)
                ax2.plot(common_times, mean_surv, label="Moyenne", color="red", linewidth=2.5)

        ax2.set_title(f"Courbes de survie (RSF) - {len(surv_fns)} relais")
        ax2.set_xlabel("Temps (années)")
        ax2.set_ylabel("Probabilité de survie")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Erreur dans l'analyse RSF : {e}")

def gradient_boosting_survival(df):
    """Gradient Boosting Survival Analysis"""
    st.header("Gradient Boosting Survival Analysis (GBSA)")
    
    if not GBSA_AVAILABLE:
        st.error("GBSA n'est pas disponible. Veuillez installer la version complète de scikit-survival.")
        return
    
    try:
        # Vérification des colonnes
        if not all(col in df.columns for col in ['ACTIF', 'censure']):
            st.error("Colonnes ACTIF et censure requises")
            return

        df_clean = df.dropna(subset=["ACTIF", "censure"])
        
        # Features simples
        features = ["ACTIF"]
        if 'AGE_ETAT' in df_clean.columns:
            features.append('AGE_ETAT')
            
        X = df_clean[features]
        y = np.array([(bool(e), t) for e, t in zip(df_clean["censure"], df_clean["ACTIF"])], 
                     dtype=[("event", bool), ("time", float)])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = GradientBoostingSurvivalAnalysis(n_estimators=80, learning_rate=0.2, 
                                               max_depth=3, random_state=42)
        
        with st.spinner("Entraînement GBSA..."):
            model.fit(X_train, y_train)

        predicted_risks = model.predict(X_test)
        cindex = concordance_index_censored(y_test["event"], y_test["time"], predicted_risks)[0]
        st.success(f"C-index GBSA : {cindex:.4f}")

        # Courbes de survie
        n_plot = min(50, len(X_test))
        surv_functions = model.predict_survival_function(X_test.iloc[:n_plot])

        fig, ax = plt.subplots(figsize=(12, 8))
        for fn in surv_functions:
            ax.step(fn.x, fn.y, where="post", alpha=0.3)
        ax.set_title("Courbes de survie (Gradient Boosting Survival)")
        ax.set_xlabel("Temps (durée en service)")
        ax.set_ylabel("Probabilité de survie")
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur dans l'analyse GBSA : {e}")

def cox_ph(df):
    """Cox Proportional Hazards Analysis"""
    st.header("Cox Proportional Hazards (CoxPH)")
    
    try:
        # Colonnes nécessaires
        base_cols = ['ACTIF', 'censure']
        optional_cols = ['AGE_ETAT', 'lib_constr', 'lib_lettre']
        
        available_cols = [col for col in base_cols + optional_cols if col in df.columns]
        df_clean = df[available_cols].dropna()
        
        if len(df_clean) < 100:
            st.warning("Pas assez de données pour l'analyse Cox")
            return

        # Encodage des variables catégorielles
        categorical_cols = [col for col in ['lib_constr', 'lib_lettre'] if col in df_clean.columns]
        if categorical_cols:
            df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
        else:
            df_encoded = df_clean.copy()

        # Suppression des colonnes hautement corrélées
        feature_cols = [col for col in df_encoded.columns if col not in ['ACTIF', 'censure']]
        if len(feature_cols) > 1:
            X = df_encoded[feature_cols]
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
            feature_cols = [col for col in feature_cols if col not in to_drop]

        df_final = df_encoded[['ACTIF', 'censure'] + feature_cols]

        # Modèle Cox
        cph = CoxPHFitter(penalizer=0.1)
        
        with st.spinner("Entraînement du modèle CoxPH..."):
            cph.fit(df_final, duration_col='ACTIF', event_col='censure')
        
        st.write("### Résumé du modèle")
        st.dataframe(cph.summary)

        # Courbes de survie pour un échantillon
        n_sample = min(30, len(df_final))
        sample = df_final.drop(columns=['ACTIF', 'censure']).iloc[:n_sample]
        
        if len(sample) > 0:
            surv = cph.predict_survival_function(sample)

            fig, ax = plt.subplots(figsize=(12, 8))
            for i in range(min(20, len(surv.columns))):
                ax.step(surv.index, surv.iloc[:, i], where="post", alpha=0.7)
            ax.set_title("Courbes de survie - Modèle de Cox")
            ax.set_xlabel("Temps (années)")
            ax.set_ylabel("Probabilité de survie")
            ax.grid(True)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur dans l'analyse Cox : {e}")

def lognormal_monte_carlo(df):
    """Log-Normal Monte Carlo Simulation"""
    st.header("Log-Normal Monte Carlo Simulation")
    
    try:
        if not all(col in df.columns for col in ['ACTIF', 'censure']):
            st.error("Colonnes ACTIF et censure requises")
            return

        lognorm_df = df[["ACTIF", "censure"]].dropna()
        
        if len(lognorm_df) < 50:
            st.warning("Pas assez de données pour l'analyse Log-Normal")
            return

        # Fit Log-Normal
        lognorm_fitter = LogNormalFitter()
        with st.spinner("Ajustement du modèle Log-Normal..."):
            lognorm_fitter.fit(durations=lognorm_df["ACTIF"], event_observed=lognorm_df["censure"])

        st.write(f"**Paramètres Log-Normal :** μ = {lognorm_fitter.mu_:.2f}, σ = {lognorm_fitter.sigma_:.2f}")

        # Courbe de survie
        fig_fit, ax_fit = plt.subplots(figsize=(10, 6))
        lognorm_fitter.plot_survival_function(ax=ax_fit)
        ax_fit.set_title("Fonction de survie Log-Normal ajustée")
        ax_fit.grid(True)
        st.pyplot(fig_fit)

        def generate_lifetime_lognormal(current_age):
            """Génère durée de vie Log-Normal"""
            mu = lognorm_fitter.mu_
            sigma = lognorm_fitter.sigma_
            t = np.random.lognormal(mean=mu, sigma=sigma)
            return max(0, t - current_age)

        # Paramètres simulation
        N_simulations = st.number_input("Nombre de simulations", min_value=10, max_value=1000, 
                                       value=300, key="ln_sim")
        N_years = st.number_input("Nombre d'années à simuler", min_value=1, max_value=50, 
                                 value=25, key="ln_years")

        # Ages actuels
        ages_actuels = df[df['censure'] == 0]['ACTIF'].dropna().values
        if len(ages_actuels) == 0:
            ages_actuels = np.random.uniform(0, 20, size=1000)

        parc_initial = list(ages_actuels)
        st.write(f"Nombre initial de relais : {len(parc_initial)}")

        # Simulation Monte Carlo
        consommation_annuelle = []

        with st.spinner("Simulation Monte Carlo Log-Normal en cours..."):
            progress_bar = st.progress(0)
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
                progress_bar.progress((sim + 1) / N_simulations)

        # Résultats
        conso_array = np.array(consommation_annuelle)
        moyenne_annuelle = conso_array.mean(axis=0)
        std_annuelle = conso_array.std(axis=0)
        years = np.arange(2025, 2025 + N_years)

        # Graphique principal
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(years, moyenne_annuelle, label="Consommation moyenne", color="navy", linewidth=2)
        ax.fill_between(years, moyenne_annuelle - std_annuelle, moyenne_annuelle + std_annuelle,
                        alpha=0.4, color="red", label="± 1 écart-type")
        ax.set_xlabel("Année")
        ax.set_ylabel("Nombre de relais remplacés")
        ax.set_title("Prévision annuelle de consommation moyenne (Log-Normal Monte Carlo)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Violin plot
        df_violin = pd.DataFrame(conso_array, columns=years)
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.violinplot(data=df_violin, inner="quartile", cut=0, ax=ax2)
        ax2.set_title("Distribution annuelle de la consommation des relais")
        ax2.set_xlabel("Année")
        ax2.set_ylabel("Relais remplacés")
        ax2.grid(True)
        plt.setp(ax2.get_xticklabels(), rotation=45)
        st.pyplot(fig2)

        # Statistiques récapitulatives
        st.write("### Statistiques récapitulatives")
        stats_df = pd.DataFrame({
            'Année': years,
            'Moyenne': moyenne_annuelle,
            'Écart-type': std_annuelle,
            'Min': conso_array.min(axis=0),
            'Max': conso_array.max(axis=0)
        })
        st.dataframe(stats_df)

    except Exception as e:
        st.error(f"Erreur dans l'analyse Log-Normal : {e}")

# LOGIQUE PRINCIPALE
def main():
    """Fonction principale de l'application"""
    
    if uploaded_file is not None:
        # Chargement des données
        df = load_data(uploaded_file)
        
        if df.empty:
            st.error("Impossible de charger les données du fichier.")
            return
            
        # Validation des données avec détection automatique
        is_valid, message, detected_cols = validate_dataframe(df)
        if not is_valid:
            st.error(f"Données invalides : {message}")
            
            # Interface de sélection manuelle des colonnes
            st.write("### Sélection manuelle des colonnes")
            st.write("Veuillez sélectionner les colonnes correspondantes dans votre fichier :")
            
            col1, col2 = st.columns(2)
            
            with col1:
                duration_col = st.selectbox(
                    "Colonne de durée/temps :",
                    options=[""] + list(df.columns),
                    help="Colonne contenant la durée de vie ou le temps d'observation"
                )
            
            with col2:
                event_col = st.selectbox(
                    "Colonne d'événement/censure :",
                    options=[""] + list(df.columns),
                    help="Colonne indiquant si l'événement s'est produit (1) ou si l'observation est censurée (0)"
                )
            
            if duration_col and event_col:
                # Mise à jour des colonnes détectées
                detected_cols = {
                    'duration': duration_col,
                    'censure': event_col
                }
                
                # Vérification des données dans les colonnes sélectionnées
                try:
                    duration_data = pd.to_numeric(df[duration_col], errors='coerce')
                    event_data = pd.to_numeric(df[event_col], errors='coerce')
                    
                    if duration_data.isna().all():
                        st.error(f"La colonne '{duration_col}' ne contient pas de données numériques valides.")
                        return
                    
                    if event_data.isna().all():
                        st.error(f"La colonne '{event_col}' ne contient pas de données numériques valides.")
                        return
                    
                    # Affichage des statistiques sur les colonnes sélectionnées
                    st.write("#### Aperçu des colonnes sélectionnées :")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**{duration_col} (durée) :**")
                        st.write(f"- Valeurs valides : {duration_data.count()}")
                        st.write(f"- Moyenne : {duration_data.mean():.2f}")
                        st.write(f"- Min/Max : {duration_data.min():.2f} / {duration_data.max():.2f}")
                    
                    with col2:
                        st.write(f"**{event_col} (événement) :**")
                        st.write(f"- Valeurs valides : {event_data.count()}")
                        event_counts = event_data.value_counts().sort_index()
                        st.write(f"- Répartition : {dict(event_counts)}")
                    
                    if st.button("Valider la sélection et continuer"):
                        is_valid = True
                        
                except Exception as e:
                    st.error(f"Erreur lors de la validation des colonnes : {e}")
                    return
            
            if not is_valid:
                return
            
        # Préprocessing avec les colonnes détectées
        df = preprocess_data(df, detected_cols)
        
        # Vérification finale après préprocessing
        if 'ACTIF' not in df.columns or 'censure' not in df.columns:
            st.error("Erreur lors du préprocessing des données.")
            return
        
        # Nettoyage final des données
        df_clean = df[['ACTIF', 'censure']].dropna()
        df_clean = df_clean[df_clean['ACTIF'] > 0]  # Suppression des durées négatives ou nulles
        
        if len(df_clean) == 0:
            st.error("Aucune donnée valide après nettoyage.")
            return
        
        # Affichage des informations sur les données
        st.sidebar.write("### Informations sur les données")
        st.sidebar.write(f"**Nombre de lignes :** {len(df)}")
        st.sidebar.write(f"**Données valides :** {len(df_clean)}")
        st.sidebar.write(f"**Colonnes disponibles :** {list(df.columns)}")
        
        # Statistiques sur les événements
        if 'censure' in df.columns:
            event_counts = df['censure'].value_counts()
            st.sidebar.write(f"**Événements observés :** {event_counts.get(1, 0)}")
            st.sidebar.write(f"**Observations censurées :** {event_counts.get(0, 0)}")
        
        # Aperçu des données
        if st.sidebar.checkbox("Afficher aperçu des données"):
            st.write("### Aperçu des données")
            st.dataframe(df.head(10))
            
            # Statistiques descriptives
            st.write("### Statistiques descriptives")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe())
            
            # Histogramme de la durée
            if 'ACTIF' in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                df['ACTIF'].hist(bins=50, ax=ax, alpha=0.7, edgecolor='black')
                ax.set_xlabel("Durée (ACTIF)")
                ax.set_ylabel("Fréquence")
                ax.set_title("Distribution des durées de service")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        # Routage vers les différents modèles
        try:
            if model_choice == "Weibull Double + Monte Carlo":
                weibull_double_monte_carlo(df_clean)
            elif model_choice == "Weibull Competing Risks + Monte Carlo":
                weibull_competing_risks(df_clean)
            elif model_choice == "Random Survival Forest (RSF)":
                random_survival_forest(df_clean)
            elif model_choice == "Gradient Boosting Survival Analysis (GBSA)":
                gradient_boosting_survival(df_clean)
            elif model_choice == "Cox Proportional Hazards (CoxPH)":
                cox_ph(df_clean)
            elif model_choice == "Log-Normal Monte Carlo Simulation":
                lognormal_monte_carlo(df_clean)
            else:
                st.warning("Veuillez sélectionner un modèle dans le menu latéral.")
                
        except Exception as e:
            st.error(f"Erreur lors de l'exécution du modèle {model_choice} : {e}")
            st.exception(e)  # Pour le debugging
            
    else:
        # Page d'accueil sans données
        st.info("### 👋 Bienvenue dans l'outil d'analyse de survie des relais ferroviaires")
        st.write("""
        Cet outil vous permet d'effectuer différents types d'analyses de survie :
        
        **📊 Modèles disponibles :**
        - **Weibull Double + Monte Carlo** : Analyse paramétrique avec simulation
        - **Weibull Competing Risks** : Modélisation des risques concurrents
        - **Random Survival Forest** : Méthode d'ensemble non-paramétrique
        - **Gradient Boosting Survival** : Algorithme de boosting pour la survie
        - **Cox Proportional Hazards** : Modèle de régression de Cox
        - **Log-Normal Monte Carlo** : Simulation avec distribution log-normale
        
        **📁 Format des données requis :**
        - Fichier CSV avec une colonne de durée et une colonne d'événement/censure
        - L'outil détecte automatiquement les colonnes ou permet une sélection manuelle
        - Colonnes optionnelles : dates, constructeur, type, etc.
        
        **🚀 Pour commencer :**
        1. Uploadez votre fichier CSV ci-dessus
        2. Sélectionnez les colonnes si nécessaire
        3. Choisissez un modèle dans le menu latéral
        4. Configurez les paramètres et lancez l'analyse
        """)
        
        # Exemple de structures de données acceptées
        st.write("### 📋 Exemples de structures de données acceptées")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Format standard :**")
            example_data1 = pd.DataFrame({
                'ACTIF': [5.2, 8.1, 12.3, 3.7, 15.8],
                'censure': [1, 0, 1, 1, 0],
                'DTETAT': ['2020-01-15', '2018-03-22', '2015-07-08', '2021-11-03', '2012-05-17']
            })
            st.dataframe(example_data1)
        
        with col2:
            st.write("**Format alternatif :**")
            example_data2 = pd.DataFrame({
                'duree_service': [5.2, 8.1, 12.3, 3.7, 15.8],
                'evenement': [1, 0, 1, 1, 0],
                'constructeur': ['ALSTOM', 'SIEMENS', 'ALSTOM', 'THALES', 'SIEMENS']
            })
            st.dataframe(example_data2)
        
        st.write("""
        **📝 Description des colonnes :**
        - **Colonne de durée** : Temps de service, âge, durée d'observation (valeurs numériques positives)
        - **Colonne d'événement** : Indicateur de défaillance (1 = panne observée, 0 = censuré/toujours en service)
        - **Colonnes optionnelles** : Date de mise en service, constructeur, type, localisation, etc.
        
        **⚠️ Notes importantes :**
        - Les valeurs manquantes seront automatiquement supprimées
        - Les durées doivent être numériques et positives
        - L'événement doit être binaire (0 ou 1)
        - L'outil s'adapte automatiquement aux noms de colonnes courants
        """)

# Exécution de l'application
if __name__ == "__main__":
    main()
