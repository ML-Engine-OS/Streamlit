# ====================================================================
# config/database_config.py - Configuration sécurisée de la base
# ====================================================================

import os
import streamlit as st
from sqlalchemy import create_engine, text
import pandas as pd
from typing import List, Optional

class DatabaseConfig:
    """Configuration sécurisée pour la base PostgreSQL"""
    
    def __init__(self):
        self.db_host = os.getenv('DB_HOST', 'posqresql')
        self.db_port = os.getenv('DB_PORT', '5432')
        self.db_name = os.getenv('DB_NAME', 'dtiesss')
        self.db_user = os.getenv('DB_USER', 'Integ')
        self.db_password = os.getenv('DB_PASSWORD')
        
        if not self.db_password:
            # Fallback pour Streamlit secrets
            try:
                self.db_password = st.secrets["database"]["password"]
                self.db_host = st.secrets["database"]["host"]
                self.db_port = st.secrets["database"]["port"]
                self.db_name = st.secrets["database"]["name"]
                self.db_user = st.secrets["database"]["user"]
            except:
                raise ValueError("Credentials de base de données manquants")
    
    def get_connection_string(self):
        """Retourne la chaîne de connexion PostgreSQL"""
        return f"postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    @st.cache_resource
    def get_engine(_self):
        """Retourne l'engine SQLAlchemy avec mise en cache"""
        try:
            engine = create_engine(
                _self.get_connection_string(),
                pool_pre_ping=True,
                pool_recycle=300,
                echo=False  # Pas de logs SQL en production
            )
            # Test de connexion
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine
        except Exception as e:
            st.error(f"Erreur de connexion à la base : {str(e)}")
            return None

# ====================================================================
# utils/relais_data_loader.py - Fonctions spécialisées pour vos données
# ====================================================================

class RelaisDataLoader:
    """Chargeur de données spécialisé pour les relais"""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
        self.engine = db_config.get_engine()
    
    def load_relais_data(self, 
                        symb: str = "79540230", 
                        constructeurs: List[str] = ["ANSA", "CSEE", "HITA"],
                        limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Charge les données de relais avec paramètres sécurisés
        
        Args:
            symb: Code symbole (défaut: "79540230")
            constructeurs: Liste des constructeurs (défaut: ["ANSA", "CSEE", "HITA"])
            limit: Limite du nombre de lignes (optionnel)
        
        Returns:
            DataFrame avec les données ou None en cas d'erreur
        """
        if not self.engine:
            st.error("Connexion à la base non établie")
            return None
        
        try:
            # Validation des paramètres d'entrée
            symb_clean = self._validate_symb(symb)
            constructeurs_clean = self._validate_constructeurs(constructeurs)
            
            # Construction de la requête sécurisée avec paramètres
            query = self._build_relais_query(limit)
            
            # Exécution avec paramètres sécurisés
            with self.engine.connect() as conn:
                df = pd.read_sql(
                    query, 
                    conn, 
                    params={
                        'symb_param': symb_clean,
                        'constructeurs_tuple': tuple(constructeurs_clean)
                    }
                )
            
            if df.empty:
                st.warning(f"Aucune donnée trouvée pour SYMB={symb_clean} et constructeurs={constructeurs_clean}")
                return None
            
            # Post-traitement pour votre analyse de fiabilité
            df_processed = self._prepare_for_reliability_analysis(df)
            
            st.success(f"Données chargées : {len(df_processed)} relais trouvés")
            return df_processed
            
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {str(e)}")
            return None
    
    def _validate_symb(self, symb: str) -> str:
        """Valide et nettoie le code SYMB"""
        # Suppression des caractères dangereux
        symb_clean = str(symb).replace("'", "").replace('"', "").replace(";", "")
        if not symb_clean.isalnum():
            raise ValueError(f"Code SYMB invalide : {symb}")
        return symb_clean
    
    def _validate_constructeurs(self, constructeurs: List[str]) -> List[str]:
        """Valide et nettoie la liste des constructeurs"""
        if not isinstance(constructeurs, list):
            raise ValueError("Les constructeurs doivent être une liste")
        
        constructeurs_clean = []
        allowed_constructeurs = ["ANSA", "CSEE", "HITA", "ALSTOM", "SIEMENS", "ABB"]  # Liste autorisée
        
        for constr in constructeurs:
            constr_clean = str(constr).upper().replace("'", "").replace('"', "")
            if constr_clean in allowed_constructeurs:
                constructeurs_clean.append(constr_clean)
            else:
                st.warning(f"Constructeur non autorisé ignoré : {constr}")
        
        if not constructeurs_clean:
            raise ValueError("Aucun constructeur valide fourni")
        
        return constructeurs_clean
    
    def _build_relais_query(self, limit: Optional[int] = None) -> str:
        """Construit la requête SQL sécurisée avec paramètres"""
        base_query = """
        SELECT 
            a.*,
            b.*
        FROM "AGRSIG_PBI"."ASTOTS" a 
        LEFT JOIN "AGRSIG_PBI"."FAS_DUREE_VIE_4" b 
            ON a."CLE" = b."CLE_TB_AS" 
        WHERE a.symb = :symb_param 
            AND a.lib_constr = ANY(:constructeurs_tuple)
        """
        
        if limit and isinstance(limit, int) and limit > 0:
            base_query += f" LIMIT {min(limit, 50000)}"  # Max 50k pour sécurité
        
        return base_query
    
    def _prepare_for_reliability_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare les données pour l'analyse de fiabilité
        Adapte vos colonnes vers le format attendu (ACTIF, censure)
        """
        df_processed = df.copy()
        
        # Mapping des colonnes selon votre schéma de base
        # ADAPTEZ CES MAPPINGS SELON VOS VRAIES COLONNES
        column_mappings = {
            'duree_vie': 'ACTIF',           # Remplacez par votre vraie colonne d'âge
            'statut_defaillance': 'censure', # Remplacez par votre vraie colonne de statut
            # Ajoutez d'autres mappings si nécessaire
        }
        
        # Application des mappings
        for old_col, new_col in column_mappings.items():
            if old_col in df_processed.columns:
                df_processed[new_col] = df_processed[old_col]
        
        # Vérification des colonnes requises
        required_cols = ['ACTIF', 'censure']
        missing_cols = [col for col in required_cols if col not in df_processed.columns]
        
        if missing_cols:
            st.error(f"Colonnes manquantes après mapping : {missing_cols}")
            st.info("Colonnes disponibles : " + ", ".join(df_processed.columns.tolist()))
            return None
        
        # Nettoyage des données
        df_processed = df_processed.dropna(subset=['ACTIF', 'censure'])
        df_processed['ACTIF'] = pd.to_numeric(df_processed['ACTIF'], errors='coerce')
        df_processed['censure'] = pd.to_numeric(df_processed['censure'], errors='coerce').astype(int)
        
        # Filtrage des valeurs aberrantes
        df_processed = df_processed[
            (df_processed['ACTIF'] > 0) & 
            (df_processed['ACTIF'] < 100) &  # Âge max réaliste
            (df_processed['censure'].isin([0, 1]))
        ]
        
        return df_processed
    
    def get_available_symb_codes(self, limit: int = 100) -> List[str]:
        """Récupère les codes SYMB disponibles"""
        if not self.engine:
            return []
        
        try:
            query = '''
            SELECT DISTINCT symb 
            FROM "AGRSIG_PBI"."ASTOTS" 
            WHERE symb IS NOT NULL 
            ORDER BY symb 
            LIMIT :limit_param
            '''
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={'limit_param': limit})
            
            return df['symb'].tolist()
            
        except Exception as e:
            st.error(f"Erreur récupération codes SYMB : {str(e)}")
            return []
    
    def get_available_constructeurs(self) -> List[str]:
        """Récupère les constructeurs disponibles"""
        if not self.engine:
            return []
        
        try:
            query = '''
            SELECT DISTINCT lib_constr 
            FROM "AGRSIG_PBI"."ASTOTS" 
            WHERE lib_constr IS NOT NULL 
            ORDER BY lib_constr
            '''
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            
            return df['lib_constr'].tolist()
            
        except Exception as e:
            st.error(f"Erreur récupération constructeurs : {str(e)}")
            return []

# ====================================================================
# dashboard_fiabilite.py - Intégration dans le dashboard principal
# ====================================================================

# Ajout au début de votre fichier principal
from config.database_config import DatabaseConfig
from utils.relais_data_loader import RelaisDataLoader

# Modification de la section sidebar
def main():
    # ... (début inchangé)
    
    elif data_source == "Base PostgreSQL":
        with st.sidebar.expander("🔧 Configuration Base de Données", expanded=True):
            try:
                # Initialisation sécurisée
                db_config = DatabaseConfig()
                data_loader = RelaisDataLoader(db_config)
                
                st.success("✅ Connexion établie")
                
                # Interface pour vos paramètres spécifiques
                st.subheader("Paramètres de Requête")
                
                # Sélection SYMB
                available_symb = data_loader.get_available_symb_codes()
                if available_symb:
                    symb_selected = st.selectbox(
                        "Code SYMB",
                        available_symb,
                        index=available_symb.index("79540230") if "79540230" in available_symb else 0
                    )
                else:
                    symb_selected = st.text_input("Code SYMB", value="79540230")
                
                # Sélection constructeurs
                available_constructeurs = data_loader.get_available_constructeurs()
                if available_constructeurs:
                    constructeurs_selected = st.multiselect(
                        "Constructeurs",
                        available_constructeurs,
                        default=["ANSA", "CSEE", "HITA"] if all(c in available_constructeurs for c in ["ANSA", "CSEE", "HITA"]) else available_constructeurs[:3]
                    )
                else:
                    st.warning("Constructeurs par défaut utilisés")
                    constructeurs_selected = ["ANSA", "CSEE", "HITA"]
                
                # Limite optionnelle
                use_limit = st.checkbox("Limiter le nombre de lignes")
                limit_value = st.number_input("Limite", min_value=100, max_value=10000, value=5000) if use_limit else None
                
                # Chargement des données
                if st.button("📥 Charger les données relais"):
                    with st.spinner("Chargement en cours..."):
                        df = data_loader.load_relais_data(
                            symb=symb_selected,
                            constructeurs=constructeurs_selected,
                            limit=limit_value
                        )
                
            except Exception as e:
                st.error(f"Erreur configuration : {str(e)}")
                df = None
    
    # ... (suite inchangée)

# ====================================================================
# .env.example - Template des variables d'environnement
# ====================================================================
"""
# Copiez ce fichier vers .env et remplissez vos valeurs
DB_HOST=posqresql
DB_PORT=5432
DB_NAME=dtiesss
DB_USER=Integ
DB_PASSWORD=your_password_here
"""

# ====================================================================
# .streamlit/secrets.toml - Pour Streamlit Cloud
# ====================================================================
"""
[database]
host = "posqresql"
port = 5432
name = "dtiesss" 
user = "Integ"
password = "your_password_here"
"""
