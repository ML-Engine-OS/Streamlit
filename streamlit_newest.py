# ====================================================================
# IMPORTS ADDITIONNELS POUR POSTGRESQL
# ====================================================================

from sqlalchemy import create_engine, text
import psycopg2
from datetime import datetime

# ====================================================================
# FONCTION DE CONNEXION POSTGRESQL
# ====================================================================

@st.cache_resource
def init_postgres_connection():
    """Initialise la connexion PostgreSQL avec mise en cache"""
    try:
        db_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(
            db_string,
            pool_pre_ping=True,  # Vérification automatique de la connexion
            pool_recycle=300     # Recyclage des connexions après 5 minutes
        )
        # Test de connexion
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).fetchone()
        return engine
    except Exception as e:
        st.error(f"Erreur de connexion PostgreSQL : {str(e)}")
        return None

@st.cache_data(ttl=600)  # Cache pendant 10 minutes
def load_data_from_postgres(query, _engine=None):
    """Charge les données depuis PostgreSQL avec mise en cache"""
    if _engine is None:
        return None
    
    try:
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erreur lors de la lecture PostgreSQL : {str(e)}")
        return None

def get_available_tables(engine):
    """Récupère la liste des tables disponibles"""
    try:
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        ORDER BY table_name;
        """
        df_tables = pd.read_sql(query, engine)
        return df_tables['table_name'].tolist()
    except:
        return []

def get_table_columns(engine, table_name):
    """Récupère les colonnes d'une table"""
    try:
        query = f"""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}' 
        ORDER BY ordinal_position;
        """
        df_cols = pd.read_sql(query, engine)
        return df_cols
    except:
        return pd.DataFrame()

# ====================================================================
# MODIFICATION DE LA FONCTION PRINCIPALE load_and_validate_data
# ====================================================================

@st.cache_data
def load_and_validate_data(uploaded_file=None, postgres_engine=None, table_name=None, query=None):
    """Charge et valide les données depuis fichier ou PostgreSQL"""
    df = None
    
    # Chargement depuis fichier
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier : {str(e)}")
            return None
    
    # Chargement depuis PostgreSQL
    elif postgres_engine is not None:
        if query:
            # Requête personnalisée
            df = load_data_from_postgres(query, postgres_engine)
        elif table_name:
            # Sélection de table complète
            query = f"SELECT * FROM {table_name} LIMIT 10000;"  # Limite de sécurité
            df = load_data_from_postgres(query, postgres_engine)
    
    if df is None:
        return None
    
    # Validation des colonnes requises
    required_cols = ['ACTIF', 'censure']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Colonnes manquantes : {missing_cols}")
        st.info("Format attendu : 'ACTIF' (âge en années), 'censure' (0=vivant, 1=défaillant)")
        
        # Affichage des colonnes disponibles pour aide
        st.write("**Colonnes disponibles dans les données :**")
        st.write(list(df.columns))
        return None
        
    return df

# ====================================================================
# MODIFICATION DE LA SECTION SIDEBAR - CONFIGURATION
# ====================================================================

def main():
    # ... (début inchangé)
    
    # ====================================================================
    # SIDEBAR - CONFIGURATION MODIFIÉE
    # ====================================================================
    
    st.sidebar.header("📋 Configuration des Données")
    
    # Sélection de la source de données
    data_source = st.sidebar.radio(
        "📊 Source des données",
        ["Fichier local", "Base PostgreSQL", "Données synthétiques"],
        index=2  # Par défaut sur synthétiques
    )
    
    df = None
    postgres_engine = None
    
    # ====================================================================
    # GESTION DES SOURCES DE DONNÉES
    # ====================================================================
    
    if data_source == "Fichier local":
        # Upload de fichier existant
        uploaded_file = st.sidebar.file_uploader(
            "📁 Charger un fichier",
            type=['csv', 'xlsx'],
            help="Format: colonnes 'ACTIF' (âge), 'censure' (0=vivant, 1=défaillant)"
        )
        
        if uploaded_file:
            df = load_and_validate_data(uploaded_file=uploaded_file)
            if df is not None:
                st.success(f"✅ **Fichier chargé** : {len(df)} relais")
    
    elif data_source == "Base PostgreSQL":
        # Configuration PostgreSQL
        with st.sidebar.expander("🔧 Connexion PostgreSQL", expanded=True):
            # Initialisation de la connexion
            postgres_engine = init_postgres_connection()
            
            if postgres_engine is None:
                st.error("❌ Connexion PostgreSQL échouée")
            else:
                st.success("✅ Connexion PostgreSQL établie")
                
                # Sélection de la méthode d'accès
                access_method = st.radio(
                    "Méthode d'accès",
                    ["Sélection de table", "Requête personnalisée"]
                )
                
                if access_method == "Sélection de table":
                    # Liste des tables disponibles
                    available_tables = get_available_tables(postgres_engine)
                    
                    if available_tables:
                        selected_table = st.selectbox(
                            "📋 Sélectionner une table",
                            [""] + available_tables
                        )
                        
                        if selected_table:
                            # Affichage des colonnes de la table
                            cols_info = get_table_columns(postgres_engine, selected_table)
                            if not cols_info.empty:
                                st.write("**Colonnes disponibles :**")
                                st.dataframe(cols_info, hide_index=True)
                            
                            # Chargement des données
                            if st.button("📥 Charger la table"):
                                df = load_and_validate_data(
                                    postgres_engine=postgres_engine,
                                    table_name=selected_table
                                )
                                if df is not None:
                                    st.success(f"✅ **Table chargée** : {len(df)} enregistrements")
                    else:
                        st.warning("⚠️ Aucune table trouvée")
                
                elif access_method == "Requête personnalisée":
                    # Requête SQL personnalisée
                    custom_query = st.text_area(
                        "📝 Requête SQL",
                        placeholder="SELECT ACTIF, censure FROM ma_table WHERE condition...",
                        height=100
                    )
                    
                    if custom_query.strip():
                        if st.button("🔍 Exécuter la requête"):
                            df = load_and_validate_data(
                                postgres_engine=postgres_engine,
                                query=custom_query
                            )
                            if df is not None:
                                st.success(f"✅ **Requête exécutée** : {len(df)} lignes")
    
    else:  # Données synthétiques
        st.sidebar.info("💡 Utilisation des données synthétiques")
        df = generate_synthetic_data()
        st.info("📝 **Données synthétiques chargées** (2500 relais) pour démonstration")
    
    # Vérification finale des données
    if df is None:
        st.warning("⚠️ Aucune donnée chargée. Sélectionnez une source de données.")
        return
    
    # ====================================================================
    # INFORMATIONS ADDITIONNELLES SUR LES DONNÉES POSTGRESQL
    # ====================================================================
    
    if data_source == "Base PostgreSQL" and postgres_engine is not None:
        # Métadonnées de la connexion
        with st.sidebar.expander("ℹ️ Informations de connexion"):
            st.write("**Serveur :** posqresql:5432")
            st.write("**Base :** dtiesss")
            st.write("**Utilisateur :** Integ")
            st.write(f"**Dernière connexion :** {datetime.now().strftime('%H:%M:%S')}")
            
            # Test de latence
            if st.button("🏃 Tester la latence"):
                start_time = time.time()
                try:
                    with postgres_engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                    latency = (time.time() - start_time) * 1000
                    st.success(f"✅ Latence : {latency:.0f} ms")
                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")
    
    # ====================================================================
    # SUITE DU CODE INCHANGÉE
    # ====================================================================
    # ... (le reste de votre fonction main() reste identique)

# ====================================================================
# MISE À JOUR DU REQUIREMENTS.TXT
# ====================================================================

"""
