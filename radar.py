import random
import sys
from datetime import datetime
import re
from bs4 import BeautifulSoup
import openai
import requests
from scholarly import scholarly
import re
import folium
from streamlit_folium import folium_static
import itertools
import pycountry  # Pour récupérer la liste des pays standardisés


# importation des librairies pour le traitement des données

import streamlit as st # librairie de streamlit

# librairie de visualisation de données
import seaborn as sns
import matplotlib.pyplot as plt

# librairie de traitement de données
import pandas as pd
import csv

# st.set_option('server.enableCORS', True)

# J'ai ajouté cette ligne pour éviter un warning de Streamlit
#st.set_option('deprecation.showPyplotGlobalUse', False)
# Permet de changer le titre de la page, l'icône et la mise en page
st.set_page_config(
    page_title="DataWiz | DataWiz est une application web qui permet de visualiser et de traiter des données.",
    page_icon="https://firebasestorage.googleapis.com/v0/b/hyphip-8ca89.appspot.com/o/datawiz.png?alt=media&token=5820f215-75f1-47ff-b486-b44d37aa02f7",
    layout="wide",   # Permet d'afficher d'utiliser la totalité de la page
    initial_sidebar_state="expanded",  # Permet d'afficher la sidebar par défaut
)

# Définir le style de la page
page_bg = '''  <style>
        [data-testid=stApp]  {
            background: radial-gradient(100.93% 54.6% at 17.34% 23.14%, #24004D 0%, #44006E 26.56%, #5000C0 43.75%, #5000BC 57.81%, #1E006D 77.08%, #0E193C 92.71%);
            color: white;
        }
        [data-testid=stHeader]  {
            background: #44006E;
            color: white;
        }
        [data-testid="stSidebarContent"]{
            background: #44006E;
            color: white;
        }
        h1{
          color: white
        }
        [data-testid=stHorizontalBlock]{
          justify-content: space-between;
        }
        [data-testid=baseButton-secondary] {
        background-color: transparent;
        border:none;
        }
        [data-testid=baseButton-secondary]:hover {
          font-weight: bold;
          text-decoration: underline;
          color: white;
          border: none;
          margin-top: 1px;
        }
        [data-testid=baseButton-secondary]:clicked {
          font-weight: 900;
          text-decoration: underline;
          color: white;
          border: none;
          margin-top: 1px;

        }
        [data-testid=baseButton-secondary]:visited {
          background-color: transparent;
          border:none;
        }
        [data-testid="stWidgetLabel"]{
          color: white
        }
        #data-description{
          color: white;
        }
        h2{
          color: white;
        }
        #stHeadingContainer{
          color : white;
        }
        [data-testid="stBottomBlockContainer"]{
            background: #44006E;
            color: white;
        }
        h3{
            color: white;
        }
        [data-testid="stChatMessageContent"]{
          color: white;
        }
        </style>
        '''
st.markdown(page_bg, unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .center-map {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# lien vers le logo de la page

logo = "https://firebasestorage.googleapis.com/v0/b/hyphip-8ca89.appspot.com/o/datawiz-removebg-preview.png?alt=media&token=3619a395-795c-4bf1-b054-6bc018369c87"
# fonction pour ordonner les données selon une colonne et un ordre
def order_by(dataframe, column, ascending=True):

  datacopy = dataframe.copy()
  return datacopy.sort_values(by=column, ascending=ascending)


# API Key Perplexity (à remplacer par ta clé)
PERPLEXITY_API_KEY = "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"
client = openai.OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")





# ✅ Fonction pour extraire dynamiquement le nom de l'institution
def extract_institution_name(affiliation):
    """
    Extrait automatiquement le nom de l’institution à partir de l’affiliation complète.
    Fonctionne pour toute institution académique ou entreprise.
    """
    # Liste des mots-clés indicateurs d'une institution
    keywords = ["University", "Institute", "College", "School", "Academy", "Inc.", "Corp.", "Laboratory", "Research Center"]
    
    # Séparer les termes de l'affiliation
    words = affiliation.split(",")

    for word in words:
        word = word.strip()
        if any(kw in word for kw in keywords):  # Vérifie si un mot-clé est présent
            return word

    # Si aucun mot-clé n'est trouvé, on retourne l'affiliation complète
    return affiliation




# ✅ Étape 1 : Recherche des chercheurs via un thème (Google Scholar)
def search_scholars_from_theme(theme, max_results=20):
    """
    Recherche des publications sur Google Scholar en fonction d'un thème
    et extrait les auteurs impliqués.
    Retourne deux listes : 
      - Une liste unique des auteurs
      - Une liste des publications
    """
    try:
        search_query = scholarly.search_pubs(theme)
        authors_list = set()  # Pour éviter les doublons
        publications_list = []  # Stocker les publications

        for _ in range(max_results):  # Limite pour éviter le blocage
            try:
                publication = next(search_query)
                title = publication['bib'].get('title', "Titre inconnu")
                authors = publication['bib'].get('author', [])

                if isinstance(authors, str):  # Si c'est une string, convertir en liste
                    authors = authors.split(", ")

                for author in authors:
                    authors_list.add(author.strip())

                publications_list.append(title)  # Ajouter le titre de la publication

            except StopIteration:
                break
        
        return list(authors_list), publications_list  # ✅ Retourner deux valeurs distinctes

    except Exception as e:
        print(f"Erreur lors de la recherche sur Google Scholar : {e}")
        return [], []



# ✅ Étape 2 : Utiliser Perplexity pour obtenir le NOM COMPLET
def get_scholar_names_perplexity(authors, publications):
    """
    Utilise Perplexity AI pour récupérer UNIQUEMENT le nom complet des chercheurs,
    en vérifiant que ces noms sont bien associés aux publications trouvées.
    """
    if not authors or not publications:
        return None

    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un expert en recherche académique. "
                "Ta mission est de récupérer le **nom complet** des chercheurs listés "
                "et de **vérifier qu'ils sont bien auteurs** des publications trouvées sur Google Scholar."
            ),
        },
        {   
            "role": "user",
            "content": (
                "Effectue une **recherche sur Google Scholar** pour retrouver le **nom complet** "
                "des chercheurs listés ci-dessous, en rapport avec leurs travaux.\n\n"
                "**⚠️ Vérifie impérativement que ces chercheurs sont bien les auteurs des publications ci-dessous.**\n"
                "**Si un nom trouvé n'est pas relié aux publications, NE PAS L’INCLURE.**\n\n"
                f"📚 **Publications trouvées :** {', '.join(publications)}\n\n"
                f"👨‍🔬 **Liste des chercheurs extraits de Google Scholar :** {', '.join(authors)}\n\n"
                "**Ne renvoie que la liste des noms complets valides, un par ligne.**"
            ),
        },
    ]

    response = client.chat.completions.create(
        model="sonar-pro",
        messages=messages,
    )

    if response:
        return response.choices[0].message.content
    return None



# ✅ Étape 3 : Recherche du profil Scholar via `scholarly`
def find_scholar_profile(full_name):
    """
    Recherche un chercheur sur Google Scholar via `scholarly`
    et retourne l'URL de son profil s'il est trouvé.
    """
    search_query = scholarly.search_author(full_name)
    try:
        author = next(search_query)  # Prendre le premier résultat
        scholar_id = author['scholar_id']
        return f"https://scholar.google.com/citations?user={scholar_id}"
    except StopIteration:
        return None


# ✅ Étape 4 : Scraping du profil Scholar pour récupérer Affiliation et H-index
def scrape_scholar_profile(scholar_url):
    """
    Récupère l'affiliation et le H-index d'un chercheur en scrapant son profil Google Scholar.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(scholar_url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        affiliation_tag = soup.find("div", class_="gsc_prf_il")
        affiliation = affiliation_tag.text.strip() if affiliation_tag else "Affiliation inconnue"

        stats = soup.find_all("td", class_="gsc_rsb_std")
        h_index = stats[2].text.strip() if len(stats) >= 3 else "Non disponible"  # 3e élément = H-index

        return {"Nom": full_name, "Affiliation": affiliation, "H-index": h_index, "Profil": scholar_url}

    return {"Nom": "Erreur", "Affiliation": "Erreur", "H-index": "Erreur", "Profil": scholar_url}


# ✅ Fonction pour nettoyer les affiliations avant de les envoyer à Perplexity
def clean_affiliation(affiliation):
    """
    Nettoie l'affiliation en retirant les titres et départements pour normaliser les noms des institutions.
    """
    if not affiliation or affiliation.lower() in ["non trouvée", "affiliation inconnue"]:
        return None  # Exclure les affiliations non valides

    # Supprimer les titres académiques et département
    remove_words = ["PhD Candidate", "Professor of", "Department of", "Faculty of", "Institute of", "Lab of", "Graduate Student"]
    
    for word in remove_words:
        affiliation = re.sub(rf"\b{word}\b", "", affiliation, flags=re.IGNORECASE).strip()

    return affiliation.strip()

# ✅ Fonction pour récupérer l’adresse via Perplexity
def get_affiliation_address_perplexity(affiliations):
    """
    Uses Perplexity AI to search for the full address and country of the listed institutions.
    """
    if not affiliations:
        return {}

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert in academic address retrieval. "
                "Your task is to find the **complete address** "
                "of the listed institutions by performing **Google searches**."
            ),
        },
        {   
            "role": "user",
            "content": (
                "Use Google to find the **full address** of each listed affiliation.\n\n"
                "**Return the result in this STRICT format:**\n"
                "Institution | Address | Country\n"
                "------------------------------------------------\n"
                f"{', '.join(affiliations)}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model="sonar-pro",
        messages=messages,
    )

    if response:
        raw_text = response.choices[0].message.content
        return parse_affiliation_addresses(raw_text)  # Convert the response into a usable dictionary
    return None


# ✅ Fonction pour analyser la réponse de Perplexity et extraire les adresses
def parse_affiliation_addresses(response_text):
    """
    Convertit la réponse brute de Perplexity en dictionnaire {Affiliation: (Adresse, Pays)}.
    """
    affiliation_map = {}
    lines = response_text.strip().split("\n")

    for line in lines:
        parts = line.split("|")
        if len(parts) == 3:
            institution = parts[0].strip()
            address = parts[1].strip()
            country = parts[2].strip()
            affiliation_map[institution] = (address, country)

    return affiliation_map

# ✅ Fonction pour trouver la meilleure correspondance
def find_best_match(original_affiliation, affiliation_data):
    """
    Trouve la meilleure correspondance approximative entre une affiliation originale et les résultats de Perplexity.
    """
    for key in affiliation_data.keys():
        if key.lower() in original_affiliation.lower():  # Vérification approximative
            return affiliation_data[key]  # Retourne (adresse, pays)

    return ("Non disponible", "Non disponible")



def expand_affiliation_abbreviations(affiliations):
    """
    Uses Perplexity AI to obtain the full name of abbreviated affiliations.
    """
    if not affiliations:
        return {}

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert in academic affiliations. "
                "Your task is to convert institution abbreviations into their full names "
                "using Google searches if necessary."
            ),
        },
        {
            "role": "user",
            "content": (
                "Here is a list of affiliations that may be abbreviated. "
                "Return their **full name** in this STRICT format:\n"
                "Abbreviation | Full Name\n"
                "-----------------------------------\n"
                f"{', '.join(affiliations)}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model="sonar-pro",
        messages=messages,
    )

    if response:
        raw_text = response.choices[0].message.content
        return parse_expanded_affiliations(raw_text)
    
    return {}

def parse_expanded_affiliations(response_text):
    """
    Convertit la réponse brute de Perplexity en dictionnaire {Abréviation: Nom complet}.
    """
    abbreviation_map = {}
    lines = response_text.strip().split("\n")

    for line in lines:
        parts = line.split("|")
        if len(parts) == 2:
            abbreviation = parts[0].strip()
            full_name = parts[1].strip()
            abbreviation_map[abbreviation] = full_name

    return abbreviation_map




def standardize_country(country_name):
    """
    Convertit les variantes de pays en leur nom officiel selon pycountry.
    """
    try:
        return pycountry.countries.lookup(country_name).name
    except LookupError:
        return country_name  # Garde l'original s'il n'est pas trouvé








# Fonction pour rechercher un auteur et son h-index
def search_scholar_with_h_index(query, max_articles=5):
    try:
        search_query = scholarly.search_author(query)
        author = scholarly.fill(next(search_query))  # Récupère le premier résultat
        
        # Informations principales
        author_name = author.get("name", "N/A")
        affiliation = author.get("affiliation", "N/A")
        h_index = calculate_h_index(author.get("publications", []))
        publications = author.get("publications", [])

        # Publications
        articles = [
            {
                "title": pub.get("bib", {}).get("title", "N/A"),
                "citations": pub.get("num_citations", 0),
            }
            for pub in publications[:max_articles]
        ]

        return {"name": author_name, "affiliation": affiliation, "h_index": h_index, "articles": articles}
    except StopIteration:
        st.warning("No results found for this query.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None



# fonction pour détecter le séparateur d'un fichier csv
def detect_separator(uploaded_file):
  # Convertir le fichier en string
  content = uploaded_file.getvalue().decode('utf-8')

  # Utiliser Sniffer pour détecter le séparateur
  # 8192 pour traiter 8kb(premiers caractères) de données mais on peut
  # ajuster la valeur selon le nombre de colonnes.
  # Augmenter jusqu'à fonctionnement si le nombre de colonnes est elevé
  dialect = csv.Sniffer().sniff(content[:8192])

  return dialect.delimiter

# Affichage de la matrice de corrélation
def correlation_matrix(dataframe):
  fig, ax = plt.subplots(figsize=(10, 8))
  st.subheader("Correlation Matrix")
  corr = dataframe[numeric_columns].corr()
  sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
  st.pyplot(fig)

# Affichage du graphique de comptage
def count_plot(dataframe):
  fig, ax = plt.subplots(figsize=(10, 8))
  st.subheader("Count Plot")
  x_column = st.selectbox("Select column Count Plot:", dataframe.columns)
  sns.countplot(data=dataframe[x_column], ax=ax)
  st.pyplot(fig)

# Affichage du graphique en secteur
def pie_chart(dataframe):
  fig, ax = plt.subplots(figsize=(10, 8))
  st.subheader("Pie Chart")
  x_column = st.selectbox("Select column Pie Chart:", dataframe.columns)
  dataframe[x_column].value_counts().plot(kind='pie', ax=ax)
  st.pyplot(fig)

# Affichage du graphique en barres
def bar_plot(dataframe):
  fig, ax = plt.subplots(figsize=(10, 8))
  st.subheader("Bar Plot")
  x_column = st.selectbox("Select X-axis column Bar Plot:", dataframe.columns)
  if(len(dataframe.columns) > 1):
    y_column = st.selectbox("Select Y-axis column Bar Plot:", dataframe.columns)
  sns.barplot(x=x_column, y=y_column, data=dataframe, ax=ax)
  st.pyplot(fig)

# Affichage du graphique en nuage de points
def scatter_plot(dataframe):
  fig, ax = plt.subplots(figsize=(10, 8))
  st.subheader("Scatter plot")
  x_axis = st.selectbox("Select X-axis for Scatter:", dataframe.columns)
  y_axis = st.selectbox("Select Y-axis for Scatter:", dataframe.columns)
  sns.scatterplot(x=dataframe[x_axis], y=dataframe[y_axis], ax=ax)
  ax.set_title("Scatter Plot")
  ax.set_xlabel(x_axis)
  ax.set_ylabel(y_axis)
  st.pyplot(fig)

# Affichage du graphique en ligne
def line_chart(dataframe):
  fig, ax = plt.subplots(figsize=(10, 8))
  st.subheader("Line Chart")
  x_column = st.selectbox("Select X-axis column Line Chart:", numeric_columns)
  y_column = st.selectbox("Select Y-axis column Line Chart:", numeric_columns)
  line_chart_data = dataframe[[x_column, y_column]]
  ax.plot(line_chart_data[x_column], line_chart_data[y_column])
  ax.set_xlabel(x_column)
  ax.set_ylabel(y_column)
  st.pyplot(fig)

# Affichage du graphique en pair
def pair_plot(dataframe):
  fig, ax = plt.subplots(figsize=(10, 8))
  st.subheader(f"Pair Plot")
  x_column = st.selectbox("Select X-axis column:", dataframe.columns)
  y_column = st.selectbox("Select Y-axis column:", dataframe.columns)
  st.write(f"Customized X: {x_column}, Y: {y_column}")
  sns.pairplot(dataframe, x_vars=[x_column], y_vars=[y_column], height=7)
  ax.set_xlabel(x_column)
  ax.set_ylabel(y_column)
  st.pyplot()



# Affichage du graphique en histogramme
def histogram(dataframe):
  fig, ax = plt.subplots(figsize=(10, 8))
  st.subheader("Histogram")
  column_name = st.selectbox("Select a column:", dataframe.columns)
  # Plot histogram with vertical orientation for categorical variables
  if dataframe[column_name].dtype == 'category':  # Check if the column is categorical
    ax.hist(dataframe[column_name]
    , orientation='horizontal', bins='auto')
  else:
    ax.hist(dataframe[column_name])
  ax.set_xlabel(column_name)
  st.pyplot(fig)

# Affichage du graphique en boîte
def box_plot(dataframe):
  fig, ax = plt.subplots(figsize=(10, 8))
  st.subheader("Box Plot")
  x_column = st.selectbox("Select X-axis column for Box Plot:", dataframe.columns)
  y_column = st.selectbox("Select Y-axis column for Box Plot:", dataframe.columns)
  st.write(f"Customized X: {x_column}, Y: {y_column}")
  sns.boxplot(x=x_column, y=y_column, data=dataframe, ax=ax)
  ax.set_xlabel(x_column)
  ax.set_ylabel(y_column)
  st.pyplot(fig)

# Affichage de l'en tête


def documentation_page():
  st.session_state['page'] = "documentation"

def about_page():
  st.session_state['page'] = "about"

def load_page():
  st.session_state['page'] = ""

def radar_page():
  st.session_state['page'] = "radar"

col1, col2 = st.columns(2)

with col1:
  # Ajout du logo
  st.image(logo, width=180)


with col2:
  col21, col22, col24 ,col23 = st.columns(4)
  with col21:
    st.button("Documentation", on_click=documentation_page)
  with col22:
    st.button("About Us", on_click=about_page)
  with col23:
    st.button("Load Your Data", on_click=load_page, key=load_page)
  with col24:
    st.button("🛰️ Radar", on_click=radar_page)



#Si aucun document n'est sélectionné
if 'page' not in st.session_state or st.session_state['page'] == ""  :

  markdown_text = """
    <div style="text-align: center;">
      <h2 style="color: #FFFFFF;">Welcome to DataWiz - Your Data Analysis and Simulation Hub!</h2>
      <h2 style="color: #F6F2FC; opacity: 0.4;">Explore, Analyze, and Simulate with DataWiz</h2>
    </div>
  """

  # Affichage du texte Markdown
  st.markdown(markdown_text, unsafe_allow_html=True)

  # Chargement du fichier (csv seulement pour le moment, à améliorer si intéressé(e))
  uploaded_file = st.file_uploader("Choose a CSV file to upload", type="csv")


  # Si un fichier est chargé (taille <= 200 MB) alors on continue
  if uploaded_file is not None:

      # Convertir le fichier en bytes
      bytes_data = uploaded_file.getvalue()
      try:
        # Détecter le séparateur du fichier
        separator = detect_separator(uploaded_file)
        # Convertir le fichier en dataframe
        dataframe = pd.read_csv(uploaded_file, header=[0], sep=separator)

      except:
        # Afficher un message d'erreur si le séparateur n'est pas détecté
        st.error("Unable to detect separator. Please check your file and try again. ou nombre de colonnes trop élevé")
        # Arrêter l'exécution du script
        sys.exit()

      # Nom du site
      st.sidebar.image(logo, width=180)

      # Titre de la sidebar
      st.sidebar.header("Search & metadata")

      # Copie de la base de donnée
      copy = dataframe.copy()

      # Récuperation de la taille de la base de donnée
      taille= len(copy.columns)

      # Création des filtres
      filtre_par_colonne= st.sidebar.selectbox("Select a feature", copy.columns[0:taille])

      def is_potential_date(column):
        # Define regex patterns for both French and US date formats
        p1= re.compile(r"\d{1,2}[.-]\d{1,2}[.-]\d{4}( \d{2}:\d{2}:\d{2})?") # French date format (dd-mm-yyyy)
        p2 = re.compile(r"\d{4}[.-]\d{1,2}[.-]\d{1,2}( \d{2}:\d{2}:\d{2})?") # US date format (yyyy-mm-dd)

        # Check if at least one value in the column matches either date pattern
        if column.dtype == 'O' or column.dtype == 'object' or column.dtype == 'category':
         return any(column.str.match(p1, na=False)) or any(column.str.match(p2, na=False))
        return False
     # Variable de detection de type de colonne
      colType = "string/number"

      if pd.api.types.is_bool_dtype(copy[filtre_par_colonne]):
        colType = "boolean"
        filtre_search_boolean= st.sidebar.selectbox("Select a boolean value ",("True"  ,"False"))

      elif pd.api.types.is_datetime64_any_dtype(copy[filtre_par_colonne]):
          colType = "datetime"
          start_date = datetime(1976, 1, 1)
          end_date = datetime.today()
          selected_range = st.sidebar.date_input("Select a date range",
                                    min_value=start_date,
                                    max_value=end_date,
                                    value=(start_date, end_date))
      else:
          colType = "string/number"
          if(is_potential_date(copy[filtre_par_colonne])):
            st.sidebar.write("This column seems to contain dates. Please select a date range.")
            colType = "datetime"
            start_date = datetime(1976, 1, 1)
            end_date = datetime.today()
            selected_range = st.sidebar.date_input("Select a date range",
                                    min_value=start_date,
                                    max_value=end_date,
                                    value=(start_date, end_date))
          else:
            filtre_search_word= st.sidebar.text_input("Insert a " + "'"+filtre_par_colonne+"'" " value for search ", value=None, placeholder="Type a value...")

      filtre_ascending_des= st.sidebar.selectbox("Order by ",("Ascending"  ,"Descending"))
      filtre_null= st.sidebar.selectbox("Include null values",("Yes"  ,"No"))

      #True si le filtre null est activé et False sinon
      is_accessible = True

      # Affiche ou non les valeurs nulles selon le filtre null
      if str(filtre_null)=='No' :
        is_accessible = False
        copy = copy.copy().dropna()
      else :
        is_accessible = True
        copy = copy

      # Placé ici en raison de la dépendance de la variable is_accessible
      rows_with_null = st.sidebar.checkbox('Show rows with only null values', value=False, key="accessible_checkbox_key", help="This checkbox can be toggled based on 'Include null values' state.", disabled=not is_accessible)

      filtre_null_stat= st.sidebar.selectbox("Null Values Statistics",copy.columns[0:taille])

      if colType == "string/number":
          # Traitements des données selon les filtres de recherche par mot clé et par colonne
          if filtre_search_word:
              # Vérifier si la colonne est de type string
              if copy[filtre_par_colonne].dtype == 'O' or copy[filtre_par_colonne].dtype == 'object' or copy[filtre_par_colonne].dtype == 'category':
                  copy = copy[copy[filtre_par_colonne].astype(str).str.contains(filtre_search_word, case=False, na=False)]
              else:
                  try:
                      # Essayer de convertir le mot clé en type de la colonne
                      converted_search_term = type(copy[filtre_par_colonne].iloc[0])(filtre_search_word)
                      copy = copy[copy[filtre_par_colonne] == converted_search_term]
                  except (ValueError, TypeError):
                      # Afficher un message d'erreur si le mot clé ne peut pas être convertir
                      st.sidebar.error("Unable to convert search term to string/numeric.")
                      pass

      elif colType == "boolean":
        if filtre_search_boolean:
          try:

            if filtre_search_boolean == "True":
              copy = copy[copy[filtre_par_colonne] == True]
            else:
              copy = copy[copy[filtre_par_colonne] == False]
          except (ValueError, TypeError):
              # Afficher un message d'erreur si le mot clé ne peut pas être convertir
            st.sidebar.error("Unable to convert search term to boolean.")
            pass

      elif colType == "datetime":
        if selected_range:
         try:
            # Vérifie qi marche en local
           if len(selected_range ) > 1:
            copy = copy[(pd.to_datetime(copy[filtre_par_colonne], format = "ISO8601") >= pd.to_datetime(selected_range[0], format = "ISO8601")) & (pd.to_datetime(copy[filtre_par_colonne], format = "ISO8601") <= pd.to_datetime(selected_range[1], format = "ISO8601"))]
           else:
            copy = copy[pd.to_datetime(copy[filtre_par_colonne], format = "ISO8601") >= pd.to_datetime(selected_range[0], format = "ISO8601")]
            
          # Version qui marche sur google colab
          #  if len(selected_range ) > 1:
          #   copy = copy[(pd.to_datetime(copy[filtre_par_colonne], infer_datetime_format=True) >= pd.to_datetime(selected_range[0], infer_datetime_format=True)) & (pd.to_datetime(copy[filtre_par_colonne], infer_datetime_format=True) <= pd.to_datetime(selected_range[1], infer_datetime_format=True))]
          #  else:
          #   copy = copy[pd.to_datetime(copy[filtre_par_colonne], infer_datetime_format=True) >= pd.to_datetime(selected_range[0], infer_datetime_format=True)]
            
         except Exception as e:
           st.sidebar.error(e)
           # st.sidebar.error("Unable to convert search term to date.")
           pass

      # Traiter les données pour afficher les lignes avec des valeurs nulles
      if rows_with_null:
        copy = copy[copy.isnull().any(axis=1)]

      # Affiche le pourcentage de valeurs nulles dans la colonne sélectionnée
      null_stats = copy[str(filtre_null_stat)].isnull().sum()
      total_rows = len(copy)
      null_percentage = (null_stats / total_rows) * 100
      if total_rows == 0:
        st.sidebar.write(f"{str(filtre_null_stat)}: {null_stats}/{total_rows} row(s), about 0.00%")
      else:
        st.sidebar.write(f"{str(filtre_null_stat)}: {null_stats}/{total_rows} row(s), about {null_percentage:.2f}%")

      # Appliquer le filtre d'ordre
      if str(filtre_ascending_des)=='Ascending' :
       copy = order_by(copy, filtre_par_colonne, ascending=True)
      else :
       copy = order_by(copy, filtre_par_colonne, ascending=False)

      # Copie de la base de donnée après les traitements
      after_filtre=copy.copy()

      # Affichage de la base de donnée
      st.dataframe(after_filtre, use_container_width=True, hide_index=True)
      num_rows = after_filtre.shape[0]
      st.write(f"Number of Rows: {num_rows}")

      # Affichage des informations sur la base de donnée dans la sidebar
      st.sidebar.text("Data Types")
      st.sidebar.text(after_filtre.dtypes)

      # tri et récupération des colonnes selon leur type (Pourquoi ? j'en sais rien
      # j'ai l'impression que ça sera utile plus tard ou donnera des idées)
      num, cat, bool = st.columns(3)

      # Récupération des colonnes de la base de donnée
      all_columns = after_filtre.columns

      # Initialiser une liste vide pour stocker les colonnes numériques
      numeric_columns = []


      for column in after_filtre.columns:
        # Verifier si la colonne contient des valeurs numériques
        if pd.api.types.is_numeric_dtype(after_filtre[column]) or pd.api.types.is_float_dtype(after_filtre[column]):
          numeric_columns.append(column)

     # Initialiser une liste vide pour stocker les colonnes catégoriques
      categorical_columns = []


      for column in after_filtre.columns:
        # Vérifier si la colonne contient des valeurs catégoriques
        if after_filtre[column].dtype == 'object':
          after_filtre[column] = after_filtre[column].astype('category')
          categorical_columns.append(column)


      # Initialiser une liste vide pour stocker les colonnes booléennes
      bool_columns = []

      for column in after_filtre.columns:
        if pd.api.types.is_bool_dtype(after_filtre[column]):
          bool_columns.append(column)

      # Description de la base de donnée
      st.subheader("Data Description")
      st.write(after_filtre.describe())

      # Visualisation des données
      st.subheader("Data Visualization")
      type_graphique= st.selectbox("Choose the graphic type. ",("Correlation Matrix"  ,
                                                                                                               "Count Plot", "Pie Chart",
                                                                                                               "Bar Plot", "Scatter plot",
                                                                                                              "Line Chart", "Pair Plot",
                                                                                                               "Histogram", "Box Plot"))

      if(type_graphique == "Correlation Matrix"):
        correlation_matrix(after_filtre)
      elif(type_graphique == "Count Plot"):
        count_plot(after_filtre)
      elif(type_graphique == "Pie Chart"):
        pie_chart(after_filtre)
      elif(type_graphique == "Bar Plot"):
       bar_plot(after_filtre)
      elif(type_graphique == "Scatter plot"):
        scatter_plot(after_filtre)
      elif(type_graphique == "Line Chart"):
        line_chart(after_filtre)
      elif(type_graphique == "Pair Plot"):
        pair_plot(after_filtre)
      elif(type_graphique == "Histogram"):
        histogram(after_filtre)
      elif(type_graphique == "Box Plot"):
        box_plot(after_filtre)
      # Configurez votre clé API GPT-3
      openai.api_key = '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'
      qst= dataframe.head(20).to_string()
      #st.write(qst)
      st.title("AI Assistant(Only the 20 first lines are selected)")

      if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

      if "messages" not in st.session_state:
       st.session_state.messages = []

      for message in st.session_state.messages:
       with st.chat_message(message["role"]):
         msg = message["content"].replace(qst, "")
         st.markdown(msg)

      if prompt := st.chat_input("Any question about your data?"):
        # prompt= prompt + qst
        # st.session_state.messages.append({"role": "user", "content": prompt + qst})
        st.session_state.messages.append({"role": "user", "content": prompt + qst})
        with st.chat_message("user"):
          st.markdown(prompt)

        with st.chat_message("AI Assistant"):
          message_placeholder = st.empty()
          full_response = ""
          try:
            for response in openai.chat.completions.create(
              model=st.session_state["openai_model"],
              messages=[
                  {"role": m["role"], "content": m["content"]}
                  for m in st.session_state.messages
             ],
              stream=True,
            ):
             if response.choices[0].delta.content is not None:
              full_response += response.choices[0].delta.content
              message_placeholder.markdown(full_response + "▌")
          except Exception as e:
            st.error("Ahaha you want to go further? You will have to pay because it's no longer free")
          message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Si on est sur la page documentation
elif st.session_state['page'] == "documentation":
  st.title('Tutoriel')

  st.video("https://firebasestorage.googleapis.com/v0/b/hyphip-8ca89.appspot.com/o/3c1d1e279e.mp4?alt=media&token=b0a7e0e9-1978-49fc-9c9b-40b4a45aa09c")

# Page Radar
elif st.session_state['page'] == "radar":
    st.title("Google Scholar Radar")

    # Choix de l'option de recherche
    search_option = st.radio("Choisissez une option de recherche :", ["Chercheur", "Thème de recherche"])

    if search_option == "Chercheur":
        # Interface existante pour la recherche par chercheur
        search_term = st.sidebar.text_input("Search Researcher", placeholder="Enter a researcher's name")
        max_articles = st.sidebar.slider("Max number of articles", 1, 20, 5)

        if st.sidebar.button("Search"):
            if search_term.strip():
                with st.spinner("Fetching data..."):
                    result = search_scholar_with_h_index(search_term, max_articles)

                if result:
                    st.success(f"Researcher Found: {result['name']}")
                    st.write(f"Affiliation: {result['affiliation']}")
                    st.write(f"H-index: {result['h_index']}")

                    st.subheader(f"Top {max_articles} Articles:")
                    for idx, article in enumerate(result["articles"], 1):
                        st.write(f"{idx}. {article['title']} ({article['citations']} citations)")
            else:
                st.warning("Please enter a search term.")


    elif search_option == "Thème de recherche":
      # ✅ Interface Streamlit
      st.title("Radar Google Scholar avec Perplexity AI")

      # 🔍 **Recherche par Thème**
      search_theme = st.text_input("Entrez un thème de recherche", placeholder="Ex: Intelligence Artificielle")

      # 🔍 **Filtre sur le H-index**
      h_index_min = st.number_input("Filtrer les chercheurs avec un H-index supérieur à :", min_value=0, value=0, step=1)

      # ✅ Récupérer la liste standardisée des pays (en anglais)
      country_list = sorted([country.name for country in pycountry.countries])

      # 🔍 **Filtre par Pays** (Liste déroulante multiple)
      selected_countries = st.multiselect("Filtrer par pays :", country_list, default=[])

      if st.button("Rechercher les chercheurs"):
          if search_theme:
              with st.spinner("Recherche en cours sur Google Scholar..."):
                  authors_list, publications = search_scholars_from_theme(search_theme, max_results=100)

                  if authors_list and publications:
                      with st.spinner("Récupération des noms complets via Perplexity (avec vérification des publications)..."):
                          complete_names = get_scholar_names_perplexity(authors_list, publications)

                          if complete_names:
                              scholar_info_list = []
                              affiliations_list = []

                              with st.spinner("Recherche des profils Google Scholar et scraping des données..."):
                                  for full_name in complete_names.split("\n"):
                                      scholar_url = find_scholar_profile(full_name)

                                      if scholar_url:
                                          scholar_info = scrape_scholar_profile(scholar_url)
                                          scholar_info_list.append(scholar_info)

                                          if scholar_info["Affiliation"] != "Affiliation inconnue":
                                              affiliations_list.append(scholar_info["Affiliation"])
                                      else:
                                          scholar_info_list.append({
                                              "Nom": full_name,
                                              "Affiliation": "Non trouvée",
                                              "H-index": "Non disponible",
                                              "Profil": "Non disponible",
                                              "Adresse": "Non disponible",
                                              "Pays": "Non disponible"
                                          })

                              with st.spinner("Recherche des adresses et pays des affiliations via Perplexity..."):
                                  # Nettoyage et filtrage des affiliations
                                  affiliations_list = [clean_affiliation(scholar["Affiliation"]) for scholar in scholar_info_list if scholar["Affiliation"] != "Affiliation inconnue"]
                                  affiliations_list = list(filter(None, affiliations_list))  # Supprime les None
                                  
                                  # Étape 1 : Demander à Perplexity d'expliciter les abréviations
                                  print("Affiliations avant expansion:", affiliations_list)
                                  expanded_affiliations = expand_affiliation_abbreviations(affiliations_list)
                                  print("Abréviations développées:", expanded_affiliations)

                                  # Remplacer les abréviations par leur nom complet si possible
                                  affiliations_list = [expanded_affiliations.get(aff, aff) for aff in affiliations_list]

                                  # Étape 2 : Recherche des adresses avec les affiliations corrigées
                                  print("Affiliations envoyées à Perplexity:", affiliations_list)
                                  affiliation_data = get_affiliation_address_perplexity(affiliations_list)
                                  print("Données renvoyées par Perplexity:", affiliation_data)

                                  if not isinstance(affiliation_data, dict):
                                      affiliation_data = {}

                                  # Mise à jour des chercheurs avec adresses et pays
                                  for scholar in scholar_info_list:
                                      original_affiliation = scholar["Affiliation"]
                                      cleaned_affiliation = clean_affiliation(original_affiliation)

                                      if cleaned_affiliation and cleaned_affiliation in affiliation_data:
                                          scholar["Adresse"], scholar["Pays"] = affiliation_data[cleaned_affiliation]
                                      else:
                                          scholar["Adresse"], scholar["Pays"] = find_best_match(original_affiliation, affiliation_data)

                                      print("Chercheur mis à jour:", scholar)

                              # ✅ Conversion du H-index en numérique et suppression des valeurs non disponibles
                              df = pd.DataFrame(scholar_info_list)
                              df["Pays"] = df["Pays"].apply(standardize_country)
                              df["H-index"] = pd.to_numeric(df["H-index"], errors="coerce")  # Convertir en nombre
                              df = df.dropna(subset=["H-index"])  # Supprimer les NaN

                              # ✅ Filtrage selon le H-index minimum défini par l'utilisateur
                              df = df[df["H-index"] >= h_index_min]

                              # ✅ Filtrage selon les pays sélectionnés
                              if selected_countries:
                                df = df[df["Pays"].isin(selected_countries)]
                              # ✅ Affichage des chercheurs filtrés
                              st.subheader(f"Informations Complètes sur les Chercheurs (H-index ≥ {h_index_min})")
                              st.dataframe(df, use_container_width=True, hide_index=True)

                          else:
                              st.warning("Aucune information trouvée via Perplexity.")
                  else:
                      st.warning("Aucun chercheur trouvé pour ce thème.")
          else:
              st.warning("Veuillez entrer un thème avant de rechercher.")










#Si on est sur la page About us
elif st.session_state['page'] == "about":
  # Données des personnes
  personnes = [
    {"nom": "Sopegue Soro", "prenom": "Yaya", "role": "Chef de projet MOA", "image": "https://i.pinimg.com/564x/5b/3f/56/5b3f56d89084ff2bd55cb482b752186a.jpg"},
    {"nom": "Chahet", "prenom": "Sid Ali", "role": "Chef de projet MOE", "image": "https://i.pinimg.com/564x/28/c8/f2/28c8f26756e59662e3cbcec3e8ac5922.jpg"},
    {"nom": "Camara", "prenom": "Aichetou", "role": "Architecte projet", "image": "https://i.pinimg.com/564x/3d/85/96/3d85965b24cfec4339cbe2661275bae5.jpg"},
    {"nom": "Belfekroun", "prenom": "Charaf", "role": "Expert data analyste", "image": "https://i.pinimg.com/564x/38/f6/9e/38f69e68a850e021a9b963f35fb80424.jpg"}
  ]

  # Affichage des divs
  for i in range(0, len(personnes), 2):
    col1, col2 = st.columns(2)  # Création de deux colonnes
    with col1:
        col11, col12 = st.columns(2)
        with col11:
          st.image(personnes[i]["image"], width=200)
        with col12:
          st.header(f"{personnes[i]['prenom']} {personnes[i]['nom']}")
          st.subheader(personnes[i]["role"])

    with col2:
      col21, col22 = st.columns(2)
      with col21:
        st.image(personnes[i+1]["image"], width=200)
      with col22:
        st.header(f"{personnes[i+1]['prenom']} {personnes[i+1]['nom']}")
        st.subheader(personnes[i+1]["role"])

