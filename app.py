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
# importation des librairies pour le traitement des données

import streamlit as st # librairie de streamlit

# librairie de visualisation de données
import seaborn as sns
import matplotlib.pyplot as plt

# librairie de traitement de données
import pandas as pd
import csv

# Configurez votre clé API GPT-4o
openai.api_key = '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'

OPENAI_API_KEY = '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'


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

def calculate_h_index(publications):
    publications_sorted = sorted(publications,key=lambda x: x.get("num_citations", 0), reverse=True)
    h_index = 0
    for idx, pub in enumerate(publications_sorted, start=1):
        if pub.get("num_citations", 0) >= idx:
            h_index = idx
        else:
            break
    return h_index

def get_random_location():
    """Génère une position aléatoire réaliste sur Terre (évite l'océan)."""
    lat = random.uniform(-60, 75)  # Limité pour éviter l'Antarctique
    lon = random.uniform(-180, 180)
    return round(lat, 6), round(lon, 6)


author_cache = {}

def get_random_location():
    """Génère une position aléatoire réaliste sur Terre (évite l'océan)."""
    lat = random.uniform(-60, 75)  # Limité pour éviter l'Antarctique
    lon = random.uniform(-180, 180)
    return round(lat, 6), round(lon, 6)

def get_scholar_profile_url(author_name):
    """Recherche l'auteur et récupère son URL Google Scholar. Utilise un cache."""
    if author_name in author_cache:
        return author_cache[author_name]

    search_query = scholarly.search_author(author_name)
    try:
        author_info = next(search_query)
        scholar_id = author_info['scholar_id']
        profile_url = f"https://scholar.google.fr/citations?hl=fr&user={scholar_id}"
        author_cache[author_name] = profile_url  # ✅ Stocker en cache
        return profile_url
    except StopIteration:
        return None

def get_h_index_from_scholar(profile_url):
    """Scrape le h-index depuis le profil Google Scholar. Utilise un cache."""
    if profile_url in author_cache:
        return author_cache[profile_url]

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(profile_url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        stats = soup.find_all("td", class_="gsc_rsb_std")
        h_index = int(stats[2].text.strip()) if len(stats) >= 2 else 0
        author_cache[profile_url] = h_index  # ✅ Stocker en cache
        return h_index
    return 0

# ✅ Optimisation : Recherche groupée des auteurs avec Google Search (évite plusieurs requêtes Scholar)
def get_author_profiles_from_google(authors):
    """Recherche les profils des auteurs sur Google en une seule requête."""
    query = f'site:scholar.google.com {" OR ".join([f"{author}" for author in authors])}'
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(search_url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        author_profiles = {}
        for link in soup.find_all("a", href=True):
            if "scholar.google.com/citations" in link["href"]:
                name = link.text.strip()
                author_profiles[name] = link["href"]
        return author_profiles
    return {}

# ✅ Optimisation des requêtes par thème
def search_publications_by_theme(theme, max_results=5):
    """Recherche des publications sur Google Scholar avec un nombre limité de requêtes."""
    try:
        search_query = scholarly.search_pubs(theme)
        publications = [next(search_query) for _ in range(max_results)]
        return publications
    except StopIteration:
        st.warning("Aucun résultat trouvé pour ce thème.")
        return []
  

def search_scholar_with_h_index(query, max_articles=5):
    """Recherche un auteur sur Google Scholar et récupère son h-index et ses publications."""
    try:
        search_query = scholarly.search_author(query)
        author = scholarly.fill(next(search_query))  # Récupère le premier résultat
        
        # Informations principales
        author_name = author.get("name", "N/A")
        affiliation = author.get("affiliation", "N/A")
        h_index = author.get("hindex", 0)  # Directement depuis l'API au lieu de recalculer
        publications = author.get("publications", [])
        
        # Publications limitées au nombre demandé
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
        st.title("Recherche par Thème de Projet")
        search_theme = st.text_input("Recherchez un thème", placeholder="Entrez un thème, ex: Intelligence Artificielle")
        h_index_threshold = st.number_input("Seuil minimum de h-index", min_value=0, value=10, step=1)

        if search_theme:
            try:
                search_query = scholarly.search_pubs(search_theme)
                results = []
                author_names = []  # ✅ Stocker les auteurs pour une recherche groupée

                for _ in range(5):  # ✅ Limite à 5 publications pour éviter le bannissement
                    try:
                        publication = next(search_query)
                        title = publication['bib']['title']
                        authors = publication['bib']['author'][:3]  # ✅ Limite à 3 auteurs

                        author_names.extend(authors)  # ✅ Ajout pour la recherche groupée

                    except StopIteration:
                        break

                # ✅ Recherche groupée sur Google pour trouver les profils des auteurs
                author_profiles = get_author_profiles_from_google(author_names)

                for author in author_names:
                    profile_url = author_profiles.get(author, "Non disponible")

                    if profile_url != "Non disponible":
                        h_index = get_h_index_from_scholar(profile_url)
                        latitude, longitude = get_random_location()
                    else:
                        h_index = 0
                        latitude, longitude = None, None

                    if h_index >= h_index_threshold:
                        results.append({
                            "chercheur": author,
                            "h-index": h_index,
                            "titre": title,
                            "latitude": latitude,
                            "longitude": longitude
                        })

                if results:
                    df = pd.DataFrame(results)
                    df["h-index"] = df["h-index"].astype(str)  
                    st.success(f"Résultats pour le thème : {search_theme} avec h-index > {h_index_threshold}")
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.warning(f"Aucun chercheur trouvé avec un h-index supérieur à {h_index_threshold}.")

            except Exception as e:
                st.error(f"Une erreur s'est produite lors de la recherche : {e}")

        else:
            st.info("Entrez un thème pour afficher les résultats.")





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


