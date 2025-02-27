import sys
from datetime import datetime
import re
import time
import geocoder
import openai
from scholarly import scholarly
import re
import pycountry
from serpapi import GoogleSearch
import re
from dotenv import load_dotenv
import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from geopy.distance import geodesic

# Importing libraries for data processing

import streamlit as st # Streamlit library

# Data visualization library
import seaborn as sns
import matplotlib.pyplot as plt

# Data processing library
import pandas as pd
import csv

load_dotenv()

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
          color: white;
          margin : 0px 0px -10px;
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
    .big-label {
            font-size: 17px;
            font-weight: normal;
            padding : -50px 0 0;
            margin : 0px 0px -4px;
        }
    </style>
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





###########################################  WORK M2MIAI -> PROJECT RADAR  ###########################################################



############################################### API KEY PERPLEXITY ###################################################################


client = openai.OpenAI(api_key=os.getenv("PERPLEXITY_API_KEY"), base_url="https://api.perplexity.ai")

######################################################################################################################################





###########################################  STEP 1 : SEARCH THEME FROM GOOGLE SCHOLAR  ##############################################
def search_scholars_from_theme(theme, max_results):
    """Search for publications on Google Scholar based on a topic extract the involved authors.
    Return two lists : 
      - A unique list of authors
      - A list of publications"""
    try:
        search_query = scholarly.search_pubs(theme)
        authors_list = set()
        publications_list = []  # Stock all publications

        for _ in range(max_results):  # Limit for time execution
            try:
                publication = next(search_query)
                title = publication['bib'].get('title', "Unknown title")
                authors = publication['bib'].get('author', [])

                if isinstance(authors, str):  # If it's a string, convert it into a list
                    authors = authors.split(", ")

                for author in authors:
                    authors_list.add(author.strip())

                publications_list.append(title)  # Add the publication title

            except StopIteration:
                break
        
        return list(authors_list), publications_list  # Return two distinct values

    except Exception as e:
        print(f"Error during search on Google Scholar : {e}")
        return [], []



###################################  STEP 2: USE PERPLEXITY FOR FOUND ALL THE NAMES AND SURNAME  ##################################
def get_scholar_names_perplexity(authors, publications):
    """Uses Perplexity AI to retrieve ONLY the full names of researchers, verifying that these names are indeed associated with the found publications."""
    if not authors or not publications:
        return None

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert in academic research. "
                "Your task is to retrieve the **full names** of the listed researchers "
                "and **verify that they are indeed the authors** of the publications found on Google Scholar."
            ),
        },
        {   
            "role": "user",
            "content": (
                "Perform a **search on Google Scholar** to retrieve the **full names** "
                "of the researchers listed below, based on their work.\n\n"
                "**⚠️ It is crucial to verify that these researchers are indeed the authors of the publications listed below.**\n"
                "**If a found name is not linked to the publications, DO NOT INCLUDE IT.**\n\n"
                f"📚 **Found publications:** {', '.join(publications)}\n\n"
                f"👨‍🔬 **List of researchers extracted from Google Scholar:** {', '.join(authors)}\n\n"
                "**Return only the list of valid full names, one per line.**"
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



########################################## STEP 3: FOUND THE PROFILE WITH SCHOLARLY  ###########################################################
def find_scholar_profile(full_name):
    """Search for a researcher on Google Scholar using scholarly and return the URL of their profile if found..."""
    search_query = scholarly.search_author(full_name)
    try:
        author = next(search_query)  # Take the first result
        scholar_id = author['scholar_id']
        return f"https://scholar.google.com/citations?user={scholar_id}"
    except StopIteration:
        return None




#####################################  STEP 4: SCRAP H-INDEX AND AFFILIATION  #################################################################

def get_scholar_profile_serpapi(scholar_url):
    """Use SerpAPI to retrieve the affiliation and H-index of a researcher via their Google Scholar profile."""

    # # Extract ID user with URL profile GOOGLE SCHOLAR using REGEX syntax
    match = re.search(r"user=([a-zA-Z0-9_-]+)", scholar_url)
    if not match:
        return {
            "Name": "Error",
            "Affiliation": "Error",
            "H-index": "Error",
            "Profile": scholar_url
        }
    
    scholar_id = match.group(1)

    params = {
        "engine": "google_scholar_author",
        "author_id": scholar_id,
        "api_key": os.getenv("SERPAPI_KEY"),
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    if "author" in results:
        profile = results["author"]
        full_name = profile.get("name", "Unknown name")
        affiliation = profile.get("affiliations", "Unknown affiliation")

        # Extract H-index
        h_index = "Not available"
        cited_by_table = results.get("cited_by", {}).get("table", [])

        for entry in cited_by_table:
            if "h_index" in entry:
                h_index = entry["h_index"].get("all", "Not available")
                break  # Stop loop when we found the info

        return {
            "Name": full_name,
            "Affiliation": affiliation,
            "H-index": h_index,
            "Profile": scholar_url
        }

    return {
        "Name": "Error",
        "Affiliation": "Error",
        "H-index": "Error",
        "Profile": scholar_url
    }


###################################  STEP 5: CLEAN THE RESULT GIVE BY PERPLEXITY -> INDEXATION RESEARCH ################################# 
def clean_affiliation(affiliation):
    """Clean affiliation by removing titles and departments to normalize the institution names."""
    if not affiliation or affiliation.lower() in ["Not found", "Unknown affiliation"]:
        return None

    # Remove academic titles and departments
    remove_words = ["PhD Candidate", "Professor of", "Department of", "Faculty of", "Institute of", "Lab of", "Graduate Student"]
    
    for word in remove_words:
        affiliation = re.sub(rf"\b{word}\b", "", affiliation, flags=re.IGNORECASE).strip()

    return affiliation.strip()



#####################################  STEP 6: PARSE AFFILIATION ADDRESSES ############################################################# 

def parse_affiliation_addresses(response_text):
    """Convert raw response from Perplexity into a dictionary {Affiliation: (Address, Country)}."""
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




###################################  STEP 7: GET ADDRESSES WITH PERPLEXITY -> SCRAP INTELLIGENT / BI-WORD ######################################### 
def get_affiliation_address_perplexity(affiliations):
    """Uses Perplexity AI to search for the full address and country of the listed institutions using bi-word indexing."""
    if not affiliations:
        return {}

    # Create bi-words for improve the accuracy of research
    bi_word_affiliations = [f"{affiliations[i]} {affiliations[i+1]}" for i in range(len(affiliations)-1)]
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert in academic address retrieval. "
                "Your task is to find the **complete address** "
                "of the listed institutions/societies by performing **Google searches**."
                "Use bi-word indexing to refine the search and improve accuracy."
                "If you cannot find the exact address, **at least determine the country**."
            ),
        },
        {   
            "role": "user",
            "content": (
                "Use Google to find the **full address** of each listed affiliation.\n\n"
                "**Return the result in this STRICT format:**\n"
                "Institution/Society | Address (if found) | Country (mandatory)\n"
                "------------------------------------------------\n"
                f"{', '.join(bi_word_affiliations)}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model="sonar-pro",
        messages=messages,
    )

    if response:
        raw_text = response.choices[0].message.content
        return parse_affiliation_addresses(raw_text)
    return None



#####################################  STEP 8: MATCHING PROCESS BETWEEN AFFILIATION DATA AND PERPLEXITY ################################ 

# Function to find the best match.
def find_best_match(original_affiliation, affiliation_data):
    """Find the best approximate match between an original affiliation and Perplexity results."""
    for key in affiliation_data.keys():
        if key.lower() in original_affiliation.lower():  # Approximate verification
            return affiliation_data[key]  # Returns (address, country)

    return ("Not available", "Not available")




#####################################  STEP 9: PARSE ABBREVIATED AFFILIATION DATA ################################################ 

def parse_expanded_affiliations(response_text):
    """Convert raw response from Perplexity into a dictionary {Abbreviation: Full name}."""
    abbreviation_map = {}
    lines = response_text.strip().split("\n")

    for line in lines:
        parts = line.split("|")
        if len(parts) == 2:
            abbreviation = parts[0].strip()
            full_name = parts[1].strip()
            abbreviation_map[abbreviation] = full_name

    return abbreviation_map



#####################################  STEP 10 : PROCESS FOR GIVE FULL NAME TO ABBREVIATED AFFILIATION DATA ############################### 

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


#####################################  STEP 11 : STANDARD COUNTRY LIBRARIES ############################################ 


def standardize_country(country_name):
    """Convert variants of countries to their official name using pycountry."""
    try:
        return pycountry.countries.lookup(country_name).name
    except LookupError:
        return country_name 


######################################################################################################################



def calculate_h_index(publications):
    publications_sorted = sorted(publications,key=lambda x: x.get("num_citations", 0), reverse=True)
    h_index = 0
    for idx, pub in enumerate(publications_sorted, start=1):
        if pub.get("num_citations", 0) >= idx:
            h_index = idx
        else:
            break
    return h_index


# Function to search for an author and their H-index
def search_scholar_with_h_index(query, max_articles=5):
    try:
        search_query = scholarly.search_author(query)
        author = scholarly.fill(next(search_query))  # Retrieve the first result
        
        # Main Information
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

##################################################################################################################################






def get_coordinates_from_address(address):
    """Convert latitude and longitude with Geopy (Nominatim)."""
    geolocator = Nominatim(user_agent="researcher_locator")
    try:
        location = geolocator.geocode(address, timeout=10)
        if location:
            return location.latitude, location.longitude
    except GeocoderTimedOut:
        st.warning(f"⏳ Timeout for the adresse : {address}")
    return None, None

def display_researcher_map(df, user_lat, user_lon, search_radius):
    """View the world map with the position of the researchers."""
    if df.empty:
        st.warning("No searchers found to display on map.")
        return

    # Center to the user
    m = folium.Map(location=[user_lat, user_lon], zoom_start=3)

    # Add blue cercle
    folium.Circle(
        location=[user_lat, user_lon],
        radius=search_radius * 1000,
        color="blue",
        fill=True,
        fill_opacity=0.2,
    ).add_to(m)

    # Add searchers to the map
    for _, row in df.iterrows():
        if pd.notnull(row["Address"]):  # Verify the adress is available
            lat, lon = row.get("Latitude"), row.get("Longitude")

            # If no latitude/longitude, so search for
            if pd.isnull(lat) or pd.isnull(lon):
                lat, lon = get_coordinates_from_address(row["Address"])
                df.at[_, "Latitude"], df.at[_, "Longitude"] = lat, lon  # Met à jour le DataFrame

            if lat and lon:
                researcher_loc = (lat, lon)
                user_loc = (user_lat, user_lon)

                # Calculate the distance between user and location
                distance_km = geodesic(user_loc, researcher_loc).km

                # Check if the searcher is within the search radius
                if distance_km <= search_radius:
                    folium.Marker(
                        location=researcher_loc,
                        popup=f"<b>{row['Name']}</b><br>{row['Affiliation']}<br><b>H-index:</b> {row['H-index']}<br><b>Address:</b> {row['Address']}<br><b>Distance:</b> {distance_km:.1f} km",max_width=75,
                        icon=folium.Icon(color="red", icon="glyphicon-user"),
                    ).add_to(m)

    # Viewing the map in Streamlit
    st.subheader("🌍 Obtained Results As a Map (Based on Radius Informations)")
    folium_static(m)




# Function to get the user's location dynamically
def get_user_location():
    try:
        g = geocoder.ip("me")  # Get location from IP
        if g.ok:
            return g.latlng  # Returns (latitude, longitude)
    except:
        pass
    return (48.8566, 2.3522)  # Default to Paris if location fails




##################################################################################################################################
##################################################################################################################################



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
  col21, col22, col23 ,col24 = st.columns(4)
  with col21:
    st.button("Load Your Data", on_click=load_page, key=load_page)
  with col22:
    st.button("🛰️ Radar", on_click=radar_page)
  with col23:
    st.button("Documentation", on_click=documentation_page)
  with col24:
    st.button("About Us", on_click=about_page)



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
  st.markdown("[🌐 Accédez à notre GitHub](https://github.com/D-Davinson/PROJET-RADAR)")



# Radar Page
# Initialize session state for storing the researcher dataframe if not already set
if "scholar_df" not in st.session_state:
    st.session_state["scholar_df"] = None

elif st.session_state['page'] == "radar":
    st.title("🕵🏽‍♂️ - Welcome to RADAR project!")

    # Choice of search option (By Topic or By Name)
    st.markdown('<p class="big-label">Select an option of your research :</p>', unsafe_allow_html=True)
    search_option = st.radio(" ", ["By Topic", "By Name"], horizontal=True)

    if search_option == "By Name":
        # Interface for searching a researcher by name
        st.subheader("🔍 Search Researcher's")
        st.markdown("<p class='big-label'>Enter a researcher's name</p>", unsafe_allow_html=True)
        search_term = st.text_input(" ", placeholder="Ex: Michael Jordan")
        st.markdown("<p class='big-label'>Max number of articles</p>", unsafe_allow_html=True)
        max_articles = st.slider(" ", 1, 20, 5)

        if st.button("Launch ↯"):
            if search_term.strip():
                with st.spinner("Fetching data..."):
                    result = search_scholar_with_h_index(search_term, max_articles)

                if result:
                    st.success(f"Researcher Found: {result['name']}")
                    st.success(f"Affiliation: {result['affiliation']}")
                    st.success(f"H-index: {result['h_index']}")
                    st.subheader(f"Top {max_articles} Articles:")
                    for idx, article in enumerate(result["articles"], 1):
                        st.success(f"{idx}. {article['title']} ({article['citations']} citations)")

    elif search_option == "By Topic":
        # Interface for searching researchers by topic
        st.title("🧭 - Finder's compass")
        col1, col2 = st.columns([3, 1])
        col3, col4 = st.columns([2, 2])
        col5, col6 = st.columns([2, 2])  
       
        # Search by Topic
        with col1:
            st.markdown('<p class="big-label">Enter a research topic:</p>', unsafe_allow_html=True)
            search_theme = st.text_input(" ", placeholder="Ex: Artificial Intelligence")

        # Filter by H-index
        with col2:
            st.markdown('<p class="big-label">Filter researchers with an H-index > = </p>', unsafe_allow_html=True)
            h_index_min = st.number_input(" ", min_value=0, value=0, step=1)

        # Retrieve standardized country list
        with col3:
            st.markdown('<p class="big-label">Filter by country:</p>', unsafe_allow_html=True)
            country_list = sorted([country.name for country in pycountry.countries])
            selected_countries = st.multiselect(" ", country_list, default=[])

        # Filter by number of publications
        with col4:
            st.markdown('<p class="big-label">Filter by publications:</p>', unsafe_allow_html=True)
            max_publications = st.number_input(" ", min_value=1, value=35, step=1)

        # User's location and search radius
        st.subheader("Your current position")
        user_lat, user_lon = get_user_location()
        with col5:
           st.markdown('<p class="big-label">Latitude</p>', unsafe_allow_html=True)
           user_lat = st.number_input(" ", value=user_lat, format="%.6f") 
        with col6:
          st.markdown('<p class="big-label">Longitude</p>', unsafe_allow_html=True)
          user_lon = st.number_input(" ", value=user_lon, format="%.6f")  
        st.markdown('<p class="big-label">Your perimeter (km)</p>', unsafe_allow_html=True)
        search_radius = st.slider(" ", 10, 50000, 100) 

        if st.button("Launch ↯"):
            if search_theme:
                with st.spinner("Searching Google Scholar..."):
                    authors_list, publications = search_scholars_from_theme(search_theme, max_results=max_publications)

                    if authors_list and publications:
                        with st.spinner("Retrieving full names via Perplexity..."):
                            complete_names = get_scholar_names_perplexity(authors_list, publications)

                            if complete_names:
                                scholar_info_list = []
                                affiliations_list = []

                                with st.spinner("Searching for Google Scholar profiles and extracting data..."):
                                    for full_name in complete_names.split(" "):
                                        scholar_url = find_scholar_profile(full_name)

                                        if scholar_url:
                                            scholar_info = get_scholar_profile_serpapi(scholar_url)
                                            scholar_info_list.append(scholar_info)

                                            if scholar_info["Affiliation"] != "Unknown affiliation":
                                                affiliations_list.append(scholar_info["Affiliation"])
                                        else:
                                            scholar_info_list.append({
                                                "Name": full_name,
                                                "Affiliation": "Not found",
                                                "H-index": "Not available",
                                                "Address": "Not available",
                                                "Country": "Not available"
                                            })

                                with st.spinner("Fetching addresses and countries for affiliations..."):
                                    affiliations_list = list(filter(None, affiliations_list))
                                    affiliation_data = get_affiliation_address_perplexity(affiliations_list)

                                    if not isinstance(affiliation_data, dict):
                                        affiliation_data = {}

                                    for scholar in scholar_info_list:
                                        original_affiliation = scholar["Affiliation"]
                                        scholar["Address"], scholar["Country"] = affiliation_data.get(original_affiliation, ("Not available", "Not available"))

                                # Convert H-index to numeric and filter unavailable values
                                df = pd.DataFrame(scholar_info_list)
                                df["Country"] = df["Country"].apply(standardize_country)
                                df["H-index"] = pd.to_numeric(df["H-index"], errors="coerce")
                                df = df.dropna(subset=["H-index"])

                                # Apply H-index filter
                                df = df[df["H-index"] >= h_index_min]

                              #  if "Latitude" not in df.columns or "Longitude" not in df.columns:
                              #      df["Latitude"], df["Longitude"] = None, None

                              #  for index, row in df.iterrows():
                              #      if pd.isnull(row["Latitude"]) or pd.isnull(row["Longitude"]):
                              #          lat, lon = get_coordinates_from_address(row["Address"])
                              #          df.at[index, "Latitude"] = lat
                              #          df.at[index, "Longitude"] = lon


                                # Sort researchers by H-index
                                df = df.sort_values(by="H-index", ascending=False)

                                # Apply country filter
                                if selected_countries:
                                    df = df[df["Country"].isin(selected_countries)]
                                
                                # Remove unnecessary columns
                                df = df.drop(columns=["Profile"], errors="ignore")
                                df = df.drop_duplicates(subset=["Name", "H-index"])
                                #df = df.drop(columns=["Latitude"], errors="Ignore")
                                #df = df.drop(columns=["Longitude"], errors="Ignore")

                                # Store results in session state to persist across changes

                                st.session_state["scholar_df"] = df.copy()

                            else:
                                st.warning("No information found via Perplexity.")
                    else:
                        st.warning("No researcher found for this topic.")
            else:
                st.warning("Please enter a topic before searching.")


    # Ensure that the results table persists and remains visible
    if st.session_state["scholar_df"] is not None:
        st.subheader("📋 Table Of Obtained Researchers")
        st.dataframe(st.session_state["scholar_df"], use_container_width=True, hide_index=True)

        # Display the map ONLY if a new search has been made
        #if st.button("Show Map 🌍"):
        #  with st.spinner("Rendering the map..."):
        #     display_researcher_map(st.session_state["scholar_df"], user_lat, user_lon, search_radius)




#Si on est sur la page About us
elif st.session_state['page'] == "about":
  # Données des personnes
  personnes = [
    {"nom": "Sopegue Soro", "prenom": "Yaya", "role": "Chef de projet MOA", "image": "https://i.pinimg.com/564x/5b/3f/56/5b3f56d89084ff2bd55cb482b752186a.jpg"},
    {"nom": "Chahet", "prenom": "Sid Ali", "role": "Chef de projet MOE", "image": "https://i.pinimg.com/564x/28/c8/f2/28c8f26756e59662e3cbcec3e8ac5922.jpg"},
    {"nom": "Camara", "prenom": "Aichetou", "role": "Architecte projet", "image": "https://i.pinimg.com/564x/3d/85/96/3d85965b24cfec4339cbe2661275bae5.jpg"},
    {"nom": "Belfekroun", "prenom": "Charaf", "role": "Expert data analyste", "image": "https://i.pinimg.com/564x/38/f6/9e/38f69e68a850e021a9b963f35fb80424.jpg"},
    {"nom": "DOGLAS PRINCE", "prenom": "Davinson", "role": "Lead Developpeur IA", "image": "assets/Profil_davinson.png"},
    {"nom": "POSENEL", "prenom": "Théo", "role": "Developpeur Front-End", "image": "assets/Profil_theo.webp"},
    {"nom": "HAMLETTE", "prenom": "Nahla", "role": "Data scientist", "image": "assets/Profil_nahla.webp"},

  ]

  # Affichage des divs
  for i in range(0, len(personnes), 2):
    col1, col2 = st.columns(2)  # Création de deux colonnes
    with col1:
        col11, col12 = st.columns(2)
        with col11:
          st.image(personnes[i]["image"], width=300)
        with col12:
          st.header(f"{personnes[i]['prenom']} {personnes[i]['nom']}")
          st.subheader(personnes[i]["role"])

    if i + 1 < len(personnes):  # Vérifier que i+1 est dans la plage valide
        with col2:
            col21, col22 = st.columns(2)
            with col21:
                st.image(personnes[i+1]["image"], width=300)
            with col22:
                st.header(f"{personnes[i+1]['prenom']} {personnes[i+1]['nom']}")
                st.subheader(personnes[i+1]["role"])