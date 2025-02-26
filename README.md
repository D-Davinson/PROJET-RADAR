#  📡 _PROJECT_ - RADAR - Researcher Automatic Detection & Address Retrieval

## 💫 Objective

### RADAR automates the search for researchers based on a research topic or a specific researcher's name.
It utilizes **Google Scholar** (via SerpAPI) and **Perplexity AI** to retrieve:

- Researcher profiles
- H-index values
- Affiliations
- Institution addresses
- Geolocation mapping of researchers

The system also supports:

✔ Filtering (H-index, country, location-based search)

✔ Automated researcher name validation via Perplexity AI

✔ Visualization on an interactive world map

## 📦 Libraries Used :

### 🔧 Dependencies
| Type of package | Import |
|-|-|
|Standard| ```os``` , ```re``` , ```time``` , ```random``` , ```json``` |
|PyPi| ```streamlit``` , ```scholarly``` , ```google_search_result``` ,```streamlit-folium```, ```folium``` ,```python-dotenv```,```certifi```, ```geopy``` , ```pandas``` , ```pycountry``` , ```openai``` |

### 🔧 Installation

To install all dependencies, run:

```
$ pip install streamlit scholarly google-search-results streamlit-folium folium python-dotenv certifi geopy pandas pycountry openai
```

## 💡 Project Structure

### 📂 Main Components

| Functions | _Description_ |
|-|-|
|```search_scholars_from_theme(theme,max_result)```|Extracts authors and publications related to a research theme.|
|```get_scholar_names_perplexity(authors, publications)```|Retrieves full researcher names while verifying publication authorship.|
|```find_scholar_profile(full_name)```|Finds a researcher's Google Scholar profile via SerpAPI.|
|```get_scholar_profile_serpapi(scholar_url)```|Extracts affiliation and H-index from the profile.|
|```clean_affiliation(affiliation)```|Clean affiliation by removing titles and departments to normalize the institution names|
|```parse_affiliation_addresses(response_text)```|Convert raw response from Perplexity into a dictionary ```{Affiliation: (Address, Country)}```|
|```get_affiliation_address_perplexity(affiliations)```|Uses Perplexity AI to search for the full address and country of the listed institutions using bi-word indexing|
|```find_best_match(original_affiliation, affiliation_data)```|Find the best approximate match between an original affiliation and Perplexity results|
|```parse_expanded_affiliations(response_text)```|Convert raw response from Perplexity into a dictionary ```{Abbreviation: Full name}```|
|```expand_affiliation_abbreviations(affiliations)```|Uses Perplexity AI to obtain the full name of abbreviated affiliations|
|```standardize_country(country_name)```|Convert variants of countries to their official name using pycountry|
|```calculate_h_index(publications)```|	Computes the H-index from a researcher’s list of publications|
|```search_scholar_with_h_index(query, max_articles)```|Retrieves a researcher’s profile and top articles based on their H-index|
|```get_coordinates_from_address(address)```|Converts an address into latitude and longitude using Geopy|
|```display_researcher_map(df, user_lat, user_lon, search_radius)```|Displays researchers on an interactive world map, filtering them by distance|


## 💫 Implementation

### Run

To compile the program :
```
$ clone https://github.com/D-Davinson/PROJET-RADAR.git
```
_Open the reposotory PROJET-RADAR_

```
$ streamlit run radar.py
```

## 🤖 Deployment

The deployment is done on Streamlit Cloud and is accessible at the following address : https://datawizv1.streamlit.app/ 

## 👁️ View

<img width="1509" alt="Capture d’écran 2025-02-26 à 12 50 24" src="https://github.com/user-attachments/assets/acb5ed12-4144-43ce-a145-4cd07ebfee27" />
<img width="1502" alt="Capture d’écran 2025-02-26 à 13 06 53" src="https://github.com/user-attachments/assets/c51034cc-bd44-4c5e-a6ea-aed75c385712" />
<img width="1510" alt="Capture d’écran 2025-02-26 à 12 51 32" src="https://github.com/user-attachments/assets/5a0f9ed4-63eb-44d6-a585-b5c321ee1d5e" />
<img width="1512" alt="Capture d’écran 2025-02-26 à 13 11 07" src="https://github.com/user-attachments/assets/fef4fa6f-fd8e-4019-b17b-aa6009e52a6b" />

## 🧑🏽‍💻 Authors

- [@Davinson DOGLAS PRINCE](https://github.com/D-Davinson)
- [@Théo POSENEL](https://github.com/TheoPosenel)
- [@Nahla HAMLETTE](https://github.com/Nahla213)
