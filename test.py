import requests
import pandas as pd
from bs4 import BeautifulSoup

# URL cible
url = "https://scholar.google.fr/citations?hl=fr&user=ek6HUpgAAAAJ"

# Récupérer le contenu de la page
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
response = requests.get(url, headers=headers)

# Vérifier si la requête a réussi
if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")

    # Trouver la première table
    table = soup.find("table")

    # Extraire les données de la table
    data = []
    if table:
        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all(["td", "th"])
            cols = [col.text.strip() for col in cols]
            data.append(cols)

        # Convertir en DataFrame
        df = pd.DataFrame(data)
        print(df)
        print(soup.find("td", class_="gsc_rsb_std"))