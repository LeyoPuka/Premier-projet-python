
from datetime import date, timedelta
from typing import List, Optional
import pandas as pd
import streamlit as st
from .utils import sanitize_columns

# Optional deps
try:     # Ce bloc teste si la bibliothèque yfinance est disponible dans ton environnement Python. 
    import yfinance as yf  # type: ignore
    HAS_YF = True   
except Exception:
    HAS_YF = False

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMZN", "^GSPC"] 

@st.cache_data(show_spinner=False)     # Elle met en cache le résultat de la fonction. L'animation de 'chargement en cours' est desactivée.
def fetch_prices_yf(tickers: List[str], start: date, end: date) -> pd.DataFrame:  # On définit le début et la fin de données avec 'start' et 'end'.
   if not HAS_YF: raise RuntimeError("yfinance non installé.")   # Si yfinance n'est pas disponible, la fonction s'arrete.

data = yf.download(tickers=tickers, start=start, end=end + timedelta(days=1),   # Téléchargement de données.
                       progress=False, auto_adjust=True, threads=True)
    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:  # Test servant à garder uniquement la colonne “Adj Close” 
        prices = data["Adj Close"].copy()                               # (prix ajusté, plus fiable pour les analyses).
    else:
        prices = data.copy()
    prices = prices.dropna(how="all")    # Supprime les lignes vides
    prices.index = pd.to_datetime(prices.index).tz_localize(None)  # Retire les informations de fuseau horaire pour éviter des erreurs d’alignement temporel.
    return sanitize_columns(prices)   # La fonction retourne un DataFrame avec des index (dates), tickers (ex. AAPL, MSFT) et des prix ajustés.

def load_from_csv(upload) -> pd.DataFrame:
    df = pd.read_csv(upload)   # Lit un fichier CSV chargé. 
    if "Date" not in df.columns:   # Le fichier doit obligatoirement avoir une colonne nommée "Date". 
        raise ValueError("CSV must contain a 'Date' column.")   # Sinon, la fonction s’arrête et affiche une erreur explicite.     
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")   # Transforme la colonne "Date" (souvent sous forme de texte) en datetime.
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()  # Les dates deviennent l’axe principal du tableau, comme dans les données financières classiques.
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")   # Tente de convertir chaque colonne en float. S’il y a du texte (ex. "N/A", "--"), il devient NaN (valeur manquante).
    return sanitize_columns(df.dropna(how="all"))       # Cela permet d’éviter des erreurs lors des calculs.
