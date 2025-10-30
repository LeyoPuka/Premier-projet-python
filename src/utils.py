
from typing import List
import numpy as np
import pandas as pd

def sanitize_columns(df: pd.DataFrame, sep: str = "_") -> pd.DataFrame:  # Fonction qui nettoie et normalise les noms de colonnes d’un DataFrame pandas.
    if isinstance(df.columns, pd.MultiIndex):   # On vérifie si df.columns est un MultiIndex (colonnes à plusieurs niveaux).
        df.columns = [sep.join(map(str, lvl)) for lvl in df.columns.values]  # Pour chaque tuple de colonnes (lvl), on convertit chaque élément en chaîne (map(str, lvl)),
    else:
        df.columns = [str(c) for c in df.columns]  # Si les colonnes ne sont pas un MultiIndex, on s’assure juste qu’elles sont toutes des chaînes (str).
    return df

def pct_returns(prices: pd.DataFrame) -> pd.DataFrame:  # La fonction pct_returns calcule les rendements journaliers (ou périodiques) en pourcentage d’un DataFrame contenant des prix d’actifs.
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)  # gère les divisions par zéro, et fillna(0.0) évite les valeurs manquantes.

def to_weights_equal(n: int) -> np.ndarray:  # Fonction qui crée un vecteur de poids égaux.
    return np.ones(n)/n if n>0 else np.array([])
