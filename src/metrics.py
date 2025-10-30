
from typing import Tuple
import numpy as np
import pandas as pd

def annualized_return(series: pd.Series, periods_per_year: int = 252) -> float:  # sert à calculer le rendement annualisé (CAGR) à partir d’une série temporelle
    s = series.dropna()    # enlève les valeurs manquantes
    if s.empty: return float("nan")    # Si la série devient vide afficher 'nan'.
    total = s.iloc[-1] / s.iloc[0] - 1.0  # Donne le rendement total cumulé sur la période complète.
    years = len(s) / periods_per_year  # Calcul la quantité d'années sur la période choisi.
    if years <= 0: return float("nan")
    return (1 + total) ** (1 / years) - 1 # Formule classique du taux de croissance annualisé composé (CAGR) :

def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:   # Calcule la volatilité annualisée d’une série de rendements périodiques (souvent journaliers).
    r = returns.dropna()
    if r.empty: return float("nan")
    return float(r.std(ddof=0) * (periods_per_year ** 0.5))  # Résultat de la volatilité annualisée.

def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:  # Calcule le ratio de Sharpe d’une série de rendements financiers.
    r = returns.dropna()  # Supprime les valeurs manquantes pour éviter les erreurs dans les calculs.
    if r.empty: return float("nan")  # Si après suppression r est vide, la fonction retourne NaN.
    rf_p = (1 + rf) ** (1/periods_per_year) - 1  # Convertit le taux sans risque annuel rf en taux par période (ex. journalier).
    excess = r - rf_p  # Calcul des rendements excédentaires par rapport au taux sans risque. Autrement dit le ratio Sharpe.
    vol = excess.std(ddof=0)  # écart-type des rendements excédentaires. L’écart-type représente le risque du portefeuille.
    if vol == 0 or pd.isna(vol): return float("nan")  # Si la volatilité = 0 ou n’est pas définie, on retourne NaN pour éviter une division par zéro.
    return float((excess.mean() / vol) * (periods_per_year ** 0.5))  # Retourne le ratio de Sharpe annualisé, qui mesure le rendement ajusté au risque du portefeuille.

def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:  # Ratio de Sortino. 
    if r.empty: return float("nan")  # Nettoyage de données.
    rf_p = (1 + rf) ** (1/periods_per_year) - 1  # Même logique que dans le ratio Sharpe, on transforme le taux sans risque annuel en taux journalier (ou hebdo).
    excess = r - rf_p
    downside = excess[excess < 0]
    dd = downside.std(ddof=0)  # Écart-type des rendements négatifs.
    if dd == 0 or pd.isna(dd): return float("nan")
    return float((excess.mean() / dd) * (periods_per_year ** 0.5))  # Rendement moyen excédentaire annualisé.

def max_drawdown(equity: pd.Series) -> Tuple[float, pd.Series]:  # Mesure très utilisée en finance pour quantifier la pire chute d’un portefeuille depuis son plus haut historique.
    e = equity.dropna()
    if e.empty: return float("nan"), pd.Series(dtype=float, name="Drawdown")
    roll_max = e.cummax()
    dd = e / roll_max - 1.0
    return float(dd.min()), dd.rename("Drawdown")  # En résumé, Drawdown répond à la question : Quelle a été la pire perte subie avant de retrouver un nouveau sommet ?

def hist_var(returns: pd.Series, level: float = 0.95) -> float:  # Calcul de la variance historique.
    r = returns.dropna()   
    if r.empty: return float("nan")
    alpha = 1 - level
    return float(np.quantile(r, alpha))

def beta_alpha(returns: pd.Series, bench: pd.Series, rf: float = 0.0, periods_per_year: int = 252):  # calcul du bêta et de l’alpha d’un actif ou portefeuille par rapport à un indice de référence (benchmark).
    r = returns.dropna()
    b = bench.dropna()
    idx = r.index.intersection(b.index)
    if len(idx) < 10: return float("nan"), float("nan")
    r = r.loc[idx]
    b = b.loc[idx]
    rf_p = (1 + rf) ** (1/periods_per_year) - 1  # Conversion du taux sans risque en taux par période
    r_ex = r - rf_p
    b_ex = b - rf_p
    cov = float(np.cov(r_ex, b_ex)[0,1])  # Covariance entre le portefeuille et le benchmark.
    varb = float(np.var(b_ex))   # variance du benchmark.
    if varb == 0: return float("nan"), float("nan")
    beta = cov / varb   # Calcul du beta.
    alpha = (r_ex.mean() - beta*b_ex.mean()) * periods_per_year  # Calcul du alpha.
    return float(beta), float(alpha)
