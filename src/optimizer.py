import numpy as np
import pandas as pd

try:  # On essaie d'importer la fonction minimize depuis Scipy.
    from scipy.optimize import minimize  # type: ignore
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# Implémentation de la frontière d'efficiente de Markowitz, ici pour trouver le portefeuille avec le ratio de Sharpe maximal.
def markowitz_max_sharpe(returns_df: pd.DataFrame, rf: float=0.0, short_allowed: bool=False) -> dict[str, float]:
    r = returns_df.copy()      # DataFrame contenant les rendements historiques de plusieurs actifs (colonnes = actifs).
    mu = r.mean().values       # vecteur des rendements moyens de chaque actif.
    cov = np.cov(r.values.T)   # matrice de covariance des rendements → mesure la corrélation entre actifs.
    n = r.shape[1]             # nombre de colonnes (actifs)
    if n==0: return {}
    def obj(w):
        w = np.array(w)    # vecteur des poids des actifs (somme = 1).
        port_mu = mu.dot(w)   # rendement attendu du portefeuille.
        port_sigma = float(np.sqrt(w.T.dot(cov).dot(w)))   # risque (écart-type du portefeuille).
        if port_sigma == 0: return 1e6
        rf_p = (1+rf)**(1/252) - 1    # taux sans risque journalier (puisqu’on suppose 252 jours de bourse/an).
        sharpe = (port_mu - rf_p) / port_sigma   # ratio de Sharpe.
        return -sharpe   # On retourne -sharpe car le solveur minimize() cherche un minimum — donc pour maximiser le Sharpe, on minimise son opposé.
    bounds = (None, None) if short_allowed else (0.0, 1.0)
    bnds = [bounds]*n
    cons = [{'type':'eq','fun': lambda w: float(np.sum(w)-1.0)}]
    x0 = np.ones(n)/n
    if not HAS_SCIPY:    # Si HAS_SCIPY n'est pas disponible.    
        best_s, best_w = -1e9, x0  
        rng = np.random.default_rng(0)
        for _ in range(10000):
            w = rng.random(n); w /= w.sum()
            s = -obj(w)
            if s>best_s: best_s, best_w = s, w
        return {k: float(v) for k,v in zip(r.columns, best_w)}
    res = minimize(obj, x0, bounds=bnds, constraints=cons, method='SLSQP', options={'maxiter':500})
    w_opt = x0 if not res.success else res.x
    return {k: float(v) for k,v in zip(r.columns, w_opt)}
