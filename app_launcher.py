# -----------------------------------------------------------------------------------------------------------------------------
# Streamlit launcher for the modular Portfolio Strategy Simulator (v2)
# -----------------------------------------------------------------------------------------------------------------------------
import os, json
from datetime import date, timedelta
from typing import Optional, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.Yfinance import DEFAULT_TICKERS, fetch_prices_yf, load_from_csv
from src.utils import sanitize_columns, pct_returns
from src.metrics import annualized_return, annualized_vol, sharpe_ratio, sortino_ratio, max_drawdown, hist_var, beta_alpha
from src.strategies import BuyAndHoldStrategy, MACrossoverStrategy, VolTargetStrategy
from src.optimizer import markowitz_max_sharpe

# ----------------------------------------------------------------------------------------------------------------------------
# Configuration de la page d'accueil
# ----------------------------------------------------------------------------------------------------------------------------

st.set_page_config(page_title="Portfolio Strategy Simulator v2 (Modular)", page_icon="💹", layout="wide")
st.title("💹 Portfolio Strategy Simulator — v2")
st.caption("Stratégies élargies + KPIs avancés + Optimisation Markowitz. **Éducatif uniquement.**")

# ----------------------------------------------------------------------------------------------------------------------------
# Affichage de la colonne de gauche de l'interface
# ----------------------------------------------------------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Paramètres")
    mode = st.radio("Source des données", ["Télécharger via yfinance", "Uploader un CSV"])
    custom = st.text_input("Ajouter des tickers séparés par des virgules (optionnel)", "")
    tickers_default = DEFAULT_TICKERS.copy()
    # Selection de tickers  
    if custom.strip():   
        extras = [t.strip().upper() for t in custom.split(",") if t.strip()]
        tickers_default = list(dict.fromkeys(tickers_default + extras))
    tickers = st.multiselect("Sélectionnez des tickers", options=tickers_default, default=["AAPL","MSFT","NVDA"])
    # Selection de dates   
    col_dates = st.columns(2)    # deux colonnes
    with col_dates[0]:
        start_date = st.date_input("Date de début", date.today() - timedelta(days=365*5))
    with col_dates[1]:
        end_date = st.date_input("Date de fin", date.today())
    # Il est possible de cibler le niveau de risque
    rf_rate = st.number_input("Taux sans risque annuel", min_value=0.0, value=0.02, step=0.005, format="%.3f")
    # Sélection du Benchmark
    benchmark = st.selectbox("Benchmark (beta/alpha)",
                             options=["^GSPC","AAPL","MSFT"] + [t for t in tickers if t not in ["^GSPC","AAPL","MSFT"]],
                             index=0)
    st.divider()
    # Choix de stratégie
    st.subheader("Stratégie")
    strategy_name = st.selectbox("Choix de la stratégie", ["Buy & Hold", "MA Crossover", "Volatility Target (EQ)"])
    # Choix des paramètres techniques 
    initial_capital = st.number_input("Capital initial (€)", min_value=100.0, value=10000.0, step=100.0)
    fast = st.number_input("MA rapide (MA Crossover)", min_value=5, value=50, step=5)
    slow = st.number_input("MA lente (MA Crossover)", min_value=20, value=200, step=10)
    tgt_vol = st.number_input("Vol cible (Vol Target)", min_value=0.02, value=0.10, step=0.01, format="%.2f")
    lookback = st.number_input("Lookback Vol (jours)", min_value=5, value=20, step=1)
    st.divider()
    # Possibilité d'enregistrer
    config_name = st.text_input("Nom de config (save/load)", "demo_config")
    btn_save = st.button("💾 Sauver config")  # Bouton interactif pour sauvegarder. TRUE si l'utilisateur appuie, FALSE sinon.
    btn_load = st.button("📂 Charger config")   # Bouton interactif pour charger
    run = st.button("▶️ Lancer", use_container_width=True)

prices = None

if run:
    try:
        if mode == "Télécharger via yfinance":
            if not tickers:
                st.warning("Sélectionnez au moins un ticker.")  # Affiche un message d'erreur si on ne choisit pas des tickers
            else:
                prices = fetch_prices_yf(tickers, start_date, end_date)
        else:
            uploaded = st.file_uploader("Charger un CSV (Date + colonnes)", type=["csv"])
            if uploaded is None:
                st.warning("Veuillez uploader un CSV.")  # Affichage d'un message d'erreur
            else:
                prices = load_from_csv(uploaded)
                cols = [c for c in tickers if c in prices.columns]  # Pour chaque élément dans tickers, si c est présent dans prices.columns,
                if cols: prices = prices[cols]  #  alors ajoute c à la liste finale. 
                prices = sanitize_columns(prices)

        if prices is None or prices.empty:
            st.stop()

        # Save/Load config
        if btn_save:   # Si le boutton est selectionné
            cfg = {    # Il crée un dictionnaire 'cfg' contenant tous les paramètres de la stratégie.
                "tickers": tickers,
                "start": str(start_date),
                "end": str(end_date),
                "rf_rate": rf_rate,
                "benchmark": benchmark,
                "strategy": strategy_name,
                "initial_capital": initial_capital,
                "fast": int(fast),
                "slow": int(slow),
                "tgt_vol": tgt_vol,
                "lookback": int(lookback),
            }
            with open(f"config_{config_name}.json","w",encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            st.success(f"Config '{config_name}' sauvegardée.")   # Affiche un message de succès dans Streamlit.
        if btn_load:
            try:
                with open(f"config_{config_name}.json","r",encoding="utf-8") as f:
                    cfg = json.load(f)   # Lit le fichier JSON et le transforme dans un dictionnaire Python.
                st.success(f"Config '{config_name}' chargée."); st.json(cfg)  # Affiche un résultat positif si le chargement a succès.
            except Exception as e:
                st.error(f"Impossible de charger la config: {e}")     # Affiche erreur.
# ------------------------------------------------------------------------------------------------------------------------------
# Tabs Calcul
# ------------------------------------------------------------------------------------------------------------------------------
        tabs = st.tabs(["📈 Résultats", "🧐 Analyse", "🧮 Optimisation"])  # Crée plusieurs onglets cliquables dans l'application.

        # Strategy selection
        if strategy_name == "Buy & Hold":
            strategy = BuyAndHoldStrategy()
        elif strategy_name == "MA Crossover":
            strategy = MACrossoverStrategy(fast=int(fast), slow=int(slow))
        else:
            strategy = VolTargetStrategy(target_vol_ann=tgt_vol, lookback=int(lookback))

        res = strategy.run(prices, capital=initial_capital)   # Lance la stratégie choisit.

        equity = res.equity.rename("Equity")    # Series ou DataFrame avec l’évolution du portefeuille.
        rets = res.periodic_returns.rename("Returns")  # calcul des rendements périodiques à partir de la variable 'res'.

        # KPIs. Des fonctions définits dans le fichier 'metrics'.
        cagr = annualized_return(equity)
        vol = annualized_vol(rets)
        sharpe = sharpe_ratio(rets, rf=rf_rate)
        sortino = sortino_ratio(rets, rf=rf_rate)
        mdd, dd_series = max_drawdown(equity)
        var95 = hist_var(rets, 0.95)

        # Benchmark
        bench_series = None  # Au départ, aucune série de benchmark est disponible.
        if benchmark in res.prices.columns:  # Si le benchmark est déjà inclus dans nôtre liste, alors :
            bench_series = res.prices[benchmark].pct_change().fillna(0.0)  # On prend une série des prix du benchmark en calculant sa variation.
        else:  # Sinon on va télécharger les données pour le benchmark choisi avec la fonction fetch_prices_yt().
            try:  
                bench_px = fetch_prices_yf([benchmark], res.prices.index.min().date(), res.prices.index.max().date())
                bench_series = bench_px.iloc[:,0].pct_change().fillna(0.0)
            except Exception:
                bench_series = None
        beta, alpha = (np.nan, np.nan)
        if bench_series is not None:   # Si la serie bench_series existe, alors on calcule son beta (sensibilité au marché) et alpha (sur ou sous performance)
            beta, alpha = beta_alpha(rets, bench_series, rf=rf_rate)
# -----------------------------------------------------------------------------------------------------------------------------
# Affichage du résultat
# -----------------------------------------------------------------------------------------------------------------------------
        with tabs[0]:  # Tout le code à l'intérieur de la table 'Résultats' (tabs[0])
            st.success("Données chargées: " + ", ".join(map(str, list(res.prices.columns))))  # affiche un message indiquant que les données ont bien été chargées.  
            st.dataframe(res.prices.tail())  # Affiche les dernières lignes du DataFrame contenant les prix.
            c1, c2, c3, c4, c5, c6 = st.columns(6)  # Crée six colonnes côte à côte pour afficher les métriques principales.
            c1.metric("CAGR", f"{cagr*100:,.2f}%")
            c2.metric("Vol ann.", f"{vol*100:,.2f}%")
            c3.metric("Sharpe", f"{sharpe:,.2f}")
            c4.metric("Sortino", f"{sortino:,.2f}")
            c5.metric("Max DD", f"{mdd*100:,.2f}%")
            c6.metric("VaR 95% (quotid.)", f"{var95*100:,.2f}%")
            if not np.isnan(beta):  # Si le beta ne'st pas manquant on affiche :
                st.caption(f"β vs {benchmark}: **{beta:.2f}**, α (ann.): **{alpha*100:.2f}%**")
            # Graphique de l'évolution du portefeuille
            fig_eq = px.line(equity.reset_index(), x=equity.index.name or "index", y="Equity",
                             labels={"x":"Date","Equity":"Valeur (€)"})
            # graphique linéaire interactif montrant la valeur du portefeuille dans le temps.
            fig_eq.update_layout(height=380, hovermode="x unified")
            st.plotly_chart(fig_eq, use_container_width=True)  # le graphique s'adapte à la largeur de la page.
            # Graphique du drawdown. Drawdown = pertes cumulées depuis le dernier pic de capital.
            dd_df = dd_series.rename("Drawdown").reset_index()
            fig_dd = px.area(dd_df, x=dd_df.columns[0], y="Drawdown")
            fig_dd.update_layout(height=220, hovermode="x unified")
            st.plotly_chart(fig_dd, use_container_width=True)
            # Exporter le résultat.
            out = pd.DataFrame({"Equity": equity, "Returns": rets})
            st.download_button("⬇️ Equity & Returns (CSV)", out.to_csv().encode("utf-8"),
                               file_name="equity_returns_v2.csv", mime="text/csv") # Télécharge un fichier .csv prêt à être ouvert dans Excel ou Python.

        with tabs[1]:  # Table 'Analyse'
            st.subheader("Corrélations (rendements)")  # Titre du table de corrélations
            rets_assets = pct_returns(res.prices)  # Elle calcule les rendements en pourcentage à partir des prix. 
            corr = rets_assets.corr()  # Matrice de corrélations
            fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto")  # affiche une matrice (heatmap) avec les valeurs de corrélation.
            st.plotly_chart(fig_corr, use_container_width=True)  # Intègre la heatmap interactive dans l'application Streamlit.
            st.subheader("Distribution des rendements du portefeuille")  # Titre du distribution du rendement du portefeuille.
            st.plotly_chart(px.histogram(rets, nbins=60), use_container_width=True)  # Histogramme montrant la distribution du rendement du portefeuille.

        with tabs[2]: # Table 'Optimisation'. Applique un modèle de Markowitz pour trouver la meilleure combinaison d’actifs.
            st.subheader("Optimisation de portefeuille — Max Sharpe (Markowitz)")
            if res.prices.shape[1] >= 2:  # On ne peut pas faire une optimisation de portefeuille avec un seul actif
                returns_df = pct_returns(res.prices).iloc[1:]  # Calcul des rendements journalières à partir du prix. Supprime la première ligne.
                weights_opt = markowitz_max_sharpe(returns_df, rf=rf_rate, short_allowed=False)  # Fonction personalisée optimisant le ratio sharpe.
                if weights_opt:
                    w_ser = pd.Series(weights_opt)  #dictionnaire ou vecteur de poids (ex. {'AAPL': 0.4, 'MSFT': 0.6}) 
                    st.write("Poids optimisés:")
                    st.dataframe(w_ser.to_frame("Weight").T if len(w_ser)>8 else w_ser)
                    st.plotly_chart(px.bar(w_ser, title="Poids optimisés (Max Sharpe)"), use_container_width=True) # Utilise Plotly Express (px.bar) pour afficher un bar chart des poids.
                else:
                    st.info("Optimisation indisponible.")
            else:
                st.info("Sélectionnez ≥2 tickers.")  # Demande de sélectionner au moins deux tickers.

        # Run log
        os.makedirs("results", exist_ok=True)
        run_log = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "tickers": list(res.prices.columns),
            "start": str(res.prices.index.min().date()),
            "end": str(res.prices.index.max().date()),
            "strategy": strategy_name,
            "initial_capital": initial_capital,
            "metrics": {"CAGR": float(cagr) if pd.notna(cagr) else None,
                        "Vol": float(vol) if pd.notna(vol) else None,
                        "Sharpe": float(sharpe) if pd.notna(sharpe) else None,
                        "Sortino": float(sortino) if pd.notna(sortino) else None,
                        "MaxDD": float(mdd) if pd.notna(mdd) else None,
                        "VaR95": float(var95) if pd.notna(var95) else None,
                        "Beta": float(beta) if not np.isnan(beta) else None,
                        "Alpha_ann": float(alpha) if not np.isnan(alpha) else None},
        }
        with open(os.path.join("results", "runs_log_v2.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(run_log) + "\n")

    except Exception as e:  # Attrape toutes les erreurs qui héritent de la classe Exception.
        st.error(f"Erreur: {e}")  # Affiche l'erreur trouvé.

st.divider()  # Commande pour faire une séparation dans l'interface.
# -----------------------------------------------------------------------------------------------------------------------------
# Message instructif s'affichant dans la page d'accueil
# -----------------------------------------------------------------------------------------------------------------------------
st.markdown("""
**Guide rapide v2**  
- Fichiers séparés : `src/` (données, stratégies, métriques, optimisateur) + `app.py` (lanceur).  
- Onglets: Résultats / Analyse / Optimisation.  
- KPIs: Sortino, VaR(95%), β/α vs benchmark.  
- Stratégies: MA crossover, Volatility targeting + Buy&Hold.  
- Optimisation Markowitz (max Sharpe).  
""")
