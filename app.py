# -*- coding: utf-8 -*-
"""
📊 Plan financier simplifié par leviers — Assurance individuelle (vie)
Databricks App (Streamlit) · Phase 2 du POC · IFRS 17

⚠️ Données synthétiques à des fins de démonstration.

Même moteur et mêmes tables que le notebook Phase 1 (schéma plan_assurance_ind_v1).
- Les curseurs recalculent tout INSTANTANÉMENT (aucune écriture requise).
- Le bouton « Écrire le scénario » fait le write-back Delta (DELETE ciblé + INSERT)
  via le SQL warehouse rattaché à l'app (ressource `sql_warehouse` de app.yaml).
- Si le warehouse n'est pas joignable, l'app bascule en MODE LOCAL : tout fonctionne
  sauf le write-back et la comparaison des scénarios déjà écrits.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Connecteur Databricks (présent dans l'app ; absent en test local -> mode local)
try:
    from databricks import sql as dbsql
    from databricks.sdk.core import Config
    DBX_DISPONIBLE = True
except ImportError:
    DBX_DISPONIBLE = False

# ==============================================================================
# 1. CONSTANTES MÉTIER (identiques au notebook Phase 1)
# ==============================================================================
SCHEMA_DEFAUT = "plan_assurance_ind_v1"

# ---- Source unique de vérité : moteur_plan.py (partagé notebook <-> app) --------
from moteur_plan import (
    ECHELLE, ANNEES, N, PRODUITS, CANAUX, RSI_MATRICE, VENTES_BASE,
    BASE_COUTS, POLICES_2025, POLICES_2030, COUL,
    fmt_fr, fmt_m, fmt_pct,
    calculer_scenario, PARAMS_BASE, construire_lignes, generer_faits_saillants,
)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": COUL["fond"],
    "axes.edgecolor": "#DDDDDD", "axes.grid": True, "grid.color": "#E3E3E0",
    "grid.linewidth": 0.7, "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold",
})

# ==============================================================================
# 2-3. BASELINE + MOTEUR : importés de moteur_plan.py
# ==============================================================================
@st.cache_data(show_spinner=False)
def resultat_baseline():
    return calculer_scenario("Base", **PARAMS_BASE)

# ==============================================================================
# 4. CONNEXION AU SQL WAREHOUSE + WRITE-BACK DELTA
# ==============================================================================
@st.cache_resource(show_spinner=False)
def connexion():
    """Connexion au warehouse rattaché à l'app (ressource sql_warehouse d'app.yaml).
    Retourne None si indisponible -> mode local."""
    if not DBX_DISPONIBLE:
        return None
    wid = os.getenv("DATABRICKS_WAREHOUSE_ID")
    if not wid:
        return None
    try:
        cfg = Config()  # authentification automatique du principal de service de l'app
        return dbsql.connect(
            server_hostname=cfg.host.replace("https://", ""),
            http_path=f"/sql/1.0/warehouses/{wid}",
            credentials_provider=lambda: cfg.authenticate,
        )
    except Exception:
        return None

def requete(sql_txt):
    conn = connexion()
    with conn.cursor() as cur:
        cur.execute(sql_txt)
        cols = [c[0] for c in cur.description] if cur.description else []
        return pd.DataFrame(cur.fetchall(), columns=cols)

def executer(sql_txt):
    conn = connexion()
    with conn.cursor() as cur:
        cur.execute(sql_txt)

def esc(s):
    return str(s).replace("'", "''")

@st.cache_data(show_spinner=False, ttl=3600)
def detecter_catalog():
    try:
        return requete("SELECT current_catalog() AS c")["c"].iloc[0]
    except Exception:
        return "workspace"

def qualifier(table, catalog, schema):
    return f"`{catalog}`.`{schema}`.`{table}`"

# --- Schémas des tables (créées si le notebook Phase 1 n'a pas encore tourné) ---
DDL = {
    "overlay_drivers_slv": ("(scenario_id STRING, levier STRING, valeur DOUBLE, "
                            "valeur_baseline DOUBLE, horodatage STRING)"),
    "forecast_output_gld": ("(scenario_id STRING, annee BIGINT, section STRING, "
                            "ligne STRING, ordre BIGINT, montant_m DOUBLE)"),
    "kpi_gld": ("(scenario_id STRING, annee BIGINT, niveau STRING, produit STRING, "
                "canal STRING, kpi STRING, valeur DOUBLE, unite STRING)"),
    "dim_scenario_gld": ("(scenario_id STRING, horodatage STRING, facteur_volume DOUBLE, "
                         "accent_croissance DOUBLE, part_independants DOUBLE, "
                         "part_desjardins DOUBLE, part_agents DOUBLE, rendement_placement DOUBLE, "
                         "croiss_couts_acquisition DOUBLE, croiss_couts_attribuables DOUBLE, "
                         "croiss_couts_non_attribuables DOUBLE)"),
}

def inserer(table, colonnes, lignes, catalog, schema, lot=400):
    """INSERT INTO par lots ; les chaînes sont échappées, None -> NULL."""
    def val(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "NULL"
        if isinstance(v, str):
            return f"'{esc(v)}'"
        return f"{v}"
    for i in range(0, len(lignes), lot):
        vals = ", ".join("(" + ", ".join(val(v) for v in ligne) + ")"
                         for ligne in lignes[i:i + lot])
        executer(f"INSERT INTO {qualifier(table, catalog, schema)} "
                 f"({', '.join(colonnes)}) VALUES {vals}")

def ecrire_scenario(res, params, scenario_id, catalog, schema):
    """Write-back Delta par scénario : CREATE IF NOT EXISTS + DELETE ciblé + INSERT.
    La baseline et les autres scénarios ne bougent jamais. Idempotent."""
    executer(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")
    for t, ddl in DDL.items():
        executer(f"CREATE TABLE IF NOT EXISTS {qualifier(t, catalog, schema)} {ddl}")

    overlay, forecast, kpi, dim = construire_lignes(res, params, scenario_id)
    sid = esc(scenario_id)
    jeux = [
        ("overlay_drivers_slv",
         ["scenario_id", "levier", "valeur", "valeur_baseline", "horodatage"], overlay),
        ("forecast_output_gld",
         ["scenario_id", "annee", "section", "ligne", "ordre", "montant_m"], forecast),
        ("kpi_gld",
         ["scenario_id", "annee", "niveau", "produit", "canal", "kpi", "valeur", "unite"], kpi),
        ("dim_scenario_gld",
         ["scenario_id", "horodatage", "facteur_volume",
          "part_independants", "part_desjardins", "part_agents", "rendement_placement",
          "croiss_couts_acquisition", "croiss_couts_attribuables",
          "croiss_couts_non_attribuables"], dim),
    ]
    for table, cols, lignes in jeux:
        executer(f"DELETE FROM {qualifier(table, catalog, schema)} "
                 f"WHERE scenario_id = '{sid}'")
        inserer(table, cols, lignes, catalog, schema)
    return sum(len(l) for _, _, l in jeux)

@st.cache_data(show_spinner=False, ttl=20)
def lire_kpi_scenarios(catalog, schema):
    return requete(
        f"SELECT scenario_id, annee, kpi, valeur FROM {qualifier('kpi_gld', catalog, schema)} "
        f"WHERE niveau = 'global' AND kpi IN "
        f"('rsi_global_pct','resultat_net_m','van_totale_m','csm_solde_fin_m')"
    )

# ==============================================================================
# 5. VISUELS (matplotlib — identiques à la Phase 1)
# ==============================================================================
def fig_waterfall(etiquettes, valeurs, titre, plafond=None):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    n = len(valeurs)
    ax.set_ylim(0, plafond or max(np.cumsum(valeurs[:-1]).max(), valeurs[0], valeurs[-1]) * 1.22)
    cumul = 0.0
    for i, (lab, v) in enumerate(zip(etiquettes, valeurs)):
        est_total = (i == n - 1) or (i == 0 and "Solde" in lab)
        if est_total:
            ax.bar(i, v, bottom=0, color=COUL["bleu"], width=0.62, zorder=3)
            sommet = v
            if i == 0:
                cumul = v
        else:
            ax.bar(i, v, bottom=cumul, color=COUL["vert"] if v >= 0 else COUL["rouge"],
                   width=0.62, zorder=3)
            sommet = cumul + max(v, 0)
            cumul += v
        ax.text(i, sommet + ax.get_ylim()[1] * 0.015, fmt_fr(v, 0),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        if not est_total and i < n - 1:
            ax.hlines(cumul, i + 0.31, i + 1 - 0.31, color=COUL["gris"],
                      linewidth=1, linestyle=":", zorder=2)
    ax.set_xticks(range(n))
    ax.set_xticklabels([e.replace(" ", "\n", 1) for e in etiquettes], fontsize=9)
    ax.set_title(titre)
    ax.set_ylabel("M$")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_fr(x)))
    fig.tight_layout()
    return fig

def fig_heatmap(rsi_adj, rendement, scenario_id):
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    donnees = np.ma.masked_invalid(rsi_adj)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#E8E8E4")
    im = ax.imshow(donnees, cmap=cmap, vmin=-12, vmax=20, aspect="auto")
    ax.set_xticks(range(len(CANAUX)), CANAUX, fontsize=10.5)
    ax.set_yticks(range(len(PRODUITS)), PRODUITS, fontsize=9.5)
    for i in range(len(PRODUITS)):
        for j in range(len(CANAUX)):
            v = rsi_adj[i, j]
            if np.isnan(v):
                ax.text(j, i, "n.d.", ha="center", va="center",
                        color=COUL["gris"], fontsize=8.5)
            else:
                ax.text(j, i, fmt_pct(v), ha="center", va="center", fontsize=9.5,
                        fontweight="bold", color="black" if -8 < v < 30 else "white")
    ax.set_title(f"RSI des ventes par produit × canal — « {scenario_id} » · "
                 f"rendement {rendement*100:.2f} %")
    fig.colorbar(im, ax=ax, shrink=0.85).set_label("RSI (%)")
    ax.grid(False)
    fig.tight_layout()
    return fig

def fig_trajectoires(res, res_base, scenario_id):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (titre, cle, unite) in zip(axes, [
        ("Résultat net (M$)", "resultat_net", "M$"),
        ("RSI global (%)", "rsi_global", "%"),
        ("Coût d'acquisition / police ($)", "cout_par_police", "$"),
    ]):
        ax.plot(ANNEES, res_base[cle], marker="o", linewidth=2, color=COUL["gris"],
                linestyle="--", label="Base")
        ax.plot(ANNEES, res[cle], marker="o", linewidth=2.5, color=COUL["vert"],
                label=scenario_id)
        ax.set_title(titre)
        ax.set_xticks(ANNEES)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8.5)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_fr(x)))
    fig.tight_layout()
    return fig

def afficher_fig(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)

# ==============================================================================
# 6. INTERFACE
# ==============================================================================
st.set_page_config(page_title="Plan financier par leviers — Assurance individuelle",
                   page_icon="📊", layout="wide")

st.title("📊 Plan financier simplifié par leviers — Assurance individuelle")
st.caption("⚠️ **Données synthétiques à des fins de démonstration** · IFRS 17 · "
           "2025 (estimé) → 2030 · POC Databricks App")

# ---- 🧭 Démarche (storyline mise en évidence, façon SUMMARY/STORYLINE) ----------
_etapes = ["1️⃣ Ventes d'affaires nouvelles", "2️⃣ CSM des ventes + VAN",
           "3️⃣ État des résultats IFRS 17", "4️⃣ Capital à rémunérer",
           "5️⃣ RSI (ROE)", "6️⃣ Sources de bénéfices", "7️⃣ Roll-forward CSM",
           "🎛️ Leviers → recalcul"]
st.markdown(
    "<div style='background:#EAF3EE;border-left:6px solid #00874E;border-radius:6px;"
    "padding:10px 14px;margin-bottom:6px;'>"
    "<b>🧭 Démarche</b> &nbsp;·&nbsp; " +
    " <span style='color:#00874E;font-weight:bold'>→</span> ".join(
        f"<span style='white-space:nowrap'>{e}</span>" for e in _etapes) +
    "</div>",
    unsafe_allow_html=True,
)
with st.expander("Voir le détail de la démarche (passifs / actifs / consolidé)"):
    st.markdown("""
| Bloc | Approche | Outil |
|---|---|---|
| **Passifs — In-force** | Baseline synthétique du bloc en vigueur (relâche CSM + RA, expérience), agrégée au niveau des 6 familles de produits | Delta (médaillon) |
| **Passifs — Affaires nouvelles** | Ventes par famille × canal → marge CSM et VAN@12 % ; leviers de volume (A), de croissance par famille (B) et de mix canal (C) en overlay | Moteur pandas |
| **Combiné (In-force + AN)** | Alimente sans couture l'état des résultats IFRS 17 et le roll-forward du CSM | Moteur pandas |
| **Actifs** | Résultat financier ≈ actifs investis × rendement (levier D) − accrétion des passifs | Moteur pandas |
| **Consolidé** | Capital à rémunérer → RSI (ROE), sources de bénéfices, coûts unitaires (levier E) ; write-back par scénario dans le Gold | Delta + app |

**Principe clé : baseline immuable + overlay de leviers → recalcul du Gold par scénario.**
""")

# ---- Clés des widgets (permettent le rechargement d'un scénario écrit) ---------
FAMILLES_COURTES = ["VE participation", "VE paiements limités", "Autres VE",
                    "Temporaires", "Maladies graves", "Autres inv. et maladie"]
CLES_B = [f"k_b_{i}" for i in range(len(PRODUITS))]
CLES_DEFAUTS = {"k_scen": "Base", "k_vol": 1.00, "k_ind": 60, "k_dsj": 25, "k_agt": 15,
                "k_rdt": 3.50, "k_gacq": 3.5, "k_gattr": 3.0, "k_gna": 2.0,
                **{c: 0.0 for c in CLES_B}}
for cle, defaut in CLES_DEFAUTS.items():
    st.session_state.setdefault(cle, defaut)

# Application d'un scénario à recharger (préparé au clic du bouton 📂, voir plus bas)
if "_a_charger" in st.session_state:
    for cle, valeur in st.session_state.pop("_a_charger").items():
        st.session_state[cle] = valeur

# ---- Barre latérale : leviers ------------------------------------------------
st.sidebar.header("🎛️ Leviers du scénario")
scenario_id = st.sidebar.text_input("Nom du scénario", key="k_scen",
                                    help="Clé du write-back : « Base » = baseline.").strip() or "Base"

fact_volume = st.sidebar.slider("A — Volume de ventes global (×)", 0.80, 1.30,
                                step=0.01, key="k_vol")

with st.sidebar.expander("B — Croissance par famille (pts de %/an vs plan)", expanded=False):
    st.caption("0 = trajectoire du plan ; +2 = la famille croît 2 pts plus vite chaque année.")
    croiss_fam = [st.slider(FAMILLES_COURTES[i], -10.0, 10.0, step=0.5, key=CLES_B[i])
                  for i in range(len(PRODUITS))]

with st.sidebar.expander("C — Mix canal (%)", expanded=False):
    p_ind = st.slider("Indépendants", 0, 100, step=1, key="k_ind")
    p_dsj = st.slider("Desjardins", 0, 100, step=1, key="k_dsj")
    p_agt = st.slider("Agents Desjardins", 0, 100, step=1, key="k_agt")
    total_c = max(1, p_ind + p_dsj + p_agt)
    parts_canal = [p_ind / total_c, p_dsj / total_c, p_agt / total_c]
    st.caption("Renormalisé : " + " / ".join(f"{p*100:.0f} %" for p in parts_canal))

rendement = st.sidebar.slider("D — Rendement de placement (%)", 2.50, 5.00,
                              step=0.05, key="k_rdt") / 100.0

with st.sidebar.expander("E — Croissance des dépenses (%/an)", expanded=False):
    g_acq = st.slider("Coûts d'acquisition", 0.0, 8.0, step=0.1, key="k_gacq") / 100.0
    g_attr = st.slider("Coûts attribuables", 0.0, 8.0, step=0.1, key="k_gattr") / 100.0
    g_na = st.slider("Coûts non attribuables", 0.0, 8.0, step=0.1, key="k_gna") / 100.0

params = dict(fact_volume=float(fact_volume), croiss_fam=[float(c) for c in croiss_fam],
              parts_canal=[float(p) for p in parts_canal], rendement=float(rendement),
              g_acq=float(g_acq), g_attr=float(g_attr), g_na=float(g_na))

# ---- Recalcul instantané (moment « wow » A) ------------------------------------
res = calculer_scenario(scenario_id, **params)
res_base = resultat_baseline()

# ---- Connexion / write-back -----------------------------------------------------
conn_ok = connexion() is not None
if conn_ok:
    CATALOG = detecter_catalog()
else:
    CATALOG = "workspace"
with st.sidebar.expander("⚙️ Avancé (catalog / schéma)"):
    CATALOG = st.text_input("Catalog", CATALOG)
    SCHEMA = st.text_input("Schéma", SCHEMA_DEFAUT)

st.sidebar.divider()
if conn_ok:
    # ---- 📂 Recharger un scénario déjà écrit dans les curseurs -------------------
    try:
        scen_dispo = requete(
            f"SELECT DISTINCT scenario_id FROM {qualifier('dim_scenario_gld', CATALOG, SCHEMA)} "
            f"ORDER BY scenario_id")["scenario_id"].tolist()
    except Exception:
        scen_dispo = []
    if scen_dispo:
        with st.sidebar.expander("📂 Recharger un scénario écrit", expanded=False):
            choix = st.selectbox("Scénario", scen_dispo, key="k_choix_chargement")
            if st.button("Charger dans les curseurs", width="stretch"):
                try:
                    ov = requete(
                        f"SELECT levier, valeur FROM "
                        f"{qualifier('overlay_drivers_slv', CATALOG, SCHEMA)} "
                        f"WHERE scenario_id = '{esc(choix)}'")
                    lev = dict(zip(ov["levier"], ov["valeur"].astype(float)))
                    total = (lev.get("C_part_independants", 0.60)
                             + lev.get("C_part_desjardins", 0.25)
                             + lev.get("C_part_agents", 0.15)) or 1.0
                    charge = {
                        "k_scen": choix,
                        "k_vol": round(float(lev.get("A_facteur_volume", 1.0)), 2),
                        "k_ind": int(round(lev.get("C_part_independants", 0.60) / total * 100)),
                        "k_dsj": int(round(lev.get("C_part_desjardins", 0.25) / total * 100)),
                        "k_agt": int(round(lev.get("C_part_agents", 0.15) / total * 100)),
                        "k_rdt": round(lev.get("D_rendement_placement", 0.035) * 100, 2),
                        "k_gacq": round(lev.get("E_croiss_couts_acquisition", 0.035) * 100, 1),
                        "k_gattr": round(lev.get("E_croiss_couts_attribuables", 0.030) * 100, 1),
                        "k_gna": round(lev.get("E_croiss_couts_non_attrib", 0.020) * 100, 1),
                    }
                    for i, p in enumerate(PRODUITS):   # levier B par famille (0 si absent
                        charge[CLES_B[i]] = round(     # -> scénario écrit avant la refonte B)
                            float(lev.get(f"B_croiss_{p}", 0.0)), 1)
                    st.session_state["_a_charger"] = charge
                    st.rerun()
                except Exception as e:
                    st.error(f"Chargement impossible : {e}")

    if st.sidebar.button("💾 Écrire le scénario dans Delta", type="primary",
                         width="stretch"):
        try:
            with st.spinner(f"Write-back du scénario « {scenario_id} »…"):
                nb = ecrire_scenario(res, params, scenario_id, CATALOG, SCHEMA)
            lire_kpi_scenarios.clear()
            st.sidebar.success(f"✅ « {scenario_id} » écrit ({nb} lignes). "
                               f"Baseline et autres scénarios intacts.")
        except Exception as e:
            st.sidebar.error(f"Échec du write-back : {e}")
else:
    st.sidebar.warning("🔌 **Mode local** — SQL warehouse non joignable : les curseurs "
                       "fonctionnent, mais pas le write-back ni la lecture des scénarios "
                       "déjà écrits. Vérifier la ressource `sql_warehouse` de l'app et les "
                       "droits du principal de service sur le schéma.")

# ---- Cartes KPI (avec écart vs Base) --------------------------------------------
annee_focus = st.select_slider("Année mise en vedette", options=ANNEES, value=2030)
jf = ANNEES.index(annee_focus)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Résultat net", fmt_m(res["resultat_net"][jf]),
          delta=fmt_m(res["resultat_net"][jf] - res_base["resultat_net"][jf]) + " vs Base")
c2.metric("RSI global (ROE)", fmt_pct(res["rsi_global"][jf]),
          delta=fmt_fr(res["rsi_global"][jf] - res_base["rsi_global"][jf], 1, " pt vs Base"))
c3.metric("VAN des affaires nouvelles", fmt_m(res["van_tot_m"][jf], 1),
          delta=fmt_m(res["van_tot_m"][jf] - res_base["van_tot_m"][jf], 1) + " vs Base")
c4.metric("Solde CSM (fin)", fmt_m(res["csm_close"][jf]),
          delta=fmt_m(res["csm_close"][jf] - res_base["csm_close"][jf]) + " vs Base")

# ---- 💡 Faits saillants automatisés ----------------------------------------------
with st.container(border=True):
    st.markdown(f"**💡 Faits saillants — « {scenario_id} » · {annee_focus}** "
                f"*(générés automatiquement à partir du scénario courant)*")
    for fait in generer_faits_saillants(res, res_base, params, jf, annee_focus, scenario_id):
        st.markdown(f"- {fait}")

# ---- Onglets ---------------------------------------------------------------------
ong1, ong2, ong3, ong4 = st.tabs(
    ["📊 Tableau de bord", "🔁 Roll-forward CSM", "⚖️ Comparaison de scénarios", "📄 Détail"]
)

with ong1:
    gauche, droite = st.columns([1.15, 1])
    with gauche:
        afficher_fig(fig_waterfall(
            ["Profit attendu (CSM + RA)", "Impact des ventes", "Expérience",
             "Dépenses", "Intérêt et marché", "Résultat d'exploitation"],
            [res["profit_attendu"][jf], res["impact_ventes"][jf], res["experience"][jf],
             res["depenses"][jf], res["interet_marche"][jf], res["exploitation"][jf]],
            f"Sources de bénéfices {annee_focus} — « {scenario_id} » (M$)"))
    with droite:
        afficher_fig(fig_heatmap(res["rsi_adj"], rendement, scenario_id))
    afficher_fig(fig_trajectoires(res, res_base, scenario_id))

with ong2:
    afficher_fig(fig_waterfall(
        ["Solde CSM départ", "Profit attendu relâché", "Impact des ventes profitables",
         "Intérêt et marché", "Expérience", "Changements d'hypothèses", "Solde CSM fin"],
        [res["csm_open"][jf], -res["csm_release"][jf], res["nb_csm"][jf],
         res["csm_interet"][jf], res["exp_csm"], res["chg_hyp"][jf], res["csm_close"][jf]],
        f"Roll-forward du CSM {annee_focus} — « {scenario_id} » (M$)",
        plafond=max(res["csm_open"][jf], res["csm_close"][jf]) * 1.18))
    st.dataframe(pd.DataFrame({
        "Année": ANNEES,
        "Solde départ (M$)": res["csm_open"].round(0),
        "Relâche (M$)": (-res["csm_release"]).round(0),
        "Ventes profitables (M$)": res["nb_csm"].round(0),
        "Intérêt (M$)": res["csm_interet"].round(0),
        "Solde fin (M$)": res["csm_close"].round(0),
    }), hide_index=True, width="stretch")

with ong3:
    st.markdown("**Scénarios écrits dans `kpi_gld`** (notebook Phase 1 ou bouton "
                "« Écrire le scénario »). Le scénario courant *(non écrit)* est superposé "
                "en pointillé.")
    if conn_ok:
        try:
            kpi_all = lire_kpi_scenarios(CATALOG, SCHEMA)
        except Exception:
            kpi_all = pd.DataFrame(columns=["scenario_id", "annee", "kpi", "valeur"])
    else:
        kpi_all = pd.DataFrame(columns=["scenario_id", "annee", "kpi", "valeur"])

    scenarios = sorted(kpi_all["scenario_id"].unique()) if len(kpi_all) else []
    couleurs = [COUL["bleu"], COUL["vert"], COUL["or"], COUL["rouge"], "#8E44AD", "#16A085"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, (kpi_nom, titre, cle_local) in zip(axes, [
        ("rsi_global_pct", "RSI global (%)", "rsi_global"),
        ("resultat_net_m", "Résultat net (M$)", "resultat_net"),
    ]):
        for s_i, sc in enumerate(scenarios):
            d = (kpi_all[(kpi_all["scenario_id"] == sc) & (kpi_all["kpi"] == kpi_nom)]
                 .sort_values("annee"))
            ax.plot(d["annee"], d["valeur"], marker="o", linewidth=2.3,
                    color=couleurs[s_i % len(couleurs)], label=sc)
        ax.plot(ANNEES, res[cle_local], marker="s", linewidth=2, linestyle=":",
                color="black", label=f"{scenario_id} (courant)")
        ax.set_title(titre)
        ax.set_xticks(ANNEES)
        ax.legend(fontsize=8.5, title="Scénario")
    fig.tight_layout()
    afficher_fig(fig)

    if len(kpi_all):
        recap = (kpi_all[kpi_all["annee"] == 2030]
                 .pivot_table(index="scenario_id", columns="kpi", values="valeur")
                 .rename(columns={"resultat_net_m": "Résultat net 2030 (M$)",
                                  "rsi_global_pct": "RSI 2030 (%)",
                                  "van_totale_m": "VAN 2030 (M$)",
                                  "csm_solde_fin_m": "CSM fin 2030 (M$)"})
                 .round(1).reset_index().rename(columns={"scenario_id": "Scénario"}))
        st.dataframe(recap, hide_index=True, width="stretch")
    elif conn_ok:
        st.info("Aucun scénario écrit pour l'instant : utiliser 💾 dans la barre latérale.")

with ong4:
    st.markdown(f"**État des résultats IFRS 17 — « {scenario_id} » (M$)**")
    pnl = pd.DataFrame({
        "Ligne": ["Produits d'assurance", "Charges d'assurance", "Réassurance nette",
                  "Résultats des activités d'assurance", "Résultat financier",
                  "Résultats autres", "Résultat d'exploitation", "Impôts", "Résultat net"],
        **{str(a): np.round([res["produits_ass"][j], -res["charges_ass"][j],
                             res["reassurance"][j], res["activites"][j],
                             res["interet_marche"][j], res["depenses"][j],
                             res["exploitation"][j], -res["impots"][j],
                             res["resultat_net"][j]], 1)
           for j, a in enumerate(ANNEES)},
    })
    st.dataframe(pnl, hide_index=True, width="stretch")

    st.markdown("**Ventes par produit (K$)**")
    ventes_df = pd.DataFrame(res["ventes_scen"].round(0), index=PRODUITS,
                             columns=[str(a) for a in ANNEES]).reset_index(names="Produit")
    st.dataframe(ventes_df, hide_index=True, width="stretch")

    st.download_button(
        "⬇️ Télécharger le P&L (CSV)",
        pnl.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
        file_name=f"pnl_{scenario_id}.csv", mime="text/csv",
    )

st.caption("Baseline immuable + overlay de leviers → recalcul du Gold par scénario · "
           f"Schéma Unity Catalog : `{CATALOG}.{SCHEMA}` · Données synthétiques.")
