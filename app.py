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
    COUSSINS, POIDS_COUSSINS, CATEGORIES_COUTS, ALLOC_BLOCS_VERS_CATEGORIES,
    BLOCS_COUTS, CLASSES_ACTIFS, POIDS_ACTIFS, RDT_CLASSES_DEFAUT,
    rendement_pondere, ORACLE_MAPPING, MOIS, PROFILS_MENSUELS, mensualiser,
    VP_CATEGORIES, CAT_BASE_2025, ALLOC_CATEGORIES_VERS_BLOCS,
    CROISS_CATEGORIES_DEFAUT, generer_faits_par_onglet,
)

def etiqueter_points(ax, xs, ys, dec=0, couleur="#333333", dy=6):
    """Étiquettes de données sur une série de points (lignes/marqueurs)."""
    for x, y in zip(xs, ys):
        ax.annotate(fmt_fr(float(y), dec), (x, float(y)), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=8, color=couleur,
                    fontweight="bold")

def etiqueter_totaux(ax, xs, totaux, dec=0):
    """Étiquettes de total au sommet de barres empilées."""
    for x, t in zip(xs, totaux):
        ax.annotate(fmt_fr(float(t), dec), (x, float(t)), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8.5, fontweight="bold",
                    color="#333333")

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
        etiqueter_points(ax, ANNEES, res[cle],
                         dec=1 if "RSI" in titre else 0, couleur=COUL["vert"])
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

st.markdown("""<style>
div[data-testid="stMetric"]{background:#FFFFFF;border:1px solid #E8E8E4;
  border-radius:12px;padding:14px 18px;box-shadow:0 1px 3px rgba(0,0,0,.05);}
div[data-testid="stMetric"] label{color:#7F8C8D;}
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:12px;}
button[data-baseweb="tab"]{border-radius:10px 10px 0 0;padding:8px 14px;}
button[data-baseweb="tab"][aria-selected="true"]{background:#EAF3EE;
  border-bottom:3px solid #00874E;font-weight:600;}
div[role="radiogroup"] label{background:#FFFFFF;border:1px solid #E8E8E4;
  border-radius:8px;padding:4px 12px;margin-right:6px;}
</style>""", unsafe_allow_html=True)

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
                **{f"k_e_{i}": round(float(CROISS_CATEGORIES_DEFAUT[i]), 1)
                   for i in range(len(CATEGORIES_COUTS))},
                **{f"k_rc_{i}": RDT_CLASSES_DEFAUT[i] for i in range(len(CLASSES_ACTIFS))},
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

with st.sidebar.expander("D — Rendement par classe d'actifs (%)", expanded=False):
    rdt_classes = [st.slider(f"{CLASSES_ACTIFS[i]} ({POIDS_ACTIFS[i]*100:.0f} %)",
                             0.5, 12.0, step=0.05, key=f"k_rc_{i}")
                   for i in range(len(CLASSES_ACTIFS))]
    rendement = rendement_pondere(rdt_classes)
    st.caption(f"Rendement global pondéré : **{rendement*100:.2f} %**")

with st.sidebar.expander("E — Coûts pré-allocation par VP (%/an)", expanded=False):
    st.caption("La croissance de chaque enveloppe VP se propage aux blocs "
               "post-allocation, aux dépenses du P&L et au RSI.")
    croiss_cats = [st.slider(f"{CATEGORIES_COUTS[i]} · {VP_CATEGORIES[i]}",
                             0.0, 8.0, step=0.1, key=f"k_e_{i}")
                   for i in range(len(CATEGORIES_COUTS))]
# Taux de blocs post-allocation DÉRIVÉS (traçabilité dim_scenario_gld)
_cat30 = CAT_BASE_2025 * (1 + np.array(croiss_cats) / 100.0) ** 5
_b25 = ALLOC_CATEGORIES_VERS_BLOCS.T @ CAT_BASE_2025
_b30 = ALLOC_CATEGORIES_VERS_BLOCS.T @ _cat30
g_acq, g_attr, g_na = ((_b30 / _b25) ** (1 / 5) - 1).tolist()

params = dict(fact_volume=float(fact_volume), croiss_fam=[float(c) for c in croiss_fam],
              parts_canal=[float(p) for p in parts_canal], rendement=float(rendement),
              rdt_classes=[float(r) for r in rdt_classes],
              croiss_categories=[float(c) for c in croiss_cats],
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

                    }
                    if any(l.startswith("E_croiss_") and not l.startswith("E_croiss_couts")
                           for l in lev):
                        for i, cat in enumerate(CATEGORIES_COUTS):
                            charge[f"k_e_{i}"] = round(
                                lev.get(f"E_croiss_{cat}",
                                        CROISS_CATEGORIES_DEFAUT[i] / 100.0) * 100, 1)
                    else:   # scénario hérité (3 blocs) : projection sur les catégories
                        g_b = np.array([lev.get("E_croiss_couts_acquisition", 0.035),
                                        lev.get("E_croiss_couts_attribuables", 0.030),
                                        lev.get("E_croiss_couts_non_attrib", 0.020)])
                        for i in range(len(CATEGORIES_COUTS)):
                            charge[f"k_e_{i}"] = round(
                                float(ALLOC_CATEGORIES_VERS_BLOCS[i] @ g_b) * 100, 1)
                    if any(l.startswith("D_rdt_") for l in lev):
                        for i, cl in enumerate(CLASSES_ACTIFS):
                            charge[f"k_rc_{i}"] = round(
                                lev.get(f"D_rdt_{cl}", RDT_CLASSES_DEFAUT[i] / 100.0) * 100, 2)
                    else:   # ancien scénario (levier D scalaire) : mise à l'échelle des défauts
                        ratio = lev.get("D_rendement_placement", 0.0351) / 0.0351
                        for i in range(len(CLASSES_ACTIFS)):
                            charge[f"k_rc_{i}"] = round(RDT_CLASSES_DEFAUT[i] * ratio, 2)
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

# ---- 💡 Faits saillants automatisés (calculés une fois, distribués par onglet) -----
faits_ong = generer_faits_par_onglet(res, res_base, params, jf, annee_focus, scenario_id)

def bloc_faits(cle, titre):
    with st.container(border=True):
        st.markdown(f"**💡 Faits saillants — {titre}** *(générés du scénario courant)*")
        for fait in faits_ong[cle]:
            st.markdown(f"- {fait}")

# ---- Onglets ---------------------------------------------------------------------
ong1, ong_cap, ong_cout, ong2, ong_mois, ong3, ong4 = st.tabs(
    ["📊 Tableau de bord", "🏛️ Capital", "💸 Coûts", "🔁 Roll-forward CSM",
     "📅 Mensualisation", "⚖️ Comparaison", "📄 Détail"]
)

def valeurs_vue(series_list, jf, cumulatif):
    """Vue annuelle (valeur de l'année focus) ou cumulative (somme 2025 -> focus)."""
    return [float(np.sum(s[:jf + 1])) if cumulatif else float(s[jf]) for s in series_list]

with ong1:
    bloc_faits("tableau_de_bord", f"« {scenario_id} » · {annee_focus}")
    vue = st.radio("Vue des waterfalls", ["Annuelle", "Cumulative depuis 2025"],
                   horizontal=True, key="k_vue_src")
    cumul = vue.startswith("Cumulative")
    suffixe = f"cumul 2025-{annee_focus}" if cumul else str(annee_focus)
    gauche, droite = st.columns([1.15, 1])
    with gauche:
        vals_src = valeurs_vue([res["profit_attendu"], res["impact_ventes"],
                                res["experience"], res["depenses"],
                                res["interet_marche"], res["exploitation"]], jf, cumul)
        afficher_fig(fig_waterfall(
            ["Profit attendu (CSM + RA)", "Impact des ventes", "Expérience",
             "Dépenses", "Intérêt et marché", "Résultat d'exploitation"],
            vals_src,
            f"Sources de bénéfices {suffixe} — « {scenario_id} » (M$)"))
    with droite:
        afficher_fig(fig_heatmap(res["rsi_adj"], rendement, scenario_id))
    afficher_fig(fig_trajectoires(res, res_base, scenario_id))

with ong_cap:
    st.markdown(f"**Capital requis par catégorie de coussin — « {scenario_id} »** "
                "*(poids par coussin : hypothèses de démonstration)*")
    cc = res["capital_coussins"]           # (6 coussins x années), Diversification < 0
    couleurs_c = [COUL["bleu"], COUL["vert"], COUL["or"], "#8E44AD", "#16A085", COUL["rouge"]]
    g1, g2 = st.columns([1.15, 1])
    with g1:
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        bas = np.zeros(N)
        for i, c in enumerate(COUSSINS):
            if POIDS_COUSSINS[i] >= 0:
                ax.bar(ANNEES, cc[i], bottom=bas, color=couleurs_c[i], width=0.62,
                       label=c, zorder=3)
                bas += cc[i]
            else:
                ax.bar(ANNEES, cc[i], bottom=0, color=couleurs_c[i], width=0.62,
                       label=f"{c} (réduction)", hatch="//", zorder=3)
        ax.plot(ANNEES, res["capital"], marker="o", color="black", linewidth=2.2,
                label="Capital net à rémunérer", zorder=4)
        etiqueter_totaux(ax, ANNEES, bas)                       # sommet des coussins bruts
        etiqueter_points(ax, ANNEES, res["capital"], dy=-14)   # capital net
        ax.axhline(0, color=COUL["gris"], linewidth=0.8)
        ax.set_title("Empilement des coussins et capital net (M$)")
        ax.set_xticks(ANNEES)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_fr(x)))
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        afficher_fig(fig)
    with g2:
        vals_cap = [float(cc[i][jf]) for i in range(len(COUSSINS))] + [float(res["capital"][jf])]
        afficher_fig(fig_waterfall(
            COUSSINS + ["Capital net"], vals_cap,
            f"Composition du capital {annee_focus} (M$)",
            plafond=float(cc[:5, jf].sum()) * 1.2))
    bloc_faits("capital", "Capital")
    tbl_cap = pd.DataFrame(cc.round(0), index=COUSSINS,
                           columns=[str(a) for a in ANNEES])
    tbl_cap.loc["Capital net à rémunérer"] = res["capital"].round(0)
    st.dataframe(tbl_cap.reset_index(names="Coussin (M$)"), hide_index=True, width="stretch")

with ong_cout:
    st.markdown(f"**Cost module — piloté PRÉ-ALLOCATION par enveloppe de VP (levier E)** "
                f"*(« {scenario_id} » : chaque catégorie croît à son taux, puis "
                f"s'alloue vers les blocs post-allocation → dépenses du P&L → RSI)*")
    ca = res["couts_avant"]                # (8 catégories x années)
    couleurs_k = ["#1F5673", "#00874E", "#B8860B", "#8E44AD", "#16A085",
                  "#C0392B", "#7F8C8D", "#2C3E50"]
    g1, g2 = st.columns([1.15, 1])
    with g1:
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        bas = np.zeros(N)
        for i, c in enumerate(CATEGORIES_COUTS):
            ax.bar(ANNEES, ca[i], bottom=bas, color=couleurs_k[i], width=0.62,
                   label=c, zorder=3)
            bas += ca[i]
        etiqueter_totaux(ax, ANNEES, ca.sum(axis=0))
        ax.set_title("Coûts PRÉ-allocation par catégorie / VP (M$) — la source")
        ax.set_xticks(ANNEES)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_fr(x)))
        ax.legend(fontsize=7.5, ncol=2)
        fig.tight_layout()
        afficher_fig(fig)
    with g2:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for k, b in enumerate(BLOCS_COUTS):
            ax.plot(ANNEES, res["couts_blocs"][k], marker="o", linewidth=2.2,
                    color=[COUL["bleu"], COUL["vert"], COUL["or"]][k], label=b)
            etiqueter_points(ax, ANNEES, res["couts_blocs"][k],
                             couleur=[COUL["bleu"], COUL["vert"], COUL["or"]][k])
        ax.set_title("Blocs POST-allocation dérivés (M$)")
        ax.set_xticks(ANNEES)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_fr(x)))
        ax.legend(fontsize=8.5)
        fig.tight_layout()
        afficher_fig(fig)
    bloc_faits("couts", "Coûts")
    tbl_ca = pd.DataFrame(ca.round(1),
                          index=[f"{c} — {vp}" for c, vp in
                                 zip(CATEGORIES_COUTS, VP_CATEGORIES)],
                          columns=[str(a) for a in ANNEES])
    tbl_ca.loc["Total"] = ca.sum(axis=0).round(1)
    st.dataframe(tbl_ca.reset_index(names="Catégorie — VP responsable (M$)"),
                 hide_index=True, width="stretch")
    with st.expander("Matrice d'allocation (catégories → blocs, hypothèses ; lignes = 100 %)"):
        mat = pd.DataFrame(ALLOC_CATEGORIES_VERS_BLOCS * 100,
                           index=[f"{c} ({vp})" for c, vp in
                                  zip(CATEGORIES_COUTS, VP_CATEGORIES)],
                           columns=BLOCS_COUTS).round(1)
        st.dataframe(mat.reset_index(names="Catégorie \\ Bloc (%)"),
                     hide_index=True, width="stretch")

with ong2:
    bloc_faits("csm", "Roll-forward CSM")
    vue_csm = st.radio("Vue", ["Annuelle", "Cumulative depuis 2025"],
                       horizontal=True, key="k_vue_csm")
    cumul_csm = vue_csm.startswith("Cumulative")
    depart = res["csm_open"][0] if cumul_csm else res["csm_open"][jf]
    flux = valeurs_vue([-res["csm_release"], res["nb_csm"], res["csm_interet"],
                        np.full(N, res["exp_csm"]), res["chg_hyp"]], jf, cumul_csm)
    suff_csm = f"cumul 2025-{annee_focus}" if cumul_csm else str(annee_focus)
    afficher_fig(fig_waterfall(
        ["Solde CSM départ", "Profit attendu relâché", "Impact des ventes profitables",
         "Intérêt et marché", "Expérience", "Changements d'hypothèses", "Solde CSM fin"],
        [depart] + flux + [res["csm_close"][jf]],
        f"Roll-forward du CSM {suff_csm} — « {scenario_id} » (M$)",
        plafond=max(depart, res["csm_close"][jf]) * 1.18))
    st.dataframe(pd.DataFrame({
        "Année": ANNEES,
        "Solde départ (M$)": res["csm_open"].round(0),
        "Relâche (M$)": (-res["csm_release"]).round(0),
        "Ventes profitables (M$)": res["nb_csm"].round(0),
        "Intérêt (M$)": res["csm_interet"].round(0),
        "Solde fin (M$)": res["csm_close"].round(0),
    }), hide_index=True, width="stretch")

with ong_mois:
    st.markdown(f"**Mensualisation exploratoire — « {scenario_id} »** "
                "*(profils de saisonnalité : hypothèses ajustables ; non persisté au Gold)*")
    c_sel1, c_sel2, c_sel3, c_sel4 = st.columns([1, 1.6, 1.6, 1.2])
    annee_m = c_sel1.selectbox("Année", ANNEES, index=len(ANNEES) - 1, key="k_m_annee")
    jm = ANNEES.index(annee_m)
    series_m = {
        "Ventes totales (K$)": (res["ventes_scen"].sum(axis=0),
                                "Saisonnalité ventes (REER + automne)", 0),
        "Résultat d'exploitation (M$)": (res["exploitation"],
                                         "Saisonnalité ventes (REER + automne)", 1),
        "Résultat net (M$)": (res["resultat_net"],
                              "Saisonnalité ventes (REER + automne)", 1),
        "Coûts totaux (M$)": (res["couts_blocs"].sum(axis=0),
                              "Charge de fin d'année (coûts)", 1),
    }
    choix_m = c_sel2.selectbox("Ligne à mensualiser", list(series_m.keys()), key="k_m_serie")
    serie_ann, profil_defaut, dec_m = series_m[choix_m]
    profil_m = c_sel3.selectbox("Profil de saisonnalité", list(PROFILS_MENSUELS.keys()),
                                index=list(PROFILS_MENSUELS.keys()).index(profil_defaut),
                                key="k_m_profil")
    intensite = c_sel4.slider("Intensité du profil", 0.0, 1.0, 1.0, 0.05, key="k_m_int")

    mens = mensualiser(serie_ann[jm], profil_m, intensite)
    cum = np.cumsum(mens)

    # Panneau sombre à barres vertes + ligne de cumul (inspiré du visuel fourni)
    fig, ax = plt.subplots(figsize=(12.5, 4.4))
    fig.patch.set_facecolor("#2B3A4A")
    ax.set_facecolor("#2B3A4A")
    ax.bar(range(12), mens, color="#8FD98F", width=0.55, zorder=3,
           label=f"{choix_m} — mensuel")
    for x_b, v_b in enumerate(mens):
        ax.annotate(fmt_fr(float(v_b), dec_m), (x_b, float(v_b)),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=8, color="white", fontweight="bold")
    ax2 = ax.twinx()
    ax2.plot(range(12), cum, color="#F5B041", marker="o", linewidth=2.2,
             label="Cumul depuis janvier", zorder=4)
    for a in (ax, ax2):
        a.grid(False)
        for cote in a.spines.values():
            cote.set_visible(False)
        a.tick_params(colors="white", labelsize=9.5)
    ax.set_xticks(range(12), MOIS)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_fr(x, dec_m)))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_fr(x, dec_m)))
    ax.set_title(f"{choix_m} — {annee_m} · profil « {profil_m} » (intensité "
                 f"{intensite:.0%}) · scénario « {scenario_id} »",
                 color="white", fontsize=12.5, fontweight="bold")
    l1, e1 = ax.get_legend_handles_labels()
    l2, e2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, e1 + e2, fontsize=9, facecolor="#2B3A4A",
              labelcolor="white", edgecolor="#44576B")
    fig.tight_layout()
    afficher_fig(fig)

    tbl_m = pd.DataFrame({"Mois": MOIS, choix_m: np.round(mens, dec_m),
                          "Cumul": np.round(cum, dec_m),
                          "Poids (%)": np.round(mens / mens.sum() * 100, 1)})
    cg, cd = st.columns([1.4, 1])
    cg.dataframe(tbl_m, hide_index=True, width="stretch")
    with cd:
        st.metric(f"Total {annee_m}", fmt_fr(float(serie_ann[jm]), dec_m))
        pointe = int(np.argmax(mens))
        st.metric("Mois de pointe", MOIS[pointe],
                  delta=f"{mens[pointe] / mens.sum() * 100:.1f} % de l'année")
        st.download_button("⬇️ Télécharger (CSV)",
                           tbl_m.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                           file_name=f"mensualisation_{scenario_id}_{annee_m}.csv",
                           mime="text/csv", key="k_m_dl")
    st.caption("💡 Prochaine itération possible : persister la vue mensuelle par scénario "
               "dans une table Gold (`forecast_mensuel_gld`) pour Genie et Power BI.")

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
            if len(d):
                etiqueter_points(ax, [d["annee"].iloc[-1]], [d["valeur"].iloc[-1]],
                                 dec=1, couleur=couleurs[s_i % len(couleurs)])
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

    st.markdown("**Correspondance dimensions FP → Oracle EPM** "
                "*(format d'export : Account · Entity · Scenario · Version · Period · Year · LOB · Amount)*")
    st.dataframe(ORACLE_MAPPING.rename(columns={
        "ligne_fp": "Ligne du plan", "compte_oracle": "Compte Oracle",
        "description_oracle": "Description EPM"}), hide_index=True, width="stretch")

    st.download_button(
        "⬇️ Télécharger le P&L (CSV)",
        pnl.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
        file_name=f"pnl_{scenario_id}.csv", mime="text/csv",
    )

st.caption("Baseline immuable + overlay de leviers → recalcul du Gold par scénario · "
           f"Schéma Unity Catalog : `{CATALOG}.{SCHEMA}` · Données synthétiques.")
