# Databricks notebook source
# MAGIC %md
# MAGIC # 📊 Plan financier simplifié par leviers — Assurance individuelle (vie) · **v2**
# MAGIC ### Preuve de concept · Databricks Free Edition · IFRS 17
# MAGIC
# MAGIC > ⚠️ **Données synthétiques à des fins de démonstration.** Tous les chiffres sont
# MAGIC > fictifs, à une échelle de démonstration volontairement décalée de tout ordre de
# MAGIC > grandeur réel. Ils ne représentent aucune donnée de l'organisation.
# MAGIC
# MAGIC ---
# MAGIC ## 🧭 Démarche
# MAGIC **1️⃣ Ventes d'affaires nouvelles → 2️⃣ CSM des ventes + VAN → 3️⃣ État des résultats
# MAGIC IFRS 17 → 4️⃣ Capital à rémunérer → 5️⃣ RSI (ROE) → 6️⃣ Sources de bénéfices →
# MAGIC 7️⃣ Roll-forward du CSM → 🎛️ Leviers → tout se recalcule**
# MAGIC
# MAGIC | Bloc | Approche | Outil |
# MAGIC |---|---|---|
# MAGIC | **Passifs — In-force** | Baseline synthétique du bloc en vigueur (relâche CSM + RA, expérience), 6 familles de produits | Delta (médaillon) |
# MAGIC | **Passifs — Affaires nouvelles** | Ventes par famille × canal → marge CSM et VAN@12 % ; leviers A (volume), B (croissance par famille), C (mix canal) en overlay | Moteur pandas |
# MAGIC | **Combiné (In-force + AN)** | Alimente l'état des résultats IFRS 17 et le roll-forward du CSM | Moteur pandas |
# MAGIC | **Actifs** | Résultat financier ≈ actifs investis × rendement (levier D) − accrétion des passifs | Moteur pandas |
# MAGIC | **Consolidé** | Capital à rémunérer → RSI, sources de bénéfices, coûts unitaires (levier E) ; write-back par scénario | Delta + app |
# MAGIC
# MAGIC **Principe clé : baseline immuable + overlay de leviers → recalcul du Gold par scénario.**
# MAGIC
# MAGIC ---
# MAGIC ## ⚠️ Prérequis v2 : le module `moteur_plan.py`
# MAGIC Le moteur (constantes, calculs, faits saillants) vit dans **`moteur_plan.py`**,
# MAGIC **source unique de vérité partagée avec l'app Streamlit**. Déposer ce fichier dans
# MAGIC le **même dossier Workspace que ce notebook** (Workspace → Import → File).
# MAGIC
# MAGIC ## ▶️ Mode d'emploi
# MAGIC 1. **Run all** → crée/rafraîchit le scénario nommé dans `01_scenario_id`.
# MAGIC 2. Changer le nom + bouger des leviers → **Run all** → nouveau scénario côte à côte.
# MAGIC 3. La section Comparaison lit tous les scénarios de `kpi_gld` (notebook **et** app).

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ 0. Configuration — import du moteur partagé

# COMMAND ----------

import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import moteur_plan as mp
    importlib.reload(mp)   # recharge les modifications éventuelles du module
except ImportError as e:
    raise ImportError(
        "❌ Module « moteur_plan.py » introuvable. Le déposer dans le MÊME dossier "
        "Workspace que ce notebook (Workspace → Import → File), puis relancer."
    ) from e

# Alias vers la source unique de vérité
ANNEES, N = mp.ANNEES, mp.N
PRODUITS, CANAUX = mp.PRODUITS, mp.CANAUX
COUL, fmt_fr, fmt_m, fmt_pct = mp.COUL, mp.fmt_fr, mp.fmt_m, mp.fmt_pct

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": COUL["fond"],
    "axes.edgecolor": "#DDDDDD", "axes.grid": True, "grid.color": "#E3E3E0",
    "grid.linewidth": 0.7, "font.size": 11, "axes.titlesize": 14,
    "axes.titleweight": "bold",
})

print(f"Moteur partagé chargé ✅  ({len(PRODUITS)} familles · horizon {ANNEES[0]}-{ANNEES[-1]})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎛️ 1. Leviers ajustables (widgets)
# MAGIC | Levier | Widget(s) | Effet |
# MAGIC |---|---|---|
# MAGIC | — | `01_scenario_id` | Nom du scénario (clé du write-back) |
# MAGIC | **A** Volume | `02_facteur_volume` | Facteur multiplicatif global (0,80 → 1,30) |
# MAGIC | **B** Croissance par famille | `03_croiss_familles` | 6 valeurs séparées par « ; » en **pts de %/an vs plan**, dans l'ordre : VE participation ; VE paiements limités ; Autres VE ; Temporaires ; Maladies graves ; Autres inv. et maladie |
# MAGIC | **C** Mix canal | `04_`/`05_`/`06_part_*` | Parts Indépendants / Desjardins / Agents (renormalisées) |
# MAGIC | **D** Rendement | `07_rendement_pct` | Rendement de placement (défaut 3,50 %) |
# MAGIC | **E** Dépenses | `08_`/`09_`/`10_croiss_*` | Croissance annuelle des 3 blocs de coûts |

# COMMAND ----------

# ⚠️ Si vous changez les NOMS des widgets, exécuter UNE FOIS : dbutils.widgets.removeAll()

dbutils.widgets.text("01_scenario_id", "Base", "01 · Nom du scénario")
dbutils.widgets.text("02_facteur_volume", "1.00", "02 · A — Facteur volume")
dbutils.widgets.text("03_croiss_familles", "0; 0; 0; 0; 0; 0",
                     "03 · B — Croiss./famille (pts/an, 6 valeurs ;)")
dbutils.widgets.text("04_part_independants_pct", "60", "04 · C — Part Indépendants (%)")
dbutils.widgets.text("05_part_desjardins_pct", "25", "05 · C — Part Desjardins (%)")
dbutils.widgets.text("06_part_agents_pct", "15", "06 · C — Part Agents Desj. (%)")
dbutils.widgets.text("07_rendement_pct", "3.50", "07 · D — Rendement placement (%)")
dbutils.widgets.text("08_croiss_couts_acquisition_pct", "3.5", "08 · E — Croiss. coûts acquisition (%/an)")
dbutils.widgets.text("09_croiss_couts_attribuables_pct", "3.0", "09 · E — Croiss. coûts attribuables (%/an)")
dbutils.widgets.text("10_croiss_couts_non_attribuables_pct", "2.0", "10 · E — Croiss. coûts non attr. (%/an)")

def _f(nom, defaut):
    try:
        return float(dbutils.widgets.get(nom).replace(",", ".").strip())
    except Exception:
        return defaut

SCENARIO_ID = dbutils.widgets.get("01_scenario_id").strip() or "Base"
FACT_VOLUME = max(0.5, min(1.5, _f("02_facteur_volume", 1.0)))

_brut = dbutils.widgets.get("03_croiss_familles").split(";")
CROISS_FAM = []
for v in _brut:
    v = v.strip().replace(",", ".")
    try:
        CROISS_FAM.append(max(-10.0, min(10.0, float(v))))
    except Exception:
        CROISS_FAM.append(0.0)
CROISS_FAM = (CROISS_FAM + [0.0] * len(PRODUITS))[: len(PRODUITS)]

_parts = np.array([_f("04_part_independants_pct", 60.0),
                   _f("05_part_desjardins_pct", 25.0),
                   _f("06_part_agents_pct", 15.0)])
PARTS_CANAL = (_parts / _parts.sum()).tolist()
RENDEMENT = max(0.005, _f("07_rendement_pct", 3.5) / 100.0)
G_ACQ = _f("08_croiss_couts_acquisition_pct", 3.5) / 100.0
G_ATTR = _f("09_croiss_couts_attribuables_pct", 3.0) / 100.0
G_NA = _f("10_croiss_couts_non_attribuables_pct", 2.0) / 100.0

PARAMS = dict(fact_volume=FACT_VOLUME, croiss_fam=CROISS_FAM, parts_canal=PARTS_CANAL,
              rendement=RENDEMENT, g_acq=G_ACQ, g_attr=G_ATTR, g_na=G_NA)

print(f"Scénario « {SCENARIO_ID} » | A={FACT_VOLUME:.2f} | B={CROISS_FAM} pts/an")
print(f"C={['%.0f %%' % (p*100) for p in PARTS_CANAL]} | D={RENDEMENT*100:.2f} % | "
      f"E={G_ACQ*100:.1f}/{G_ATTR*100:.1f}/{G_NA*100:.1f} %/an")

# COMMAND ----------

# ---- Catalog auto-détecté + schéma versionné + write-back ----------------------
CATALOG = spark.sql("SELECT current_catalog()").collect()[0][0]
SCHEMA = "plan_assurance_ind_v1"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")
spark.sql(f"USE `{CATALOG}`.`{SCHEMA}`")

def nom_table(t):
    return f"`{CATALOG}`.`{SCHEMA}`.`{t}`"

def ecrire_reference(pdf, table, commentaire=""):
    """Table de référence (bronze / dimensions) : déterministe, overwrite complet."""
    sdf = spark.createDataFrame(pdf)
    sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(nom_table(table))
    if commentaire:
        com = commentaire.replace("'", "''")
        spark.sql(f"COMMENT ON TABLE {nom_table(table)} IS '{com}'")

def ecrire_par_scenario(pdf, table, scenario_id):
    """Write-back PAR SCÉNARIO : DELETE ciblé + append (baseline immuable, idempotent)."""
    sdf = spark.createDataFrame(pdf)
    if spark.catalog.tableExists(nom_table(table)):
        sid = scenario_id.replace("'", "''")
        spark.sql(f"DELETE FROM {nom_table(table)} WHERE scenario_id = '{sid}'")
        sdf.write.mode("append").saveAsTable(nom_table(table))
    else:
        sdf.write.mode("overwrite").saveAsTable(nom_table(table))

# ---- Migration v1 -> v2 (une seule fois) -----------------------------------------
# La v1 avait créé ventes_conformees_slv avec la colonne « ventes_baseline_k » ;
# la v2 écrit « ventes_k ». On recrée la table si l'ancien schéma est détecté
# (elle ne contient que des scénarios à l'ancienne échelle, à réécrire de toute façon).
if spark.catalog.tableExists(nom_table("ventes_conformees_slv")):
    cols_v1 = [c.name for c in spark.table(nom_table("ventes_conformees_slv")).schema]
    if "ventes_baseline_k" in cols_v1:
        spark.sql(f"DROP TABLE {nom_table('ventes_conformees_slv')}")
        print("Migration v1 -> v2 : ventes_conformees_slv recréée (nouveau schéma).")

print(f"Catalog : {CATALOG} | Schéma : {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🥉 2. Couche Bronze — intrants synthétiques (échelle de démonstration)
# MAGIC Écrite depuis les constantes du module partagé : **les niveaux visibles dans le
# MAGIC Catalog sont donc eux aussi à l'échelle fictive.**

# COMMAND ----------

ventes_brz = pd.DataFrame(
    [(p, a, round(float(mp.VENTES_BASE[i, j]), 1))
     for i, p in enumerate(PRODUITS) for j, a in enumerate(ANNEES)],
    columns=["produit", "annee", "ventes_k"],
)

hypotheses_brz = pd.DataFrame(
    [
        ("mix_canal_defaut_independants", 0.60, "HYPOTHÈSE de démonstration (levier C)"),
        ("mix_canal_defaut_desjardins", 0.25, "HYPOTHÈSE de démonstration (levier C)"),
        ("mix_canal_defaut_agents", 0.15, "HYPOTHÈSE de démonstration (levier C)"),
        ("rendement_placement_defaut", 0.035, "Rendement de base (levier D)"),
        ("taux_impot_effectif", 0.24, "Taux d'impôt effectif synthétique"),
        ("taux_relache_csm", 0.071, "Relâche annuelle du profit attendu (CSM)"),
        ("taux_interet_csm", 0.0315, "Accroissement d'intérêt sur le solde CSM"),
        ("marge_csm_ventes_2025", 0.75, "CSM des ventes / ventes (2025), croît vers 1.03"),
        ("csm_solde_depart_m", 1937.0 * mp.ECHELLE, "Solde CSM au 1er janv. 2025 (échelle démo)"),
        ("capital_a_remunerer_2025_m", 464.0 * mp.ECHELLE, "Capital moyen 2025 (échelle démo)"),
        ("capital_a_remunerer_2030_m", 581.0 * mp.ECHELLE, "Capital moyen 2030 (échelle démo)"),
        ("ratio_actifs_sur_capital", 5.6, "Actifs investis ≈ 5,6 x capital (proxy)"),
    ],
    columns=["hypothese", "valeur", "commentaire"],
)

couts_brz = pd.DataFrame(
    [
        ("Coûts d'acquisition attribuables", float(mp.BASE_COUTS[0]), "Différés en partie ; levier E-acq"),
        ("Coûts attribuables récurrents", float(mp.BASE_COUTS[1]), "Administration ; levier E-attr"),
        ("Coûts non attribuables", float(mp.BASE_COUTS[2]), "Frais généraux ; levier E-non-attr"),
    ],
    columns=["bloc_cout", "montant_2025_m", "commentaire"],
)

oracle_mapping_brz = pd.DataFrame(
    [
        ("Produits d'assurance", "AC4100", "Revenus d'assurance"),
        ("Charges d'assurance", "AC4200", "Charges de services d'assurance"),
        ("Réassurance nette", "AC4300", "Résultat net de réassurance"),
        ("Résultats des activités d'assurance", "AC4900", "Sous-total activités d'assurance"),
        ("Résultat financier", "AC5100", "Produits financiers nets IFRS 17"),
        ("Résultats autres", "AC5200", "Autres produits et charges"),
        ("Résultat d'exploitation", "AC5900", "Sous-total exploitation"),
        ("Impôts", "AC6100", "Charge d'impôt"),
        ("Résultat net", "AC6900", "Résultat net attribuable"),
        ("Capital à rémunérer moyen", "KP7100", "Capital économique moyen"),
        ("Solde CSM fin", "KP7200", "Marge de service contractuelle"),
    ],
    columns=["ligne_fp", "compte_oracle", "description_oracle"],
)

ecrire_reference(ventes_brz, "ventes_brz", "Bronze - ventes synthétiques (échelle démo, 6 familles)")
ecrire_reference(hypotheses_brz, "hypotheses_brz", "Bronze - hypothèses de base (synthétiques)")
ecrire_reference(couts_brz, "couts_brz", "Bronze - modèle de coûts synthétique (échelle démo)")
ecrire_reference(oracle_mapping_brz, "oracle_mapping_brz", "Bronze - correspondance FP -> Oracle EPM")
print("Tables Bronze écrites (échelle de démonstration).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧮 3. Moteur (module partagé) + couches Silver et Gold

# COMMAND ----------

# ---- Moteur : LE MÊME calcul que l'app Streamlit --------------------------------
res = mp.calculer_scenario(SCENARIO_ID, **PARAMS)
S = res

print(f"Moteur exécuté pour « {SCENARIO_ID} » (module partagé) :")
print(f"  Résultat net : {fmt_m(S['resultat_net'][0])} (2025) → {fmt_m(S['resultat_net'][-1])} (2030)")
print(f"  RSI global   : {fmt_pct(S['rsi_global'][0])} → {fmt_pct(S['rsi_global'][-1])}")
print(f"  Solde CSM    : {fmt_m(S['csm_open'][0])} → {fmt_m(S['csm_close'][-1])}")

# ---- Lignes Gold préparées par le module (mêmes formats que l'app) ---------------
overlay_l, forecast_l, kpi_l, dim_l = mp.construire_lignes(res, PARAMS, SCENARIO_ID)

overlay_pdf = pd.DataFrame(overlay_l, columns=["scenario_id", "levier", "valeur",
                                               "valeur_baseline", "horodatage"])
forecast_pdf = pd.DataFrame(forecast_l, columns=["scenario_id", "annee", "section",
                                                 "ligne", "ordre", "montant_m"])
kpi_pdf = pd.DataFrame(kpi_l, columns=["scenario_id", "annee", "niveau", "produit",
                                       "canal", "kpi", "valeur", "unite"])
dim_pdf = pd.DataFrame(dim_l, columns=["scenario_id", "horodatage", "facteur_volume",
                                       "part_independants", "part_desjardins", "part_agents",
                                       "rendement_placement", "croiss_couts_acquisition",
                                       "croiss_couts_attribuables", "croiss_couts_non_attribuables"])
# Compatibilité avec la table existante (colonne de l'ancien levier B, désormais vide)
dim_pdf.insert(3, "accent_croissance", np.nan)

# ---- Silver ----------------------------------------------------------------------
lignes = [(SCENARIO_ID, p, c, a, round(float(res["ventes_pc"][i, j, c_i]), 1))
          for i, p in enumerate(PRODUITS) for j, a in enumerate(ANNEES)
          for c_i, c in enumerate(CANAUX)]
ventes_conf_pdf = pd.DataFrame(lignes, columns=["scenario_id", "produit", "canal",
                                                "annee", "ventes_k"])
taux = [G_ACQ, G_ATTR, G_NA]
couts_post_pdf = pd.DataFrame(
    [(SCENARIO_ID, b, a, round(float(mp.BASE_COUTS[k] * (1 + taux[k]) ** t), 2))
     for k, b in enumerate(couts_brz["bloc_cout"]) for t, a in enumerate(ANNEES)],
    columns=["scenario_id", "bloc_cout", "annee", "montant_m"],
)

ecrire_par_scenario(ventes_conf_pdf, "ventes_conformees_slv", SCENARIO_ID)
ecrire_par_scenario(overlay_pdf, "overlay_drivers_slv", SCENARIO_ID)
ecrire_par_scenario(couts_post_pdf, "couts_post_alloc_slv", SCENARIO_ID)
ecrire_reference(oracle_mapping_brz, "oracle_mapping_slv", "Silver - mapping FP -> Oracle EPM conformé")

# ---- Gold (write-back par scénario) -----------------------------------------------
ecrire_par_scenario(forecast_pdf, "forecast_output_gld", SCENARIO_ID)
ecrire_par_scenario(kpi_pdf, "kpi_gld", SCENARIO_ID)
ecrire_par_scenario(dim_pdf, "dim_scenario_gld", SCENARIO_ID)

business_drivers_dim = pd.DataFrame(
    [
        ("A", "Volume de ventes global", "Facteur multiplicatif sur les ventes totales", "0.80 - 1.30"),
        ("B", "Croissance par famille", "Points de %/an vs la trajectoire du plan, par famille de produit", "-10 - +10 pts/an"),
        ("C", "Mix canal", "Parts Indépendants / Desjardins / Agents (somme 100 %)", "0 - 100 %"),
        ("D", "Rendement / spread de placement", "Pilote le résultat financier, l'intérêt CSM et le RSI", "2.5 % - 5.0 %"),
        ("E", "Dépenses ventilées", "Croissance annuelle des 3 blocs de coûts", "0 % - 8 %/an"),
    ],
    columns=["levier", "nom", "description", "plage_suggeree"],
)
ecrire_reference(business_drivers_dim, "business_drivers_dim_gld", "Gold - dimension des leviers (v2)")

nb = spark.table(nom_table("dim_scenario_gld")).count()
print(f"Silver + Gold écrits. Scénarios dans le modèle : {nb}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 📈 4. Tableau de bord du scénario courant

# COMMAND ----------

ANNEE_FOCUS = 2030
jf = ANNEES.index(ANNEE_FOCUS)

# ---- 4.1 Cartes KPI ---------------------------------------------------------------
cartes = [
    ("Résultat net", fmt_m(S["resultat_net"][jf]), COUL["vert"],
     f"vs {fmt_m(S['resultat_net'][0])} en 2025"),
    ("RSI global (ROE)", fmt_pct(S["rsi_global"][jf]), COUL["bleu"],
     f"vs {fmt_pct(S['rsi_global'][0])} en 2025"),
    ("VAN des affaires nouvelles", fmt_m(S["van_tot_m"][jf], 1),
     COUL["rouge"] if S["van_tot_m"][jf] < 0 else COUL["vert"],
     "VAN @ 12 % — négatif = destruction de valeur"),
    ("Solde CSM (fin)", fmt_m(S["csm_close"][jf]), COUL["or"],
     f"départ 2025 : {fmt_m(S['csm_open'][0])}"),
]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
fig.suptitle(f"Indicateurs clés {ANNEE_FOCUS} — scénario « {SCENARIO_ID} »  ·  données synthétiques",
             fontsize=15, fontweight="bold", y=1.04)
for ax, (titre, valeur, couleur, sous) in zip(axes, cartes):
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.9, transform=ax.transAxes,
                               facecolor="white", edgecolor="#DDDDDD", linewidth=1.2))
    ax.text(0.5, 0.78, titre, ha="center", va="center", fontsize=12,
            color=COUL["gris"], transform=ax.transAxes)
    ax.text(0.5, 0.45, valeur, ha="center", va="center", fontsize=22,
            fontweight="bold", color=couleur, transform=ax.transAxes)
    ax.text(0.5, 0.15, sous, ha="center", va="center", fontsize=9.5,
            color=COUL["gris"], transform=ax.transAxes)
plt.tight_layout()
plt.show()

# ---- 4.2 💡 Faits saillants automatisés (module partagé) ---------------------------
faits = mp.generer_faits_saillants(res, mp.calculer_scenario("Base", **mp.PARAMS_BASE),
                                   PARAMS, jf, ANNEE_FOCUS, SCENARIO_ID)
print("💡 FAITS SAILLANTS —", SCENARIO_ID, "·", ANNEE_FOCUS)
for f in faits:
    print("  •", f.replace("**", ""))

# COMMAND ----------

# ---- 4.3 Waterfall des sources de bénéfices + bridge CSM ---------------------------
def waterfall(ax, etiquettes, valeurs, titre, plafond=None):
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
    ax.set_xticklabels([e.replace(" ", "\n", 1) for e in etiquettes], fontsize=9.5)
    ax.set_title(titre)
    ax.set_ylabel("M$")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_fr(x)))

fig, ax = plt.subplots(figsize=(13, 5.2))
waterfall(ax, ["Profit attendu (CSM + RA)", "Impact des ventes", "Expérience",
               "Dépenses", "Intérêt et marché", "Résultat d'exploitation"],
          [S["profit_attendu"][jf], S["impact_ventes"][jf], S["experience"][jf],
           S["depenses"][jf], S["interet_marche"][jf], S["exploitation"][jf]],
          f"Sources de bénéfices {ANNEE_FOCUS} — « {SCENARIO_ID} » (M$, synthétique)")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(13, 5.2))
waterfall(ax, ["Solde CSM départ", "Profit attendu relâché", "Impact des ventes profitables",
               "Intérêt et marché", "Expérience", "Changements d'hypothèses", "Solde CSM fin"],
          [S["csm_open"][jf], -S["csm_release"][jf], S["nb_csm"][jf], S["csm_interet"][jf],
           S["exp_csm"], S["chg_hyp"][jf], S["csm_close"][jf]],
          f"Roll-forward du CSM {ANNEE_FOCUS} — « {SCENARIO_ID} » (M$, synthétique)",
          plafond=max(S["csm_open"][jf], S["csm_close"][jf]) * 1.18)
plt.tight_layout()
plt.show()

# COMMAND ----------

# ---- 4.4 Heatmap RSI produit × canal + trajectoires --------------------------------
fig, ax = plt.subplots(figsize=(9.5, 5.6))
donnees = np.ma.masked_invalid(res["rsi_adj"])
cmap_rsi = plt.get_cmap("RdYlGn").copy()
cmap_rsi.set_bad(color="#E8E8E4")
im = ax.imshow(donnees, cmap=cmap_rsi, vmin=-12, vmax=20, aspect="auto")
ax.set_xticks(range(len(CANAUX)), CANAUX, fontsize=11)
ax.set_yticks(range(len(PRODUITS)), PRODUITS, fontsize=10)
for i in range(len(PRODUITS)):
    for j in range(len(CANAUX)):
        v = res["rsi_adj"][i, j]
        if np.isnan(v):
            ax.text(j, i, "n.d.", ha="center", va="center", color=COUL["gris"], fontsize=9)
        else:
            ax.text(j, i, fmt_pct(v), ha="center", va="center", fontsize=10.5,
                    fontweight="bold", color="black" if -8 < v < 30 else "white")
ax.set_title(f"RSI des ventes par produit × canal — « {SCENARIO_ID} » · "
             f"rendement {RENDEMENT*100:.2f} % · données synthétiques", fontsize=12.5)
fig.colorbar(im, ax=ax, shrink=0.85).set_label("RSI (%)")
ax.grid(False)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
fig.suptitle(f"Trajectoires 2025-2030 — « {SCENARIO_ID} »  ·  données synthétiques",
             fontsize=14, fontweight="bold")
for ax, (titre, cle) in zip(axes, [("Résultat net (M$)", "resultat_net"),
                                   ("RSI global (%)", "rsi_global"),
                                   ("Coût d'acquisition / police ($)", "cout_par_police")]):
    ax.plot(ANNEES, S[cle], marker="o", color=COUL["vert"], linewidth=2.5)
    ax.set_title(titre)
    ax.set_xticks(ANNEES)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_fr(x)))
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # ⚖️ 5. Comparaison des scénarios (lit `kpi_gld` — notebook **et** app)

# COMMAND ----------

kpi_all = (
    spark.table(nom_table("kpi_gld"))
    .filter("niveau = 'global' AND kpi IN ('rsi_global_pct','resultat_net_m','van_totale_m')")
    .toPandas()
)
scenarios = sorted(kpi_all["scenario_id"].unique())
couleurs_scen = [COUL["bleu"], COUL["vert"], COUL["or"], COUL["rouge"], "#8E44AD", "#16A085"]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
fig.suptitle(f"Comparaison de scénarios ({len(scenarios)} scénario(s))  ·  données synthétiques",
             fontsize=14, fontweight="bold")
for k, (kpi_nom, titre) in enumerate([("rsi_global_pct", "RSI global (%)"),
                                      ("resultat_net_m", "Résultat net (M$)")]):
    ax = axes[k]
    for s_i, sc in enumerate(scenarios):
        d = (kpi_all[(kpi_all["scenario_id"] == sc) & (kpi_all["kpi"] == kpi_nom)]
             .sort_values("annee"))
        ax.plot(d["annee"], d["valeur"], marker="o", linewidth=2.5,
                color=couleurs_scen[s_i % len(couleurs_scen)], label=sc)
    ax.set_title(titre)
    ax.set_xticks(ANNEES)
    ax.legend(title="Scénario", fontsize=10)
plt.tight_layout()
plt.show()

recap = (kpi_all[kpi_all["annee"] == 2030]
         .pivot_table(index="scenario_id", columns="kpi", values="valeur")
         .rename(columns={"resultat_net_m": "Résultat net 2030 (M$)",
                          "rsi_global_pct": "RSI 2030 (%)",
                          "van_totale_m": "VAN 2030 (M$)"})
         .round(1).reset_index())
display(spark.createDataFrame(recap))

# 🧹 Ménage optionnel : purger un scénario écrit avec l'ANCIENNE version (échelle x1,
# 8 familles) qui fausserait la comparaison. Décommenter, ajuster le nom, exécuter :
# for t in ["kpi_gld", "forecast_output_gld", "overlay_drivers_slv",
#           "dim_scenario_gld", "ventes_conformees_slv", "couts_post_alloc_slv"]:
#     spark.sql(f"DELETE FROM {nom_table(t)} WHERE scenario_id = 'V2'")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # 🧞 6. Genie en français
# MAGIC 1. Menu de gauche → **Genie** → **New** ; ajouter **`kpi_gld`**, **`forecast_output_gld`**
# MAGIC    (et `dim_scenario_gld`) du schéma `plan_assurance_ind_v1`.
# MAGIC 2. Instructions du Space : *« Réponds en français. Montants en M$ sauf colonnes
# MAGIC    suffixées _k (K$) et _$ (dollars). RSI en %. Données synthétiques de démonstration. »*
# MAGIC
# MAGIC **Questions de démo :** « Quel canal a le meilleur RSI en 2027 ? » ·
# MAGIC « Quel produit détruit de la valeur en 2026 et de combien ? » ·
# MAGIC « Compare le résultat net des scénarios entre 2025 et 2030. »

# COMMAND ----------

# ---- STUB — Export Oracle EPM (baseline seulement) ---------------------------------
fc = spark.table(nom_table("forecast_output_gld")).toPandas()
mp_oracle = spark.table(nom_table("oracle_mapping_slv")).toPandas()
base = fc[(fc["scenario_id"] == "Base")
          & (fc["section"].isin(["etat_resultats", "capital", "csm_rollforward"]))]
base = base.merge(mp_oracle, left_on="ligne", right_on="ligne_fp", how="inner")
if len(base) > 0:
    oracle_out = pd.DataFrame({
        "Account": base["compte_oracle"], "Entity": "ASSUR_IND", "Scenario": "Plan",
        "Version": "Working", "Period": "YearTotal",
        "Year": "FY" + base["annee"].astype(str).str[2:],
        "LOB": "Assurance individuelle", "Amount": base["montant_m"],
    })
    ecrire_reference(oracle_out, "forecast_output_oracle_gld",
                     "Gold - export baseline au format Oracle EPM (stub)")
    print(f"forecast_output_oracle_gld : {len(oracle_out)} lignes (baseline).")
else:
    print("⚠️ Écrire d'abord un scénario « Base » (Run all avec 01_scenario_id = Base).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### STUB — Rolling forecast (à venir)
# MAGIC Actuals jusqu'à la coupure → projection ré-ancrée → nouvel overlay stocké comme
# MAGIC nouveau scénario (ex. `RF_2026T3_Base`) via le même write-back. Visuels, app et
# MAGIC Genie fonctionnent sans modification.
# MAGIC
# MAGIC ---
# MAGIC # ✅ Définition de « terminé » (v2)
# MAGIC - [x] **Source unique de vérité** : `moteur_plan.py` partagé notebook ↔ app (chiffres identiques garantis).
# MAGIC - [x] 6 familles · levier B par famille · échelle de démonstration · 2025 lissé.
# MAGIC - [x] Run all idempotent, médaillon visible au Catalog, mention « données synthétiques ».
# MAGIC - [x] Faits saillants automatisés (même génération que l'app).
# MAGIC
# MAGIC 💡 **Scénario de démo (2ᵉ run)** : `01_scenario_id = Croissance MG`,
# MAGIC `03_croiss_familles = 0; 0; 0; -2; 3; 0`, rendement `4.00` → les Maladies graves
# MAGIC accélèrent, les Temporaires ralentissent, RSI et CSM remontent.

# COMMAND ----------

display(spark.sql(f"SHOW TABLES IN `{CATALOG}`.`{SCHEMA}`"))
