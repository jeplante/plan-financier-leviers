# -*- coding: utf-8 -*-
"""
moteur_plan.py — SOURCE UNIQUE DE VÉRITÉ du POC « Plan financier par leviers »
================================================================================
Constantes métier, baseline synthétique, moteur de calcul, préparation des lignes
Gold et faits saillants. Importé PAR LES DEUX livrables :
  - le notebook Databricks (Phase 1)  ->  import moteur_plan as mp
  - l'app Streamlit (Phase 2)         ->  from moteur_plan import ...

⚠️ Données 100 % synthétiques. ECHELLE décale volontairement tous les niveaux
monétaires de tout ordre de grandeur réel (les ratios restent cohérents).

Aucune dépendance à Streamlit ni à Spark : numpy + pandas seulement.
"""

import numpy as np
import pandas as pd
from datetime import datetime

SCHEMA_DEFAUT = "plan_assurance_ind_v1"
ANNEES = list(range(2025, 2031))
N = len(ANNEES)

PRODUITS = [
    "Vie entière avec participation", "Vie entière à paiements limités",
    "Autres vies entières", "Temporaires", "Maladies graves",
    "Autres invalidité et maladie",
]
CANAUX = ["Indépendants", "Desjardins", "Agents Desjardins"]

# ---- Facteur d'échelle de camouflage -----------------------------------------
# Tous les NIVEAUX monétaires (ventes, P&L, CSM, capital, coûts, polices) sont
# multipliés par ECHELLE pour s'éloigner de tout ordre de grandeur réel.
# Les RATIOS (RSI, marges, coût par police) restent cohérents entre eux.
ECHELLE = 2.0

# 6 familles (ODS et « Autres » retirées du périmètre à la demande du métier)
VENTES_2025 = np.array([39510, 5400, 8660, 24723, 20940, 4300], dtype=float) * ECHELLE
VENTES_2030 = np.array([68674, 8520, 13970, 35680, 32020, 6330], dtype=float) * ECHELLE

RSI_MATRICE = pd.DataFrame(
    {
        "Indépendants":      [5.6, 1.8, 9.2, 7.3, 17.1, np.nan],
        "Desjardins":        [3.1, 16.7, 55.6, 6.9, 5.2, np.nan],
        "Agents Desjardins": [np.nan, 1.4, -1.2, -10.5, -0.6, -5.3],
    },
    index=PRODUITS,
)
VAN_2025 = np.array([-12848, 3005, 900, 1177, 748, -2024], dtype=float) * ECHELLE
MARGE_VAN = VAN_2025 / VENTES_2025

BASE_COUTS = np.array([95.0, 48.0, 14.6]) * ECHELLE   # acquisition / attribuables / non attr. (M$)
# Échelle distincte pour les polices -> le coût unitaire par police est lui aussi
# décalé des valeurs d'origine (≈ 4 034 $ -> 3 381 $ au lieu de 3 026 -> 2 536 $).
POLICES_2025, POLICES_2030 = 52083.0 * 1.5, 72000.0 * 1.5

# Palette inspirée de la marque Databricks : Lava, Navy, vert, ambre, fonds Oat.
COUL = {"vert": "#00A972", "rouge": "#FF3621", "bleu": "#1B3139",
        "gris": "#6B7A80", "or": "#FFAB00", "fond": "#F9F7F4"}

# ---- Capital requis par catégorie de coussin (HYPOTHÈSES de démonstration) -------
# Poids appliqués au capital à rémunérer ; la diversification est négative. Somme = 1.
COUSSINS = ["Assurance", "Crédit et marché", "Opérationnel",
            "Intérêt", "PfAD", "Diversification"]
POIDS_COUSSINS = np.array([0.52, 0.30, 0.08, 0.13, 0.17, -0.20])

# ---- Cost module : rétro-allocation vers les catégories AVANT allocation ----------
CATEGORIES_COUTS = ["Manufacture", "Distribution et appuis à la distribution",
                    "Opérations", "Réclamations", "Communication et marketing",
                    "Efficacité opérationnelle", "Autres fonctions de soutien", "TI"]
# Matrice de rétro-allocation : part de chaque BLOC post-allocation
# (acquisition / attribuables récurrents / non attribuables) provenant de chaque
# catégorie avant allocation. Chaque ligne somme à 1,00. Hypothèses de démonstration.
ALLOC_BLOCS_VERS_CATEGORIES = np.array([
    #  Manuf  Distr  Opér   Récl   Comm   Effic  Autres TI
    [0.10, 0.55, 0.05, 0.00, 0.12, 0.02, 0.06, 0.10],   # Coûts d'acquisition
    [0.08, 0.10, 0.30, 0.20, 0.04, 0.06, 0.10, 0.12],   # Coûts attribuables récurrents
    [0.05, 0.05, 0.10, 0.05, 0.10, 0.10, 0.35, 0.20],   # Coûts non attribuables
])
BLOCS_COUTS = ["Coûts d'acquisition", "Coûts attribuables récurrents",
               "Coûts non attribuables"]

# ---- VP fictives responsables de chaque catégorie (démonstration) ------------------
VP_CATEGORIES = ["VP Produits et actuariat", "VP Distribution", "VP Opérations",
                 "VP Réclamations", "VP Marketing", "VP Transformation",
                 "VP Services corporatifs", "VP Technologies"]

# ---- Modèle de coûts PILOTÉ PRÉ-ALLOCATION ------------------------------------------
# Les 8 catégories (une par VP) sont la SOURCE ; l'allocation vers les 3 blocs
# post-allocation en découle, puis alimente le P&L et la rentabilité.
# Bases 2025 par catégorie = rétro-allocation des blocs 2025 (cohérence historique).
_BLOCS_2025 = np.array([95.0, 48.0, 14.6]) * ECHELLE
CAT_BASE_2025 = ALLOC_BLOCS_VERS_CATEGORIES.T @ _BLOCS_2025            # (8,)
# Matrice d'allocation catégories -> blocs (inversion bayésienne : chaque ligne = 1)
ALLOC_CATEGORIES_VERS_BLOCS = (ALLOC_BLOCS_VERS_CATEGORIES
                               * _BLOCS_2025[:, None]).T / CAT_BASE_2025[:, None]
# Croissances par défaut par catégorie (%/an) = CAGR implicite de l'ancien modèle
# (blocs à 3,5 / 3,0 / 2,0 %/an) -> la baseline reste quasi identique.
_blocs_2030 = _BLOCS_2025 * (1 + np.array([0.035, 0.030, 0.020])) ** 5
_cat_2030 = ALLOC_BLOCS_VERS_CATEGORIES.T @ _blocs_2030
CROISS_CATEGORIES_DEFAUT = ((_cat_2030 / CAT_BASE_2025) ** (1 / 5) - 1) * 100  # en %

# ---- Levier D granularisé par classe d'actifs --------------------------------------
CLASSES_ACTIFS = ["Revenu fixe", "Revenus variables (actions)",
                  "Immobilier", "Placements alternatifs"]
POIDS_ACTIFS = np.array([0.70, 0.15, 0.10, 0.05])       # composition du portefeuille
RDT_CLASSES_DEFAUT = [2.7, 5.5, 4.5, 7.0]               # % ; pondéré ≈ 3,51 %

def rendement_pondere(rdt_classes_pct):
    """Rendement global = moyenne des rendements par classe pondérée par POIDS_ACTIFS."""
    return float(np.dot(POIDS_ACTIFS, np.asarray(rdt_classes_pct, dtype=float)) / 100.0)

# ---- Mensualisation (profils de saisonnalité, hypothèses de démonstration) ---------
MOIS = ["Janv", "Févr", "Mars", "Avr", "Mai", "Juin",
        "Juil", "Août", "Sept", "Oct", "Nov", "Déc"]
PROFILS_MENSUELS = {
    "Uniforme": np.full(12, 1 / 12),
    # Ventes d'assurance individuelle : pointe REER (févr.-mars), creux estival,
    # remontée d'automne.
    "Saisonnalité ventes (REER + automne)": np.array(
        [0.085, 0.105, 0.110, 0.080, 0.075, 0.070,
         0.060, 0.060, 0.085, 0.090, 0.095, 0.085]),
    # Coûts : régulier avec charge de fin d'année (régularisations, projets).
    "Charge de fin d'année (coûts)": np.array(
        [0.078, 0.078, 0.078, 0.078, 0.078, 0.078,
         0.078, 0.078, 0.078, 0.078, 0.100, 0.120]),
}

def mensualiser(valeur_annuelle, profil="Uniforme", intensite=1.0):
    """Répartit une valeur annuelle sur 12 mois.
    intensite ∈ [0, 1] : 0 = uniforme, 1 = profil plein ; entre les deux, mélange."""
    base = PROFILS_MENSUELS["Uniforme"]
    cible = PROFILS_MENSUELS.get(profil, base)
    poids = (1.0 - intensite) * base + intensite * cible
    poids = poids / poids.sum()
    return float(valeur_annuelle) * poids

# ---- Correspondance dimensions FP -> Oracle EPM -------------------------------------
ORACLE_MAPPING = pd.DataFrame(
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

def fmt_fr(x, dec=0, suffixe=""):
    return f"{x:,.{dec}f}".replace(",", " ").replace(".", ",") + suffixe

def fmt_m(x, dec=0):
    return fmt_fr(x, dec, " M$")

def fmt_pct(x, dec=1):
    return fmt_fr(x, dec, " %")

def ventes_baseline():
    """Interpolation géométrique 2025->2030 par famille + léger bruit (seed 42).
    Périmètre 6 familles : total 2025 ≈ 103,5 M$, 2030 ≈ 165,2 M$ (~9,8 %/an)."""
    rng = np.random.default_rng(42)
    tx = (VENTES_2030 / VENTES_2025) ** (1 / (N - 1))
    v = np.array([VENTES_2025 * tx**t for t in range(N)]).T
    bruit = 1 + rng.uniform(-0.015, 0.015, size=v.shape)
    bruit[:, 0] = 1.0
    bruit[:, -1] = 1.0
    v *= bruit
    return v

VENTES_BASE = ventes_baseline()

def calculer_scenario(scenario_id, fact_volume, croiss_fam, parts_canal,
                      rendement, g_acq, g_attr, g_na, rdt_classes=None,
                      croiss_categories=None, facteur_capital=1.0,
                      choc_taux_pb=0.0, choc_mortalite=0.0,
                      choc_decheance=0.0, choc_morbidite=0.0):
    """croiss_fam : ajustement de croissance PAR FAMILLE, en points de %/an vs le plan
    (ex. +2.0 -> la famille croît 2 pts plus vite chaque année que sa trajectoire de base).
    rdt_classes : rendements par classe d'actifs (%) — informatif ; le calcul utilise
    `rendement` (déjà pondéré via rendement_pondere), les classes sont tracées en overlay.
    croiss_categories : croissances PRÉ-ALLOCATION par catégorie/VP (%/an). Si fourni,
    les 8 catégories sont la SOURCE et les blocs post-allocation en découlent ;
    sinon (mode hérité, notebook), les blocs sont pilotés par g_acq/g_attr/g_na.
    choc_taux_pb : choc de taux en points de base. L'ALM adosse ~90 %% du résultat
    financier ; la hausse bonifie toutefois les produits PERMANENTS la 1re année
    (VAN, CSM des ventes, RSI), puis le repricing de l'industrie normalise.
    choc_mortalite / choc_decheance / choc_morbidite : écart réel vs hypothèses en %%
    (positif = DÉFAVORABLE) -> Expérience du P&L + Changements d'hypothèses du CSM."""
    t_idx = np.arange(N)
    rsi_mat = RSI_MATRICE.to_numpy()

    # Levier B : chaque famille suit sa trajectoire de base × (1 + ajustement)^t
    adj = np.array(croiss_fam, dtype=float) / 100.0            # points -> fraction
    facteur_fam = (1.0 + adj[:, None]) ** t_idx[None, :]        # (familles x années)
    ventes_scen = VENTES_BASE * fact_volume * facteur_fam

    ventes_tot_m = ventes_scen.sum(axis=0) / 1000.0
    ventes_pc = ventes_scen[:, :, None] * np.array(parts_canal)[None, None, :]

    boost_taux = choc_taux_pb / 10000.0            # choc en fraction
    IDX_PERM = [0, 1, 2]                            # familles permanentes (vie entière)
    rsi_adj = rsi_mat + (rendement - 0.035) * 100 * 0.8
    # G — 1re année : la hausse de taux bonifie le RSI des permanents (avant repricing)
    rsi_adj[IDX_PERM, :] = rsi_adj[IDX_PERM, :] + boost_taux * 100 * 0.8
    masque = ~np.isnan(rsi_mat)
    rsi_nb = np.array([
        np.nansum(np.where(masque, rsi_adj, 0) * ventes_pc[:, j, :]) /
        max(1e-9, (ventes_pc[:, j, :] * masque).sum())
        for j in range(N)
    ])

    van_k = MARGE_VAN[:, None] * ventes_scen + ventes_scen * (rendement - 0.035) * 2.0
    # G — VAN des permanents bonifiée la 1re année seulement (repricing ensuite)
    van_k[IDX_PERM, 0] = van_k[IDX_PERM, 0] + ventes_scen[IDX_PERM, 0] * boost_taux * 3.0
    van_tot_m = van_k.sum(axis=0) / 1000.0

    marge_csm = np.linspace(0.75, 1.03, N)
    nb_csm = marge_csm * ventes_tot_m
    part_perm = ventes_scen[IDX_PERM, 0].sum() / max(1e-9, ventes_scen[:, 0].sum())
    nb_csm[0] = nb_csm[0] * (1 + boost_taux * 4.0 * part_perm)   # G — 1re année                      # ventes déjà à l'échelle
    # Changements d'hypothèses lissés (plus de choc -37 en 2025 -> trajectoire douce)
    chg_hyp = np.array([-12, -8, -4, 0, 6, 12], dtype=float) * ECHELLE
    # H — renforcement d'hypothèses si l'expérience est défavorable (persistant)
    chg_hyp = chg_hyp - (0.90 * choc_mortalite + 1.10 * choc_morbidite
                         + 1.30 * choc_decheance) * ECHELLE
    exp_csm = 12.0 * ECHELLE
    csm_open, csm_close, csm_release, csm_interet = [], [], [], []
    solde = 1937.0 * ECHELLE
    for j in range(N):
        rel = 0.071 * solde
        inte = 0.0315 * solde * (rendement / 0.035)
        fin = solde - rel + nb_csm[j] + inte + exp_csm + chg_hyp[j]
        csm_open.append(solde); csm_release.append(rel)
        csm_interet.append(inte); csm_close.append(fin)
        solde = fin
    csm_open, csm_close = np.array(csm_open), np.array(csm_close)
    csm_release, csm_interet = np.array(csm_release), np.array(csm_interet)

    ra_release = 31.0 * ECHELLE * 1.05 ** t_idx
    profit_attendu = csm_release + ra_release
    impact_ventes = 0.11 * van_tot_m
    experience = ((15.0 + 1.5 * t_idx)
                  - (0.40 * choc_mortalite + 0.55 * choc_morbidite
                     + 0.15 * choc_decheance)) * ECHELLE

    if croiss_categories is not None:
        # Modèle piloté PRÉ-ALLOCATION : chaque catégorie (VP) croît à son taux,
        # puis s'alloue vers les 3 blocs -> dépenses du P&L -> rentabilité.
        g_cat = np.asarray(croiss_categories, dtype=float) / 100.0
        couts_avant = CAT_BASE_2025[:, None] * (1 + g_cat[:, None]) ** t_idx[None, :]
        couts_blocs = ALLOC_CATEGORIES_VERS_BLOCS.T @ couts_avant
        acq, attr, na = couts_blocs
    else:
        # Mode hérité (widgets du notebook) : blocs pilotés directement
        acq = BASE_COUTS[0] * (1 + g_acq) ** t_idx
        attr = BASE_COUTS[1] * (1 + g_attr) ** t_idx
        na = BASE_COUTS[2] * (1 + g_na) ** t_idx
        couts_blocs = np.vstack([acq, attr, na])
        couts_avant = ALLOC_BLOCS_VERS_CATEGORIES.T @ couts_blocs
    depenses_src = -(attr + na + 0.30 * acq)

    # Levier F : optimisation du capital (réassurance, ALM, redéploiement) —
    # réduit le capital à rémunérer, ses coussins ET les actifs adossés (donc le
    # résultat financier baisse un peu, mais le RSI monte : dénominateur plus petit).
    capital = (np.linspace(464.0, 581.0, N) * ECHELLE
               * (0.85 + 0.15 * fact_volume) * facteur_capital)
    actifs = capital * 5.6
    # Décomposition du capital par catégorie de coussin (poids constants, hypothèse)
    capital_coussins = POIDS_COUSSINS[:, None] * capital[None, :]        # (6 x années)
    # Accrétion des passifs LINÉAIRE : plus de « boom » 2025 -> trajectoire lisse,
    # calibrée pour un RSI ~25 % qui glisse doucement vers ~22 %.
    accretion_passifs = np.linspace(30.0, 49.0, N) * ECHELLE
    # G — ALM : ~90 % du bilan est adossé ; seul un résidu (~10 %) réagit au choc
    interet_marche = (actifs * rendement - accretion_passifs
                      + actifs * boost_taux * 0.10)

    activites = profit_attendu + impact_ventes + experience
    exploitation = activites + interet_marche + depenses_src
    impots = np.where(exploitation > 0, 0.24 * exploitation, 0.0)
    resultat_net = exploitation - impots

    echelle_vol = 0.85 + 0.15 * fact_volume
    produits_ass = np.linspace(669.0, 806.0, N) * ECHELLE * echelle_vol
    charges_ass = np.linspace(450.0, 561.0, N) * ECHELLE * echelle_vol
    reassurance = activites - (produits_ass - charges_ass)
    rsi_global = resultat_net / capital * 100.0

    polices = np.linspace(POLICES_2025, POLICES_2030, N) * fact_volume
    cout_par_police = (acq + attr + na) * 1_000_000.0 / polices

    return {
        "scenario_id": scenario_id, "ventes_scen": ventes_scen, "ventes_pc": ventes_pc,
        "rsi_adj": rsi_adj, "masque": masque, "van_k": van_k,
        "resultat_net": resultat_net, "exploitation": exploitation,
        "rsi_global": rsi_global, "rsi_nb": rsi_nb, "van_tot_m": van_tot_m,
        "csm_open": csm_open, "csm_close": csm_close, "csm_release": csm_release,
        "csm_interet": csm_interet, "nb_csm": nb_csm, "chg_hyp": chg_hyp,
        "exp_csm": exp_csm, "profit_attendu": profit_attendu,
        "impact_ventes": impact_ventes, "experience": experience,
        "depenses": depenses_src, "interet_marche": interet_marche,
        "capital": capital, "cout_par_police": cout_par_police, "polices": polices,
        "capital_coussins": capital_coussins, "couts_avant": couts_avant,
        "couts_blocs": couts_blocs,
        "produits_ass": produits_ass, "charges_ass": charges_ass,
        "reassurance": reassurance, "impots": impots, "activites": activites,
    }

PARAMS_BASE = dict(fact_volume=1.0, croiss_fam=[0.0] * len(PRODUITS),
                   parts_canal=[0.60, 0.25, 0.15],
                   rendement=rendement_pondere(RDT_CLASSES_DEFAUT),   # ≈ 3,51 %
                   rdt_classes=list(RDT_CLASSES_DEFAUT),
                   g_acq=0.035, g_attr=0.030, g_na=0.020,
                   croiss_categories=[round(float(g), 4) for g in CROISS_CATEGORIES_DEFAUT],
                   facteur_capital=1.0,
                   choc_taux_pb=0.0, choc_mortalite=0.0,
                   choc_decheance=0.0, choc_morbidite=0.0)

def construire_lignes(res, params, scenario_id):
    """Prépare les lignes overlay / forecast / kpi / dim (mêmes formats que Phase 1)."""
    horo = datetime.now().isoformat(timespec="seconds")
    pc = params["parts_canal"]

    overlay = [
        (scenario_id, "A_facteur_volume", params["fact_volume"], 1.00, horo),
        (scenario_id, "C_part_independants", pc[0], 0.60, horo),
        (scenario_id, "C_part_desjardins", pc[1], 0.25, horo),
        (scenario_id, "C_part_agents", pc[2], 0.15, horo),
        (scenario_id, "D_rendement_placement", params["rendement"], 0.035, horo),
        (scenario_id, "E_croiss_couts_acquisition", params["g_acq"], 0.035, horo),
        (scenario_id, "E_croiss_couts_attribuables", params["g_attr"], 0.030, horo),
        (scenario_id, "E_croiss_couts_non_attrib", params["g_na"], 0.020, horo),
    ] + [
        (scenario_id, f"B_croiss_{p}", float(params["croiss_fam"][i]), 0.00, horo)
        for i, p in enumerate(PRODUITS)
    ]

    sections = {
        "etat_resultats": [
            ("Produits d'assurance", res["produits_ass"]),
            ("Charges d'assurance", -res["charges_ass"]),
            ("Réassurance nette", res["reassurance"]),
            ("Résultats des activités d'assurance", res["activites"]),
            ("Résultat financier", res["interet_marche"]),
            ("Résultats autres", res["depenses"]),
            ("Résultat d'exploitation", res["exploitation"]),
            ("Impôts", -res["impots"]),
            ("Résultat net", res["resultat_net"]),
        ],
        "sources_benefices": [
            ("Profit attendu (CSM + RA)", res["profit_attendu"]),
            ("Impact des ventes", res["impact_ventes"]),
            ("Expérience", res["experience"]),
            ("Dépenses", res["depenses"]),
            ("Intérêt et marché", res["interet_marche"]),
            ("Résultat d'exploitation", res["exploitation"]),
        ],
        "csm_rollforward": [
            ("Solde CSM départ", res["csm_open"]),
            ("Profit attendu relâché", -res["csm_release"]),
            ("Impact des ventes profitables", res["nb_csm"]),
            ("Intérêt et marché", res["csm_interet"]),
            ("Expérience", np.full(N, res["exp_csm"])),
            ("Changements d'hypothèses", res["chg_hyp"]),
            ("Solde CSM fin", res["csm_close"]),
        ],
        "capital": [
            ("Capital à rémunérer moyen", res["capital"]),
            ("Actifs investis (proxy)", res["capital"] * 5.6),
        ],
        "capital_coussins": [
            (c, res["capital_coussins"][i]) for i, c in enumerate(COUSSINS)
        ],
        "couts_avant_allocation": [
            (c, res["couts_avant"][i]) for i, c in enumerate(CATEGORIES_COUTS)
        ],
    }
    for nom_l, cle_l in [("G_choc_taux_pb", "choc_taux_pb"),
                         ("H_choc_mortalite", "choc_mortalite"),
                         ("H_choc_decheance", "choc_decheance"),
                         ("H_choc_morbidite", "choc_morbidite")]:
        overlay.append((scenario_id, nom_l, float(params.get(cle_l, 0.0)), 0.0, horo))
    overlay.append((scenario_id, "F_facteur_capital",
                    float(params.get("facteur_capital", 1.0)), 1.00, horo))
    # Levier E pré-allocation : une ligne d'overlay par catégorie/VP (si fourni)
    if params.get("croiss_categories"):
        for i, cat in enumerate(CATEGORIES_COUTS):
            overlay.append((scenario_id, f"E_croiss_{cat}",
                            float(params["croiss_categories"][i]) / 100.0,
                            float(CROISS_CATEGORIES_DEFAUT[i]) / 100.0, horo))
    # Levier D granularisé : une ligne d'overlay par classe d'actifs (si fourni)
    if params.get("rdt_classes"):
        for i, cl in enumerate(CLASSES_ACTIFS):
            overlay.append((scenario_id, f"D_rdt_{cl}",
                            float(params["rdt_classes"][i]) / 100.0,
                            RDT_CLASSES_DEFAUT[i] / 100.0, horo))
    forecast = []
    for section, lignes_sec in sections.items():
        for ordre, (ligne, serie) in enumerate(lignes_sec, start=1):
            for j, a in enumerate(ANNEES):
                forecast.append((scenario_id, a, section, ligne, ordre,
                                 round(float(serie[j]), 2)))

    kpi = []
    for j, a in enumerate(ANNEES):
        for nom, valr, unite in [
            ("resultat_net_m", res["resultat_net"][j], "M$"),
            ("resultat_exploitation_m", res["exploitation"][j], "M$"),
            ("rsi_global_pct", res["rsi_global"][j], "%"),
            ("rsi_nouvelles_affaires_pct", res["rsi_nb"][j], "%"),
            ("van_totale_m", res["van_tot_m"][j], "M$"),
            ("csm_solde_fin_m", res["csm_close"][j], "M$"),
            ("capital_a_remunerer_m", res["capital"][j], "M$"),
            ("cout_acquisition_par_police_$", res["cout_par_police"][j], "$"),
            ("polices_emises", res["polices"][j], "nb"),
            ("ventes_totales_k", res["ventes_scen"][:, j].sum(), "K$"),
        ]:
            kpi.append((scenario_id, a, "global", "Tous", "Tous", nom,
                        round(float(valr), 2), unite))
        for i, p in enumerate(PRODUITS):
            kpi.append((scenario_id, a, "produit", p, "Tous", "ventes_k",
                        round(float(res["ventes_scen"][i, j]), 1), "K$"))
            kpi.append((scenario_id, a, "produit", p, "Tous", "van_k",
                        round(float(res["van_k"][i, j]), 1), "K$"))
            for c_i, c in enumerate(CANAUX):
                kpi.append((scenario_id, a, "produit_canal", p, c, "ventes_k",
                            round(float(res["ventes_pc"][i, j, c_i]), 1), "K$"))
                if res["masque"][i, c_i]:
                    kpi.append((scenario_id, a, "produit_canal", p, c, "rsi_pct",
                                round(float(res["rsi_adj"][i, c_i]), 2), "%"))

    # accent_croissance (ancien levier B) omis -> NULL ; les leviers B par famille
    # vivent désormais dans overlay_drivers_slv (format long, flexible).
    dim = [(scenario_id, horo, params["fact_volume"],
            pc[0], pc[1], pc[2], params["rendement"], params["g_acq"],
            params["g_attr"], params["g_na"])]
    return overlay, forecast, kpi, dim

# ---- Rolling forecast : recalibration des leviers sur les réels ---------------------
# Des « réels » synthétiques divergent progressivement du plan ; à chaque date de
# coupure, l'information accumulée croît -> l'ajustement des leviers implicites aussi.
# Principe démontré : rolling forecast = baseline immuable + NOUVEL overlay recalibré.
# Coupure mensuelle : l'information accumulée (et donc la recalibration des leviers)
# croît linéairement de Déc 2025 (0,50) à Déc 2026 (1,00).
COUPURES_RF = {"Déc 2025": {"intensite": 0.50, "annee_x": 2025.96}}
for _m in range(1, 13):
    COUPURES_RF[f"{MOIS[_m - 1]} 2026"] = {
        "intensite": round(0.50 + 0.50 * _m / 12, 3),
        "annee_x": round(2026.0 + _m / 12 - 0.02, 3),
    }
NARRATIF_RF = ("Les réels montrent : ventes sous le plan (pression concurrentielle sur "
               "les temporaires), coûts TI et Distribution au-dessus (projets de "
               "modernisation), rendement des actions au-dessus (marchés favorables).")

def params_rolling_forecast(coupure):
    """Leviers implicites recalibrés sur les réels observés à la date de coupure.
    Retourne (params_rf, narratif, position_x_de_la_coupure)."""
    k = COUPURES_RF[coupure]["intensite"]
    p = {kk: (list(v) if isinstance(v, list) else v) for kk, v in PARAMS_BASE.items()}
    p["fact_volume"] = round(1.0 - 0.045 * k, 3)                 # ventes sous le plan
    p["croiss_categories"][7] = round(p["croiss_categories"][7] + 1.6 * k, 2)  # TI
    p["croiss_categories"][1] = round(p["croiss_categories"][1] + 0.8 * k, 2)  # Distribution
    p["rdt_classes"][1] = round(p["rdt_classes"][1] + 0.9 * k, 2)  # actions favorables
    p["rendement"] = rendement_pondere(p["rdt_classes"])
    return p, NARRATIF_RF, COUPURES_RF[coupure]["annee_x"]

GROUPES_LEVIERS = [
    ("A · Volume", ["fact_volume"]),
    ("B · Familles", ["croiss_fam"]),
    ("C · Mix canal", ["parts_canal"]),
    ("D · Rendement", ["rendement", "rdt_classes"]),
    ("E · Coûts VP", ["croiss_categories", "g_acq", "g_attr", "g_na"]),
    ("F · Capital", ["facteur_capital"]),
    ("G · Taux (ALM)", ["choc_taux_pb"]),
    ("H · Hyp. actuarielles", ["choc_mortalite", "choc_decheance", "choc_morbidite"]),
]

def attribution_par_levier(params_cible, jf, params_base=None):
    """Décompose l'écart de résultat net (année d'indice jf) entre la baseline et un
    scénario cible, levier par levier (application séquentielle A -> F ; les effets
    d'interaction sont portés par l'ordre d'application)."""
    base = dict(params_base or PARAMS_BASE)
    net_prec = calculer_scenario("_attr", **base)["resultat_net"][jf]
    net_base = net_prec
    etiquettes, deltas = [], []
    courant = dict(base)
    for nom, cles in GROUPES_LEVIERS:
        for c in cles:
            if c in params_cible:
                courant[c] = params_cible[c]
        net = calculer_scenario("_attr", **courant)["resultat_net"][jf]
        etiquettes.append(nom)
        deltas.append(float(net - net_prec))
        net_prec = net
    return float(net_base), etiquettes, deltas, float(net_prec)

def overlay_vers_params(lev):
    """Reconstruit un dict de params moteur depuis les lignes d'overlay {levier: valeur}
    d'un scénario écrit (gère les scénarios hérités avec valeurs par défaut)."""
    parts = np.array([lev.get("C_part_independants", 0.60),
                      lev.get("C_part_desjardins", 0.25),
                      lev.get("C_part_agents", 0.15)])
    parts = parts / parts.sum()
    if any(k.startswith("E_croiss_") and not k.startswith("E_croiss_couts") for k in lev):
        cc = [lev.get(f"E_croiss_{c}", CROISS_CATEGORIES_DEFAUT[i] / 100.0) * 100
              for i, c in enumerate(CATEGORIES_COUTS)]
    else:
        g_b = np.array([lev.get("E_croiss_couts_acquisition", 0.035),
                        lev.get("E_croiss_couts_attribuables", 0.030),
                        lev.get("E_croiss_couts_non_attrib", 0.020)])
        cc = (ALLOC_CATEGORIES_VERS_BLOCS @ g_b * 100).tolist()
    if any(k.startswith("D_rdt_") for k in lev):
        rc = [lev.get(f"D_rdt_{cl}", RDT_CLASSES_DEFAUT[i] / 100.0) * 100
              for i, cl in enumerate(CLASSES_ACTIFS)]
    else:
        ratio = lev.get("D_rendement_placement", rendement_pondere(RDT_CLASSES_DEFAUT)) \
                / rendement_pondere(RDT_CLASSES_DEFAUT)
        rc = [d * ratio for d in RDT_CLASSES_DEFAUT]
    return dict(
        fact_volume=float(lev.get("A_facteur_volume", 1.0)),
        croiss_fam=[float(lev.get(f"B_croiss_{p}", 0.0)) for p in PRODUITS],
        parts_canal=parts.tolist(),
        rendement=rendement_pondere(rc), rdt_classes=rc,
        croiss_categories=[float(x) for x in cc],
        g_acq=0.035, g_attr=0.030, g_na=0.020,
        facteur_capital=float(lev.get("F_facteur_capital", 1.0)),
        choc_taux_pb=float(lev.get("G_choc_taux_pb", 0.0)),
        choc_mortalite=float(lev.get("H_choc_mortalite", 0.0)),
        choc_decheance=float(lev.get("H_choc_decheance", 0.0)),
        choc_morbidite=float(lev.get("H_choc_morbidite", 0.0)),
    )

def generer_faits_par_onglet(res, res_base, params, jf, annee_focus, scenario_id):
    """Faits saillants générés automatiquement, organisés PAR ONGLET de l'app.
    Alternative légère et 100 % fiable en démo à un Genie embarqué."""
    faits = []

    # 1) Écart vs Base + levier dominant
    d_net = res["resultat_net"][jf] - res_base["resultat_net"][jf]
    ecarts_lev = {
        "le volume (A)": abs(params["fact_volume"] - 1.0) / 0.10,
        "la croissance par famille (B)": max(abs(c) for c in params["croiss_fam"]) / 2.0,
        "le mix canal (C)": max(abs(params["parts_canal"][0] - 0.60),
                                abs(params["parts_canal"][1] - 0.25),
                                abs(params["parts_canal"][2] - 0.15)) / 0.05,
        "le rendement de placement (D)": abs(params["rendement"] - 0.035) / 0.0025,
        "l'optimisation du capital (F)": abs(params.get("facteur_capital", 1.0) - 1.0) / 0.02,
        "le choc de taux (G)": abs(params.get("choc_taux_pb", 0.0)) / 25.0,
        "les hypothèses actuarielles (H)": max(
            abs(params.get("choc_mortalite", 0.0)), abs(params.get("choc_decheance", 0.0)),
            abs(params.get("choc_morbidite", 0.0))) / 2.0,
        "les coûts pré-allocation (E)": (
            max(abs(params["croiss_categories"][i] - CROISS_CATEGORIES_DEFAUT[i])
                for i in range(len(CATEGORIES_COUTS))) / 0.5
            if params.get("croiss_categories") else
            max(abs(params["g_acq"] - 0.035), abs(params["g_attr"] - 0.030),
                abs(params["g_na"] - 0.020)) / 0.005),
    }
    levier_dom = max(ecarts_lev, key=ecarts_lev.get)
    if max(ecarts_lev.values()) < 0.01:
        faits.append(f"⚖️ Le scénario « {scenario_id} » est aligné sur la baseline : "
                     f"aucun levier n'a été bougé.")
    else:
        sens = "au-dessus" if d_net >= 0 else "en dessous"
        faits.append(f"⚖️ Résultat net {annee_focus} : **{fmt_m(abs(d_net))} {sens} de la "
                     f"baseline** ; principal levier actionné : **{levier_dom}**.")

    # 2) Meilleure et pire cellule RSI produit × canal
    rsi = res["rsi_adj"]
    i_max = np.unravel_index(np.nanargmax(rsi), rsi.shape)
    i_min = np.unravel_index(np.nanargmin(rsi), rsi.shape)
    faits.append(f"🏆 Meilleur RSI des ventes : **{PRODUITS[i_max[0]]} × {CANAUX[i_max[1]]}** "
                 f"à **{fmt_pct(rsi[i_max])}** ; le plus faible : "
                 f"{PRODUITS[i_min[0]]} × {CANAUX[i_min[1]]} à {fmt_pct(rsi[i_min])}.")

    # 3) Produits destructeurs de valeur (VAN négative, année focus)
    van_focus = res["van_k"][:, jf] / 1000.0
    negatifs = [(PRODUITS[i], van_focus[i]) for i in range(len(PRODUITS)) if van_focus[i] < -0.5]
    if negatifs:
        pire = min(negatifs, key=lambda x: x[1])
        faits.append(f"🔻 {len(negatifs)} famille(s) détruisent de la valeur en {annee_focus} "
                     f"(VAN totale {fmt_m(res['van_tot_m'][jf], 1)}) ; la plus déficitaire : "
                     f"**{pire[0]}** à {fmt_m(pire[1], 1)}.")
    else:
        faits.append(f"🟢 Aucune famille ne détruit de valeur en {annee_focus} — "
                     f"VAN totale : {fmt_m(res['van_tot_m'][jf], 1)}.")

    # 4) Trajectoire CSM et coûts unitaires sur l'horizon
    d_csm = res["csm_close"][-1] - res["csm_open"][0]
    faits.append(f"📈 Le solde CSM passe de {fmt_m(res['csm_open'][0])} à "
                 f"**{fmt_m(res['csm_close'][-1])}** sur l'horizon "
                 f"({'+' if d_csm >= 0 else ''}{fmt_fr(d_csm / res['csm_open'][0] * 100, 1)} %).")
    d_cpp = res["cout_par_police"][-1] - res["cout_par_police"][0]
    tendance = "baisse" if d_cpp < 0 else "hausse"
    fait_cpp = (f"💰 Coût d'acquisition moyen par police en **{tendance}** : "
                f"{fmt_fr(res['cout_par_police'][0])} $ → "
                f"{fmt_fr(res['cout_par_police'][-1])} $ "
                f"(effet volume vs croissance des dépenses).")

    # ---- Faits dédiés : Capital ----------------------------------------------------
    cc = res["capital_coussins"]
    i_dom = int(np.argmax(POIDS_COUSSINS))
    d_capital = res["capital"][-1] - res["capital"][0]
    faits_capital = [
        f"🏛️ Coussin dominant : **{COUSSINS[i_dom]}** à {fmt_m(cc[i_dom][jf])} "
        f"({POIDS_COUSSINS[i_dom]*100:.0f} % du capital net) en {annee_focus} ; la "
        f"diversification réduit le besoin de {fmt_m(abs(cc[-1][jf]))}.",
        f"📊 Capital net à rémunérer : {fmt_m(res['capital'][0])} → "
        f"{fmt_m(res['capital'][-1])} sur l'horizon "
        f"({'+' if d_capital >= 0 else ''}{fmt_fr(d_capital / res['capital'][0] * 100, 1)} %).",
    ]

    # ---- Faits dédiés : Coûts (pré-allocation / VP) ---------------------------------
    ca = res["couts_avant"]
    i_gros = int(np.argmax(ca[:, jf]))
    faits_couts = [fait_cpp,
                   f"🏢 Plus grosse enveloppe pré-allocation en {annee_focus} : "
                   f"**{CATEGORIES_COUTS[i_gros]}** ({VP_CATEGORIES[i_gros]}) à "
                   f"{fmt_m(ca[i_gros][jf], 1)}."]
    if params.get("croiss_categories"):
        g = np.asarray(params["croiss_categories"], dtype=float)
        i_rapide = int(np.argmax(g))
        faits_couts.append(
            f"📈 Catégorie à plus forte croissance : **{CATEGORIES_COUTS[i_rapide]}** "
            f"({VP_CATEGORIES[i_rapide]}) à {fmt_fr(g[i_rapide], 1)} %/an — se propage "
            f"aux blocs post-allocation et au RSI.")

    depasse = np.where(res["nb_csm"] > res["csm_release"])[0]
    if len(depasse):
        fait_croisement = (f"⚔️ L'impact des ventes profitables dépasse la relâche du "
                           f"profit attendu dès **{2025 + int(depasse[0])}** — le stock "
                           f"de CSM se reconstitue plus vite qu'il ne se consomme.")
    else:
        fait_croisement = ("⚔️ La relâche du profit attendu dépasse l'impact des ventes "
                           "profitables sur tout l'horizon : le stock de CSM se consomme.")

    return {
        "tableau_de_bord": faits[:3],
        "capital": faits_capital,
        "couts": faits_couts,
        "csm": [faits[3], fait_croisement],
    }

def generer_faits_saillants(res, res_base, params, jf, annee_focus, scenario_id):
    """Compatibilité (notebook) : liste à plat de tous les faits."""
    d = generer_faits_par_onglet(res, res_base, params, jf, annee_focus, scenario_id)
    return d["tableau_de_bord"] + d["csm"] + d["capital"] + d["couts"]
