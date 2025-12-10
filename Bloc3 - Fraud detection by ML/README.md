# 🎓 Certification AIA - Architecte en Intelligence Artificielle

[![RNCP](https://img.shields.io/badge/RNCP-38777-blue)](https://www.francecompetences.fr/recherche/rncp/38777/)
[![Niveau](<https://img.shields.io/badge/Niveau-7_(BAC+5)-green>)](https://www.francecompetences.fr/recherche/rncp/38777/)
[![Statut](https://img.shields.io/badge/Statut-En_cours-orange)]()

---

## 📋 Table des matières

- [Bloc 3 - Pipeline de Données](#bloc-3---pipeline-de-données)
  - [Objectifs](#objectifs-bloc-3)
  - [Contenu](#projet-pipeline-fraud-detection)
  - [Ressources](d)
- [Contact](#contact)

---

## 🟡 Bloc 3 - Pipeline de Données et ML en Production

### Objectifs {#objectifs-bloc-3}

Concevoir et mettre en œuvre un pipeline de données industriel complet, depuis l'ingestion jusqu'au déploiement de modèles de machine learning en production avec monitoring continu.

### Contenu

- Conception de pipelines de données temps réel
- ETL/ELT et orchestration (Apache Airflow)
- Feature engineering et preprocessing
- Entraînement et évaluation de modèles ML
- MLOps : tracking (MLflow), versioning, CI/CD
- Conteneurisation (Docker) et déploiement
- Monitoring et observabilité des données
- Qualité des données (Great Expectations)

### Ressources

Les ressources et exercices pratiques sont organisés dans les dossiers de ce bloc.

### Projet : Pipeline Fraud Detection {#projet-pipeline-fraud-detection}

**Contexte** : Mise en production d'un système de détection de fraude bancaire temps réel capable de traiter 21 millions de transactions avec une latence inférieure à 50ms par prédiction.

#### Objectifs du projet

1. Concevoir un système de gestion de données temps réel
2. Établir un pipeline ETL/ELT complet
3. Automatiser les flux de données
4. Surveiller la qualité et la conformité
5. Développer des procédures de contrôle qualité

#### Livrables attendus

| Livrable                | Format      | Contenu                                     |
| ----------------------- | ----------- | ------------------------------------------- |
| Pipeline de données     | Code Python | ETL/ELT, feature engineering, orchestration |
| Modèles ML entraînés    | MLflow      | 4 modèles comparés (RF, LogReg, SVC)        |
| Application déployée    | Docker/HF   | Flask + FastAPI + MLflow en production      |
| Documentation technique | PDF         | Architecture, justifications, performances  |
| Présentation            | Slides      | Synthèse 15 min pour le jury                |
| Vidéo démonstration     | MP4/Link    | Scoring temps réel en fonctionnement        |

#### Architecture déployée

**Stack technologique :**

[XXX] inserer le pipeline image

#### Dataset utilisé

🔗 [Dataset Hugging Face](https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv)

#### Résultats obtenus

**Comparaison des modèles :**

| Modèle                    | AUC-ROC | Precision | Recall | F1-Score | Latence |
| ------------------------- | ------- | --------- | ------ | -------- | ------- |
| **RandomForest** ⭐       | 1.000   | 100.0%    | 100.0% | 100.0%   | 8.3ms   |
| LogisticRegression (100)  | 0.740   | 93.22%    | 100.0% | 96.48%   | 0.6ms   |
| LogisticRegression (1000) | 0.730   | 93.75%    | 100.0% | 96.79%   | 0.6ms   |

**Modèle sélectionné** : RandomForest

#### Points clés développés

- ✅ **ETL/ELT complet** : Feature engineering, transformations, multi-bases
- ✅ **Automatisation** : Orchestration Supervisor, CI/CD GitHub Actions
- ✅ **Monitoring** : Evidently AI (data drift), Grafana (métriques)
- ✅ **Qualité** : Great Expectations, validation continue
- ✅ **MLOps** : MLflow tracking, model registry, versioning
- ✅ **Conteneurisation** : Docker multi-stage, déploiement HF Spaces
- ✅ **Conformité** : GDPR, PCI-DSS, AI Act (explicabilité)

#### Infrastructure déployée

**Services en production :**

| Service   | URL                                           | Description               |
| --------- | --------------------------------------------- | ------------------------- |
| Streamlit | https://davidrambeau-bloc3-streamlit.hf.space | Interface utilisateur     |
| FastAPI   | https://davidrambeau-bloc3-fastapi.hf.space   | API de scoring            |
| MLflow    | https://davidrambeau-bloc3-mlflow.hf.space    | Tracking & model registry |
| Flask     | https://davidrambeau-bloc3-flask.hf.space     | Monitoring (optionnel)    |

**Stockage externe :**

- **NeonDB** : PostgreSQL serverless (métadonnées MLflow)
- **AWS S3** : bucket-laposte-david (artefacts modèles)


## 📝 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 📧 Contact {#contact}

- **Candidat** : David RAMBEAU
- **Formation** : Lead Data Science & Engineering
- **Certification** : AIA - RNCP38777
- **Email** : david.rambeau@gmail.com
- **LinkedIn** : https://www.linkedin.com/in/davidrambeau/
- **GitHub** : https://github.com/AByGCreation


**Dernière mise à jour** : Décembre 2025
**Version** : 1.0.0
**Statut global** : 🟡 En cours de
