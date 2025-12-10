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

Concevoir et mettre en œuvre un pipeline de données industriel complet, depuis l'ingestion jusqu'au déploiement de modèles de machine learning en production.

### Ressources

Les ressources et exercices pratiques sont organisés dans les dossiers de ce bloc.

### Projet : Pipeline Fraud Detection {#projet-pipeline-fraud-detection}

**Contexte** : Mise en production d'un système de détection de fraude bancaire temps réel par prédiction ML.

#### Objectifs du projet

1. Concevoir un système de gestion de données temps réel
2. Établir un pipeline ETL/ELT complet
3. Automatiser les flux de données
4. Surveiller la qualité et la conformité
5. Développer des procédures de contrôle qualité

#### Architecture déployée

**Stack technologique :**
Schema global du Pipeline
[<img src="assets/pipeline.png">](https://github.com/AByGCreation/Jedha_certification_IAI/blob/master/Bloc3%20-%20Fraud%20detection%20by%20ML/Dossier/assets/pipeline.png)




#### Dataset utilisé

🔗 [Dataset Hugging Face](https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv)

#### Résultats obtenus

**Comparaison des modèles :**

| Modèle                    | AUC-ROC | Precision | Recall | F1-Score |
| ------------------------- | ------- | --------- | ------ | -------- | ------- |
| **RandomForest** ⭐       | 1.000   | 100.0%    | 100.0% | 100.0%   |
| LogisticRegression (100)  | 0.740   | 93.22%    | 100.0% | 96.48%   |
| LogisticRegression (1000) | 0.730   | 93.75%    | 100.0% | 96.79%   |

**Modèle sélectionné** : RandomForest

#### Points clés développés

- ✅ **ETL/ELT complet** : Feature engineering, transformations, multi-bases
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
