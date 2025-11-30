# 🎓 Certification AIA - Architecte en Intelligence Artificielle

[![RNCP](https://img.shields.io/badge/RNCP-38777-blue)](https://www.francecompetences.fr/recherche/rncp/38777/)
[![Niveau](https://img.shields.io/badge/Niveau-7_(BAC+5)-green)](https://www.francecompetences.fr/recherche/rncp/38777/)
[![Statut](https://img.shields.io/badge/Statut-En_cours-orange)]()

---

## 📋 Table des matières

- [Vue d'ensemble](#-vue-densemble)
- [Structure de la certification](#-structure-de-la-certification)
- [Organisation du repository](#-organisation-du-repository)
- [Bloc 1 - Gouvernance des données](#-bloc-1---gouvernance-des-données)
- [Bloc 2 - Architecture de données](#-bloc-2---architecture-de-données)
- [Bloc 3 - Pipelines de données](#-bloc-3---pipelines-de-données)
- [Bloc 4 - Solutions d'IA](#-bloc-4---solutions-dia)
- [Technologies utilisées](#-technologies-utilisées)
- [Installation et déploiement](#-installation-et-déploiement)
- [Calendrier de certification](#-calendrier-de-certification)
- [Ressources et documentation](#-ressources-et-documentation)
- [Contact](#-contact)

---

## 🎯 Vue d'ensemble

### Objectif de la certification

La certification **Architecte en Intelligence Artificielle (AIA)** - RNCP38777 - Niveau 7 (équivalent BAC+5) vise à former des professionnels capables de :

- Concevoir et piloter la **gouvernance des données**
- Déployer des **architectures de données** scalables et sécurisées
- Mettre en œuvre des **pipelines de données** automatisés
- Construire, déployer et piloter des **solutions d'IA** en production

### Format de certification

**Option retenue** : Passage individuel des blocs (4 blocs distincts)

| Bloc | Durée | Format |
|------|-------|--------|
| Bloc 1 | 30 min | 15 min présentation + 15 min Q&A |
| Bloc 2 | 20 min | 5 min présentation + 15 min Q&A |
| Bloc 3 | 20 min | 5 min présentation + 15 min Q&A |
| Bloc 4 | 15 min | 5 min présentation + 10 min Q&A |

**Durée totale** : 1h25 (si passage complet)

---

## 📚 Structure de la certification

### Compétences évaluées

#### 🔵 Bloc 1 : Concevoir et piloter la gouvernance des données
- Concevoir une politique de Data Gouvernance conforme aux régulations
- Collaborer avec les parties prenantes pour la mise en œuvre
- Former et sensibiliser les collaborateurs (inclusion handicap)
- Réaliser des audits réguliers de conformité
- Évaluer et gérer les risques liés aux données

#### 🟢 Bloc 2 : Concevoir et déployer des architectures de données
- Identifier les besoins architecturaux (contraintes techniques/opérationnelles)
- Élaborer un cahier des charges d'architecture
- Concevoir des modèles de données (logiques et physiques)
- Déployer des infrastructures cloud/on-premise
- Mettre en place des outils de surveillance et monitoring
- Documenter l'architecture de manière accessible

#### 🟡 Bloc 3 : Concevoir et mettre en œuvre des pipelines de données
- Concevoir un système de gestion de données temps réel
- Établir des pipelines ETL/ELT entre bases de données
- Automatiser les flux de données
- Surveiller la qualité et la conformité des données
- Développer des procédures de contrôle qualité

#### 🔴 Bloc 4 : Construire, déployer et piloter des solutions d'IA
- Rédiger un cahier des charges pour solution IA
- Créer des algorithmes d'IA adaptés aux données
- Adapter l'infrastructure via des API
- Concevoir des pipelines CI/CD pour l'IA
- Développer des scripts de réentraînement automatique
- Piloter la performance en production (monitoring)

---

## 📁 Organisation du repository

```
certification-aia-rncp38777/
│
├── README.md                          # Ce fichier
├── LICENSE
├── .gitignore
│
├── bloc-1-data-governance/            # 🔵 BLOC 1
│   ├── README.md
│   ├── documentation/
│   │   ├── plan-gouvernance.md
│   │   ├── matrice-amdec.xlsx
│   │   ├── matrice-raci.xlsx
│   │   └── presentation.pptx
│   ├── cas-etude-spotify/
│   │   ├── contexte.md
│   │   ├── audit-initial.md
│   │   └── recommandations.md
│   └── livrables/
│       ├── plan-gouvernance-spotify.pdf
│       └── presentation-jury.pdf
│
├── bloc-2-architecture-donnees/       # 🟢 BLOC 2
│   ├── README.md
│   ├── documentation/
│   │   ├── cahier-des-charges.md
│   │   ├── architecture-diagrams/
│   │   └── presentation.pptx
│   ├── projet-stripe/
│   │   ├── architecture/
│   │   │   ├── aws-production/
│   │   │   └── docker-poc/
│   │   ├── terraform/
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   └── outputs.tf
│   │   ├── docker/
│   │   │   ├── docker-compose.yml
│   │   │   └── Dockerfile
│   │   └── scripts/
│   │       └── setup.sh
│   ├── demos/
│   │   └── video-demo-production.mp4
│   └── livrables/
│       ├── architecture-stripe.pdf
│       └── code-deployment.zip
│
├── bloc-3-pipelines-donnees/          # 🟡 BLOC 3
│   ├── README.md
│   ├── documentation/
│   │   ├── infrastructure-plan.md
│   │   ├── pipeline-diagrams/
│   │   └── presentation.pptx
│   ├── projet-fraud-detection/
│   │   ├── architecture/
│   │   │   ├── kafka-streaming/
│   │   │   ├── postgresql/
│   │   │   └── redis/
│   │   ├── terraform/
│   │   ├── airflow/
│   │   │   ├── dags/
│   │   │   └── plugins/
│   │   ├── data-quality/
│   │   │   └── great_expectations/
│   │   └── monitoring/
│   │       └── grafana-dashboards/
│   ├── demos/
│   │   └── video-pipeline-live.mp4
│   └── livrables/
│       ├── infrastructure-fraud.pdf
│       └── code-pipeline.zip
│
├── bloc-4-solutions-ia/               # 🔴 BLOC 4
│   ├── README.md
│   ├── documentation/
│   │   ├── cahier-des-charges.md
│   │   ├── model-documentation.md
│   │   └── presentation.pptx
│   ├── projet-final/
│   │   ├── notebooks/
│   │   │   ├── 01-exploration.ipynb
│   │   │   ├── 02-preprocessing.ipynb
│   │   │   └── 03-modeling.ipynb
│   │   ├── src/
│   │   │   ├── models/
│   │   │   ├── api/
│   │   │   └── training/
│   │   ├── mlflow/
│   │   ├── cicd/
│   │   │   ├── .github/workflows/
│   │   │   └── Jenkinsfile
│   │   ├── monitoring/
│   │   │   ├── evidently/
│   │   │   └── prometheus/
│   │   └── deployment/
│   │       ├── kubernetes/
│   │       └── docker/
│   ├── demos/
│   │   └── video-solution-production.mp4
│   └── livrables/
│       ├── solution-ia.pdf
│       ├── code-model.zip
│       └── code-deployment.zip
│
├── resources/                          # 📚 RESSOURCES COMMUNES
│   ├── templates/
│   │   ├── presentation-template.pptx
│   │   └── documentation-template.md
│   ├── datasets/
│   │   └── links.md
│   └── references/
│       ├── RGPD-guide.pdf
│       └── aws-best-practices.md
│
└── docs/                               # 📖 DOCUMENTATION GLOBALE
    ├── certification-guide.md
    ├── qa-preparation/
    │   ├── bloc1-questions.md
    │   ├── bloc2-questions.md
    │   ├── bloc3-questions.md
    │   └── bloc4-questions.md
    └── evaluation-criteria.md
```

---

## 🔵 Bloc 1 - Gouvernance des données

### Projet : Data Governance Spotify

**Contexte** : Spotify connaît une croissance exponentielle et fait face à un silotage important entre ses départements, particulièrement au niveau Marketing.

### Objectifs du projet

1. Diagnostiquer l'état actuel de la gouvernance
2. Concevoir un plan de gouvernance complet
3. Définir les rôles et responsabilités (RACI)
4. Analyser les risques (matrice AMDEC)
5. Proposer une transformation organisationnelle (Embedded → Centre d'Excellence)

### Livrables attendus

| Livrable | Format | Contenu |
|----------|--------|---------|
| Plan de gouvernance | Word/Google Doc | Politique complète, processus, standards |
| Présentation | PowerPoint/Slides | Synthèse 15 min pour le jury |
| Matrice AMDEC | Excel | 14 risques identifiés avec actions |
| Matrice RACI | Excel | Rôles et responsabilités |

### Points clés à développer

- ✅ Transformation Embedded → Centre d'Excellence
- ✅ Gestion des risques TOHE (Technique, Organisationnel, Humain, Économique, Légal)
- ✅ Accessibilité et inclusion (personnes en situation de handicap)
- ✅ Exercice des droits utilisateurs (RGPD/CCPA)
- ✅ Audits réguliers et amélioration continue

### Critères d'évaluation

- Pertinence du diagnostic
- Complétude du plan de gouvernance
- Gestion des parties prenantes
- Prise en compte de la conformité réglementaire
- Capacité à gérer les risques

---

## 🟢 Bloc 2 - Architecture de données

### Projet : From SQL to NoSQL - Migration Stripe

**Contexte** : Migration d'une plateforme de paiement Stripe vers une architecture hybride SQL/NoSQL pour améliorer les performances et la scalabilité.

### Objectifs du projet

1. Concevoir une architecture de données hybride
2. Déployer l'infrastructure en production (AWS) et en POC (Docker)
3. Assurer la conformité PCI-DSS et RGPD
4. Optimiser les performances et la scalabilité
5. Documenter l'architecture complète

### Livrables attendus

| Livrable | Format | Contenu |
|----------|--------|---------|
| Diagramme architecture | PowerPoint/Draw.io | Architecture complète (SQL, NoSQL, cache) |
| Code Infrastructure | GitHub | Terraform pour AWS, Docker Compose pour POC |
| Vidéo démo | MP4 | Infrastructure en production (5-10 min) |
| Documentation | Markdown | Guide de déploiement et troubleshooting |

### Stack technique proposée

**Production (AWS)** :
- RDS PostgreSQL (données transactionnelles)
- DocumentDB (données semi-structurées)
- Keyspaces (analytics temps réel)
- Neptune (graphe de fraude)
- ElastiCache Redis (cache)
- S3 (stockage objets)

**POC Local (Docker)** :
- PostgreSQL
- MongoDB
- Cassandra
- Neo4j
- Redis
- MinIO

### Points clés à développer

- ✅ Choix techniques justifiés (SQL vs NoSQL)
- ✅ Conformité PCI-DSS pour données de paiement
- ✅ Architecture résiliente et scalable
- ✅ Monitoring et alerting
- ✅ Documentation accessible

### Critères d'évaluation

- Pertinence des choix techniques
- Respect des contraintes de sécurité
- Qualité du code d'infrastructure
- Clarté de la documentation
- Démonstration fonctionnelle

---

## 🟡 Bloc 3 - Pipelines de données

### Projet : Automatic Fraud Detection

**Contexte** : Système de détection de fraude bancaire en temps réel avec gestion de flux de données massifs.

### Objectifs du projet

1. Concevoir un système de gestion temps réel
2. Implémenter des pipelines ETL/ELT
3. Automatiser les flux de données
4. Mettre en place le monitoring qualité
5. Assurer la conformité RGPD

### Livrables attendus

| Livrable | Format | Contenu |
|----------|--------|---------|
| Diagramme pipeline | PowerPoint/Draw.io | Architecture de streaming complète |
| Code Pipeline | GitHub | Kafka, Airflow, scripts ETL |
| Vidéo démo | MP4 | Pipeline en fonctionnement (5-10 min) |
| Documentation | Markdown | Guide opérationnel |

### Stack technique proposée

**Streaming** :
- Apache Kafka / MSK (ingestion temps réel)
- Apache Flink (traitement streaming)

**Orchestration** :
- Apache Airflow (orchestration batch)
- Prefect (alternative moderne)

**Stockage** :
- PostgreSQL (données structurées)
- Redis (cache rapide)
- S3 (data lake)

**Qualité** :
- Great Expectations (validation)
- dbt (transformations)

**Monitoring** :
- Grafana + Prometheus
- CloudWatch (AWS)

### Dataset recommandé

**CiferAI/Cifer-Fraud-Detection-Dataset-AF** (Hugging Face)
- 21 millions de transactions
- Données réalistes de fraude
- Déséquilibre de classes (fraude rare)

### Points clés à développer

- ✅ Architecture temps réel vs batch
- ✅ Data Quality Framework
- ✅ Gestion des volumes massifs
- ✅ Automatisation complète
- ✅ Monitoring et alerting

### Critères d'évaluation

- Architecture temps réel fonctionnelle
- Qualité du code de pipeline
- Robustesse du système
- Monitoring effectif
- Documentation opérationnelle

---

## 🔴 Bloc 4 - Solutions d'IA

### Projet : Final Project - Solution IA complète

**Contexte** : Développement, déploiement et monitoring d'une solution d'IA en production avec CI/CD complet.

### Objectifs du projet

1. Rédiger un cahier des charges IA
2. Développer un modèle ML/DL performant
3. Créer une API de prédiction
4. Mettre en place un pipeline CI/CD
5. Automatiser le réentraînement
6. Monitorer les performances en production

### Livrables attendus

| Livrable | Format | Contenu |
|----------|--------|---------|
| Présentation solution | PowerPoint | Solution IA complète (5 min) |
| Code modèle | GitHub | Notebooks + code source |
| Code déploiement | GitHub | API, CI/CD, monitoring |
| Vidéo démo | MP4 | Solution en production (5-10 min) |

### Stack technique proposée

**Développement** :
- Python (scikit-learn, TensorFlow, PyTorch)
- Jupyter Notebooks
- MLflow (tracking expériences)

**API** :
- FastAPI ou Flask
- Docker

**CI/CD** :
- GitHub Actions ou Jenkins
- Tests automatisés (pytest)

**Déploiement** :
- AWS SageMaker ou EC2
- Kubernetes (optionnel)

**Monitoring** :
- Evidently AI (drift detection)
- Prometheus + Grafana
- CloudWatch

**Réentraînement** :
- Airflow DAG ou Lambda
- Déclenchement automatique sur drift

### Points clés à développer

- ✅ Cahier des charges complet
- ✅ Expérimentation rigoureuse (MLflow)
- ✅ API RESTful performante
- ✅ CI/CD automatisé
- ✅ Monitoring production (data drift, model drift)
- ✅ Réentraînement automatique

### Critères d'évaluation

- Qualité du modèle (métriques)
- Architecture de déploiement
- Automatisation CI/CD
- Monitoring effectif
- Documentation technique

---

## 🛠️ Technologies utilisées

### Cloud & Infrastructure

| Technologie | Usage | Blocs |
|-------------|-------|-------|
| AWS (RDS, S3, EC2, Lambda) | Infrastructure cloud | 2, 3, 4 |
| Terraform | Infrastructure as Code | 2, 3 |
| Docker / Docker Compose | Containerisation | 2, 3, 4 |
| Kubernetes | Orchestration (optionnel) | 4 |

### Bases de données

| Technologie | Type | Usage | Blocs |
|-------------|------|-------|-------|
| PostgreSQL | SQL | Données transactionnelles | 2, 3 |
| MongoDB / DocumentDB | NoSQL Document | Données semi-structurées | 2 |
| Cassandra / Keyspaces | NoSQL Wide-column | Analytics temps réel | 2 |
| Neo4j / Neptune | Graph | Détection de fraude | 2 |
| Redis / ElastiCache | Cache | Performance | 2, 3 |

### Data Engineering

| Technologie | Usage | Blocs |
|-------------|-------|-------|
| Apache Kafka / MSK | Streaming | 3 |
| Apache Airflow | Orchestration | 3, 4 |
| dbt | Transformations | 3 |
| Great Expectations | Qualité données | 3 |

### Machine Learning

| Technologie | Usage | Blocs |
|-------------|-------|-------|
| scikit-learn | ML classique | 4 |
| TensorFlow / PyTorch | Deep Learning | 4 |
| MLflow | Tracking expériences | 4 |
| FastAPI | API ML | 4 |

### Monitoring & Observabilité

| Technologie | Usage | Blocs |
|-------------|-------|-------|
| Grafana + Prometheus | Métriques infrastructure | 2, 3, 4 |
| Evidently AI | Monitoring ML | 4 |
| CloudWatch | Logs AWS | 2, 3, 4 |

### CI/CD

| Technologie | Usage | Blocs |
|-------------|-------|-------|
| GitHub Actions | Pipeline CI/CD | 2, 3, 4 |
| pytest | Tests automatisés | 3, 4 |
| pre-commit | Quality gates | 2, 3, 4 |

---

## 🚀 Installation et déploiement

### Prérequis

```bash
# Système
- macOS / Linux / Windows (WSL2)
- Python 3.9+
- Docker Desktop
- Git

# Cloud (optionnel pour production)
- Compte AWS
- AWS CLI configuré
- Terraform installé
```

### Installation locale

```bash
# Cloner le repository
git clone https://github.com/[username]/certification-aia-rncp38777.git
cd certification-aia-rncp38777

# Configuration environnement Python
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configuration Docker
docker --version
docker-compose --version
```

### Déploiement par bloc

#### Bloc 2 - Architecture

```bash
cd bloc-2-architecture-donnees/projet-stripe

# POC Local (Docker)
docker-compose up -d

# Production (AWS)
cd terraform
terraform init
terraform plan
terraform apply
```

#### Bloc 3 - Pipelines

```bash
cd bloc-3-pipelines-donnees/projet-fraud-detection

# Démarrer l'infrastructure locale
docker-compose up -d

# Démarrer Airflow
cd airflow
airflow db init
airflow webserver -p 8080
```

#### Bloc 4 - Solution IA

```bash
cd bloc-4-solutions-ia/projet-final

# Lancer les notebooks
jupyter lab

# Lancer l'API
cd src/api
uvicorn main:app --reload

# CI/CD
git push origin main  # Déclenche GitHub Actions
```

---

## 📅 Calendrier de certification

### Planning prévisionnel

| Bloc | Date cible | Statut | Priorité |
|------|------------|--------|----------|
| Bloc 1 | Janvier 2026 | 🟡 En cours | P0 |
| Bloc 2 | Février 2026 | 🟡 En cours | P0 |
| Bloc 3 | Mars 2026 | 🔴 À faire | P1 |
| Bloc 4 | Avril 2026 | 🔴 À faire | P1 |

### Jalons importants

- ✅ **Novembre 2025** : Matrice AMDEC Bloc 1 complétée
- ✅ **Novembre 2025** : Architecture AWS Bloc 2 déployée
- 🟡 **Décembre 2025** : POC Docker Bloc 2 finalisé
- 🔴 **Janvier 2026** : Dataset Bloc 3 sélectionné
- 🔴 **Janvier 2026** : Présentation Bloc 1 au jury
- 🔴 **Février 2026** : Infrastructure Bloc 3 déployée
- 🔴 **Mars 2026** : Modèle ML Bloc 4 entraîné
- 🔴 **Avril 2026** : Certification complète obtenue

---

## 📖 Ressources et documentation

### Documentation officielle

- [France Compétences - RNCP38777](https://www.francecompetences.fr/recherche/rncp/38777/)
- [Guide de certification AIA](./docs/certification-guide.md)
- [Critères d'évaluation](./docs/evaluation-criteria.md)

### Réglementations

- [RGPD - Texte officiel](https://www.cnil.fr/fr/reglement-europeen-protection-donnees)
- [CCPA - California Consumer Privacy Act](https://oag.ca.gov/privacy/ccpa)
- [PCI-DSS - Payment Card Industry](https://www.pcisecuritystandards.org/)

### Technologies

- [AWS Documentation](https://docs.aws.amazon.com/)
- [Terraform Documentation](https://www.terraform.io/docs)
- [Docker Documentation](https://docs.docker.com/)
- [Airflow Documentation](https://airflow.apache.org/docs/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

### Datasets

- [Hugging Face Datasets](https://huggingface.co/datasets)
- [CiferAI Fraud Detection](https://huggingface.co/datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

### Préparation Q&A

- [Questions Bloc 1](./docs/qa-preparation/bloc1-questions.md)
- [Questions Bloc 2](./docs/qa-preparation/bloc2-questions.md)
- [Questions Bloc 3](./docs/qa-preparation/bloc3-questions.md)
- [Questions Bloc 4](./docs/qa-preparation/bloc4-questions.md)

---

## 📊 Avancement du projet

### Bloc 1 - Gouvernance (70% complété)

- [x] Contexte et problématique Spotify
- [x] Matrice AMDEC (14 risques)
- [x] Matrice RACI
- [x] Plan de transformation Embedded → CoE
- [ ] Documentation finale
- [ ] Présentation PowerPoint
- [ ] Préparation Q&A jury

### Bloc 2 - Architecture (60% complété)

- [x] Architecture AWS production
- [x] Code Terraform
- [x] Architecture Docker POC
- [ ] Vidéo démo production
- [ ] Documentation complète
- [ ] Présentation PowerPoint

### Bloc 3 - Pipelines (30% complété)

- [x] Sélection dataset (CiferAI)
- [x] Architecture technique définie
- [ ] Infrastructure déployée
- [ ] Pipelines ETL/ELT
- [ ] Data Quality Framework
- [ ] Monitoring
- [ ] Documentation
- [ ] Vidéo démo

### Bloc 4 - Solutions IA (10% complété)

- [ ] Cahier des charges
- [ ] Exploration de données
- [ ] Développement modèle
- [ ] API déployée
- [ ] CI/CD configuré
- [ ] Monitoring ML
- [ ] Documentation
- [ ] Vidéo démo

---

## 🤝 Contribution

Ce repository est personnel dans le cadre de la certification AIA. Les contributions externes ne sont pas acceptées, mais les suggestions et retours sont bienvenus.

### Standards de code

- **Python** : PEP 8, black formatter, type hints
- **Infrastructure** : Terraform best practices
- **Documentation** : Markdown avec diagrammes Mermaid
- **Git** : Commits conventionnels (feat, fix, docs, etc.)

### Structure des commits

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Exemples :
```
feat(bloc1): add AMDEC risk matrix
fix(bloc2): correct Terraform RDS configuration
docs(bloc3): update pipeline documentation
```

---

## 📝 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 📧 Contact

**Candidat** : David Rambeau  
**Formation** : Lead Data Science & Engineering  
**Certification** : AIA - RNCP38777  
**Email** : [votre-email]  
**LinkedIn** : [votre-profil-linkedin]  
**GitHub** : [votre-github]

---

## 🎯 Objectifs de ce repository

1. **Centraliser** tous les livrables des 4 blocs de certification
2. **Documenter** le travail réalisé de manière professionnelle
3. **Démontrer** les compétences techniques acquises
4. **Faciliter** la révision avant les jurys de certification
5. **Partager** (après certification) les bonnes pratiques et méthodologies

---


---

**Dernière mise à jour** : Novembre 2025  
**Version** : 1.0.0  
**Statut du projet** : 🟡 En cours de développement

---