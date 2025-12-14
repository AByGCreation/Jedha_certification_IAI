# Security Policy

## 🔐 Politique de Sécurité - Système de Détection de Fraude Bancaire

Ce document définit la politique de sécurité pour le projet de détection de fraude bancaire (Certification AIA - RNCP38777).

---

## 📋 Table des Matières

- [Versions Supportées](#versions-supportées)
- [Signalement de Vulnérabilités](#signalement-de-vulnérabilités)
- [Mesures de Sécurité Implémentées](#mesures-de-sécurité-implémentées)
- [Conformité Réglementaire](#conformité-réglementaire)
- [Gestion des Secrets](#gestion-des-secrets)
- [Sécurité du Pipeline CI/CD](#sécurité-du-pipeline-cicd)
- [Protection des Données](#protection-des-données)
- [Contact](#contact)

---

## 🛡️ Versions Supportées

| Version | Supportée          | Fin de Support |
| ------- | ------------------ | -------------- |
| 1.x.x   | :white_check_mark: | En cours       |
| < 1.0   | :x:                | Non supporté   |

**Version actuelle en production** : `v1.0.0`

---

## 🚨 Signalement de Vulnérabilités

### Comment Signaler une Vulnérabilité

Si vous découvrez une vulnérabilité de sécurité, **NE PAS** créer d'issue publique.

**Procédure de signalement** :

1. **Email sécurisé** : Envoyez un rapport détaillé à `report@securitymail.com`
2. **Objet** : `[SECURITY] Vulnérabilité - [Description courte]`
3. **Informations requises** :
   - Description détaillée de la vulnérabilité
   - Étapes de reproduction
   - Impact potentiel (criticité)
   - Versions affectées
   - Preuve de concept (si applicable)

### Délais de Réponse

- **Accusé de réception** : Sous 48 heures ouvrées
- **Évaluation initiale** : Sous 5 jours ouvrés
- **Correction** :
  - Critique : 7 jours
  - Haute : 14 jours
  - Moyenne : 30 jours
  - Faible : 90 jours

### Divulgation Responsable

Nous nous engageons à :

- Reconnaître publiquement les chercheurs en sécurité (sauf demande contraire)
- Publier un avis de sécurité après correction
- Respecter la confidentialité jusqu'à publication du correctif

---

## 🔒 Mesures de Sécurité Implémentées

### 1. Infrastructure

#### API FastAPI

- ✅ **HTTPS obligatoire** en production
- ✅ **CORS** : Origines restreintes
- ✅ **Validation Pydantic** : Tous les inputs validés
- ✅ **Sanitization** : Échappement SQL via ORM

#### Base de Données (NeonDB PostgreSQL)

- ✅ **Connexion SSL/TLS** obligatoire
- ✅ **Credentials** : Stockés dans GitHub Secrets
- ✅ **Principe du moindre privilège** : Rôles limités
- ✅ **Backup automatique** : Quotidien (rétention 30 jours)
- ✅ **Audit logging** : Toutes les requêtes tracées

#### Stockage S3 (AWS)

- ✅ **Encryption at rest** : AES-256
- ✅ **Encryption in transit** : TLS 1.2+
- ✅ **IAM roles** : Permissions minimales
- ✅ **Bucket policies** : Accès restreint
- ✅ **Versioning** : Activé sur artefacts MLflow

### 2. Authentification & Autorisation

#### Accès MLflow

- ✅ **Token-based auth** : Tokens rotatifs
- ✅ **Expiration** : 90 jours
- ✅ **Permissions** : Read-only pour API, Write pour training

#### Accès API

- ✅ **API Keys** : Authentification requise en production
- ✅ **Refresh tokens** : Expiration 7 jours

#### Accès Admin

- ✅ **MFA obligatoire** : GitHub, AWS Console
- ✅ **IP whitelisting** : Accès restreint
- ✅ **Session timeout** : 15 minutes d'inactivité

### 3. Surveillance & Monitoring

#### Apitally

- ✅ **Surveillance temps réel** : Endpoints critiques
- ✅ **Alertes Slack** : Anomalies détectées
- ✅ **Métriques** : Latence, erreurs, volumétrie
- ✅ **Retention logs** : 90 jours

---

## ⚖️ Conformité Réglementaire

### RGPD (Règlement Général sur la Protection des Données)

#### Article 5 - Principes

- ✅ **Minimisation** : Collecte uniquement des données nécessaires
- ✅ **Limitation de conservation** : Purge automatique après 2 ans
- ✅ **Intégrité** : Encryption + checksums

#### Article 22 - Décision Automatisée

- ✅ **Droit à l'explication** : Features et scores tracés
- ✅ **Intervention humaine** : Transactions > 5000€ revues manuellement

#### Article 32 - Sécurité

- ✅ **Pseudonymisation** : cc_num hashé (SHA-256)
- ✅ **Encryption** : TLS 1.3, AES-256
- ✅ **Tests réguliers** : Pentests annuels

### PCI-DSS (Payment Card Industry Data Security Standard)

#### Exigence 3 - Protection des Données

- ✅ **Masquage** : Seuls les 4 derniers chiffres visibles
- ✅ **Pas de stockage CVV** : Jamais conservé
- ✅ **Encryption** : Algorithmes approuvés PCI

#### Exigence 10 - Journalisation

- ✅ **Audit trail** : Toutes modifications tracées
- ✅ **Horodatage** : NTP synchronisé
- ✅ **Retention** : 1 an minimum (2 ans implémenté)

#### Exigence 11 - Tests de Sécurité

- ✅ **Scans trimestriels** : Automated security scans
- ✅ **Pentests annuels** : Par organisme certifié

### AI Act (Règlement européen sur l'IA)

#### Systèmes à Haut Risque

- ✅ **Documentation technique** : Architecture complète
- ✅ **Journalisation** : Traçabilité des décisions
- ✅ **Supervision humaine** : Revue des cas critiques
- ✅ **Robustesse** : Tests adversariaux implémentés

---

## 🔑 Gestion des Secrets

### GitHub Secrets (Production)

#### Secrets Obligatoires

```
✅ AWS_ACCESS_KEY_ID          - Accès S3 pour modèles
✅ AWS_SECRET_ACCESS_KEY      - Credentials AWS
✅ AWS_DEFAULT_REGION         - Région AWS (eu-north-1)
✅ HF_TOKEN                   - Token Hugging Face (Write)
✅ MLFLOW_TRACKING_URI        - URI serveur MLflow
✅ NEONDB_CONNECTION_STRING   - PostgreSQL connection string
✅ APITALLY_CLIENT_ID         - Monitoring API key
```

### Rotation des Secrets

| Secret                   | Fréquence       | Dernière Rotation |
| ------------------------ | --------------- | ----------------- |
| AWS_ACCESS_KEY_ID        | 90 jours        | 2025-12-01        |
| HF_TOKEN                 | 180 jours       | 2025-11-15        |
| NEONDB_CONNECTION_STRING | À chaque breach | 2025-10-01        |
| APITALLY_CLIENT_ID       | 365 jours       | 2025-09-01        |

### Détection de Fuites

- ✅ **GitHub Secret Scanning** : Activé
- ✅ **Pre-commit hooks** : Scan local avant push
- ✅ **GitGuardian** : Surveillance continue

---

## 🔐 Sécurité du Pipeline CI/CD

### GitHub Actions

#### Workflow Hardening

- ✅ **Permissions minimales** : `contents: read`, `actions: write`
- ✅ **Pinned actions** : Versions SHA-256 (pas de @latest)
- ✅ **Secrets masqués** : Jamais loggés en clair
- ✅ **Environnement isolé** : Runners éphémères

#### Code Signing

- ✅ **Commits signés** : GPG obligatoire pour merges
- ✅ **Tags signés** : Vérification avant déploiement
- ✅ **SBOM** : Software Bill of Materials généré

### Portes de Contrôle

#### Gate 1 - Tests Unitaires

- ✅ **Accuracy** : ≥92% requis
- ✅ **F1-Score** : ≥85% requis
- ✅ **Coverage** : ≥80% requis

#### Gate 2 - Tests Intégration

- ✅ **Traçabilité** : Tous logs écrits
- ✅ **Latence P99** : <100ms
- ✅ **Sécurité** : Pas de secrets dans logs

#### Gate 3 - Tests Smoke

- ✅ **Healthcheck** : Endpoints répondent
- ✅ **Monitoring** : Apitally actif

---

## 🛡️ Protection des Données

### Données en Transit

- ✅ **TLS 1.3** : Protocole minimum
- ✅ **Certificate Pinning** : Validation stricte

### Données au Repos

- ✅ **Database** : Encryption PostgreSQL native
- ✅ **S3** : SSE-S3 (AES-256)
- ✅ **Backups** : Encryptés (GPG)

### Rétention des Données

| Type de Donnée     | Durée     | Justification       |
| ------------------ | --------- | ------------------- |
| Prédictions (logs) | 2 ans     | RGPD Article 5.1.e  |
| Métriques agrégées | 5 ans     | Analyse long terme  |
| Modèles ML         | Permanent | Auditabilité        |
| Audit trails       | 7 ans     | Conformité bancaire |

---

## 🔍 Tests de Sécurité

### Automatisés (CI/CD)

#### SAST (Static Application Security Testing)

- ✅ **Bandit** : Scan Python (daily)
- ✅ **Safety** : Vulnérabilités dépendances
- ✅ **Trivy** : Scan containers Docker

#### DAST (Dynamic Application Security Testing)

- ✅ **OWASP ZAP** : Scan API (hebdo le dimanche entre 3h et 4h)

### Manuels (Trimestriels)

- ✅ **Code Review** : Revue par pairs obligatoire
- ✅ **Architecture Review** : Validation sécurité
- ✅ **Threat Modeling** : STRIDE analysis

### Pentests (Annuels)

- ✅ **Black Box** : Test en aveugle
- ✅ **Grey Box** : Accès partiel
- ✅ **Red Team** : Simulation attaque complète

---

## 🚨 Gestion des Incidents

### Plan de Réponse

#### 1. Détection (0-15min)

- Alertes Apitally/Slack
- Monitoring automatique
- Logs centralisés

#### 2. Évaluation (15-30min)

- Classification criticité (P0-P4)
- Impact assessment
- Équipe d'astreinte notifiée

#### 3. Containment (30min-2h)

- Isolation système compromis
- Rollback si nécessaire
- Blocage attaquant (IP/User)

#### 4. Éradication (2h-24h)

- Correction vulnérabilité
- Patch déployé
- Vérification sécurité

#### 5. Récupération (24h-72h)

- Restauration service
- Surveillance accrue
- Communication stakeholders

#### 6. Post-Mortem (1 semaine)

- Rapport incident
- Leçons apprises
- Mesures préventives

### Équipe d'Astreinte

| Rôle          | Contact                   | Disponibilité  |
| ------------- | ------------------------- | -------------- |
| Security Lead | security@securitymail.com | 24/7           |
| DevOps        | devops@securitymail.com   | Heures ouvrées |
| Legal/DPO     | dpo@securitymail.com      | Sur demande    |

---

## 📞 Contact

### Sécurité

- **Email** : `report@securitymail.com`

### Divulgation Publique

Les vulnérabilités corrigées seront publiées dans :

- **GitHub Security Advisories** : [Lien]
- **CHANGELOG.md** : Avec référence CVE si applicable

### Bug Bounty

Actuellement **aucun programme** de bug bounty actif (projet académique).

---

## 📚 Ressources

### Documentation Interne

- [Architecture Security Design](./docs/security-architecture.md)
- [Incident Response Playbook](./docs/incident-response.md)
- [Compliance Matrix](./docs/compliance-matrix.md)

### Standards & Références

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [ANSSI Guide](https://www.ssi.gouv.fr/)

---

## 📜 Historique des Mises à Jour

| Date       | Version | Changements                                 |
| ---------- | ------- | ------------------------------------------- |
| 2025-12-12 | 1.0.0   | Version initiale - Certification AIA Bloc 4 |

---

## ⚖️ Licence

Ce projet est développé dans le cadre de la certification **Architecte en Intelligence Artificielle (RNCP38777)**.

**Confidentialité** : Les données de production sont soumises au secret bancaire et ne sont pas incluses dans ce repository.

---

**Dernière mise à jour** : 12 décembre 2025  
**Responsable Sécurité** : David RAMBEAU  
**Version** : 1.0.0
