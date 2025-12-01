# 🔄 APACHE AIRFLOW - ETL/ELT ORCHESTRATION

## Vue d'ensemble

Apache Airflow orchestre les pipelines ETL/ELT de la plateforme de détection de fraude.

## 🌐 Accès

**URL:** http://localhost:8080  
**Login:** admin  
**Password:** stripe_password

## 📊 DAGs Disponibles

### 1. **etl_postgres_to_clickhouse** 
**Fréquence:** Quotidienne (2h du matin)

Pipeline ETL principal : PostgreSQL (OLTP) → ClickHouse (OLAP)

**Étapes:**
1. **Extract** - Extraction transactions PostgreSQL
2. **Transform** - Conversion types, nettoyage
3. **Load** - Chargement batch ClickHouse
4. **Validate** - Vérification qualité (count match)
5. **Notify** - Notification de fin

**Visualisation:**
```
extract_from_postgres >> transform_transactions >> load_to_clickhouse 
                      >> validate_data_quality >> send_completion_notification
```

---

### 2. **data_quality_checks**
**Fréquence:** Quotidienne (6h du matin)

Validation automatisée de la qualité des données.

**Vérifications:**
- ✅ Valeurs NULL interdites
- ✅ Types de données (amount > 0, fraud_score 0-100)
- ✅ Intégrité référentielle (FK valides)
- ⚠️  Règles métier (fraud_score vs is_fraud cohérent)

---

### 3. **daily_aggregations**
**Fréquence:** Quotidienne (3h du matin, après ETL)

Pré-calcul des métriques analytics pour dashboards Grafana.

**Tables créées:**
- `daily_stats` - Statistiques globales quotidiennes
- `merchant_daily_stats` - Métriques par marchand
- `user_daily_stats` - Métriques par utilisateur
- `hourly_patterns` - Patterns horaires (détection anomalies)

**Optimisation:** OPTIMIZE TABLE automatique après agrégations

---

## 🚀 Démarrage Rapide

### Lancer Airflow avec la plateforme

```bash
# Lancer tous les services (inclut Airflow)
docker-compose up -d

# Vérifier qu'Airflow est prêt
docker-compose logs -f airflow-webserver

# Ouvrir l'interface
open http://localhost:8080
```

### Activer les DAGs

Par défaut, les DAGs sont **désactivés**. Pour les activer :

1. Ouvrir http://localhost:8080
2. Se connecter (admin / stripe_password)
3. Cliquer sur le toggle à gauche de chaque DAG
4. Les DAGs s'exécuteront selon leur schedule

### Exécution manuelle

Pour tester immédiatement un DAG :

1. Cliquer sur le nom du DAG
2. Cliquer sur le bouton "▶️ Trigger DAG" en haut à droite
3. Observer l'exécution en temps réel dans "Graph View"

---

## 📈 Monitoring

### Logs des tâches

1. Cliquer sur un DAG
2. Cliquer sur un run
3. Cliquer sur une tâche
4. Onglet "Logs" pour voir les détails

### Métriques de performance

- **Duration:** Temps d'exécution de chaque tâche
- **Success Rate:** Taux de réussite historique
- **Next Run:** Prochaine exécution programmée

---

## 🛠️ Développement de DAGs

### Structure d'un DAG

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta

default_args = {
    'owner': 'david_rambeau',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'mon_dag',
    default_args=default_args,
    schedule_interval='0 2 * * *',  # Cron expression
    catchup=False,
    tags=['custom'],
) as dag:
    
    task = PythonOperator(
        task_id='ma_tache',
        python_callable=ma_fonction,
    )
```

### Ajouter un nouveau DAG

1. Créer un fichier Python dans `/airflow/dags/`
2. Redémarrer Airflow : `docker-compose restart airflow-scheduler`
3. Le DAG apparaît automatiquement dans l'interface

---

## 🔧 Configuration

### Variables d'environnement

Définies dans `docker-compose.yml` :

```yaml
POSTGRES_HOST: postgres
CLICKHOUSE_HOST: clickhouse
MONGODB_HOST: mongodb
...
```

### Connexions

Airflow se connecte directement aux bases via les variables d'environnement.  
Pas besoin de configurer les "Connections" dans l'UI.

---

## 📊 Intégration avec Grafana

Les tables agrégées par Airflow alimentent les dashboards Grafana :

- `daily_stats` → Dashboard "Transaction Volume"
- `merchant_daily_stats` → Dashboard "Top Merchants"
- `hourly_patterns` → Dashboard "Fraud Patterns by Hour"

---

## 🎓 Certification AIA - Bloc 3

Ces DAGs démontrent les compétences suivantes :

✅ **Orchestration ETL** - Apache Airflow  
✅ **Pipelines automatisés** - Scheduling quotidien  
✅ **Data Quality** - Validation automatisée  
✅ **Agrégations batch** - Pré-calcul métriques  
✅ **Monitoring** - Logs, alertes, retry  
✅ **Scalabilité** - Traitement par batch  

---

## 📞 Support

En cas de problème :

```bash
# Vérifier les logs Airflow
docker-compose logs airflow-scheduler
docker-compose logs airflow-webserver

# Redémarrer Airflow
docker-compose restart airflow-scheduler airflow-webserver

# Réinitialiser la BDD Airflow (attention : perte historique)
docker-compose down
docker volume rm dataarch_airflow_postgres_data
docker-compose up -d
```

---

**Airflow transforme votre architecture en un système de données industrialisé et automatisé ! 🚀**
