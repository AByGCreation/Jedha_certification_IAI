import os
import pytest
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import subprocess


from dotenv import load_dotenv, find_dotenv

# Trouve le fichier .env
dotenv_path = find_dotenv()
print(f"Fichier .env trouvé : {dotenv_path}")

# Charge le fichier
load_dotenv(dotenv_path)

# Vérifie la variable
print(f"NEONDB_CONNECTION_STRING = {os.getenv('NEONDB_CONNECTION_STRING')}")

# ========================================
# CONNEXION DATABASE
# ========================================

@pytest.fixture(scope="session")
def db_connection():
    """Connexion partagée pour toute la session de tests"""
    conn = psycopg2.connect(
        os.getenv("NEONDB_CONNECTION_STRING"),
        cursor_factory=RealDictCursor
    )
    yield conn
    conn.close()

# ========================================
# CRÉATION DU TEST RUN
# ========================================

@pytest.fixture(scope="session")
def test_run_id(db_connection):
    """Crée un test_run et retourne son ID"""
    cursor = db_connection.cursor()
    
    # Informations contextuelles
    git_commit = get_git_commit()
    git_branch = get_git_branch()
    ci_run_id = os.getenv("GITHUB_RUN_ID", None)
    environment = "ci" if ci_run_id else "local"
    created_by = os.getenv("GITHUB_ACTOR", os.getenv("USER", "unknown"))
    
    # Créer le test_run
    cursor.execute("""
        INSERT INTO test_runs (
            git_commit_hash, git_branch, ci_run_id, 
            environment, started_at, created_by,
            total_tests, passed_tests, failed_tests, skipped_tests
        ) VALUES (%s, %s, %s, %s, %s, %s, 0, 0, 0, 0)
        RETURNING run_id
    """, (git_commit, git_branch, ci_run_id, environment, datetime.now(), created_by))
    
    run_id = cursor.fetchone()['run_id']
    db_connection.commit()
    
    print(f"\n📊 Test Run ID: {run_id}")
    print(f"   Branch: {git_branch}")
    print(f"   Commit: {git_commit[:8]}")
    print(f"   Environment: {environment}\n")
    
    yield run_id
    
    # À la fin de la session, mettre à jour les totaux
    cursor.execute("""
        UPDATE test_runs 
        SET 
            completed_at = %s,
            total_tests = (SELECT COUNT(*) FROM test_logs WHERE run_id = %s),
            passed_tests = (SELECT COUNT(*) FROM test_logs WHERE run_id = %s AND status = 'passed'),
            failed_tests = (SELECT COUNT(*) FROM test_logs WHERE run_id = %s AND status = 'failed'),
            skipped_tests = (SELECT COUNT(*) FROM test_logs WHERE run_id = %s AND status = 'skipped'),
            total_duration_ms = (SELECT SUM(duration_ms) FROM test_logs WHERE run_id = %s),
            overall_status = CASE 
                WHEN (SELECT COUNT(*) FROM test_logs WHERE run_id = %s AND status = 'failed') > 0 
                THEN 'failed' 
                ELSE 'passed' 
            END
        WHERE run_id = %s
    """, (datetime.now(), run_id, run_id, run_id, run_id, run_id, run_id, run_id))
    db_connection.commit()
    
    print(f"\n✅ Test Run {run_id} terminé et mis à jour")

# ========================================
# HOOK PYTEST POUR LOGGER CHAQUE TEST
# ========================================

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook exécuté après chaque test pour logger le résultat"""
    
    # Exécuter le test
    outcome = yield
    report = outcome.get_result()
    
    # Logger uniquement après l'exécution du test (pas setup/teardown)
    if report.when == "call":
        # Récupérer les fixtures
        db_connection = item.funcargs.get('db_connection')
        test_run_id = item.funcargs.get('test_run_id')
        
        if db_connection and test_run_id:
            log_test_result(
                db_connection=db_connection,
                run_id=test_run_id,
                item=item,
                report=report
            )

# ========================================
# FONCTION DE LOGGING
# ========================================

def log_test_result(db_connection, run_id, item, report):
    """Enregistre le résultat d'un test dans test_logs"""
    
    cursor = db_connection.cursor()
    
    # Extraire les informations du test
    test_name = item.name
    test_file = str(item.fspath.relative_to(item.config.rootdir))
    
    # Déterminer la catégorie et le gate
    if "unit" in test_file:
        test_category = "unit"
        test_gate = 1
    elif "integration" in test_file:
        test_category = "integration"
        test_gate = 2
    elif "smoke" in test_file:
        test_category = "smoke"
        test_gate = 3
    else:
        test_category = "other"
        test_gate = None
    
    # Statut et durée
    status = report.outcome  # passed | failed | skipped
    duration_ms = int(report.duration * 1000)
    
    # Erreur (si échec)
    error_message = None
    error_traceback = None
    if report.failed:
        error_message = str(report.longreprtext)[:500] if hasattr(report, 'longreprtext') else str(report.longrepr)[:500]
        error_traceback = str(report.longrepr)[:2000] if hasattr(report, 'longrepr') else None
    
    # Métriques spécifiques (si attachées au test)
    model_version = getattr(item, 'model_version', None)
    model_accuracy = getattr(item, 'model_accuracy', None)
    model_f1_score = getattr(item, 'model_f1_score', None)
    api_latency_p99 = getattr(item, 'api_latency_p99', None)
    
    # Contexte Git et CI
    git_commit = get_git_commit()
    git_branch = get_git_branch()
    ci_run_id = os.getenv("GITHUB_RUN_ID", None)
    environment = "ci" if ci_run_id else "local"
    created_by = os.getenv("GITHUB_ACTOR", os.getenv("USER", "unknown"))
    
    # Insertion dans test_logs
    try:
        cursor.execute("""
            INSERT INTO test_logs (
                run_id, test_name, test_file, test_category, test_gate,
                status, duration_ms, error_message, error_traceback,
                model_version, model_accuracy, model_f1_score, api_latency_p99,
                git_commit_hash, git_branch, ci_run_id, environment, created_at, created_by
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
        """, (
            run_id, test_name, test_file, test_category, test_gate,
            status, duration_ms, error_message, error_traceback,
            model_version, model_accuracy, model_f1_score, api_latency_p99,
            git_commit, git_branch, ci_run_id, environment, datetime.now(), created_by
        ))
        db_connection.commit()
        
        # Affichage console
        status_icon = "✅" if status == "passed" else "❌" if status == "failed" else "⏭️"
        print(f"{status_icon} {test_name} ({duration_ms}ms) → Logged to DB")
        
    except Exception as e:
        print(f"❌ Erreur logging test {test_name}: {e}")
        db_connection.rollback()

# ========================================
# UTILITAIRES GIT
# ========================================

def get_git_commit():
    """Récupère le hash du commit Git actuel"""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        return "unknown"

def get_git_branch():
    """Récupère la branche Git actuelle"""
    # En CI, utiliser la variable d'environnement
    branch = os.getenv("GITHUB_REF_NAME")
    if branch:
        return branch
    
    # En local, utiliser git
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        return "unknown"