import os
import sys
import pytest
import psycopg2
import subprocess
from pathlib import Path
from psycopg2.extras import RealDictCursor
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.append('./Bloc4 - Final project/Projet/Architecture')

# Load environment FIRST
load_dotenv(find_dotenv(), override=True)
load_dotenv(".env", override=True)

print(f"\n{'='*60}")
print(f"🔧 CONFTEST SETUP")
print(f"{'='*60}")
print(f"MLFLOW_TRACKING_URI: {os.getenv('MLFLOW_TRACKING_URI', 'NOT SET')}")
print(f"NEONDB: {os.getenv('NEONDB_CONNECTION_STRING', 'NOT SET')[:50]}...")
print(f"{'='*60}\n")

# ========================================
# IMPORT APP (après setup paths)
# ========================================

try:
    from App.Dockers.fastapi.main import app, getMyModel, getModelRunID
    print("✅ FastAPI app imported successfully\n")
except Exception as e:
    print(f"❌ Failed to import FastAPI app: {e}\n")
    app = None
    getMyModel = None
    getModelRunID = None

# ========================================
# SESSION-WIDE MODEL LOADING
# ========================================

@pytest.fixture(scope="session", autouse=True)
def load_model_once():
    """
    Charge le modèle UNE SEULE FOIS pour toute la session de tests.
    En CI, si le chargement échoue, continue sans modèle (tests skip).
    """
    if app is None or getMyModel is None:
        print("⚠️ FastAPI app not available, skipping model load\n")
        yield None
        return
    
    print("\n🤖 Loading MLflow model (once for all tests)...")
    
    # Vérifier si on est en CI
    is_ci = os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"
    

    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
    
    if aws_access_key and aws_secret_key:
        print(f"🔑 AWS credentials found (key: {aws_access_key[:10]}***)")
        print(f"🌍 AWS region: {aws_region}")
        
        # Configurer boto3 explicitement
        import boto3
        boto3.setup_default_session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        
        # OU configurer via variables d'environnement
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key
        os.environ["AWS_DEFAULT_REGION"] = aws_region
        
        print("✅ Boto3 configured with credentials")
    else:
        print("⚠️ AWS credentials not found in environment")



    try:
        model = getMyModel()
        
        if model is None:
            if is_ci:
                print("⚠️ Model is None in CI - tests requiring model will be skipped\n")
            else:
                print("❌ Model is None locally - stopping tests\n")
                pytest.exit("❌ Failed to load model")
        else:
            # Injecter dans app.state
            app.state.loaded_model = model
            print("✅ Model loaded and injected into app.state\n")
        
        yield model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}\n")
        import traceback
        traceback.print_exc()
        
        # En CI, continuer sans modèle (skip tests)
        # En local, arrêter
        if is_ci:
            print("⚠️ In CI: Continuing without model (tests will skip)\n")
            yield None
        else:
            print("❌ Local: Stopping tests\n")
            pytest.exit(f"❌ Failed to load model: {e}")

# ========================================
# MODEL FIXTURES
# ========================================

@pytest.fixture(scope="session")
def model(load_model_once):
    """Get the loaded model"""
    return load_model_once

@pytest.fixture(scope="session")
def model_available(load_model_once):
    """Check if model is available"""
    return load_model_once is not None

@pytest.fixture
def require_model(model_available):
    """Skip test if model not available"""
    if not model_available:
        pytest.skip("Model not available - skipping test")

# ========================================
# FASTAPI CLIENT
# ========================================

@pytest.fixture(scope="session")
def client():
    """FastAPI test client - shared across all tests"""
    if app is None:
        pytest.skip("FastAPI app not available")
    
    from fastapi.testclient import TestClient
    return TestClient(app)

# ========================================
# DATABASE CONNECTION
# ========================================

@pytest.fixture(scope="session")
def db_connection():
    """Database connection - optional"""
    try:
        conn_str = os.getenv("NEONDB_CONNECTION_STRING")
        if not conn_str:
            print("⚠️ NEONDB_CONNECTION_STRING not set, skipping DB tests\n")
            yield None
            return
            
        conn = psycopg2.connect(conn_str, cursor_factory=RealDictCursor)
        print("✅ Connected to Neon database\n")
        yield conn
        conn.close()
        print("✅ Database connection closed\n")
    except Exception as e:
        print(f"⚠️ Database unavailable: {e}\n")
        yield None

# ========================================
# TEST RUN TRACKING
# ========================================

@pytest.fixture(scope="session")
def test_run_id(db_connection):
    """Create test run in database - optional"""
    if db_connection is None:
        yield None
        return
    
    cursor = db_connection.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO test_runs (
                git_commit_hash, git_branch, ci_run_id, 
                environment, started_at, created_by,
                total_tests, passed_tests, failed_tests, skipped_tests
            ) VALUES (%s, %s, %s, %s, %s, %s, 0, 0, 0, 0)
            RETURNING run_id
        """, (
            get_git_commit(),
            get_git_branch(),
            os.getenv("GITHUB_RUN_ID"),
            "ci" if os.getenv("GITHUB_RUN_ID") else "local",
            datetime.now(),
            os.getenv("GITHUB_ACTOR", os.getenv("USER", "unknown"))
        ))
        
        run_id = cursor.fetchone()['run_id']
        db_connection.commit()
        
        print(f"📊 Test Run ID: {run_id}\n")
        yield run_id
        
        # Update totals at end
        try:
            cursor.execute("""
                UPDATE test_runs 
                SET completed_at = %s,
                    total_tests = (SELECT COUNT(*) FROM test_logs WHERE run_id = %s),
                    passed_tests = (SELECT COUNT(*) FROM test_logs WHERE run_id = %s AND status = 'passed'),
                    failed_tests = (SELECT COUNT(*) FROM test_logs WHERE run_id = %s AND status = 'failed'),
                    skipped_tests = (SELECT COUNT(*) FROM test_logs WHERE run_id = %s AND status = 'skipped')
                WHERE run_id = %s
            """, (datetime.now(), run_id, run_id, run_id, run_id, run_id))
            db_connection.commit()
        except:
            pass
        
    except Exception as e:
        print(f"⚠️ Test run creation failed: {e}\n")
        yield None

# ========================================
# PYTEST HOOKS FOR LOGGING
# ========================================

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Log test results to database"""
    outcome = yield
    report = outcome.get_result()
    
    if report.when != "call":
        return
    
    if not hasattr(item, 'funcargs'):
        return
        
    db_connection = item.funcargs.get('db_connection')
    test_run_id = item.funcargs.get('test_run_id')
    
    if db_connection is None or test_run_id is None:
        return
    
    try:
        log_test_result(db_connection, test_run_id, item, report)
    except Exception as e:
        print(f"⚠️ Error logging test: {e}")

def log_test_result(db_connection, run_id, item, report):
    """Log test result to database"""
    cursor = db_connection.cursor()
    
    test_name = item.name
    test_file = str(item.fspath.relative_to(item.config.rootdir))
    
    if "unit" in test_file:
        test_category = "unit"
        test_gate = 1
    elif "integration" in test_file:
        test_category = "integration"
        test_gate = 2
    else:
        test_category = "other"
        test_gate = None
    
    status = report.outcome
    duration_ms = int(report.duration * 1000)
    error_message = str(report.longrepr)[:500] if report.failed else None
    
    try:
        cursor.execute("""
            INSERT INTO test_logs (
                run_id, test_name, test_file, test_category, test_gate,
                status, duration_ms, error_message,
                git_commit_hash, git_branch, ci_run_id, environment, created_at, created_by
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            run_id, test_name, test_file, test_category, test_gate,
            status, duration_ms, error_message,
            get_git_commit(), get_git_branch(), 
            os.getenv("GITHUB_RUN_ID"), 
            "ci" if os.getenv("GITHUB_RUN_ID") else "local",
            datetime.now(),
            os.getenv("GITHUB_ACTOR", os.getenv("USER", "unknown"))
        ))
        db_connection.commit()
        
        icon = "✅" if status == "passed" else "❌" if status == "failed" else "⏭️"
        print(f"{icon} {test_name} ({duration_ms}ms)")
    except Exception as e:
        print(f"❌ Logging error: {e}")

# ========================================
# GIT UTILITIES
# ========================================

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except:
        return os.getenv("GITHUB_SHA", "unknown")

def get_git_branch():
    branch = os.getenv("GITHUB_REF_NAME")
    if branch:
        return branch
    try:
        return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except:
        return "unknown"