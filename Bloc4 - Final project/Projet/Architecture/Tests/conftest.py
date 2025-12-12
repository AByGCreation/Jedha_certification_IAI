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

# Load environment
dotenv_path = find_dotenv()
load_dotenv(".env")
load_dotenv(dotenv_path)

print(f"\n✅ .env loaded from: {dotenv_path}")
print(f"NEONDB_CONNECTION_STRING: {'SET' if os.getenv('NEONDB_CONNECTION_STRING') else 'NOT SET'}")
print(f"MLFLOW_TRACKING_URI: {os.getenv('MLFLOW_TRACKING_URI', 'NOT SET')}\n")

# Global model storage
_LOADED_MODEL = None
_MODEL_LOAD_ERROR = None

# ========================================
# DATABASE CONNECTION
# ========================================

@pytest.fixture(scope="session")
def db_connection():
    """Shared database connection for entire test session"""
    try:
        conn_str = os.getenv("NEONDB_CONNECTION_STRING")
        if not conn_str:
            print("⚠️ NEONDB_CONNECTION_STRING not set - skipping database logging\n")
            yield None
            return
            
        conn = psycopg2.connect(conn_str, cursor_factory=RealDictCursor)
        print("✅ Connected to Neon database\n")
        yield conn
        conn.close()
    except Exception as e:
        print(f"⚠️ Database connection failed: {e}\n")
        yield None

# ========================================
# TEST RUN CREATION
# ========================================

@pytest.fixture(scope="session")
def test_run_id(db_connection):
    """Create a test_run and return its ID"""
    if db_connection is None:
        print("⚠️ Skipping test_run creation - database not available\n")
        yield None
        return
    
    cursor = db_connection.cursor()
    
    git_commit = get_git_commit()
    git_branch = get_git_branch()
    ci_run_id = os.getenv("GITHUB_RUN_ID", None)
    environment = "ci" if ci_run_id else "local"
    created_by = os.getenv("GITHUB_ACTOR", os.getenv("USER", "unknown"))
    
    try:
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
        
        print(f"📊 Test Run Created:")
        print(f"   ID: {run_id}")
        print(f"   Branch: {git_branch}")
        print(f"   Commit: {git_commit[:8] if git_commit else 'unknown'}")
        print(f"   Environment: {environment}\n")
        
        yield run_id
        
        # Update totals at end
        cursor.execute("""
            UPDATE test_runs 
            SET 
                completed_at = %s,
                total_tests = (SELECT COUNT(*) FROM test_logs WHERE run_id = %s),
                passed_tests = (SELECT COUNT(*) FROM test_logs WHERE run_id = %s AND status = 'passed'),
                failed_tests = (SELECT COUNT(*) FROM test_logs WHERE run_id = %s AND status = 'failed'),
                skipped_tests = (SELECT COUNT(*) FROM test_logs WHERE run_id = %s AND status = 'skipped'),
                total_duration_ms = (SELECT COALESCE(SUM(duration_ms), 0) FROM test_logs WHERE run_id = %s),
                overall_status = CASE 
                    WHEN (SELECT COUNT(*) FROM test_logs WHERE run_id = %s AND status = 'failed') > 0 
                    THEN 'failed' 
                    ELSE 'passed' 
                END
            WHERE run_id = %s
        """, (datetime.now(), run_id, run_id, run_id, run_id, run_id, run_id, run_id))
        db_connection.commit()
        print(f"✅ Test Run {run_id} updated\n")
        
    except Exception as e:
        print(f"❌ Error with test_run: {e}\n")
        yield None

# ========================================
# MODEL LOADING - OPTIONAL, NON-BLOCKING
# ========================================

@pytest.fixture(scope="session", autouse=True)
def load_model():
    """Try to load model but don't block tests if it fails"""
    global _LOADED_MODEL, _MODEL_LOAD_ERROR
    
    print("🤖 Attempting to load MLflow model...")
    try:
        from App.Dockers.fastapi.main import app, getMyModel
        
        model = getMyModel()
        if model is not None:
            app.state.loaded_model = model
            _LOADED_MODEL = model
            print("✅ Model loaded successfully\n")
        else:
            _MODEL_LOAD_ERROR = "Model is None"
            print("⚠️ Model returned None (may not be registered in MLflow)\n")
            
    except Exception as e:
        _MODEL_LOAD_ERROR = str(e)
        print(f"⚠️ Model loading failed (tests will continue): {e}\n")
    
    yield
    _LOADED_MODEL = None

@pytest.fixture(scope="session")
def model():
    """Fixture to provide model to tests"""
    global _LOADED_MODEL
    return _LOADED_MODEL

@pytest.fixture(scope="session")
def model_available():
    """Check if model is available"""
    global _LOADED_MODEL
    return _LOADED_MODEL is not None

# ========================================
# PYTEST HOOK FOR LOGGING
# ========================================

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook executed after each test to log results"""
    
    outcome = yield
    report = outcome.get_result()
    
    if report.when == "call":
        if hasattr(item, 'funcargs'):
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
# LOGGING FUNCTION
# ========================================

def log_test_result(db_connection, run_id, item, report):
    """Log test result to test_logs table"""
    
    if db_connection is None:
        return
    
    cursor = db_connection.cursor()
    test_name = item.name
    test_file = str(item.fspath.relative_to(item.config.rootdir))
    
    # Determine category and gate
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
    
    error_message = None
    if report.failed:
        try:
            error_message = str(report.longrepr)[:500]
        except:
            pass
    
    git_commit = get_git_commit()
    git_branch = get_git_branch()
    ci_run_id = os.getenv("GITHUB_RUN_ID", None)
    environment = "ci" if ci_run_id else "local"
    created_by = os.getenv("GITHUB_ACTOR", os.getenv("USER", "unknown"))
    
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
            git_commit, git_branch, ci_run_id, environment, datetime.now(), created_by
        ))
        db_connection.commit()
        status_icon = "✅" if status == "passed" else "❌" if status == "failed" else "⏭️"
        print(f"{status_icon} {test_name}")
        
    except Exception as e:
        print(f"❌ Error logging {test_name}: {e}")

# ========================================
# GIT UTILITIES
# ========================================

def get_git_commit():
    """Get current Git commit hash"""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        return os.getenv("GITHUB_SHA", "unknown")

def get_git_branch():
    """Get current Git branch"""
    branch = os.getenv("GITHUB_REF_NAME")
    if branch:
        return branch
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        return "unknown"