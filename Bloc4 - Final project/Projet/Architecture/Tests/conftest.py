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

print(f"\n✅ .env loaded")
print(f"MLFLOW_TRACKING_URI: {os.getenv('MLFLOW_TRACKING_URI', 'NOT SET')}\n")

# ========================================
# GLOBAL MODEL STATE
# ========================================

_MODEL = None
_MODEL_LOADED = False

def try_load_model():
    """Try to load model - don't fail if it can't"""
    global _MODEL, _MODEL_LOADED
    
    if _MODEL_LOADED:
        return _MODEL
    
    print("🤖 Attempting to load model...")
    
    try:
        from App.Dockers.fastapi.main import getMyModel
        
        model = getMyModel()
        _MODEL = model
        _MODEL_LOADED = True
        
        if model is not None:
            print("✅ Model loaded successfully\n")
        else:
            print("⚠️ Model is None - will skip model tests\n")
        
        return model
        
    except Exception as e:
        print(f"⚠️ Model loading failed (will skip model tests): {e}\n")
        _MODEL_LOADED = True
        _MODEL = None
        return None

# ========================================
# MODEL FIXTURES
# ========================================

@pytest.fixture(scope="session")
def model():
    """Get the loaded model (may be None)"""
    return try_load_model()

@pytest.fixture(scope="session")
def model_available():
    """Check if model is available"""
    model = try_load_model()
    return model is not None

@pytest.fixture(scope="session")
def require_model(model_available):
    """Skip test if model is not available"""
    if not model_available:
        pytest.skip("Model not available")

# ========================================
# DATABASE CONNECTION
# ========================================

@pytest.fixture(scope="session")
def db_connection():
    """Database connection - optional"""
    try:
        conn_str = os.getenv("NEONDB_CONNECTION_STRING")
        if not conn_str:
            print("⚠️ NEONDB_CONNECTION_STRING not set\n")
            yield None
            return
            
        conn = psycopg2.connect(conn_str, cursor_factory=RealDictCursor)
        print("✅ Connected to Neon database\n")
        yield conn
        conn.close()
    except Exception as e:
        print(f"⚠️ Database unavailable: {e}\n")
        yield None

# ========================================
# TEST RUN CREATION
# ========================================

@pytest.fixture(scope="session")
def test_run_id(db_connection):
    """Create test run - optional"""
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
        
        # Update at end
        try:
            cursor.execute("""
                UPDATE test_runs 
                SET completed_at = %s
                WHERE run_id = %s
            """, (datetime.now(), run_id))
            db_connection.commit()
        except:
            pass
        
    except Exception as e:
        print(f"⚠️ Test run creation failed: {e}\n")
        yield None

# ========================================
# PYTEST HOOKS
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

# ========================================
# LOGGING FUNCTION
# ========================================

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
    error_message = None
    
    if report.failed:
        try:
            error_message = str(report.longrepr)[:500]
        except:
            pass
    
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