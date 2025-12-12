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

# ========================================
# CONNEXION DATABASE
# ========================================

@pytest.fixture(scope="session")
def db_connection():
    """Shared database connection for entire test session"""
    try:
        conn = psycopg2.connect(
            os.getenv("NEONDB_CONNECTION_STRING"),
            cursor_factory=RealDictCursor
        )
        print("✅ Connected to Neon database\n")
        yield conn
        conn.close()
    except Exception as e:
        print(f"⚠️ Database connection failed: {e}\n")
        yield None

# ========================================
# CRÉATION DU TEST RUN
# ========================================

@pytest.fixture(scope="session")
def test_run_id(db_connection):
    """Create a test_run and return its ID"""
    if db_connection is None:
        print("⚠️ Skipping test_run creation - database not available\n")
        yield None
        return
    
    cursor = db_connection.cursor()
    
    # Contextual information
    git_commit = get_git_commit()
    git_branch = get_git_branch()
    ci_run_id = os.getenv("GITHUB_RUN_ID", None)
    environment = "ci" if ci_run_id else "local"
    created_by = os.getenv("GITHUB_ACTOR", os.getenv("USER", "unknown"))
    
    try:
        # Create the test_run
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
        
        # Update totals at end of session
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
        
        print(f"✅ Test Run {run_id} completed and updated\n")
        
    except Exception as e:
        print(f"❌ Error creating test_run: {e}\n")
        yield None

# ========================================
# MODEL LOADING FIXTURE
# ========================================

@pytest.fixture(scope="session", autouse=True)
def setup_model_for_tests():
    """Load MLflow model before tests - OPTIONAL (don't fail if not available)"""
    try:
        from App.Dockers.fastapi.main import app, getMyModel
        
        print("🤖 Loading model for tests...")
        model = getMyModel()
        
        if model is None:
            print("⚠️ Model not available - tests will skip model-dependent operations\n")
            yield None
            return
        
        app.state.loaded_model = model
        print(f"✅ Model loaded successfully\n")
        yield model
        
    except Exception as e:
        print(f"⚠️ Model loading failed (optional): {e}\n")
        yield None

# ========================================
# HOOK PYTEST POUR LOGGER CHAQUE TEST
# ========================================

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook executed after each test to log results"""
    
    outcome = yield
    report = outcome.get_result()
    
    # Log only after test execution (not setup/teardown)
    if report.when == "call":
        # Get fixtures from the request
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
# FONCTION DE LOGGING
# ========================================

def log_test_result(db_connection, run_id, item, report):
    """Log test result to test_logs table"""
    
    if db_connection is None:
        return
    
    cursor = db_connection.cursor()
    
    # Extract test information
    test_name = item.name
    test_file = str(item.fspath.relative_to(item.config.rootdir))
    
    # Determine category and gate
    if "unit" in test_file:
        test_category = "unit"
        test_gate = 1
    elif "integration" in test_file:
        test_category = "integration"
        test_gate = 2
    elif "quality" in test_file:
        test_category = "quality"
        test_gate = 2
    else:
        test_category = "other"
        test_gate = None
    
    # Status and duration
    status = report.outcome  # passed | failed | skipped
    duration_ms = int(report.duration * 1000)
    
    # Error (if failed)
    error_message = None
    error_traceback = None
    if report.failed:
        try:
            error_message = str(report.longreprtext)[:500] if hasattr(report, 'longreprtext') else str(report.longrepr)[:500]
            error_traceback = str(report.longrepr)[:2000] if hasattr(report, 'longrepr') else None
        except:
            pass
    
    # Git context
    git_commit = get_git_commit()
    git_branch = get_git_branch()
    ci_run_id = os.getenv("GITHUB_RUN_ID", None)
    environment = "ci" if ci_run_id else "local"
    created_by = os.getenv("GITHUB_ACTOR", os.getenv("USER", "unknown"))
    
    # Insert into test_logs
    try:
        cursor.execute("""
            INSERT INTO test_logs (
                run_id, test_name, test_file, test_category, test_gate,
                status, duration_ms, error_message, error_traceback,
                git_commit_hash, git_branch, ci_run_id, environment, created_at, created_by
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
        """, (
            run_id, test_name, test_file, test_category, test_gate,
            status, duration_ms, error_message, error_traceback,
            git_commit, git_branch, ci_run_id, environment, datetime.now(), created_by
        ))
        db_connection.commit()
        
        status_icon = "✅" if status == "passed" else "❌" if status == "failed" else "⏭️"
        print(f"{status_icon} {test_name} ({duration_ms}ms)")
        
    except Exception as e:
        print(f"❌ Error logging {test_name}: {e}")
        try:
            db_connection.rollback()
        except:
            pass

# ========================================
# UTILITAIRES GIT
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
    # In CI, use environment variable
    branch = os.getenv("GITHUB_REF_NAME")
    if branch:
        return branch
    
    # Locally, use git
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        return "unknown"