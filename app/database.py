from sqlalchemy import create_engine, event
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import DATABASE_URL

_is_sqlite = DATABASE_URL.startswith("sqlite")

# Build engine kwargs based on database backend
_engine_kwargs = {}
if _is_sqlite:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
    # SQLite benefits from a bounded connection pool to avoid lock contention
    _engine_kwargs["poolclass"] = QueuePool
    _engine_kwargs["pool_size"] = 5
    _engine_kwargs["max_overflow"] = 10
    _engine_kwargs["pool_recycle"] = 3600  # recycle connections every hour
    _engine_kwargs["pool_pre_ping"] = True
else:
    # PostgreSQL (or other server-based databases)
    _engine_kwargs["pool_pre_ping"] = True
    _engine_kwargs["pool_size"] = 10
    _engine_kwargs["max_overflow"] = 20
    _engine_kwargs["pool_recycle"] = 1800  # recycle every 30 min to avoid stale connections

engine = create_engine(DATABASE_URL, **_engine_kwargs)


if _is_sqlite:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA synchronous=NORMAL")  # Faster writes with WAL (safe)
        cursor.execute("PRAGMA cache_size=-64000")    # 64MB page cache
        cursor.execute("PRAGMA temp_store=MEMORY")    # Temp tables in memory
        cursor.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped I/O
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
