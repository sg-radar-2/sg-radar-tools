
from pathlib import Path

from db.connection import get_connection


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def apply_schema():
    """schema.sql을 PostgreSQL에 적용한다."""

    # 1. SQL 파일 읽기
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

    print("[INFO] Applying database schema...", flush=True)

    # 2. PostgreSQL에 연결하여 SQL 실행
    with get_connection() as conn:
        conn.execute(schema_sql)

    print("[OK] Schema SQL executed.", flush=True)

    # 3. 실제 Table과 Index 생성 확인
    with get_connection() as conn:
        result = conn.execute(
            """
            SELECT
                to_regclass('public.devices'),
                to_regclass('public.events'),
                to_regclass('public.idx_events_device_time')
            """
        ).fetchone()

    if result is None or any(item is None for item in result):
        raise RuntimeError("Schema verification failed")

    print("[OK] devices table exists.", flush=True)
    print("[OK] events table exists.", flush=True)
    print("[OK] events index exists.", flush=True)
    print("[SUCCESS] Database schema applied.", flush=True)


if __name__ == "__main__":
    apply_schema()
