
import os

import psycopg


def get_connection():
    """
    Render 환경변수 DATABASE_URL을 이용하여
    PostgreSQL에 연결한다.
    """
    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    return psycopg.connect(
        database_url,
        connect_timeout=5
    )


def check_connection() -> bool:
    """
    PostgreSQL에서 SELECT 1을 실행하여
    DB 연결 상태를 확인한다.
    """
    with get_connection() as conn:
        result = conn.execute("SELECT 1").fetchone()

    return result is not None and result[0] == 1
