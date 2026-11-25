
-- RAD-2 Cloud Database Schema
-- 1. Raspberry Pi 기기 정보

CREATE TABLE IF NOT EXISTS devices (
    device_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    pair_code_hash TEXT,
    last_seen_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL
        DEFAULT NOW()
);


-- 2. Radar Event 정보

CREATE TABLE IF NOT EXISTS events (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    device_id TEXT NOT NULL
        REFERENCES devices(device_id),

    edge_event_id BIGINT NOT NULL,

    kind TEXT NOT NULL
        CHECK (kind IN ('camera', 'final')),

    person TEXT NOT NULL,

    t TIMESTAMPTZ NOT NULL,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,

    situation TEXT NOT NULL,

    prob DOUBLE PRECISION
        CHECK (prob >= 0 AND prob <= 1),

    abnormal BOOLEAN NOT NULL DEFAULT FALSE,

    photo BOOLEAN NOT NULL DEFAULT FALSE,
    photo_key TEXT,

    label TEXT DEFAULT NULL,

    details JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL
        DEFAULT NOW(),

    updated_at TIMESTAMPTZ NOT NULL
        DEFAULT NOW(),

    UNIQUE (device_id, edge_event_id)
);


-- 3. 이벤트 조회용 Index

CREATE INDEX IF NOT EXISTS idx_events_device_time
ON events (device_id, t DESC);
