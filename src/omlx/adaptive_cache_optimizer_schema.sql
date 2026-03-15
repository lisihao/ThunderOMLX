-- ThunderOMLX Adaptive Cache Optimizer Database Schema
-- Phase 1: Data Collection Infrastructure
-- Created: 2026-03-14

-- 表 1: 每次推理记录
CREATE TABLE IF NOT EXISTS agent_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- Prompt 信息
    system_prompt_length INTEGER,
    user_query_length INTEGER,
    total_prompt_length INTEGER,

    -- 缓存性能
    cache_hit_ratio REAL,
    skip_logic_type TEXT,

    -- 配置信息
    block_size INTEGER,
    padding_tokens INTEGER,
    padding_overhead REAL,

    -- 性能指标
    prefill_time_ms REAL,
    decode_time_ms REAL,
    total_time_ms REAL,

    -- 配置版本
    config_version TEXT
);

CREATE INDEX IF NOT EXISTS idx_agent_timestamp ON agent_metrics(agent_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_timestamp ON agent_metrics(timestamp);

-- 表 2: 聚合统计
CREATE TABLE IF NOT EXISTS cache_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    date DATE NOT NULL,

    request_count INTEGER,
    avg_cache_hit_ratio REAL,
    full_skip_rate REAL,
    avg_padding_overhead REAL,

    prompt_length_p50 INTEGER,
    prompt_length_p90 INTEGER,
    prompt_length_p99 INTEGER,

    avg_total_time_ms REAL,
    current_block_size INTEGER,

    UNIQUE(agent_id, date)
);

-- 表 3: 配置变更历史
CREATE TABLE IF NOT EXISTS config_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

    old_block_size INTEGER,
    new_block_size INTEGER,
    old_max_padding INTEGER,
    new_max_padding INTEGER,

    change_reason TEXT,
    optimization_score REAL,
    actual_performance_delta REAL,
    is_rolled_back BOOLEAN DEFAULT 0,

    -- 性能基线（优化前）
    baseline_avg_prefill_ms REAL,
    baseline_avg_total_ms REAL,
    baseline_avg_padding REAL,
    baseline_sample_count INTEGER,

    -- 优化后性能（监控期）
    post_avg_prefill_ms REAL,
    post_avg_total_ms REAL,
    post_avg_padding REAL,
    post_sample_count INTEGER,

    -- 回滚信息
    rollback_timestamp DATETIME,
    rollback_reason TEXT
);

-- 表 4: A/B 测试
CREATE TABLE IF NOT EXISTS optimization_experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_time DATETIME,

    control_block_size INTEGER,
    treatment_block_size INTEGER,
    treatment_ratio REAL DEFAULT 0.1,  -- 10% 流量到实验组

    -- 控制组统计
    control_sample_count INTEGER DEFAULT 0,
    control_avg_prefill_ms REAL,
    control_avg_total_ms REAL,
    control_avg_padding REAL,

    -- 实验组统计
    treatment_sample_count INTEGER DEFAULT 0,
    treatment_avg_prefill_ms REAL,
    treatment_avg_total_ms REAL,
    treatment_avg_padding REAL,

    -- 实验结果
    status TEXT DEFAULT 'running',  -- running | completed | stopped
    winner TEXT,  -- control | treatment | tie
    p_value REAL,  -- 统计显著性
    confidence REAL,  -- 置信度
    conclusion TEXT  -- 实验结论
);
