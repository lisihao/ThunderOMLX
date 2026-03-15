# 自适应缓存优化模块设计

**日期**: 2026-03-14
**目标**: 智能分析、自动优化、持续进化的缓存配置系统

---

## 🎯 核心理念

```
硬编码配置（当前）          自适应优化（目标）
      ↓                           ↓
  一次性分析              →    持续数据收集
  静态配置                →    动态策略优化
  手动调整                →    自动进化
  基于假设                →    基于真实数据
```

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│              Adaptive Cache Optimizer (ACO)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [1] 数据收集层 (Data Collection)                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  每次推理后记录：                                     │  │
│  │  • agent_id, prompt_length, cache_hit_ratio          │  │
│  │  • skip_logic_type, inference_time, padding_overhead │  │
│  │  • block_size, timestamp                             │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  [2] 历史数据库 (SQLite)                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  • agent_metrics (每次推理记录)                       │  │
│  │  • cache_performance (聚合统计)                       │  │
│  │  • config_history (配置变更历史)                      │  │
│  │  • optimization_experiments (A/B 测试结果)            │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  [3] 模式分析层 (Pattern Analysis)                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  周期性分析（每天/每周）：                            │  │
│  │  • Prompt 长度分布（P50, P90, P99）                   │  │
│  │  • Cache hit ratio 趋势                               │  │
│  │  • Padding overhead 分析                              │  │
│  │  • 异常检测（性能回退）                               │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  [4] 策略优化层 (Strategy Optimization)                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  自动生成优化方案：                                   │  │
│  │  • 推荐最优 block_size                                │  │
│  │  • 调整 padding 策略                                  │  │
│  │  • 生成 A/B 测试计划                                  │  │
│  │  • 预测性能收益                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  [5] 自适应执行层 (Adaptive Execution)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  动态配置应用：                                       │  │
│  │  • 渐进式切换（5% → 50% → 100%）                     │  │
│  │  • 性能监控（实时对比）                               │  │
│  │  • 自动回滚（性能下降时）                             │  │
│  │  • 配置版本管理                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 数据模型设计

### 表 1: agent_metrics (每次推理记录)

```sql
CREATE TABLE agent_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- Prompt 信息
    system_prompt_length INTEGER,  -- System prompt 固定部分
    user_query_length INTEGER,      -- User query 动态部分
    total_prompt_length INTEGER,    -- 总长度

    -- 缓存性能
    cache_hit_ratio REAL,           -- 0.0 - 1.0
    skip_logic_type TEXT,           -- 'FULL', 'APPROXIMATE', 'NONE'

    -- 配置信息
    block_size INTEGER,
    padding_tokens INTEGER,
    padding_overhead REAL,          -- padding / total_prompt * 100

    -- 性能指标
    prefill_time_ms REAL,
    decode_time_ms REAL,
    total_time_ms REAL,

    -- 配置版本
    config_version TEXT,

    INDEX idx_agent_timestamp (agent_id, timestamp),
    INDEX idx_timestamp (timestamp)
);
```

### 表 2: cache_performance (聚合统计)

```sql
CREATE TABLE cache_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    date DATE NOT NULL,

    -- 统计指标
    request_count INTEGER,
    avg_cache_hit_ratio REAL,
    full_skip_rate REAL,           -- FULL SKIP 触发率
    avg_padding_overhead REAL,

    -- Prompt 长度分布
    prompt_length_p50 INTEGER,
    prompt_length_p90 INTEGER,
    prompt_length_p99 INTEGER,

    -- 性能指标
    avg_total_time_ms REAL,

    -- 当前配置
    current_block_size INTEGER,

    UNIQUE(agent_id, date)
);
```

### 表 3: config_history (配置变更历史)

```sql
CREATE TABLE config_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- 配置变更
    old_block_size INTEGER,
    new_block_size INTEGER,
    old_max_padding INTEGER,
    new_max_padding INTEGER,

    -- 变更原因
    change_reason TEXT,             -- 'manual', 'auto_optimization', 'rollback'
    optimization_score REAL,        -- 预期收益分数

    -- 实际效果（后续更新）
    actual_performance_delta REAL,  -- 实际性能变化
    is_rolled_back BOOLEAN DEFAULT 0
);
```

### 表 4: optimization_experiments (A/B 测试)

```sql
CREATE TABLE optimization_experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_time DATETIME,

    -- 实验配置
    control_block_size INTEGER,     -- 对照组（当前配置）
    treatment_block_size INTEGER,   -- 实验组（新配置）

    -- 流量分配
    treatment_ratio REAL,           -- 实验组流量占比 (0.05 - 1.0)

    -- 实验结果
    control_avg_hit_ratio REAL,
    treatment_avg_hit_ratio REAL,
    control_avg_padding REAL,
    treatment_avg_padding REAL,

    -- 决策
    status TEXT,                    -- 'running', 'success', 'rollback', 'cancelled'
    winner TEXT,                    -- 'control', 'treatment'
    confidence REAL                 -- 统计置信度
);
```

---

## 🔄 核心流程

### 流程 1: 数据收集（每次推理后）

```python
# src/omlx/adaptive_cache_optimizer.py

class AdaptiveCacheOptimizer:
    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)
        self.current_config = self._load_current_config()

    def log_inference(
        self,
        agent_id: str,
        system_prompt_length: int,
        user_query_length: int,
        cache_hit_ratio: float,
        skip_logic_type: str,
        block_size: int,
        padding_tokens: int,
        prefill_time_ms: float,
        decode_time_ms: float,
    ):
        """记录每次推理的数据"""
        total_prompt_length = system_prompt_length + user_query_length
        padding_overhead = padding_tokens / total_prompt_length * 100

        self.db.execute("""
            INSERT INTO agent_metrics (
                agent_id, system_prompt_length, user_query_length,
                total_prompt_length, cache_hit_ratio, skip_logic_type,
                block_size, padding_tokens, padding_overhead,
                prefill_time_ms, decode_time_ms, total_time_ms,
                config_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            agent_id, system_prompt_length, user_query_length,
            total_prompt_length, cache_hit_ratio, skip_logic_type,
            block_size, padding_tokens, padding_overhead,
            prefill_time_ms, decode_time_ms,
            prefill_time_ms + decode_time_ms,
            self.current_config.get("version", "1.0.0")
        ))
        self.db.commit()
```

### 流程 2: 模式分析（周期性，如每天）

```python
def analyze_patterns(self, agent_id: str, days: int = 7):
    """分析最近 N 天的使用模式"""

    # 查询历史数据
    data = pd.read_sql_query(f"""
        SELECT *
        FROM agent_metrics
        WHERE agent_id = ?
          AND timestamp >= datetime('now', '-{days} days')
    """, self.db, params=(agent_id,))

    if len(data) < 100:  # 数据太少，不足以分析
        return None

    # 分析 Prompt 长度分布
    prompt_lengths = data['total_prompt_length']
    p50 = prompt_lengths.quantile(0.50)
    p90 = prompt_lengths.quantile(0.90)
    p99 = prompt_lengths.quantile(0.99)

    # 分析 Cache 性能
    avg_hit_ratio = data['cache_hit_ratio'].mean()
    full_skip_rate = (data['skip_logic_type'] == 'FULL').mean()

    # 分析 Padding 开销
    avg_padding = data['padding_overhead'].mean()

    # 当前配置
    current_block_size = data['block_size'].mode()[0]

    return {
        "agent_id": agent_id,
        "request_count": len(data),
        "prompt_length_p50": int(p50),
        "prompt_length_p90": int(p90),
        "prompt_length_p99": int(p99),
        "avg_cache_hit_ratio": avg_hit_ratio,
        "full_skip_rate": full_skip_rate,
        "avg_padding_overhead": avg_padding,
        "current_block_size": current_block_size,
    }
```

### 流程 3: 策略优化（基于模式分析）

```python
def optimize_strategy(self, pattern: dict) -> dict:
    """基于使用模式生成优化策略"""

    agent_id = pattern["agent_id"]
    p90_length = pattern["prompt_length_p90"]
    current_block_size = pattern["current_block_size"]
    avg_padding = pattern["avg_padding_overhead"]

    # 候选 block_size
    candidates = [16, 32, 64, 128, 256]

    # 评估每个候选配置
    best_config = None
    best_score = -1

    for block_size in candidates:
        # 模拟这个 block_size 下的表现
        simulated_padding = self._simulate_padding(
            p90_length, block_size
        )

        # 计算优化分数（综合考虑 padding 和快照数量）
        score = self._calculate_score(
            block_size, simulated_padding, p90_length
        )

        if score > best_score:
            best_score = score
            best_config = {
                "block_size": block_size,
                "max_padding": block_size,
                "expected_padding_overhead": simulated_padding,
            }

    # 判断是否需要变更配置
    if best_config["block_size"] != current_block_size:
        improvement = (avg_padding - best_config["expected_padding_overhead"]) / avg_padding * 100

        if improvement > 10:  # 至少 10% 改进才值得切换
            return {
                "agent_id": agent_id,
                "action": "change_config",
                "old_block_size": current_block_size,
                "new_block_size": best_config["block_size"],
                "expected_improvement": improvement,
                "confidence": best_score,
            }

    return {
        "agent_id": agent_id,
        "action": "keep_current",
        "reason": "current config is optimal"
    }
```

### 流程 4: A/B 测试（渐进式切换）

```python
def start_experiment(self, agent_id: str, new_block_size: int):
    """启动 A/B 测试"""

    current_block_size = self._get_current_block_size(agent_id)

    experiment_id = self.db.execute("""
        INSERT INTO optimization_experiments (
            agent_id,
            control_block_size,
            treatment_block_size,
            treatment_ratio,
            status
        ) VALUES (?, ?, ?, ?, ?)
    """, (agent_id, current_block_size, new_block_size, 0.05, 'running')).lastrowid

    self.db.commit()

    print(f"🧪 Experiment {experiment_id} started:")
    print(f"   Control: block_size={current_block_size}")
    print(f"   Treatment: block_size={new_block_size}")
    print(f"   Initial traffic: 5%")

    return experiment_id


def evaluate_experiment(self, experiment_id: int) -> str:
    """评估 A/B 测试结果"""

    # 查询实验配置
    exp = self.db.execute("""
        SELECT * FROM optimization_experiments WHERE id = ?
    """, (experiment_id,)).fetchone()

    # 查询实验组和对照组的实际表现
    control_metrics = self._get_metrics_by_block_size(
        exp['agent_id'], exp['control_block_size'], exp['start_time']
    )
    treatment_metrics = self._get_metrics_by_block_size(
        exp['agent_id'], exp['treatment_block_size'], exp['start_time']
    )

    # 统计检验（t-test）
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(
        control_metrics['padding_overhead'],
        treatment_metrics['padding_overhead']
    )

    # 判断 winner
    if p_value < 0.05:  # 统计显著
        if treatment_metrics['padding_overhead'].mean() < control_metrics['padding_overhead'].mean():
            winner = 'treatment'
            decision = 'rollout'  # 全量发布
        else:
            winner = 'control'
            decision = 'rollback'  # 回滚
    else:
        winner = 'control'
        decision = 'keep_current'  # 无显著差异，保持现状

    # 更新实验结果
    self.db.execute("""
        UPDATE optimization_experiments
        SET end_time = CURRENT_TIMESTAMP,
            control_avg_padding = ?,
            treatment_avg_padding = ?,
            status = ?,
            winner = ?,
            confidence = ?
        WHERE id = ?
    """, (
        control_metrics['padding_overhead'].mean(),
        treatment_metrics['padding_overhead'].mean(),
        decision,
        winner,
        1 - p_value,
        experiment_id
    ))
    self.db.commit()

    return decision
```

### 流程 5: 自动配置应用

```python
def apply_optimal_config(self, agent_id: str):
    """应用最优配置（自动或手动触发）"""

    # 分析最近 7 天的模式
    pattern = self.analyze_patterns(agent_id, days=7)

    if not pattern:
        print(f"⚠️  {agent_id}: 数据不足，无法优化")
        return

    # 生成优化策略
    strategy = self.optimize_strategy(pattern)

    if strategy["action"] == "keep_current":
        print(f"✅ {agent_id}: 当前配置已是最优")
        return

    # 启动 A/B 测试
    exp_id = self.start_experiment(
        agent_id,
        strategy["new_block_size"]
    )

    # 等待实验运行（实际中会是异步的，这里简化）
    # ... 一段时间后 ...

    # 评估实验结果
    decision = self.evaluate_experiment(exp_id)

    if decision == 'rollout':
        # 全量发布新配置
        self._update_config(agent_id, strategy["new_block_size"])
        print(f"🚀 {agent_id}: 新配置已全量发布 (block_size={strategy['new_block_size']})")
    elif decision == 'rollback':
        print(f"🔄 {agent_id}: 新配置表现不佳，已回滚")
    else:
        print(f"➡️  {agent_id}: 保持当前配置")
```

---

## 🎯 集成到 ThunderOMLX

### 修改 scheduler.py

```python
# src/omlx/scheduler.py

class Scheduler:
    def __init__(self, ...):
        # ... 现有初始化 ...

        # 初始化自适应优化器
        if config.enable_adaptive_optimization:
            from adaptive_cache_optimizer import AdaptiveCacheOptimizer
            self.aco = AdaptiveCacheOptimizer(
                db_path=config.adaptive_cache_db_path
            )
        else:
            self.aco = None

    def add_request(self, request: Request):
        # ... 现有逻辑 ...

        # 记录推理开始时间
        request._aco_start_time = time.perf_counter()
        request._aco_system_prompt_length = len(request.prompt_token_ids)  # 简化

    def _complete_request(self, request: Request):
        # ... 现有逻辑 ...

        # 记录推理数据到 ACO
        if self.aco and hasattr(request, '_aco_start_time'):
            inference_time = (time.perf_counter() - request._aco_start_time) * 1000

            self.aco.log_inference(
                agent_id=request.agent_id,
                system_prompt_length=request._aco_system_prompt_length,
                user_query_length=request.num_prompt_tokens - request._aco_system_prompt_length,
                cache_hit_ratio=request._aco_cache_hit_ratio,
                skip_logic_type=request._aco_skip_logic_type,
                block_size=self.config.paged_cache_block_size,
                padding_tokens=request._aco_padding_tokens,
                prefill_time_ms=request._aco_prefill_time,
                decode_time_ms=inference_time - request._aco_prefill_time,
            )
```

---

## 📋 实施计划

### Phase 1: 数据收集基础设施（1-2 天）

- [ ] 创建数据库 schema
- [ ] 实现 `AdaptiveCacheOptimizer` 基础类
- [ ] 集成到 scheduler（数据收集）
- [ ] 测试数据收集是否正常

### Phase 2: 模式分析引擎（2-3 天）

- [ ] 实现 `analyze_patterns()` 方法
- [ ] 实现周期性分析任务（cron job 或后台线程）
- [ ] 生成分析报告（JSON 或 Markdown）
- [ ] 可视化工具（可选）

### Phase 3: 策略优化引擎（2-3 天）

- [ ] 实现 `optimize_strategy()` 方法
- [ ] 实现配置评分算法
- [ ] 实现 A/B 测试框架
- [ ] 实现自动回滚机制

### Phase 4: 自适应执行（1-2 天）

- [ ] 动态配置加载
- [ ] 渐进式切换（5% → 50% → 100%）
- [ ] 性能监控和告警
- [ ] 配置版本管理

### Phase 5: 测试和优化（2-3 天）

- [ ] 模拟测试（生成测试数据）
- [ ] 真实环境测试
- [ ] 性能基准测试
- [ ] 文档编写

**总计**: 8-13 天

---

## 🎯 预期效果

### 短期（1-2 周）
- ✅ 自动收集所有 Agent 的使用数据
- ✅ 生成每日/每周分析报告
- ✅ 识别配置优化机会

### 中期（1-2 月）
- ✅ 自动优化 2-3 个 Agent 的配置
- ✅ A/B 测试验证优化效果
- ✅ 平均 padding 开销减少 30-50%

### 长期（3-6 月）
- ✅ 所有 Agent 达到最优配置
- ✅ 系统自动适应使用模式变化
- ✅ Zero-configuration（新 Agent 自动优化）

---

## 📊 对比：硬编码 vs 自适应

| 特性 | 硬编码方案 | 自适应方案 |
|------|-----------|-----------|
| **配置方式** | 一次性手动 | 持续自动优化 |
| **数据来源** | 静态 SOUL.md | 实际使用数据 |
| **适应性** | 无 | 自动适应变化 |
| **优化周期** | 手动重新分析 | 自动（每天/每周） |
| **风险控制** | 无 | A/B 测试 + 自动回滚 |
| **可扩展性** | 每个 Agent 手动配置 | 新 Agent 自动优化 |
| **长期维护** | 高（手动） | 低（自动） |

---

**签署**: Solar (战略家 + 治理官双签)
**日期**: 2026-03-14
**下一步**: 等待监护人确认，开始实施 Phase 1
