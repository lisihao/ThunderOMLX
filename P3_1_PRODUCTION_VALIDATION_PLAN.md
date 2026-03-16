# P3-1 生产验证计划（保守方案 A）

## 📋 验证目标

在生产环境中验证 Pydantic v2 配置系统（settings_v2）的稳定性和性能，为最终迁移做准备。

---

## 🕐 时间线

| 阶段 | 持续时间 | 里程碑 |
|------|----------|--------|
| **第 1 周** | 1-7 天 | 内部测试（开发/测试环境） |
| **第 2 周** | 8-14 天 | 生产灰度（少量用户/流量） |
| **评估期** | 第 15 天 | Go/No-Go 决策 |
| **清理期** | 第 16+ 天 | 执行阶段四清理（如果通过）|

---

## 🧪 第 1 周：内部测试

### 测试环境配置

```bash
# 开发环境
export OMLX_USE_SETTINGS_V2=true
export OMLX_LOG_LEVEL=debug

# 启动服务
omlx serve --model-dir ~/.omlx/models
```

### 测试清单

- [ ] **功能测试**
  - [ ] 服务启动成功
  - [ ] 所有 API 端点正常
  - [ ] Admin 界面加载正常
  - [ ] settings.json 加载/保存正常

- [ ] **性能测试**
  - [ ] 启动时间 < 5 秒
  - [ ] 内存占用 < v1 + 10%
  - [ ] API 响应时间无明显增加

- [ ] **兼容性测试**
  - [ ] 现有 settings.json 可正常加载
  - [ ] 环境变量覆盖正常（OMLX_*）
  - [ ] CLI 参数覆盖正常

- [ ] **压力测试**
  - [ ] 100+ 请求/秒无异常
  - [ ] 长时间运行（24 小时）无内存泄漏
  - [ ] 配置热重载正常

### 监控指标

```bash
# 性能基准
hyperfine \
  --warmup 3 \
  "OMLX_USE_SETTINGS_V2=false omlx --help" \
  "OMLX_USE_SETTINGS_V2=true omlx --help"

# 内存占用
ps aux | grep omlx

# 日志监控
tail -f ~/.omlx/logs/omlx.log | grep -i "error\|warning"
```

---

## 🚀 第 2 周：生产灰度

### 灰度策略

**方案 1: 按用户灰度**（推荐）
```bash
# 仅对特定用户启用 v2
if [[ "$USER" == "test_user_1" || "$USER" == "test_user_2" ]]; then
  export OMLX_USE_SETTINGS_V2=true
fi
```

**方案 2: 按流量灰度**
```bash
# 10% 流量使用 v2
if (( RANDOM % 100 < 10 )); then
  export OMLX_USE_SETTINGS_V2=true
fi
```

**方案 3: 按时间灰度**
```bash
# 每天 2 小时使用 v2
HOUR=$(date +%H)
if (( HOUR >= 14 && HOUR < 16 )); then
  export OMLX_USE_SETTINGS_V2=true
fi
```

### 监控指标（关键）

| 指标 | v1 基线 | v2 目标 | 实际值 | 状态 |
|------|---------|---------|--------|------|
| 启动时间 | ___ 秒 | < v1 × 3 | ___ | ⏳ |
| 内存占用 | ___ MB | < v1 + 10% | ___ | ⏳ |
| API 延迟 (p50) | ___ ms | < v1 + 5% | ___ | ⏳ |
| API 延迟 (p99) | ___ ms | < v1 + 10% | ___ | ⏳ |
| 错误率 | ___% | < v1 | ___ | ⏳ |
| CPU 占用 | ___% | < v1 + 5% | ___ | ⏳ |

### 日志收集

```bash
# 对比 v1 和 v2 的日志
grep "ERROR\|WARN" ~/.omlx/logs/omlx.log | \
  awk '{print $1, $2, $5}' | \
  sort | uniq -c
```

---

## 📊 第 15 天：Go/No-Go 决策

### 通过标准（全部满足才能 Go）

- [ ] **功能**: 所有测试用例通过，无新增 bug
- [ ] **性能**: 所有指标在目标范围内
- [ ] **稳定性**: 无 crash、无内存泄漏
- [ ] **兼容性**: 与 v1 行为完全一致
- [ ] **用户反馈**: 无重大问题报告

### Go 决策 → 执行阶段四清理

如果通过，执行以下清理步骤：

1. **重命名文件**
   ```bash
   cd /Users/lisihao/ThunderOMLX/src/omlx
   mv settings.py settings_v1_backup.py
   mv settings_v2.py settings.py
   ```

2. **移除特性开关**
   ```bash
   # 批量替换
   sed -i '' '/USE_SETTINGS_V2/d' cli.py admin/routes.py process_memory_enforcer.py
   sed -i '' '/if USE_SETTINGS_V2:/,/else:/d' cli.py
   # 手动清理剩余条件导入
   ```

3. **删除 v1 代码**（保留备份）
   ```bash
   git mv src/omlx/settings.py archive/settings_v1_backup.py
   ```

4. **更新文档**
   - 更新 README.md
   - 更新配置指南
   - 添加迁移日志

### No-Go 决策 → 回滚计划

如果不通过，执行回滚：

```bash
# 方法 1: 环境变量回滚（立即生效）
unset OMLX_USE_SETTINGS_V2

# 方法 2: 代码回滚（彻底移除）
git revert <commit-hash>
```

---

## 📝 验证报告模板

### 每周报告

```markdown
# P3-1 验证报告 - 第 X 周

**日期**: YYYY-MM-DD
**环境**: 开发/生产
**v2 启用比例**: X%

## 性能指标

| 指标 | v1 | v2 | 变化 | 状态 |
|------|----|----|------|------|
| 启动时间 | X.Xs | X.Xs | +X% | ✅/❌ |
| 内存占用 | XMB | XMB | +X% | ✅/❌ |
| API p50 | Xms | Xms | +X% | ✅/❌ |
| API p99 | Xms | Xms | +X% | ✅/❌ |

## 问题记录

- [ ] 问题 1: 描述...
- [ ] 问题 2: 描述...

## 结论

- [ ] 继续灰度
- [ ] 扩大范围
- [ ] 暂停回滚
```

---

## 🔧 故障排查

### 常见问题

**问题 1: v2 启动失败**
```bash
# 检查环境变量
echo $OMLX_USE_SETTINGS_V2

# 检查日志
tail -100 ~/.omlx/logs/omlx.log

# 回滚到 v1
unset OMLX_USE_SETTINGS_V2
```

**问题 2: 配置加载失败**
```bash
# 验证 settings.json 格式
python3 -m json.tool ~/.omlx/settings.json

# 测试 v1 兼容性
python3 -c "
from omlx.settings_compat import convert_v1_to_v2
import json
with open('~/.omlx/settings.json') as f:
    v1 = json.load(f)
v2 = convert_v1_to_v2(v1)
print('转换成功')
"
```

**问题 3: 性能回退**
```bash
# 性能分析
python3 -m cProfile -o profile.stats omlx serve

# 内存分析
python3 -m memory_profiler omlx/cli.py
```

---

## 📞 联系和支持

**问题上报**:
- 创建 GitHub Issue: `https://github.com/.../issues`
- 标签: `P3-1`, `settings-v2`, `validation`

**紧急回滚**:
```bash
unset OMLX_USE_SETTINGS_V2
systemctl restart omlx  # 或等效重启命令
```

---

**计划状态**: ✅ 准备就绪
**预计完成**: 2 周后
**负责人**: [待指定]
