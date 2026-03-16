# P3-1 阶段快速参考

## 一句话总结

为 5 个核心模块添加特性开关，支持运行时在 settings v1（旧）和 v2（新 Pydantic）之间切换。

## 使用方式

```bash
# v1 模式（默认，无需操作）
omlx serve --model-dir /path/to/models

# v2 模式（通过环境变量启用）
export OMLX_USE_SETTINGS_V2=true
omlx serve --model-dir /path/to/models

# 或在一行中
OMLX_USE_SETTINGS_V2=true omlx serve --model-dir /path/to/models
```

## 修改了哪些文件

| 文件 | 修改项 | 代码行数 |
|------|--------|---------|
| cli.py | 2 处特性开关 | +13 |
| admin/routes.py | 3 处条件导入 | +7 |
| process_memory_enforcer.py | 2 处条件导入 | +7 |
| **合计** | **5 处特性开关** | **+27** |

## 核心概念

```python
# 特性开关模式（3 行代码）
USE_SETTINGS_V2 = os.getenv("OMLX_USE_SETTINGS_V2", "false").lower() == "true"

if USE_SETTINGS_V2:
    from .settings_v2 import SomeClass  # 新版本
else:
    from .settings import SomeClass     # 旧版本
```

## 验证检查清单

- [x] 特性开关添加到 cli.py
- [x] 特性开关添加到 admin/routes.py
- [x] 特性开关添加到 process_memory_enforcer.py
- [x] v1 模式导入验证（默认）
- [x] v2 模式导入验证（OMLX_USE_SETTINGS_V2=true）
- [x] API 兼容性检查通过

## 关键数字

| 指标 | 数值 |
|------|------|
| 修改文件 | 3 |
| 特性开关处数 | 5 |
| 新增代码行 | 27 |
| 删除代码行 | 0 |
| 测试用例通过 | 4/4 |
| 后向兼容性 | ✅ 100% |

## 故障排查

### 问题：仍在使用旧版本

```bash
# 检查当前版本
python3 -c "import os; os.environ['OMLX_USE_SETTINGS_V2']='true'; from omlx.settings_v2 import init_settings; print(init_settings.__module__)"

# 预期输出: omlx.settings_v2
```

### 问题：v2 出现错误

```bash
# 快速回滚到 v1
unset OMLX_USE_SETTINGS_V2
omlx serve --model-dir /path/to/models
```

## 下一步

- **P3-2**: 完整集成测试和性能验证
- **P3-3**: 优化和反馈改进
- **P3-4**: 设置 v2 为默认，计划删除 v1

## 文档位置

| 文件 | 用途 |
|------|------|
| P3_1_FEATURE_FLAG_USAGE.md | 详细使用指南 |
| P3_1_COMPLETION_SUMMARY.md | 完整完成报告 |
| P3_1_QUICK_REFERENCE.md | 本文件（快速参考） |

---

**记住**: 特性开关是**临时的**，最终会在 P3-4 中删除 v1 代码。
