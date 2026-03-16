# 禁用 lz4 压缩测试

## 问题发现

lz4 压缩开销巨大且几乎无效：
- 每 block (~38MB) 压缩耗时 ~560ms
- 压缩率: 99.5% (只减少 0.5%)
- pp8192 总压缩时间: ~18 秒

## 测试方案

禁用 lz4 压缩，重新运行 benchmark，对比性能。

## 执行步骤

1. 检查如何禁用压缩（配置文件或环境变量）
2. 重启 server with compression disabled  
3. 重新运行 pp8192 benchmark
4. 对比结果

## 预期结果

如果压缩是瓶颈：
- pp8192 tg TPS: 从 66.2 提升到 ~75-80 tok/s
- 更接近 Native MLX 的 80.7 tok/s
