"""
持续运行自适应缓存优化 - 真实大模型推理

功能:
1. 启动真实的 Scheduler 和大模型
2. 持续生成推理请求
3. 实时监控缓存命中率、padding overhead
4. 自动发现优化机会并应用
5. 可视化优化前后的性能变化

使用方式:
    python3 scripts/run_adaptive_optimization_live.py --model-path ~/models/xxx

可选参数:
    --simulate: 使用模拟模式（无需真实模型）
    --analysis-interval: 多少次请求后分析一次（默认 50）
    --total-requests: 总请求数（默认 500）
"""

import sys
from pathlib import Path
import argparse
import time
import random
from typing import Optional, Dict, List
from datetime import datetime

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omlx.adaptive_cache_optimizer import AdaptiveCacheOptimizer


# ============================================================================
# 模拟模式（无需真实模型）
# ============================================================================

class SimulatedScheduler:
    """模拟 Scheduler，无需真实模型"""

    def __init__(self, db_path: str, initial_block_size: int = 128):
        self.aco = AdaptiveCacheOptimizer(db_path)
        self.block_size = initial_block_size
        self.agent_id = "live-test-agent"
        self.request_count = 0

        print(f"✅ 模拟 Scheduler 初始化完成")
        print(f"   初始 block_size: {self.block_size}")

    def generate_request(self) -> Dict:
        """生成模拟推理请求"""
        # 模拟不同的 prompt 长度模式
        # 使用 448 tokens，这样 block_size=128 时 padding=64，block_size=64 时 padding=0
        system_prompt_length = 448
        user_query_length = 0

        total_prompt_length = system_prompt_length + user_query_length

        # 计算 padding
        remainder = total_prompt_length % self.block_size
        if remainder == 0:
            padding_tokens = 0
        else:
            padding_needed = self.block_size - remainder
            padding_tokens = padding_needed if padding_needed <= 64 else 0

        # 模拟推理时间
        prefill_time_ms = (total_prompt_length + padding_tokens) * 0.5
        output_tokens = random.randint(50, 100)
        decode_time_ms = output_tokens * 4.0

        # Cache hit ratio
        cache_hit_ratio = system_prompt_length / total_prompt_length

        # Skip logic type
        skip_logic_type = "APPROXIMATE" if cache_hit_ratio >= 0.90 else "NONE"

        # 记录到 ACO
        self.aco.log_inference(
            agent_id=self.agent_id,
            system_prompt_length=system_prompt_length,
            user_query_length=user_query_length,
            cache_hit_ratio=cache_hit_ratio,
            skip_logic_type=skip_logic_type,
            block_size=self.block_size,
            padding_tokens=padding_tokens,
            prefill_time_ms=prefill_time_ms,
            decode_time_ms=decode_time_ms,
        )

        self.request_count += 1

        return {
            'total_prompt_length': total_prompt_length,
            'padding_tokens': padding_tokens,
            'padding_overhead': (padding_tokens / total_prompt_length * 100) if total_prompt_length > 0 else 0,
            'prefill_time_ms': prefill_time_ms,
            'decode_time_ms': decode_time_ms,
            'total_time_ms': prefill_time_ms + decode_time_ms,
            'cache_hit_ratio': cache_hit_ratio,
        }

    def apply_optimization(self, new_block_size: int, reason: str):
        """应用优化"""
        old_block_size = self.block_size
        self.block_size = new_block_size

        # 记录配置变更
        self.aco.apply_optimization(
            agent_id=self.agent_id,
            new_block_size=new_block_size,
            old_block_size=old_block_size,
            reason=reason
        )

        print(f"\n🔧 应用优化: block_size {old_block_size} → {new_block_size}")
        print(f"   原因: {reason}")


# ============================================================================
# 真实模式（使用真实模型）
# ============================================================================

class RealScheduler:
    """真实 Scheduler，使用 MLX 模型"""

    def __init__(self, model_path: str, db_path: str):
        print(f"🔄 加载模型: {model_path}")

        try:
            import mlx.core as mx
            from mlx_lm import load

            # 加载模型
            self.model, self.tokenizer = load(model_path)

            # 初始化 Scheduler
            from omlx.scheduler import Scheduler, SchedulerConfig

            config = SchedulerConfig(
                enable_adaptive_cache_optimization=True,
                adaptive_cache_db_path=db_path,
                enable_auto_apply=False,  # 手动控制
                paged_cache_block_size=128,  # 初始 block_size
            )

            self.scheduler = Scheduler(
                model=self.model,
                tokenizer=self.tokenizer,
                config=config
            )

            self.agent_id = "live-test-agent"
            self.request_count = 0

            print(f"✅ 真实 Scheduler 初始化完成")
            print(f"   初始 block_size: {config.paged_cache_block_size}")

        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            raise

    def generate_request(self) -> Dict:
        """生成真实推理请求"""
        # 使用固定的 system prompt（模拟 agent）
        system_prompt = "You are a helpful AI assistant. " * 50  # ~448 tokens
        user_query = "Hello!"

        full_prompt = system_prompt + user_query

        # 记录开始时间
        start_time = time.perf_counter()

        # 执行推理
        try:
            # 简单推理（生成少量 tokens）
            response = self.scheduler.generate(
                prompt=full_prompt,
                max_tokens=50,
                temperature=0.7,
            )

            total_time_ms = (time.perf_counter() - start_time) * 1000

            # 获取统计信息
            stats = self.scheduler.get_stats()

            self.request_count += 1

            return {
                'response': response,
                'total_time_ms': total_time_ms,
                'stats': stats,
            }

        except Exception as e:
            print(f"❌ 推理失败: {e}")
            return None

    def apply_optimization(self, new_block_size: int, reason: str):
        """应用优化"""
        # 更新 Scheduler 的 block_size
        old_block_size = self.scheduler.config.paged_cache_block_size
        self.scheduler.config.paged_cache_block_size = new_block_size

        # 记录配置变更
        self.scheduler.aco.apply_optimization(
            agent_id=self.agent_id,
            new_block_size=new_block_size,
            old_block_size=old_block_size,
            reason=reason
        )

        print(f"\n🔧 应用优化: block_size {old_block_size} → {new_block_size}")
        print(f"   原因: {reason}")


# ============================================================================
# 主流程
# ============================================================================

def print_progress_bar(current: int, total: int, width: int = 50):
    """打印进度条"""
    filled = int(width * current / total)
    bar = '█' * filled + '░' * (width - filled)
    percent = 100 * current / total
    print(f"\r进度: [{bar}] {percent:.1f}% ({current}/{total})", end='', flush=True)


def print_metrics(metrics_list: List[Dict], phase_name: str):
    """打印性能指标"""
    if not metrics_list:
        return

    avg_padding = sum(m.get('padding_tokens', 0) for m in metrics_list) / len(metrics_list)
    avg_padding_overhead = sum(m.get('padding_overhead', 0) for m in metrics_list) / len(metrics_list)
    avg_prefill = sum(m.get('prefill_time_ms', 0) for m in metrics_list) / len(metrics_list)
    avg_total = sum(m.get('total_time_ms', 0) for m in metrics_list) / len(metrics_list)

    print(f"\n{'='*70}")
    print(f"📊 {phase_name} 性能指标 (最近 {len(metrics_list)} 次请求)")
    print(f"{'='*70}")
    print(f"   平均 padding: {avg_padding:.1f} tokens ({avg_padding_overhead:.1f}%)")
    print(f"   平均 prefill 时间: {avg_prefill:.1f}ms")
    print(f"   平均总时间: {avg_total:.1f}ms")


def run_live_optimization(
    scheduler,
    analysis_interval: int = 50,
    total_requests: int = 500,
):
    """
    运行持续优化

    Args:
        scheduler: Scheduler 实例（真实或模拟）
        analysis_interval: 每 N 次请求分析一次
        total_requests: 总请求数
    """
    print(f"\n{'='*70}")
    print(f"🚀 开始持续运行自适应优化")
    print(f"{'='*70}")
    print(f"   分析间隔: 每 {analysis_interval} 次请求")
    print(f"   总请求数: {total_requests}")
    print(f"   开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    metrics_buffer = []
    phase_metrics = {
        'before_optimization': [],
        'after_optimization': [],
    }

    optimization_applied = False
    optimization_point = None

    for i in range(1, total_requests + 1):
        # 生成请求
        result = scheduler.generate_request()

        if result:
            metrics_buffer.append(result)

            # 根据是否已优化，记录到不同的 phase
            if optimization_applied:
                phase_metrics['after_optimization'].append(result)
            else:
                phase_metrics['before_optimization'].append(result)

        # 打印进度条
        print_progress_bar(i, total_requests)

        # 每隔 analysis_interval 次请求，运行分析
        if i % analysis_interval == 0:
            print()  # 换行

            # 打印当前指标
            print_metrics(metrics_buffer[-analysis_interval:], f"最近 {analysis_interval} 次请求")

            # 运行分析（仅在第一次分析时应用优化）
            if not optimization_applied:
                print(f"\n🔍 运行自适应分析...")

                recommendation = scheduler.aco.analyze_patterns(
                    scheduler.agent_id,
                    min_samples=20
                )

                if recommendation:
                    print(f"\n✅ 发现优化机会:")
                    print(f"   当前 block_size: {recommendation['current_block_size']}")
                    print(f"   推荐 block_size: {recommendation['recommended_block_size']}")
                    print(f"   当前 padding: {recommendation['current_padding_overhead']:.1f}%")
                    print(f"   优化后 padding: {recommendation['recommended_padding_overhead']:.1f}%")
                    print(f"   改进幅度: {recommendation['improvement_pct']:.1f}%")

                    # 应用优化
                    scheduler.apply_optimization(
                        new_block_size=recommendation['recommended_block_size'],
                        reason=recommendation['reason']
                    )

                    optimization_applied = True
                    optimization_point = i

                    print(f"\n🎯 优化已应用！继续运行以观察效果...")
                else:
                    print(f"\n⚠️ 暂无优化建议（样本不足或改进幅度 <2%）")

            # 清空缓冲区
            metrics_buffer = []

        # 小延迟，避免过快
        time.sleep(0.01)

    print()  # 最后换行

    # 打印最终对比
    print(f"\n{'='*70}")
    print(f"🏁 运行完成")
    print(f"{'='*70}")

    if optimization_applied:
        print(f"\n优化应用时间点: 第 {optimization_point} 次请求")

        print_metrics(phase_metrics['before_optimization'], "优化前")
        print_metrics(phase_metrics['after_optimization'], "优化后")

        # 计算改进
        if phase_metrics['before_optimization'] and phase_metrics['after_optimization']:
            before = phase_metrics['before_optimization']
            after = phase_metrics['after_optimization']

            avg_padding_before = sum(m.get('padding_tokens', 0) for m in before) / len(before)
            avg_padding_after = sum(m.get('padding_tokens', 0) for m in after) / len(after)

            avg_total_before = sum(m.get('total_time_ms', 0) for m in before) / len(before)
            avg_total_after = sum(m.get('total_time_ms', 0) for m in after) / len(after)

            padding_improvement = ((avg_padding_before - avg_padding_after) / avg_padding_before * 100) if avg_padding_before > 0 else 0
            time_improvement = ((avg_total_before - avg_total_after) / avg_total_before * 100) if avg_total_before > 0 else 0

            print(f"\n{'='*70}")
            print(f"🎯 优化效果总结")
            print(f"{'='*70}")
            print(f"   Padding 减少: {padding_improvement:.1f}%")
            print(f"   总时间加速: {time_improvement:.1f}%")
    else:
        print(f"\n⚠️ 未应用优化（可能样本不足或已是最优配置）")

    # 打印配置历史
    print(f"\n{'='*70}")
    print(f"📜 配置变更历史")
    print(f"{'='*70}")

    import sqlite3
    db_path = scheduler.aco.db_path
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("""
            SELECT timestamp, agent_id, old_block_size, new_block_size, change_reason
            FROM config_history
            ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()

    if rows:
        for row in rows:
            print(f"\n   时间: {row[0]}")
            print(f"   Agent: {row[1]}")
            print(f"   变更: block_size {row[2]} → {row[3]}")
            print(f"   原因: {row[4]}")
    else:
        print("\n   无记录")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="持续运行自适应缓存优化 - 真实大模型推理"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="模型路径（如果使用真实模式）"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="使用模拟模式（无需真实模型）"
    )
    parser.add_argument(
        "--analysis-interval",
        type=int,
        default=50,
        help="多少次请求后分析一次（默认 50）"
    )
    parser.add_argument(
        "--total-requests",
        type=int,
        default=500,
        help="总请求数（默认 500）"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="~/.cache/thunderomlx/adaptive_cache_live.db",
        help="数据库路径"
    )

    args = parser.parse_args()

    # 扩展路径
    db_path = Path(args.db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 初始化 Scheduler
    if args.simulate:
        print("🔧 使用模拟模式")
        scheduler = SimulatedScheduler(str(db_path))
    else:
        if not args.model_path:
            print("❌ 真实模式需要 --model-path 参数")
            print("   或使用 --simulate 启用模拟模式")
            sys.exit(1)

        print("🔧 使用真实模式")
        scheduler = RealScheduler(args.model_path, str(db_path))

    # 运行持续优化
    try:
        run_live_optimization(
            scheduler=scheduler,
            analysis_interval=args.analysis_interval,
            total_requests=args.total_requests,
        )

        print(f"\n{'='*70}")
        print(f"✅ 运行完成！")
        print(f"{'='*70}")
        print(f"\n数据库: {db_path}")
        print(f"可使用以下命令查看详细报告:")
        print(f"  python3 scripts/analyze_and_optimize.py --db-path {db_path}")

    except KeyboardInterrupt:
        print(f"\n\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
