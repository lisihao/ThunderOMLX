"""Phase 0 环境测试"""
import os
import sys

# 设置环境变量（避免 OpenMP 冲突）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加 src 到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_omlx_import():
    """测试 omlx 模块导入"""
    import omlx
    assert omlx.__version__ is not None
    print(f"✅ omlx version: {omlx.__version__}")

def test_fastapi_app():
    """测试 FastAPI 应用"""
    from omlx import server
    assert server.app is not None
    print(f"✅ FastAPI app: {type(server.app).__name__}")

def test_dependencies():
    """测试核心依赖"""
    import fastapi
    import uvicorn
    import httpx
    print(f"✅ fastapi: {fastapi.__version__}")
    print(f"✅ uvicorn: {uvicorn.__version__}")
    print(f"✅ httpx: {httpx.__version__}")

def test_thunderllama_binary():
    """测试 ThunderLLAMA 二进制"""
    import os.path
    llama_server = os.path.expanduser("~/ThunderLLAMA/build/bin/llama-server")
    assert os.path.exists(llama_server), "llama-server not found"
    print(f"✅ llama-server: {llama_server}")

if __name__ == "__main__":
    print("🧪 Phase 0 环境测试")
    print("=" * 50)

    try:
        test_omlx_import()
        test_fastapi_app()
        test_dependencies()
        test_thunderllama_binary()

        print("=" * 50)
        print("✅ 所有测试通过！环境就绪。")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
