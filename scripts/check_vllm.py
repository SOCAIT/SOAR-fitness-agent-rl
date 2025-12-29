
try:
    import vllm
    print("vLLM is installed")
    print(f"Version: {vllm.__version__}")
except ImportError:
    print("vLLM is NOT installed")

