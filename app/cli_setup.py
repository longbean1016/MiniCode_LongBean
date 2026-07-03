"""命令行交互式配置向导 — 设置 API Key、Base URL 和默认模型。

   对标 Hermes-agent 的终端交互式配置流程。
   通过 python -m app.main --setup 启动，与 TUI 模式分离。
"""

from app.infra.user_config import ensure_user_config, save_user_config


def run_cli_setup() -> None:
    """命令行配置向导：三步输入 API Key → Base URL → 模型选择。"""
    print()
    print("  MiniCode 首次配置")
    print("  ────────────────")
    print()

    config = ensure_user_config()

    # ── Step 1: API Key ──
    print("  Step 1/3: API Key")
    print("  输入你的 API Key，用于调用大模型服务。")
    api_key = ""
    while not api_key:
        try:
            val = input("  API Key: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  已取消。")
            return
        if val:
            api_key = val
        else:
            print("  API Key 不能为空，请重新输入。")

    # ── Step 2: Base URL ──
    print()
    print("  Step 2/3: Base URL")
    print("  输入模型服务的 API 地址。")
    print("  常用地址：")
    print("    DeepSeek: https://api.deepseek.com")
    print("    OpenAI:   https://api.openai.com/v1")
    default_url = config.base_url or "https://api.deepseek.com"
    try:
        val = input(f"  Base URL [{default_url}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n  已取消。")
        return
    base_url = val if val else default_url

    # ── Step 3: 模型选择 ──
    print()
    print("  Step 3/3: 选择默认模型")
    print()

    from app.infra.model_capabilities import detect_provider, get_models_for_provider

    provider = detect_provider(base_url)
    models = get_models_for_provider(provider)

    if not models:
        # 自定义 URL：让用户手动输入模型名
        print(f"  未识别该 URL 的厂商，请手动输入模型名称。")
        try:
            val = input("  模型名称: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  已取消。")
            return
        model = val if val else "deepseek-v4-flash"
    else:
        # 列出可选模型，让用户选择编号
        print(f"  厂商: {provider.upper()}  |  可选模型:")
        for i, m in enumerate(models, 1):
            marker = "  ●" if m == config.model else "   "
            print(f"    {i}. {marker} {m}")

        print()
        try:
            choice = input(f"  选择 [1-{len(models)}] (默认 1): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  已取消。")
            return

        try:
            idx = int(choice) - 1 if choice else 0
            if 0 <= idx < len(models):
                model = models[idx]
            else:
                model = models[0]
        except ValueError:
            model = models[0]

    # ── 保存配置：models 只包含当前厂商的模型，不混入其他厂商 ──
    config.api_key = api_key
    config.base_url = base_url
    config.model = model
    config.models = models  # models 已在上面按 provider 过滤
    save_user_config(config)

    print()
    print(f"  配置已保存到 ~/.bean/settings.json")
    print(f"    API Key:  {api_key[:12]}...")
    print(f"    Base URL: {base_url}")
    print(f"    Model:    {model}")
    print()
    print("  现在可以启动 MiniCode：")
    print("    python -m app.main")
    print()
