#!/usr/bin/env python3
"""
Kalshi 爬虫完整示例
展示如何使用不同的认证方式收集 Kalshi 预测市场数据

使用你的私钥文件: examples/swm.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swm.utils.crawler import KalshiCrawler


def example_1_private_key():
    """
    示例 1: 使用私钥文件认证（最安全，推荐）
    
    使用你的私钥文件: examples/swm.txt
    """
    print("=" * 70)
    print("示例 1: 使用私钥文件认证")
    print("=" * 70)
    
    # 私钥文件路径（相对于当前脚本）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    private_key_path = os.path.join(script_dir, 'swm.txt')
    
    # 也可以从环境变量获取
    if os.getenv('KALSHI_PRIVATE_KEY_PATH'):
        private_key_path = os.getenv('KALSHI_PRIVATE_KEY_PATH')
    
    if not os.path.exists(private_key_path):
        print(f"❌ 私钥文件不存在: {private_key_path}")
        print("请确保 examples/swm.txt 文件存在")
        return None
    
    print(f"✅ 使用私钥: {private_key_path}")
    
    crawler = KalshiCrawler(
        output_file='data/kalshi_markets_basic.jsonl',
        private_key_path=private_key_path,  # 🔐 使用私钥文件
        cache_size=50,
    )
    
    print("\n开始收集市场数据（自动包含时间序列）...\n")
    
    # 收集数据
    crawler.collect_markets(
        max_markets=100,
        status='open',
        include_history=True,
        history_limit=100,
    )
    
    print("\n✅ 使用私钥认证收集完成!")
    print("📈 每个市场都包含时间序列数据\n")
    return crawler


def example_2_api_key():
    """
    示例 2: 使用 API Key 认证
    """
    print("=" * 70)
    print("示例 2: 使用 API Key 认证")
    print("=" * 70)
    
    api_key = os.getenv('KALSHI_API_KEY')
    
    if not api_key:
        print("❌ 未找到 API key")
        print("请设置环境变量: export KALSHI_API_KEY='your-key'")
        return None
    
    print(f"✅ 使用 API key: {api_key[:10]}...")
    
    crawler = KalshiCrawler(
        output_file='data/kalshi_markets_basic.jsonl',
        api_key=api_key,  # 🔑 使用 API key
        cache_size=50,
    )
    
    print("\n开始收集市场数据...\n")
    
    crawler.collect_markets(
        max_markets=20000,
        status='open',
        include_history=True,
    )
    
    print("\n✅ 使用 API key 认证收集完成!\n")
    return crawler


def example_4_analyze_data():
    """
    示例 3: 分析收集到的时间序列数据
    """
    print("=" * 70)
    print("示例 3: 分析时间序列数据")
    print("=" * 70)
    
    import json
    from datetime import datetime
    
    data_file = 'data/kalshi_markets_basic.jsonl'
    
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        print("请先运行示例 1 或 2 收集数据\n")
        return
    
    print(f"📊 分析文件: {data_file}\n")
    
    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:  # 只显示前3个市场
                break
            
            market = json.loads(line)
            
            print(f"🔹 市场 #{i+1}: {market.get('market_id', 'N/A')}")
            print(f"   问题: {market.get('question', 'N/A')[:60]}...")
            print(f"   结果: {market.get('outcome', 'pending')}")
            
            # 检查时间序列格式（PolyMarket 兼容）
            daily_ts = market.get('daily_time_series', {})
            
            if daily_ts and 'Yes' in daily_ts:
                yes_prices = daily_ts['Yes']
                print(f"   ✅ daily_time_series: {len(yes_prices)} 个 Yes 数据点")
                if yes_prices:
                    for point in yes_prices[:2]:
                        dt = datetime.fromtimestamp(point['t'])
                        print(f"      {dt.strftime('%Y-%m-%d %H:%M')}: ${point['p']:.3f}")
            
            print()
    
    print("✅ 数据分析完成!\n")


def display_usage():
    """显示使用说明"""
    print("=" * 70)
    print("Kalshi 爬虫示例")
    print("=" * 70)
    print("\n🔑 认证方式（按优先级）:")
    print("  1. 私钥文件 (推荐): examples/swm.txt")
    print("     - 最安全的方式")
    print("     - 需要 kalshi-python SDK")
    print()
    print("  2. API Key:")
    print("     export KALSHI_API_KEY='your-key'")
    print()
    
    print("\n📚 示例列表:")
    print("  1. 使用私钥文件（推荐）")
    print("  2. 使用 API Key")
    print("  3. 分析时间序列数据")
    
    print("\n🔥 特性:")
    print("  ✅ 自动包含时间序列数据")
    print("  ✅ 使用官方 SDK 获取 Series 数据")
    print("  ✅ PolyMarket 兼容输出格式")
    
    print("\n💡 快速开始:")
    print("  cd /data/haofeiy2/social-world-model")
    print("  python3 examples/kalshi_crawler_example.py")
    print()


if __name__ == "__main__":
    # 创建数据目录
    os.makedirs('data', exist_ok=True)
    
    display_usage()
    
    print("\n" + "=" * 70)
    print("开始运行示例...")
    print("=" * 70 + "\n")
    
    # 按优先级尝试不同的认证方式
    crawler = None
    
    # 示例 1: 私钥（如果存在）
    crawler = example_1_private_key()
    
    # 如果私钥示例失败，尝试 API key
    if not crawler:
        crawler = example_2_api_key()
    
    # 分析收集到的数据
    if crawler:
        print("\n")
        example_4_analyze_data()
    
    print("\n" + "=" * 70)
    print("✅ 示例完成!")
    print("=" * 70)
    print("\n📂 数据已保存到: data/kalshi_markets_basic.jsonl")
    print("\n查看数据:")
    print("  # 查看第一个市场")
    print("  head -1 data/kalshi_markets_basic.jsonl | python3 -m json.tool | head -50")
    print()
    print("  # 查看时间序列")
    print("  python3 << 'EOF'")
    print("import json")
    print("with open('data/kalshi_markets_basic.jsonl') as f:")
    print("    market = json.loads(f.readline())")
    print("    ts = market.get('daily_time_series', {})")
    print("    print('Market:', market.get('market_id'))")
    print("    print('Time series keys:', list(ts.keys()))")
    print("    print('Yes prices:', len(ts.get('Yes', [])), 'points')")
    print("EOF")
    print()
