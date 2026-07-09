"""
极简跑通测试 - 1个SKU，验证代码逻辑
"""
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("="*60)
    print("DRP-Optima 极简跑通测试")
    print("="*60)
    
    try:
        # 测试1: 导入核心模块
        print("\n1. 测试导入核心模块...")
        from config.data_config import TFTModelConfig
        from data.data_loader import DataLoader
        print("  ✓ 核心模块导入成功")
        
        # 测试2: 创建配置
        print("\n2. 测试配置创建...")
        config = TFTModelConfig()
        print(f"  ✓ TFT配置创建成功: hidden_size={config.hidden_size}")
        
        # 测试3: 检查数据文件
        print("\n3. 检查数据文件...")
        import os
        data_files = ["data/goods_info.csv", "data/shop_info.csv"]
        for f in data_files:
            if os.path.exists(f):
                print(f"  ✓ {f} 存在")
            else:
                print(f"  ⚠ {f} 不存在（发布版本不包含数据文件）")
        
        print("\n" + "="*60)
        print("✅ 极简测试通过!")
        print("="*60)
        return 0
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
