"""
推理脚本 - inference.py (CLI 包装层)
仅保留命令行参数解析，实际业务逻辑已下沉到 integration/inference_api.py

迁移时间：2026-06-17
原始业务逻辑：inference.py (651行)
当前状态：CLI 薄包装层 (~50行)
"""

import os
import sys
import argparse
import json
import time

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 从 integration API 导入业务逻辑
from integration.inference_api import (
    run_inference,
    print_result,
    save_result,
)


def main():
    parser = argparse.ArgumentParser(
        description='需求预测与补货优化推理脚本 (Integration API 版本)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python inference.py --product-code 1010101160 --store-code 16 --weeks 4
  python inference.py --product-code 1010101160 --store-code 16 --weeks 4 --safety-days 3
  python inference.py --product-code 1010101160 --store-code 16 --weeks 4 --with-ppo
        """
    )

    # 全局参数
    parser.add_argument('--config', type=str, help='YAML 配置文件路径（推荐通过 cli.py 使用）')
    
    parser.add_argument('--product-code', type=str, required=True,
                        help='商品编码（例如：1010101160 或 SKU_0）')
    parser.add_argument('--store-code', type=str, required=True,
                        help='门店编码（例如：16 或 STORE_0）')
    parser.add_argument('--weeks', type=int, default=4,
                        help='预测周数（默认：4）')
    parser.add_argument('--current-inventory', type=float, default=0,
                        help='当前库存量（默认：0）')
    parser.add_argument('--safety-days', type=int, default=3,
                        help='安全库存天数（默认：3）')
    parser.add_argument('--lead-time-days', type=int, default=7,
                        help='补货提前期天数（默认：7）')
    parser.add_argument('--with-ppo', action='store_true',
                        help='同时运行 PPO 全局评估（耗时较长）')
    parser.add_argument('--output', type=str, default=None,
                        help='结果保存路径（可选）')

    args = parser.parse_args()
    
    # 加载 YAML 配置（如果提供）
    if args.config:
        print(f"\n📝 加载 YAML 配置文件: {args.config}")
        try:
            from config.loader import load_yaml_config, apply_yaml_to_namespace
            
            yaml_config = load_yaml_config(args.config)
            print(f"   ✅ YAML 配置加载成功")
            
            # 只应用 inference section（防止训练配置污染）
            applied = apply_yaml_to_namespace(args, yaml_config, filter_key='inference')
            
            # 如果 YAML 中有 common section，也应用（公共参数）
            if 'common' in yaml_config:
                applied += apply_yaml_to_namespace(args, yaml_config, filter_key='common')
            
            print(f"   ✅ 已应用 {applied} 个推理参数（仅 inference/common section）")
            
        except Exception as e:
            print(f"   ❌ YAML 配置加载失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 运行推理（调用 integration API）
    result = run_inference(
        product_code=args.product_code,
        store_code=args.store_code,
        weeks=args.weeks,
        current_inventory=args.current_inventory,
        safety_days=args.safety_days,
        lead_time_days=args.lead_time_days,
        with_ppo=args.with_ppo,
        output_path=args.output,
    )

    # 打印结果
    if 'error' not in result:
        print_result(
            product_code=args.product_code,
            store_code=args.store_code,
            weeks=args.weeks,
            forecast=result['forecast'],
            replenishment=result['replenishment'],
            ppo_result=result.get('ppo_evaluation'),
        )
    else:
        print(f"推理失败: {result['error']}")
        import traceback
        traceback.print_exc()

    return result


if __name__ == '__main__':
    try:
        result = main()
        sys.exit(0 if 'error' not in result else 1)
    except Exception as e:
        print(f"程序异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
