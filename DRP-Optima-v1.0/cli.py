"""
DRP-Optima 统一命令行入口
整合训练、推理、评估等功能到单一入口点
"""

import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 版本信息
VERSION = "1.0.0"
AUTHOR = "欧宁益（浙江大学工程管理2021级硕士）"


def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog="drp-optima",
        description="DRP-Optima: 医药供应链需求预测与补货优化系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练 TFT 模型
  python cli.py train --model tft --config config/train.yaml
  
  # 训练 PPO 模型
  python cli.py train --model ppo --total-steps 50000
  
  # 推理（使用训练好的模型）
  python cli.py infer --model-path models/tft_model_best.ckpt
  
  # 评估模型
  python cli.py evaluate --model-path models/tft_model_best.ckpt
  
  # 运行完整流程（数据准备 → 训练 → 推理）
  python cli.py pipeline --config config/train.yaml
  
  # 生成代码知识图谱
  python cli.py graph --output docs/code_knowledge_graph/
        """,
    )
    
    # 全局参数
    parser.add_argument("--version", action="version", version=f"DRP-Optima {VERSION}")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # ===== train 命令 =====
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--model", type=str, choices=["tft", "ppo", "integrated"], default="tft", help="模型类型")
    train_parser.add_argument("--config", type=str, help="训练配置文件路径（覆盖全局配置）")
    train_parser.add_argument("--total-steps", type=int, help="总训练步数（PPO）")
    train_parser.add_argument("--num-skus", type=int, help="SKU 数量限制")
    train_parser.add_argument("--num-stores", type=int, help="门店数量限制")
    train_parser.add_argument("--no-tensorboard", action="store_true", help="禁用 TensorBoard")
    
    # ===== infer 命令 =====
    infer_parser = subparsers.add_parser("infer", help="模型推理")
    infer_parser.add_argument("--model-path", type=str, required=True, help="模型文件路径")
    infer_parser.add_argument("--config", type=str, help="推理配置文件路径（覆盖全局配置）")
    infer_parser.add_argument("--output", type=str, help="输出目录")
    infer_parser.add_argument("--forecast-horizon", type=int, default=7, help="预测 horizon（天）")
    
    # ===== evaluate 命令 =====
    eval_parser = subparsers.add_parser("evaluate", help="评估模型")
    eval_parser.add_argument("--model-path", type=str, required=True, help="模型文件路径")
    eval_parser.add_argument("--config", type=str, help="评估配置文件路径（覆盖全局配置）")
    eval_parser.add_argument("--metrics", type=str, nargs="+", help="评估指标列表")
    
    # ===== pipeline 命令 =====
    pipeline_parser = subparsers.add_parser("pipeline", help="运行完整流程")
    pipeline_parser.add_argument("--config", type=str, help="流程配置文件路径（覆盖全局配置）")
    pipeline_parser.add_argument("--skip-train", action="store_true", help="跳过训练步骤")
    pipeline_parser.add_argument("--skip-infer", action="store_true", help="跳过推理步骤")
    
    # ===== graph 命令 =====
    graph_parser = subparsers.add_parser("graph", help="生成代码知识图谱")
    graph_parser.add_argument("--output", type=str, default="docs/code_knowledge_graph/", help="输出目录")
    graph_parser.add_argument("--open", action="store_true", help="生成后自动在浏览器中打开")
    
    # ===== config 命令 =====
    config_parser = subparsers.add_parser("config", help="配置管理")
    config_parser.add_argument("action", type=str, choices=["show", "generate"], help="操作：show（显示当前配置）/ generate（生成示例配置）")
    config_parser.add_argument("--output", type=str, help="生成配置文件的输出路径（generate 时）")
    
    return parser


def cmd_train(args):
    """训练模型命令"""
    print("🚀 开始训练模型...")
    print(f"   模型类型: {args.model}")
    
    # 加载配置文件
    config = {}
    if args.config:
        from config.loader import load_yaml_config
        config = load_yaml_config(args.config)
        print(f"   配置文件: {args.config}")
    
    try:
        if args.model == "tft":
            print("   正在训练 TFT 模型...")
            # 调用 train.py 的逻辑
            from train import main as train_main
            import sys
            # 构造命令行参数（注意：train.py 使用位置参数，不是 --model）
            sys.argv = ['train.py', 'tft']
            if args.total_steps:
                sys.argv += ['--total-steps', str(args.total_steps)]
            if args.num_skus:
                sys.argv += ['--num-skus', str(args.num_skus)]
            if args.no_tensorboard:
                sys.argv += ['--no-tensorboard']
            
            train_main()
            
        elif args.model == "ppo":
            print("   正在训练 PPO 模型...")
            # 调用 experiments/run_proposed.py
            from experiments.run_proposed import main as train_ppo
            import sys
            sys.argv = ['run_proposed.py']
            if args.total_steps:
                sys.argv += ['--total-steps', str(args.total_steps)]
            if args.num_skus:
                sys.argv += ['--num-skus', str(args.num_skus)]
            
            train_ppo()
            
        elif args.model == "integrated":
            print("   正在训练集成模型...")
            from experiments.run_end_to_end import main as train_integrated
            train_integrated()
        
        print("✅ 训练完成！")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_infer(args):
    """推理命令"""
    print("🔮 开始推理...")
    print(f"   模型路径: {args.model_path}")
    
    # 加载配置文件
    config = {}
    if args.config:
        from config.loader import load_yaml_config
        config = load_yaml_config(args.config)
        print(f"   配置文件: {args.config}")
    
    try:
        from inference import main as infer_main
        import sys
        
        # 构造命令行参数
        sys.argv = ['inference.py', '--model-path', args.model_path]
        if args.output:
            sys.argv += ['--output', args.output]
        if args.forecast_horizon:
            sys.argv += ['--forecast-horizon', str(args.forecast_horizon)]
        
        infer_main()
        print("✅ 推理完成！")
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_evaluate(args):
    """评估命令"""
    print("📊 开始评估模型...")
    print(f"   模型路径: {args.model_path}")
    
    # 加载配置文件
    config = {}
    if args.config:
        from config.loader import load_yaml_config
        config = load_yaml_config(args.config)
        print(f"   配置文件: {args.config}")
    
    try:
        from evaluation.evaluator import Evaluator
        from config import get_config
        
        # 加载配置
        app_config = get_config()
        
        # 创建评估器
        evaluator = Evaluator(app_config)
        
        # 运行评估
        metrics = evaluator.evaluate_model(args.model_path)
        
        print("✅ 评估完成！")
        print(f"   评估指标:")
        for key, value in metrics.items():
            print(f"     {key}: {value:.4f}")
        
        # 保存评估结果
        import json
        output_file = f"{args.model_path}_eval.json"
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"   评估结果已保存: {output_file}")
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_pipeline(args):
    """完整流程命令"""
    print("🔄 开始运行完整流程...")
    
    # 加载配置文件
    config = {}
    if args.config:
        from config.loader import load_yaml_config
        config = load_yaml_config(args.config)
        print(f"   配置文件: {args.config}")
    
    try:
        import sys
        import glob
        
        # 步骤1: 数据准备
        print("\n📦 步骤1: 数据准备")
        from data.data_pipeline import main as data_pipeline_main
        data_pipeline_main()
        
        # 步骤2: 训练（除非跳过）
        if not args.skip_train:
            print("\n🚀 步骤2: 训练模型")
            from train import main as train_main
            sys.argv = ['train.py', '--model', 'tft']
            train_main()
        else:
            print("\n⏭ 步骤2: 训练模型 (已跳过)")
        
        # 步骤3: 推理（除非跳过）
        if not args.skip_infer:
            print("\n🔮 步骤3: 模型推理")
            # 查找最佳模型
            model_files = glob.glob("models/**/tft_model_best.ckpt", recursive=True)
            if model_files:
                from inference import main as infer_main
                sys.argv = ['inference.py', '--model-path', model_files[0]]
                infer_main()
            else:
                print("   ⚠️ 未找到训练好的模型，跳过推理")
        else:
            print("\n⏭ 步骤3: 模型推理 (已跳过)")
        
        # 步骤4: 评估
        print("\n📊 步骤4: 模型评估")
        model_files = glob.glob("models/**/tft_model_best.ckpt", recursive=True)
        if model_files:
            from evaluation.evaluator import Evaluator
            from config import get_config
            app_config = get_config()
            evaluator = Evaluator(app_config)
            metrics = evaluator.evaluate_model(model_files[0])
            print("   评估指标:")
            for key, value in metrics.items():
                print(f"     {key}: {value:.4f}")
        
        print("\n✅ 完整流程执行完成！")
        
    except Exception as e:
        print(f"\n❌ 流程执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_graph(args):
    """生成代码知识图谱命令"""
    print("🕸️ 开始生成代码知识图谱...")
    
    try:
        import subprocess
        
        # 确保输出目录存在
        output_dir = args.output.rstrip('/')
        os.makedirs(output_dir, exist_ok=True)
        
        # 调用 scripts/code_knowledge_graph.py
        script_path = os.path.join(PROJECT_ROOT, "scripts", "code_knowledge_graph.py")
        output_file = f"{output_dir}/knowledge_graph.html"
        
        cmd = [
            sys.executable,
            script_path,
            str(PROJECT_ROOT),
            output_file
        ]
        
        print(f"   输出目录: {output_dir}")
        print(f"   运行命令: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        if result.returncode == 0:
            print("✅ 知识图谱生成成功！")
            print(f"   输出文件: {output_file}")
            
            # 显示统计信息
            import re
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            nodes_match = re.search(r'nodes:\s*\[(.*?)\]', content, re.DOTALL)
            if nodes_match:
                nodes_str = nodes_match.group(1)
                nodes = re.findall(r'\{.*?\}', nodes_str)
                print(f"   总节点数: {len(nodes)}")
            
            edges_match = re.search(r'edges:\s*\[(.*?)\]', content, re.DOTALL)
            if edges_match:
                edges_str = edges_match.group(1)
                edges = re.findall(r'\{.*?\}', edges_str)
                print(f"   总边数: {len(edges)}")
            
            # 自动打开
            if args.open:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(output_file)}")
                print("   已在浏览器中打开")
            
        else:
            print(f"❌ 知识图谱生成失败！")
            print(f"   错误输出: {result.stderr}")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ 知识图谱生成失败: {e}")
        sys.exit(1)


def cmd_config(args):
    """配置管理命令"""
    if args.action == "show":
        print("📋 当前配置:")
        try:
            from config import get_config
            from dataclasses import asdict
            
            config = get_config()
            
            # 转换为字典并显示
            def print_config_obj(obj, prefix="  "):
                """递归打印配置对象"""
                if hasattr(obj, '__dataclass_fields__'):
                    # dataclass 对象
                    for field_name in obj.__dataclass_fields__:
                        value = getattr(obj, field_name)
                        if hasattr(value, '__dataclass_fields__'):
                            print(f"{prefix}{field_name}:")
                            print_config_obj(value, prefix + "  ")
                        else:
                            print(f"{prefix}{field_name}: {value}")
                else:
                    # SimpleNamespace 或其他对象
                    for attr in dir(obj):
                        if not attr.startswith('_'):
                            value = getattr(obj, attr)
                            if not callable(value):
                                if hasattr(value, '__dataclass_fields__') or hasattr(value, '__dict__'):
                                    print(f"{prefix}{attr}:")
                                    print_config_obj(value, prefix + "  ")
                                else:
                                    print(f"{prefix}{attr}: {value}")
            
            print_config_obj(config)
            
        except Exception as e:
            print(f"❌ 无法加载配置: {e}")
            import traceback
            traceback.print_exc()
            
    elif args.action == "generate":
        print("📝 生成示例配置文件...")
        try:
            output_path = args.output or "config/example.yaml"
            from config.yaml_loader import generate_example_config
            generate_example_config(output_path)
            print(f"✅ 配置文件已生成: {output_path}")
        except Exception as e:
            print(f"❌ 生成配置文件失败: {e}")


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    # 根据命令调用对应函数
    if args.command == "train":
        cmd_train(args)
    elif args.command == "infer":
        cmd_infer(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "pipeline":
        cmd_pipeline(args)
    elif args.command == "graph":
        cmd_graph(args)
    elif args.command == "config":
        cmd_config(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
