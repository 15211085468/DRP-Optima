"""
TFT 模型评估模块 - evaluation/tft_evaluator.py

从 scripts/baseline_comparison.py 的 Phase 3 提取而来。
负责对训练好的 TFT 模型进行全量预测评估。
"""

import os
import traceback
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder, TorchNormalizer


def evaluate_tft(
    tft_ckpt_path: str,
    df_full: pd.DataFrame,
    test_weeks: int,
    output_dir: str,
    raw_metrics: dict,
    calc_smape,
    calc_mae,
    calc_rmse,
) -> tuple:
    """
    评估 TFT 模型，返回 (tft_metrics_dict, (y_true, y_pred) or None)
    
    Args:
        tft_ckpt_path: TFT checkpoint 文件路径
        df_full: Phase 1 预处理完成的完整数据
        test_weeks: 测试窗口周数
        output_dir: 结果保存目录
        raw_metrics: 逐 SKU 原始指标字典（会被就地修改，添加 'TFT (Ours)' 键）
        calc_smape, calc_mae, calc_rmse: 误差计算函数
    
    Returns:
        (tft_metrics, tft_predictions)
        - tft_metrics: dict with keys SMAPE, MAE, RMSE, n_samples
        - tft_predictions: (y_true, y_pred) 或 None（如果评估失败）
    """
    print("\n" + "=" * 60)
    print("Phase 3: TFT 全量预测")
    print("=" * 60)

    if not os.path.exists(tft_ckpt_path):
        print(f"[WARN] TFT 模型文件不存在，跳过 TFT 对比")
        return None, None

    try:
        cuda_available = torch.cuda.is_available()

        # 1. 加载 checkpoint 获取 dataset_parameters
        print("正在加载 checkpoint 获取 dataset_parameters...")
        checkpoint = torch.load(tft_ckpt_path, map_location="cpu", weights_only=False)
        dataset_params = checkpoint.get("dataset_parameters")
        if dataset_params is not None:
            print(f"  dataset_parameters 已找到（{len(dataset_params)} 个参数）")
        else:
            print("[WARN]  dataset_parameters 未找到，将使用手动配置")

        # 2. 加载模型
        tft_model = TemporalFusionTransformer.load_from_checkpoint(tft_ckpt_path)
        tft_model = tft_model.to("cuda" if cuda_available else "cpu")
        tft_model.eval()
        print("TFT 模型加载成功")

        # 3. 复用 Phase 1 已预处理的完整数据
        print("复用 Phase 1 预处理数据（df_full），跳过重新读取...")

        from data.data_preprocessor import DataPreprocessor
        from config.data_config import TFTModelConfig

        tft_config = TFTModelConfig(
            max_encoder_length=52,
            max_prediction_length=1,
            hidden_size=64,
            batch_size=256,
            loss_type="smape",
        )

        print(f"  df_full 已就绪, 维度: {df_full.shape}")

        # 过滤序列长度不足的组
        min_req = int(52 * 0.8) + 1
        gsizes = df_full.groupby(["shop_code", "goods_code"]).size()
        valid_g = gsizes[gsizes >= min_req].index
        df_full = df_full[
            df_full.set_index(["shop_code", "goods_code"]).index.isin(valid_g)
        ].copy()

        print(f"  df_full 过滤后维度: {df_full.shape}")

        # 填充衍生特征列的 NaN
        derived_cols = [c for c in df_full.columns if c.startswith(("lag_", "rolling_", "qty_wow"))]
        for col in derived_cols:
            if df_full[col].isna().any():
                df_full[col] = df_full.groupby(["shop_code", "goods_code"])[col].transform(
                    lambda s: s.ffill().bfill()
                )
                df_full[col] = df_full[col].fillna(0)

        df_full["time_idx"] = df_full["time_idx"].astype(int)

        # 关键：将 shop_code 和 goods_code 转换为字符串类型
        print("  转换 shop_code 和 goods_code 为字符串类型...")
        df_full["shop_code"] = df_full["shop_code"].astype(str)
        df_full["goods_code"] = df_full["goods_code"].astype(str)
        print("  转换完成")

        training_cutoff = df_full["time_idx"].max() - test_weeks

        # 4. 恢复或手动创建数据集
        dataset_params = checkpoint.get("dataset_parameters")

        if dataset_params is not None:
            print("  dataset_parameters 已找到，正在恢复...")
            training_ds = TimeSeriesDataSet.from_parameters(
                dataset_params,
                df_full[df_full["time_idx"] <= training_cutoff].copy(),
            )
            print(f"  训练数据集已恢复: {len(training_ds)} 样本")

            val_ds = TimeSeriesDataSet.from_dataset(
                training_ds,
                df_full,
                predict=True,
                stop_randomization=True,
            )
            print(f"  验证数据集已创建: {len(val_ds)} 样本")
        else:
            print("  dataset_parameters 未找到，使用手动配置...")
            training_ds = TimeSeriesDataSet(
                df_full[df_full["time_idx"] <= training_cutoff],
                time_idx="time_idx",
                target="sale_qty",
                group_ids=["shop_code", "goods_code"],
                max_encoder_length=52,
                min_encoder_length=42,
                max_prediction_length=1,
                static_categoricals=["shop_code", "goods_code"],
                static_reals=["encoder_length", "sale_qty_center", "sale_qty_scale"],
                time_varying_known_reals=[],
                time_varying_unknown_reals=["sale_amt", "relative_time_idx"],
                allow_missing_timesteps=True,
            )
            val_ds = TimeSeriesDataSet.from_dataset(training_ds, df_full, predict=True, stop_randomization=True)

        val_dl = val_ds.to_dataloader(train=False, batch_size=256, num_workers=0)

        # 5. 执行预测
        with torch.no_grad():
            tft_preds = tft_model.predict(
                val_dl,
                return_y=True,
                trainer_kwargs=dict(accelerator="gpu" if cuda_available else "cpu"),
            )

        tft_y_pred = tft_preds.output.cpu().numpy().flatten()
        tft_y_true = tft_preds.y[0].cpu().numpy().flatten()

        # 6. 按 SKU 聚合指标（用于统计显著性检验）
        try:
            training_cutoff = df_full["time_idx"].max() - test_weeks
            df_test = df_full[df_full["time_idx"] > training_cutoff].copy().reset_index(drop=True)

            if len(df_test) == len(tft_y_pred):
                df_test["tft_pred"] = tft_y_pred
                df_test["tft_true"] = tft_y_true

                # 按 SKU 聚合
                tft_per_sku = df_test.groupby(["shop_code", "goods_code"]).apply(
                    lambda g: pd.Series({
                        "smape": calc_smape(g["tft_true"].values, g["tft_pred"].values),
                        "mae": calc_mae(g["tft_true"].values, g["tft_pred"].values),
                        "rmse": calc_rmse(g["tft_true"].values, g["tft_pred"].values),
                    })
                ).reset_index()

                # 添加到 raw_metrics
                if "TFT (Ours)" not in raw_metrics:
                    raw_metrics["TFT (Ours)"] = {"smape": [], "mae": [], "rmse": []}

                raw_metrics["TFT (Ours)"]["smape"].extend(tft_per_sku["smape"].tolist())
                raw_metrics["TFT (Ours)"]["mae"].extend(tft_per_sku["mae"].tolist())
                raw_metrics["TFT (Ours)"]["rmse"].extend(tft_per_sku["rmse"].tolist())

                print(f"TFT 每 SKU 指标已计算: {len(tft_per_sku)} 个 SKU")
                print(f"  SMAPE 范围: [{tft_per_sku['smape'].min():.4f}, {tft_per_sku['smape'].max():.4f}]")
            else:
                print(f"[WARN] TFT 每 SKU 指标计算失败: df_test 长度({len(df_test)}) != 预测值长度({len(tft_y_pred)})")

        except Exception as agg_e:
            print(f"[WARN] TFT 每 SKU 指标计算失败: {agg_e}")
            traceback.print_exc()

        # 7. 计算整体指标
        valid_mask = np.isfinite(tft_y_pred) & np.isfinite(tft_y_true)
        tft_y_pred = tft_y_pred[valid_mask]
        tft_y_true = tft_y_true[valid_mask]

        tft_metrics = {
            "SMAPE": calc_smape(tft_y_true, tft_y_pred),
            "MAE": calc_mae(tft_y_true, tft_y_pred),
            "RMSE": calc_rmse(tft_y_true, tft_y_pred),
            "n_samples": int(valid_mask.sum()),
        }
        tft_predictions = (tft_y_true, tft_y_pred)
        print(f"TFT 评估完成：SMAPE={tft_metrics['SMAPE']:.4f}, MAE={tft_metrics['MAE']:.4f}")

        # 8. 保存 TFT 预测值
        try:
            npz_path = os.path.join(output_dir, "tft_predictions.npz")
            np.savez_compressed(
                npz_path,
                y_true=tft_y_true,
                y_pred=tft_y_pred,
                smape=tft_metrics["SMAPE"],
                mae=tft_metrics["MAE"],
                rmse=tft_metrics["RMSE"],
            )
            print(f"TFT 预测值已保存: {npz_path}")
        except Exception as save_e:
            print(f"[WARN] TFT 预测值保存失败: {save_e}")

        return tft_metrics, tft_predictions

    except Exception as e:
        print(f"[WARN] TFT 评估失败: {e}")
        traceback.print_exc()
        return None, None


def find_latest_tft_model(project_root: str) -> str or None:
    """
    自动查找最新的 TFT checkpoint 文件
    
    Args:
        project_root: 项目根目录
    
    Returns:
        checkpoint 文件路径，如果未找到返回 None
    """
    models_dir = os.path.join(project_root, "records", "models")
    if not os.path.exists(models_dir):
        return None

    ckpt_files = [f for f in os.listdir(models_dir) if f.endswith(".ckpt")]
    if not ckpt_files:
        return None

    ckpt_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    return os.path.join(models_dir, ckpt_files[0])
