"""
可视化工具模块 - utils/plotting.py

从 scripts/baseline_comparison.py 的 Phase 5 提取而来。
提供统一的预测模型性能对比可视化函数。
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_metrics_bar(
    summary_df: "pd.DataFrame",
    output_dir: str,
    colors: list = None,
    dpi: int = 150,
) -> str:
    """
    绘制预测模型误差对比柱状图（SMAPE / MAE / RMSE）
    
    Args:
        summary_df: 汇总指标 DataFrame，必须包含列：
                   Model, SMAPE_mean, MAE_mean, RMSE_mean
        output_dir: 图片保存目录
        colors: 每个模型对应的颜色列表（可选）
        dpi: 图片 DPI
    
    Returns:
        保存的文件路径
    """
    model_names = summary_df["Model"].tolist()
    smape_vals = summary_df["SMAPE_mean"].tolist()
    mae_vals = summary_df["MAE_mean"].tolist()
    rmse_vals = summary_df["RMSE_mean"].tolist()

    # 默认颜色配置（深色背景风格）
    if colors is None:
        colors = ["#5B8FF9", "#61D9AA", "#F6BD16", "#E86452", "#945FB9"]
    colors = colors[: len(model_names)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor("#121826")

    metrics = [
        (axes[0], smape_vals, "SMAPE (均值)"),
        (axes[1], mae_vals, "MAE (均值)"),
        (axes[2], rmse_vals, "RMSE (均值)"),
    ]

    for ax, metric_vals, metric_name in metrics:
        bars = ax.bar(
            range(len(model_names)), metric_vals,
            color=colors, alpha=0.88, width=0.6,
        )
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(
            model_names, rotation=20, ha="right",
            fontsize=9, color="#E2E8F0",
        )
        ax.set_title(metric_name, color="#E2E8F0", fontsize=13)
        ax.set_facecolor("#1A233A")
        ax.tick_params(colors="#94A3B8")
        ax.yaxis.label.set_color("#E2E8F0")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        # 在柱顶标注数值
        for bar, val in zip(bars, metric_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=8.5, color="#FFD700",
            )

    plt.suptitle("DRP-Optima — 预测模型性能对比", color="#E2E8F0", fontsize=15, y=1.01)
    plt.tight_layout()

    bar_path = os.path.join(output_dir, "metrics_bar.png")
    plt.savefig(bar_path, dpi=dpi, bbox_inches="tight", facecolor="#121826")
    plt.close()
    print(f"误差柱状图已保存: {bar_path}")
    return bar_path


def plot_tft_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tft_metrics: dict,
    output_dir: str,
    sample_size: int = 5000,
    dpi: int = 150,
) -> str:
    """
    绘制 TFT 预测值 vs 真实值的散点图
    
    Args:
        y_true: 真实值数组
        y_pred: 预测值数组
        tft_metrics: TFT 指标字典（用于标注 SMAPE）
        output_dir: 图片保存目录
        sample_size: 随机采样样本数（默认 5000）
        dpi: 图片 DPI
    
    Returns:
        保存的文件路径；如果出错返回空字符串
    """
    np.random.seed(42)
    actual_sample_size = min(sample_size, len(y_true))
    sample_idx = np.random.choice(len(y_true), actual_sample_size, replace=False)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#121826")
    ax.set_facecolor("#1A233A")

    ax.scatter(
        y_true[sample_idx], y_pred[sample_idx],
        alpha=0.3, s=8, c="#F6BD16",
        label=f"TFT SMAPE={tft_metrics['SMAPE']:.4f}",
    )

    max_val = max(y_true[sample_idx].max(), y_pred[sample_idx].max())
    ax.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="Perfect")

    ax.set_xlabel("Actual", color="#E2E8F0")
    ax.set_ylabel("Predicted", color="#E2E8F0")
    ax.set_title("TFT 预测散点图", color="#E2E8F0", fontsize=13)
    ax.tick_params(colors="#94A3B8")
    ax.legend(facecolor="#2a2a4e", edgecolor="#444", labelcolor="#E2E8F0")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()

    scatter_path = os.path.join(output_dir, "scatter_comparison.png")
    plt.savefig(scatter_path, dpi=dpi, bbox_inches="tight", facecolor="#121826")
    plt.close()
    print(f"散点图已保存: {scatter_path}")
    return scatter_path


def plot_baseline_comparison(
    summary_df: "pd.DataFrame",
    output_dir: str,
    tft_predictions: tuple = None,
    tft_metrics: dict = None,
    colors: list = None,
    dpi: int = 150,
) -> dict:
    """
    一站式绘制 baseline 对比的所有可视化图表
    
    Args:
        summary_df: 汇总指标 DataFrame
        output_dir: 图片保存目录
        tft_predictions: (y_true, y_pred) 元组，若为 None 则不绘制散点图
        tft_metrics: TFT 指标字典，若为 None 则尝试从 summary_df 获取
        colors: 颜色列表
        dpi: 图片 DPI
    
    Returns:
        字典：{"bar_path": str, "scatter_path": str or None}
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {"bar_path": None, "scatter_path": None}

    # 1. 指标柱状图
    results["bar_path"] = plot_metrics_bar(
        summary_df=summary_df,
        output_dir=output_dir,
        colors=colors,
        dpi=dpi,
    )

    # 2. TFT 散点图（仅当 TFT 结果可用时）
    if tft_predictions is not None:
        if tft_metrics is None:
            # 尝试从 summary_df 获取 TFT 的 SMAPE
            tft_row = summary_df[summary_df["Model"] == "TFT (Ours)"]
            if not tft_row.empty:
                tft_metrics = {"SMAPE": tft_row["SMAPE_mean"].values[0]}

        if tft_metrics is not None:
            y_true_tft, y_tft = tft_predictions
            results["scatter_path"] = plot_tft_scatter(
                y_true=y_true_tft,
                y_pred=y_tft,
                tft_metrics=tft_metrics,
                output_dir=output_dir,
                dpi=dpi,
            )

    return results
