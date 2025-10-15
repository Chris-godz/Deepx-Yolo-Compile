#!/usr/bin/env python3
"""
YOLOv5 DXNN vs YOLOv5SU 相似度对比
计算相似度指标并生成可视化图表
"""

import os
import cv2
import numpy as np
import json
import torch
from dx_engine import InferenceEngine
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def letterbox_resize(image, target_size=(640, 640), fill_value=114):
    """Letterbox resize with padding"""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    r = min(target_h / h, target_w / w)
    new_h, new_w = int(h * r), int(w * r)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((target_h, target_w, 3), fill_value, dtype=np.uint8)
    
    dy = (target_h - new_h) // 2
    dx = (target_w - new_w) // 2
    
    padded[dy:dy+new_h, dx:dx+new_w] = resized
    return padded, r, (dx, dy)

def calculate_similarity_metrics(output1, output2):
    """计算所有相似度指标"""
    flat1 = output1.flatten()
    flat2 = output2.flatten()
    
    min_len = min(len(flat1), len(flat2))
    flat1 = flat1[:min_len]
    flat2 = flat2[:min_len]
    
    metrics = {}
    
    # 1. 余弦相似度 (Cosine Similarity)
    metrics['余弦相似度'] = 1 - cosine(flat1, flat2)
    
    # 2. 皮尔逊相关系数 (Pearson Correlation)
    try:
        pearson_corr, _ = pearsonr(flat1, flat2)
        metrics['皮尔逊相关系数'] = pearson_corr
    except:
        metrics['皮尔逊相关系数'] = 0.0
    
    # 3. 均方误差 (MSE)
    metrics['均方误差(MSE)'] = mean_squared_error(flat1, flat2)
    
    # 4. 平均绝对误差 (MAE)
    metrics['平均绝对误差(MAE)'] = mean_absolute_error(flat1, flat2)
    
    # 5. 归一化均方根误差 (NRMSE)
    rmse = np.sqrt(metrics['均方误差(MSE)'])
    data_range = np.max(flat1) - np.min(flat1)
    metrics['归一化RMSE'] = rmse / data_range if data_range != 0 else 0
    
    # 6. 结构相似性指数 (SSIM-like)
    mean1, mean2 = np.mean(flat1), np.mean(flat2)
    var1, var2 = np.var(flat1), np.var(flat2)
    covar = np.mean((flat1 - mean1) * (flat2 - mean2))
    
    c1, c2 = 0.01, 0.03
    ssim = ((2 * mean1 * mean2 + c1) * (2 * covar + c2)) / \
           ((mean1**2 + mean2**2 + c1) * (var1 + var2 + c2))
    metrics['结构相似性(SSIM)'] = ssim
    
    # 7. 相对误差 (Relative Error)
    metrics['相对误差'] = np.mean(np.abs(flat1 - flat2) / (np.abs(flat1) + 1e-8))
    
    return metrics

def create_similarity_visualization(metrics):
    """创建相似度指标可视化图表"""
    
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('YOLOv5 DXNN vs YOLOv5SU Model Similarity Analysis', fontsize=16, fontweight='bold')
    
    # 1. 相似度指标条形图
    ax1 = axes[0, 0]
    similarity_metrics = {
        'Cosine Similarity': metrics['余弦相似度'],
        'Pearson Correlation': metrics['皮尔逊相关系数'],
        'SSIM': metrics['结构相似性(SSIM)']
    }
    ax1.set_title('Similarity Metrics (Closer to 1 = More Similar)', fontweight='bold')
    ax1.set_ylabel('Similarity Value')
    
    bars = ax1.bar(similarity_metrics.keys(), similarity_metrics.values(), 
                   color=['#2E8B57', '#4169E1', '#FF6347'])
    ax1.set_ylim(0, 1.1)
    
    # 添加数值标签
    for bar, value in zip(bars, similarity_metrics.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.6f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 误差指标条形图
    ax2 = axes[0, 1]
    error_metrics = {
        'MSE': metrics['均方误差(MSE)'],
        'MAE': metrics['平均绝对误差(MAE)'],
        'Normalized RMSE': metrics['归一化RMSE'],
        'Relative Error': metrics['相对误差']
    }
    ax2.set_title('Error Metrics (Closer to 0 = More Similar)', fontweight='bold')
    ax2.set_ylabel('Error Value')
    
    bars = ax2.bar(error_metrics.keys(), error_metrics.values(), 
                   color=['#FF4500', '#FF8C00', '#FFA500', '#FFD700'])
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars, error_metrics.values()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(error_metrics.values())*0.01,
                f'{value:.6f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. 综合相似度评分雷达图
    ax3 = axes[1, 0]
    ax3.remove()  # 移除原轴
    ax3 = fig.add_subplot(2, 2, 3, projection='polar')
    
    categories = ['Cosine Sim', 'Pearson Corr', 'SSIM', 'Low MSE', 'Low MAE', 'Low NRMSE']
    radar_title = 'Comprehensive Similarity Score\n(Outer Ring = Perfect Similarity)'
    
    values = [
        metrics['余弦相似度'],
        metrics['皮尔逊相关系数'],
        metrics['结构相似性(SSIM)'],
        1 - min(metrics['均方误差(MSE)'] / 10, 1),  # 反转并归一化
        1 - min(metrics['平均绝对误差(MAE)'] / 2, 1),  # 反转并归一化
        1 - min(metrics['归一化RMSE'], 1)  # 反转
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]  # 闭合图形
    angles += angles[:1]
    
    ax3.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
    ax3.fill(angles, values, alpha=0.25, color='#1f77b4')
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)
    ax3.set_ylim(0, 1)
    ax3.set_title(radar_title, pad=20, fontweight='bold')
    ax3.grid(True)
    
    # 4. 质量评估表格
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 计算总体质量评分
    avg_similarity = np.mean([metrics['余弦相似度'], metrics['皮尔逊相关系数'], metrics['结构相似性(SSIM)']])
    
    if avg_similarity > 0.99:
        quality = "EXCELLENT"
        quality_desc = "Highly Similar (>99%)"
    elif avg_similarity > 0.95:
        quality = "GOOD"
        quality_desc = "Quite Similar (>95%)"
    elif avg_similarity > 0.90:
        quality = "ACCEPTABLE"
        quality_desc = "Reasonably Similar (>90%)"
    else:
        quality = "POOR"
        quality_desc = "Significant Differences (<90%)"
    
    # 创建评估表格
    table_data = [
        ['Evaluation Item', 'Value', 'Description'],
        ['Cosine Similarity', f'{metrics["余弦相似度"]:.6f}', 'Closer to 1 = More Similar'],
        ['Pearson Correlation', f'{metrics["皮尔逊相关系数"]:.6f}', 'Linear Correlation Degree'],
        ['Structural Similarity', f'{metrics["结构相似性(SSIM)"]:.6f}', 'Structure Info Preservation'],
        ['Mean Squared Error', f'{metrics["均方误差(MSE)"]:.6f}', 'Numerical Difference Degree'],
        ['Mean Absolute Error', f'{metrics["平均绝对误差(MAE)"]:.6f}', 'Average Deviation'],
        ['', '', ''],
        ['Overall Assessment', quality, quality_desc]
    ]
    table_title = 'Detailed Evaluation Report'
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表格样式
    for i in range(len(table_data)):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # 标题行
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            elif i == len(table_data) - 1:  # 评估行
                cell.set_facecolor('#E7E6E6')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#F2F2F2')
    
    ax4.set_title(table_title, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def main():
    """主函数"""
    print("🚀 YOLOv5 DXNN vs YOLOv5SU 相似度分析")
    print("="*50)
    
    # 加载配置和模型
    config_path = './yolov5su.json'
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # DXNN模型
    ie = InferenceEngine(config["model"]["path"])
    print(f"✅ DXNN模型已加载")
    
    # YOLOv5SU模型
    from ultralytics import YOLO
    yolo_model = YOLO('./model/yolov5su.pt')
    print(f"✅ YOLOv5SU模型已加载")
    
    # 加载测试图片
    test_image = "./test.jpg"
    original_img = cv2.imread(test_image)
    preprocessed_img, _, _ = letterbox_resize(original_img, (640, 640))
    
    print(f"📷 测试图片: {os.path.basename(test_image)}")
    
    # DXNN推理
    dxnn_input = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB)
    dxnn_outputs = ie.run([dxnn_input])
    dxnn_main = dxnn_outputs[0]
    
    # YOLOv5SU推理
    yolo_input_rgb = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB)
    yolo_input_tensor = torch.from_numpy(yolo_input_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        raw_output = yolo_model.model(yolo_input_tensor)
    yolo_main = raw_output[0].cpu().numpy()
    
    print(f"✅ 推理完成")
    print(f"   DXNN输出: {dxnn_main.shape}")
    print(f"   YOLOv5SU输出: {yolo_main.shape}")
    
    # 计算相似度指标
    print("📊 计算相似度指标...")
    metrics = calculate_similarity_metrics(dxnn_main, yolo_main)
    
    # 打印结果
    print("\n📈 相似度分析结果:")
    print("-" * 50)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # 生成可视化图表
    print("🎨 生成可视化图表...")
    fig = create_similarity_visualization(metrics)
    
    # 保存图表
    output_path = "./similarity_analysis.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 分析完成!")
    print(f"📄 图表已保存: {output_path}")

if __name__ == "__main__":
    main()