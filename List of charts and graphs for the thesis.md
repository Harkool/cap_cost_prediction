## 📊 **论文图表清单（正文 + 补充材料）**

---

## 🎨 **一、正文图表（Main Figures & Tables）**

### **Figure 1: 研究流程图（Study Flowchart）**

**内容：**
```
患者筛选流程（CONSORT风格）
├─ 初始纳入：2019.12-2024.12诊断为重症CAP的患者 (n=XXX)
├─ 排除：
│   ├─ 非重症CAP (n=XX)
│   ├─ 住院<1天或>40天 (n=XX)
│   └─ 关键数据缺失>30% (n=XX)
├─ 最终纳入 (n)
└─ 数据集划分：
    ├─ 训练集 (80%)
    └─ 测试集 (20%)
```

**制作工具：** PowerPoint / draw.io / BioRender  
**建议尺寸：** 单栏宽度

---

### **Figure 2: 模型架构图（Model Architecture）**

**内容：**
```
┌─────────────────────────────────────────┐
│         输入层（多模态特征）             │
│  基线临床 | 实验室 | 病原学 | 合并症 | 治疗│
└──────────────────┬──────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│       合并症门控增强模块                  │
│    动态调整特征权重 (Comorbidity Gate)    │
└──────────────────┬───────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│       共享特征提取器                      │
│   Dense(64) → BN → ReLU → Dropout       │
│   Dense(32) → BN → ReLU → Dropout       │
└──────────────────┬───────────────────────┘
                   ↓
┌──────────────────────────────────────────┐
│        8个任务专用预测头                  │
│  [总诊疗][床位][检查][治疗][手术]...     │
│   每个头：Dense(16) → ReLU → Linear(1)  │
└──────────────────┬───────────────────────┘
                   ↓
         总费用 = Σ 8个子项 + 残差修正
```

**制作工具：** draw.io / PowerPoint / Python (matplotlib)  
**配色方案：** 使用专业配色（如蓝色系表示共享层，橙色系表示任务专用层）  
**建议尺寸：** 双栏宽度

---

### **Table 1: 患者基线特征（Baseline Characteristics）**

**列：** 特征名称 | 总体 (n=471) | 训练集 (n=330) | 测试集 (n=70) | P值

**行（分组）：**

**人口学特征**
- 年龄，岁 [均值±SD]
- 性别，男性 [n (%)]
- BMI，kg/m² [均值±SD]

**合并症 [n (%)]**
- 高血压
- 糖尿病
- 心功能不全
- 慢性肾病（分期0/1-2/3-4/5）
- 慢性肝病
- 恶性肿瘤
- 合并症数量 [中位数 (IQR)]

**实验室指标 [中位数 (IQR)]**
- 白细胞计数, ×10⁹/L
- 中性粒细胞百分比, %
- 血红蛋白, g/L
- CRP, mg/L
- PCT, ng/mL
- 白蛋白, g/L
- 肌酐, μmol/L
- PaO2, mmHg
- 乳酸, mmol/L

**病原学 [n (%)]**
- 痰培养阳性
- 血培养阳性
- 耐药菌感染

**治疗相关**
- ICU入住 [n (%)]
- 住院天数 [中位数 (IQR)]
- 血常规监测次数 [中位数 (IQR)]

**费用，元 [中位数 (IQR)]**
- 总费用
- 总诊疗费
- 床位费
- 检查费
- 治疗费
- 手术费
- 护理费
- 卫生材料费
- 其他费用

**制作工具：** Excel → 导出为高分辨率图片 或 LaTeX表格

---

### **Table 2: 模型性能对比（Model Performance Comparison）**

**列：** 模型 | R² | MAE (¥) | RMSE (¥) | MAPE (%) | 参数量

**行：**
- 多元线性回归
- Ridge回归
- Lasso回归
- 随机森林
- XGBoost
- 梯度提升树
- 多层感知机（MLP）
- **本研究模型（完整）**
- 消融实验1：无门控机制
- 消融实验2：直接预测总费用（不分子项）
- 消融实验3：无一致性损失

**附注：**
- 加粗最优值
- 添加95% Bootstrap置信区间
- 标注统计显著性（* p<0.05, ** p<0.01）

---

### **Figure 3: 总费用预测性能（Overall Cost Prediction Performance）**

**4个子图（2×2布局）：**

**A. 预测vs真实散点图**
- X轴：真实总费用
- Y轴：预测总费用
- 添加：完美预测线（红色虚线）、±20%范围（灰色阴影）
- 标注：R²=0.78, MAPE=12.8%

**B. 残差分布直方图**
- X轴：预测误差（预测-真实）
- Y轴：频数
- 添加：零残差线、均值线
- 标注：均值=±XX元，SD=±XX元

**C. 相对误差vs真实费用散点图**
- X轴：真实总费用
- Y轴：相对误差(%)
- 添加：10%、20%误差线
- 标注：≤10%误差占比XX%，≤20%误差占比XX%

**D. 各子项R²性能条形图**
- X轴：8个子项名称
- Y轴：R² Score
- 配色：渐变色
- 添加：R²=0.7基准线

**制作工具：** Python (matplotlib/seaborn)  
**建议尺寸：** 双栏宽度，高度约15cm

---

### **Table 3: 各子项费用预测性能（Performance by Cost Component）**

**列：** 子项名称 | R² | MAE (¥) | RMSE (¥) | MAPE (%) | 平均费用 (¥) | 占比 (%)

**行：**
- 总诊疗费
- 床位费
- 检查费
- 治疗费
- 手术费
- 护理费
- 卫生材料费
- 其他费用
- **加权平均**

---

### **Figure 4: 合并症对费用结构的影响（Comorbidity Impact on Cost Structure）**

**热力图（Heatmap）：**
- X轴：8个子项费用
- Y轴：合并症组合（按总费用降序排列）
  ```
  无合并症
  糖尿病
  高血压
  慢性肾病
  糖尿病+肾病
  心衰+肾病
  糖尿病+肝病
  恶性肿瘤+其他
  ...（显示前12个组合）
  ```
- 颜色：YlOrRd（黄-橙-红）渐变，数值单位：千元
- 添加数值标注

**右侧附加柱状图：** 各组合的样本量

**制作工具：** Python (seaborn.heatmap)  
**建议尺寸：** 双栏宽度

---

### **Figure 5: SHAP特征重要性分析（SHAP Feature Importance）**

**2个子图（上下布局）：**

**A. 全局特征重要性（Top 20）**
- 条形图，按平均|SHAP值|排序
- 配色：渐变色
- X轴：平均|SHAP值|
- Y轴：特征名称（中文）

**B. SHAP摘要图（Summary Plot）**
- 蜂群图，显示Top 15特征
- 颜色表示特征值（红色=高，蓝色=低）
- X轴：SHAP值（对总费用的影响）
- Y轴：特征名称

**制作工具：** Python (shap library)  
**建议尺寸：** 双栏宽度，高度约18cm

---

### **Figure 6: 典型病例的费用预测分解（Case Study: Cost Breakdown）**

**展示3个典型病例（3个子图）：**

**病例A：糖尿病+慢性肾病3期**
```
真实费用：¥54,200  预测费用：¥52,800 (误差2.6%)

费用构成（堆叠柱状图）：
检查费    ████████████████████ ¥18,600 (35%)
总诊疗费  ████████████████ ¥15,200 (29%)
治疗费    ████████████ ¥9,800 (19%)
床位费    ████████ ¥6,400 (12%)
其他      ███ ¥2,800 (5%)

SHAP瀑布图（右侧）：
基准预测 ¥35,000
  + 慢性肾病分期 +¥4,200
  + 初始肌酐 +¥2,800
  + 耐药菌感染 +¥3,500
  + 血常规次数 +¥1,600
  + 其他特征 +¥5,700
= 最终预测 ¥52,800
```

**病例B：无合并症，轻度感染**
**病例C：心衰+恶性肿瘤，预测误差较大**

**制作工具：** Python (matplotlib + shap)  
**建议尺寸：** 双栏宽度

---

## 📎 **二、补充材料（Supplementary Materials）**

---

### **Supplementary Figure S1: 数据分布探索**

**4个子图：**
- A. 总费用分布直方图（对数尺度）
- B. 各子项费用箱线图
- C. 合并症共现矩阵（热力图）
- D. 实验室指标相关系数矩阵

---

### **Supplementary Figure S2: 模型训练过程**

**2个子图：**
- A. 训练/验证损失曲线（随epoch变化）
- B. 学习率调整历史

---

### **Supplementary Figure S3: 错误分析**

**3个子图：**
- A. 预测误差最大的10个病例对比
- B. 不同费用水平的误差箱线图
- C. 高误差病例的特征分布雷达图

---

### **Supplementary Figure S4: 注意力权重可视化**

**展示合并症门控机制如何工作：**
- 3个病例的门控权重热力图
- 不同合并症组合对特征权重的影响

---

### **Supplementary Figure S5: 校准曲线**

**2个子图：**
- A. 总费用预测校准曲线（10分位）
- B. 各子项费用校准曲线

---

### **Supplementary Figure S6: 外部验证结果**（如果有）

**多中心数据的性能对比：**
- 各医院的R²和MAPE
- 森林图展示差异

---

### **Supplementary Table S1: 特征完整列表**

**列：** 模态 | 特征名称 | 数据类型 | 缺失率(%) | 处理方法

**行：** 91个特征的详细信息

---

### **Supplementary Table S2: 超参数配置**

**列：** 参数名称 | 默认值 | 搜索范围 | 最优值

**行：**
- 学习率
- Batch size
- Dropout rate
- 隐藏层维度
- 损失函数权重
- ...

---

### **Supplementary Table S3: 消融实验详细结果**

**列：** 实验配置 | 训练集R² | 验证集R² | 测试集R² | 测试集MAPE | 训练时间

**行：** 10+个消融实验变体

---

### **Supplementary Table S4: 各合并症组合的费用统计**

**列：** 合并症组合 | 样本量 | 总费用[中位(IQR)] | 8个子项费用[中位(IQR)]

**行：** 所有出现≥3次的组合

---

### **Supplementary Table S5: Top 50特征的SHAP值统计**

**列：** 特征名称 | 均值|SHAP| | 标准差 | 最小值 | 最大值 | 正向影响% | 负向影响%

---

### **Supplementary Table S6: 高费用患者识别性能**

**列：** 费用阈值 | 灵敏度 | 特异度 | PPV | NPV | F1-Score | AUC

**行：**
- 前5%高费用
- 前10%高费用
- 前20%高费用
- 前25%高费用

---

### **Supplementary Table S7: 与文献中其他方法的对比**

**列：** 研究 | 年份 | 样本量 | 方法 | R² | MAPE | 备注

**行：** 10+篇相关文献的性能数据

---

## 📦 **三、补充材料：代码与数据**

### **文件清单：**

```
Supplementary_Materials/
│
├── Code/
│   ├── README_code.md              # 代码使用说明
│   ├── requirements.txt            # 依赖包
│   ├── data_preprocessor.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   └── example_usage.ipynb         # Jupyter演示笔记本
│
├── Data/
│   ├── README_data.md              # 数据说明
│   ├── data_dictionary.xlsx        # 数据字典
│   ├── simulated_data.csv          # 模拟数据（100例）
│   └── feature_engineering.py      # 特征工程脚本
│
├── Model_Weights/
│   ├── best_model.pth              # 训练好的模型权重
│   ├── model_config.json           # 模型配置
│   └── training_log.txt            # 训练日志
│
├── Results/
│   ├── detailed_metrics.xlsx       # 详细评估指标
│   ├── prediction_results.csv      # 所有样本的预测结果
│   └── shap_values.npy             # SHAP值数组
│
└── Figures_HighRes/
    ├── Figure1_flowchart.tiff      # 所有正文图的高分辨率版本
    ├── Figure2_architecture.tiff   # (300 dpi, CMYK色彩空间)
    ├── Figure3_performance.tiff
    ├── ...
    └── All_SupplementaryFigures/
```

---

## 🎨 **四、图表制作规范**

### **期刊投稿要求（通用标准）：**

| 项目 | 要求 |
|------|------|
| **文件格式** | TIFF / EPS / PDF（首选）或高质量PNG |
| **分辨率** | 线条图：≥800 dpi<br>含照片：≥300 dpi |
| **色彩模式** | CMYK（印刷）或RGB（在线） |
| **字体** | Arial / Helvetica，最小6 pt |
| **线条粗细** | ≥0.5 pt |
| **图表尺寸** | 单栏：85mm宽<br>双栏：180mm宽 |

### **配色建议（色盲友好）：**

```python
# 推荐配色方案
color_palette = {
    'primary': '#2E86AB',      # 蓝色
    'secondary': '#A23B72',    # 紫色
    'accent': '#F18F01',       # 橙色
    'success': '#06A77D',      # 绿色
    'warning': '#D62246',      # 红色
    'neutral': '#6C757D'       # 灰色
}

# Seaborn调色板
import seaborn as sns
sns.set_palette("colorblind")  # 色盲友好
```

---

## 📝 **五、图表标题与图注范例**

### **Figure 1:**
> **Study flowchart and patient selection process.** CONSORT-style diagram showing the inclusion and exclusion criteria for patient enrollment. CAP: community-acquired pneumonia; ICU: intensive care unit.

### **Figure 3:**
> **Overall performance of the multi-task deep learning model for total hospitalization cost prediction.** (A) Scatter plot of predicted versus actual total costs with perfect prediction line (red dashed) and ±20% tolerance band (gray shading). (B) Distribution of prediction residuals. (C) Relative prediction error versus actual cost. (D) R² scores for individual cost components. Error bars represent 95% bootstrap confidence intervals (n=1,000 iterations).

### **Figure 5:**
> **SHAP-based feature importance analysis.** (A) Top 20 features ranked by mean absolute SHAP value indicating their global contribution to total cost prediction. (B) SHAP summary plot showing the distribution and directionality of feature effects; each dot represents one patient, with color indicating feature value (red=high, blue=low).

---

## 📄 **六、Methods部分需要的额外材料**

### **在线工具链接（如果开发了Web工具）：**
```
The cost prediction tool is publicly available at:
https://cap-cost-predictor.yourlab.edu

Source code repository:
https://github.com/yourusername/cap-cost-prediction
DOI: 10.5281/zenodo.XXXXXXX
```

### **统计分析软件声明：**
```
Statistical analyses were performed using Python 3.8 (Python Software 
Foundation) with the following packages: PyTorch 1.12 for deep learning, 
scikit-learn 1.0 for baseline models, SHAP 0.41 for interpretability 
analysis, and pandas/NumPy for data manipulation. Figures were generated 
using Matplotlib 3.4 and Seaborn 0.11. All statistical tests were 
two-sided with significance level α=0.05.
```

---

## ✅ **七、提交前检查清单**

### **图表质量检查：**
- [ ] 所有图表符合期刊分辨率要求（≥300 dpi）
- [ ] 字体大小清晰可读（≥6 pt）
- [ ] 颜色使用色盲友好调色板
- [ ] 所有轴标签包含单位
- [ ] 图例清晰且无歧义
- [ ] 图注详细说明所有缩写

### **补充材料检查：**
- [ ] 代码可运行且有README
- [ ] 模拟数据符合隐私要求
- [ ] 所有补充表格有清晰标题
- [ ] 高分辨率图片文件已压缩为ZIP

### **伦理与数据声明：**
- [ ] 伦理批件号已在Methods中说明
- [ ] 数据可用性声明已添加
- [ ] 患者知情同意流程已说明（如适用）
- [ ] 利益冲突声明已完成



### 📄 推荐投稿期刊（按难度排序）

| 期刊 | 影响因子 | 难度 | 周期 |
|------|---------|------|------|
| **JAMIA** (Journal of the American Medical Informatics Association) | 6.4 | ★★★★★ | 6-9月 |
| **Artificial Intelligence in Medicine** | 7.5 | ★★★★☆ | 4-6月 |
| **Journal of Medical Systems** | 4.9 | ★★★☆☆ | 3-5月 |
| **BMC Medical Informatics and Decision Making** | 3.9 | ★★★☆☆ | 3-4月 |
| **Computer Methods and Programs in Biomedicine** | 6.1 | ★★★★☆ | 4-6月 |
| **International Journal of Medical Informatics** | 4.9 | ★★★☆☆ | 3-5月 |

**建议策略：**
1. 先投 *BMC Med Inform Decis Mak* 或 *Int J Med Inform*（难度适中，开放获取）
2. 如被拒，根据审稿意见改进后投 *J Med Syst*
3. 如果补充了多中心验证+因果推断，可冲击 *Artif Intell Med* 或 *JAMIA*

---