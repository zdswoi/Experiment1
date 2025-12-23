import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier  # 新增DT
from sklearn.ensemble import RandomForestClassifier  # 新增RF
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit, cross_val_score
from imblearn.over_sampling import SMOTE
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cdist
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.ticker import MultipleLocator
import math


# 设置matplotlib后端（根据实际环境，若TkAgg有问题可换，如'QtAgg'等 ）
# 线粗4，边框5，刻度线9，刻度线与标签间距9，大标签40 32xy轴，去掉图例边框

import matplotlib
matplotlib.use('TkAgg')

# 在导入库后添加全局字体设置
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "bold"  # 全局字体粗体
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 5
plt.rcParams['legend.fontsize'] = 26  # 图例标签字体大小与标题一致

# 加载Excel数据（请确保文件路径正确 ）
file_path = f"E:/shiyan1/荧光/1111/新建 Microsoft Excel 工作表.xlsx"
sheet = pd.read_excel(file_path, sheet_name="Sheet4")

# 提取数据
data = sheet.iloc[:, 1:].values.T  # 转置使样本为行，特征为列
new_data = sheet.iloc[:, 11:12].values.T  # 新数据用于进行预测的数据
# print(new_data)

# 提取并处理标签
class_labels = sheet.columns[1:]  # 第一行是类别标签
labels = [math.trunc(float(label)) for label in class_labels]
labels = np.array(labels)  # 转换为numpy数组

class_names = ["OTC", "TC", "CTC", "DOX"]

unique_labels = np.unique(labels)
# new-data-标签
class_labels_prediction = sheet.columns[11:12]
labels_prediction = [math.trunc(float(label)) for label in class_labels_prediction]
labels_prediction = np.array(labels_prediction)
prediction = np.unique(labels_prediction)
prediction = class_names[prediction[0]-1]
# print(prediction)

# 生成sample_labels：同一类别内按序号递增（如OTC1、OTC2，TC1、TC2...）
# 1. 统计每个类别的样本数量，用于计数
class_counts = {cls: 0 for cls in class_names}  # 初始化计数器
# 建立数字标签到类别名称的映射（如1→OTC，2→TC等）
label_to_name = {unique_labels[i]: class_names[i] for i in range(len(unique_labels))}
# 原始样本的实际类别名称列表
sample_true_names = [label_to_name[label] for label in labels]
sample_labels = []
for name in sample_true_names:
    class_counts[name] += 1  # 对应类别计数+1
    sample_labels.append(f"{name}{class_counts[name]}")  # 生成标签（如OTC1、TC2）

# 数据归一化
scaler = StandardScaler()
data_norm = scaler.fit_transform(data)
new_data_norm = scaler.transform(new_data)

# PCA降维
pca = PCA(n_components=2)
score = pca.fit_transform(data_norm)
new_data_pca = pca.transform(new_data_norm) # new_data_pca是预测的数据
latent = pca.explained_variance_ratio_ * 100


# 置信椭圆函数
# 当n_std=1.96时，对应95% 置信区间（正态分布中，95% 的概率落在均值 ±1.96σ 范围内）；
# 当n_std=3.0时（代码默认值），对应约 99.7% 置信区间（正态分布中，99.7% 的概率落在均值 ±3σ 范围
def confidence_ellipse(x, y, ax, n_std=3.0, edgecolor='k',facecolor='k', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      edgecolor=edgecolor, facecolor=facecolor,** kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y)
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
custom_colors = [
    (0.992,0.784,0.592),  # 橙色 "OTC"
    (0.616,0.843,0.616),     # 绿色 "TC"
    (0.761,0.698,0.839),  # 紫色 "CTC"
    (0.604,0.733,0.953),   # 蓝色，之前是黄色 "DOX"
    (0.996, 0.722, 0.976)   # 粉色，之前是蓝色  第5号 其他
    ]
# (0.996, 0.722, 0.976)  # 粉色，之前是蓝色  第5号 其他
# 转换为matplotlib可用的颜色格式
custom_cmap = plt.cm.colors.ListedColormap(custom_colors)
color_map = {label: custom_cmap(i) for i, label in enumerate(unique_labels)}

# 1. 绘制PCA散点图
plt.figure(figsize=(10, 8))
ax = plt.gca()
# 定义颜色映射，让类别和颜色稳定对应，这里用rainbow也可换其他（如tab10等 ）
# colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

# 计算并存储各类别的中心点位置
class_centers = {}
for i, class_id in enumerate(unique_labels):
    mask = (labels == class_id)
    x = score[mask, 0]
    y = score[mask, 1]
    class_centers[class_id] = (np.mean(x), np.mean(y))

    plt.scatter(x, y, color=color_map[class_id], s=150, alpha=1.0,
                label=f'{class_names[i]} (n={len(x)})', edgecolor='k')

    if len(x) > 1:
        confidence_ellipse(x, y, ax, n_std=2.65,
                           edgecolor=color_map[class_id],
                           facecolor=color_map[class_id],
                           linewidth=1,
                           linestyle='-',
                           alpha=0.65)

# 设置标题和轴标签字体大小
plt.title('PCA classification plot', fontsize=30, fontweight='bold')
plt.xlabel(f'PC1 ({latent[0]:.1f}%)', fontsize=30, fontweight='bold')
plt.ylabel(f'PC2 ({latent[1]:.1f}%)', fontsize=30, fontweight='bold')
ax.minorticks_on()
ax.tick_params(axis='both', which='major', labelsize=28, width=5, length=8, pad=8)
# 次刻度：更密集的辅助刻度线
ax.xaxis.set_minor_locator(MultipleLocator(5))   # x 轴次刻度间隔 2
ax.yaxis.set_minor_locator(MultipleLocator(2.5)) # y 轴次刻度间隔 0.5
ax.tick_params(axis='both', which='minor', width=5, length=5)
# 坐标轴刻度标签粗体
plt.xticks(fontsize=28, fontweight='bold')
plt.yticks(fontsize=28, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()


# 2. K-Means聚类 (修正标签对齐)
kmeans = KMeans(n_clusters=17, random_state=0, n_init=10).fit(score)
centers = kmeans.cluster_centers_

# 建立聚类中心与真实类别的映射关系
dist_matrix = cdist(centers, list(class_centers.values()))
cluster_to_class = np.argmin(dist_matrix, axis=1)

# 映射K-Means预测结果到真实类别
kmeans_labels_mapped = np.array([cluster_to_class[label] for label in kmeans.labels_])

plt.figure(figsize=(10, 8))
ax = plt.gca()

for i in range(16):
    mask = (kmeans.labels_ == i)
    x = score[mask, 0]
    y = score[mask, 1]

    # 使用映射后的真实类别标签
    true_class_idx = cluster_to_class[i]
    plt.scatter(x, y, color=color_map[unique_labels[true_class_idx]], s=150, alpha=1.0,edgecolor='k',
                label=f'{class_names[true_class_idx]} (Cluster {i + 1})')
    # plt.scatter(centers[i, 0], centers[i, 1], c='black', marker='x', s=200)
    plt.scatter(x, y, color=color_map[class_id], s=125, alpha=0.7,
                label=f'{class_names[i]} (n={len(x)})', edgecolor='k')

    if len(x) > 1:
        confidence_ellipse(x, y, ax, n_std=3.0,
                           edgecolor=color_map[unique_labels[true_class_idx]],
                           facecolor=color_map[unique_labels[true_class_idx]],
                           linestyle='-',
                           linewidth=1,
                           alpha=0.65)

# 对新数据进行K-Means预测并映射标签
kmeans_pred = kmeans.predict(new_data_pca)
kmeans_pred_mapped = [class_names[cluster_to_class[p]] for p in kmeans_pred]
#
# # 添加新数据点
# plt.scatter(new_data_pca[:, 0], new_data_pca[:, 1], c='black', s=200,
#             marker='*', label=f'New Data (Pred: {kmeans_pred_mapped[0]})', edgecolor='k')
plt.xlabel(f'PC1 ({latent[0]:.1f}%)', fontsize=30, fontweight='bold')
plt.ylabel(f'PC2 ({latent[1]:.1f}%)', fontsize=30, fontweight='bold')
plt.title('K-Means classification plot', fontsize=30, fontweight='bold',pad=10)
ax.minorticks_on()
ax.tick_params(axis='both', which='major', labelsize=28, width=5, length=8, pad=8)
# 次刻度：更密集的辅助刻度线
ax.xaxis.set_minor_locator(MultipleLocator(5))   # x 轴次刻度间隔 2
ax.yaxis.set_minor_locator(MultipleLocator(1.25)) # y 轴次刻度间隔 0.5
ax.tick_params(axis='both', which='minor', width=5, length=5)
# 坐标轴刻度标签粗体
plt.xticks(fontsize=28, fontweight='bold')
plt.yticks(fontsize=28, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 3. SVM分类和决策边界图（改进部分）
# 改进1：使用分层抽样划分数据集，保证类别比例
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)  # 测试集比例调整为30%
for train_idx, test_idx in sss.split(score, labels):
    X_train, X_test = score[train_idx], score[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

# 改进2：处理数据不平衡（动态调整SMOTE参数，避免样本量不足错误）
# 计算最小类别样本数
unique, counts = np.unique(y_train, return_counts=True)
min_samples = np.min(counts)
# 确保近邻数小于最小类别样本数
n_neighbors = min(5, min_samples - 1)  # 最多5个近邻，且不超过最小样本数-1

# 只有当最小类别样本数大于1时才使用SMOTE（避免样本量过少）
if min_samples > 1:
    smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"SMOTE processed training set samples: {X_train_resampled.shape[0]} (original: {X_train.shape[0]})")
else:
    # 样本量过少时不使用SMOTE
    X_train_resampled, y_train_resampled = X_train, y_train
    print(f"Insufficient sample size, SMOTE not used (minimum class samples: {min_samples})")

# 改进3：根据样本量动态调整交叉验证折数
# 计算重采样后最小类别样本数
unique_resampled, counts_resampled = np.unique(y_train_resampled, return_counts=True)
min_samples_resampled = np.min(counts_resampled)
# 交叉验证折数不能超过最小类别样本数，最多5折
cv_folds = min(5, min_samples_resampled)
print(f"Cross-validation folds adjusted based on sample size: {cv_folds}")

# 改进4：网格搜索优化SVM参数
param_grid = {
    'C': [0.1, 1, 10, 100],  # 正则化参数
    'gamma': [0.001, 0.01, 0.1, 1],  # 核函数参数
    'kernel': ['rbf', 'linear']  # 尝试不同核函数
}
grid_search = GridSearchCV(
    SVC(class_weight='balanced'),  # 自动平衡类别权重
    param_grid,
    cv=cv_folds,  # 使用动态调整的折数
    scoring='accuracy',
    n_jobs=-1  # 并行计算
)
grid_search.fit(X_train_resampled, y_train_resampled)
print(f"SVM Optimal parameters: {grid_search.best_params_}")
print(f"SVM Cross-validation accuracy: {grid_search.best_score_:.2f}")

# 使用优化后的模型
svm = grid_search.best_estimator_

# 预测与评估
y_pred = svm.predict(X_test)
print(f"SVM Test set accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 额外测试集（保持原始逻辑）
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(score, labels, test_size=0.9, random_state=42)
y_pred_1 = svm.predict(X_test_1)
# print(f"SVM大测试集准确率: {accuracy_score(y_test_1, y_pred_1):.2f}")

# 交叉验证评估稳定性（使用调整后的折数）
cv_scores = cross_val_score(svm, score, labels, cv=cv_folds)
# print(f"SVM交叉验证准确率: {cv_scores.mean():.4f}（±{cv_scores.std():.4f}）")

# 对新数据进行SVM预测
svm_pred = svm.predict(new_data_pca)
svm_pred_mapped = [class_names[int(p) - 1] for p in svm_pred]

# 绘制SVM决策边界（仅用前两维）
plt.figure(figsize=(10, 8))
ax = plt.gca()

# 创建网格（仅用前两维）
x_min, x_max = score[:, 0].min() - 1, score[:, 0].max() + 1
y_min, y_max = score[:, 1].min() - 1, score[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# 构建虚拟数据（仅用前两维，其他维度取均值）
if pca.n_components_ > 2:
    # 其他维度填充均值
    mean_other = np.mean(score[:, 2:], axis=0) if score.shape[1] > 2 else []
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points = np.hstack([grid_points, np.tile(mean_other, (grid_points.shape[0], 1))])
else:
    grid_points = np.c_[xx.ravel(), yy.ravel()]

# 预测每个网格点
Z = svm.predict(grid_points)
Z = Z.reshape(xx.shape)

# 绘制决策边界和区域
plt.contourf(xx, yy, Z, alpha=0.65, cmap=custom_cmap)
plt.scatter(score[:, 0], score[:, 1], c=labels, cmap=custom_cmap, alpha=1.0,
            s=150, edgecolor='k', label='Training Data')

plt.xlabel(f'PC1 ({latent[0]:.1f}%)', fontsize=30, fontweight='bold')
plt.ylabel(f'PC2 ({latent[1]:.1f}%)', fontsize=30, fontweight='bold')
plt.title('SVM classification plot', fontsize=30, fontweight='bold',pad=10)
ax.minorticks_on()
ax.tick_params(axis='both', which='major', labelsize=28, width=5, length=8, pad=8)
# 次刻度：更密集的辅助刻度线
ax.xaxis.set_minor_locator(MultipleLocator(5))   # x 轴次刻度间隔 2
ax.yaxis.set_minor_locator(MultipleLocator(1)) # y 轴次刻度间隔 0.5
ax.tick_params(axis='both', which='minor', width=5, length=5)
plt.xticks(fontsize=28, fontweight='bold')
plt.yticks(fontsize=28, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0)

plt.tight_layout()


# 4. 分层聚类HCA
linked = linkage(score, 'average')
num_clusters = 4
cluster_labels = fcluster(linked, num_clusters, criterion='maxclust')

# 建立HCA聚类与真实类别的映射
hca_centers = []
for i in range(1, num_clusters + 1):
    mask = (cluster_labels == i)
    hca_centers.append((np.mean(score[mask, 0]), np.mean(score[mask, 1])))

dist_matrix_hca = cdist(hca_centers, list(class_centers.values()))
hca_to_class = np.argmin(dist_matrix_hca, axis=1)

# 映射HCA预测结果到真实类别
hca_labels_mapped = np.array([hca_to_class[label - 1] for label in cluster_labels])

# 对新数据进行HCA预测
combined_data = np.vstack([score, new_data_pca])
linked_combined = linkage(combined_data, 'average')
cluster_labels_combined = fcluster(linked_combined, num_clusters, criterion='maxclust')
hca_pred = cluster_labels_combined[-len(new_data_pca):]
hca_pred_mapped = [class_names[hca_to_class[p - 1]] for p in hca_pred]

# 绘制HCA树状图
plt.figure(figsize=(12, 6))
dendrogram(linked,  # 仅使用原始数据的 linkage
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True,
           color_threshold=10,
           labels=sample_labels)  # 使用按类别单独计数的标签（OTC1、TC2等）

# 获取当前轴对象
ax = plt.gca()
# 方法2（备选）：如果方法1效果不佳，可尝试修改集合对象  3感觉还不错可以
for collection in ax.collections:
    collection.set_linewidth(4)

# 扩大X轴和Y轴范围（核心修改）
# 获取当前轴范围
current_ylim = ax.get_ylim()

# 在现有范围基础上扩大20%（可根据需要调整比例）
y_expand = 0.4

# 设置新的轴范围（左右各扩展x_expand，上下各扩展y_expand）
ax.set_ylim(current_ylim[0]-y_expand, current_ylim[1])

# 添加聚类分割线
plt.axhline(y=10, color='r', linestyle='--', linewidth=4)

# 添加图例和标签
plt.ylabel('Distance', fontsize=30, fontweight='bold')
plt.title('Hierarchical Clustering Dendrogram with Class-based Labels', fontsize=30, fontweight='bold',pad=10)

ax.minorticks_on()
ax.tick_params(axis='y', which='major', labelsize=28, width=5, length=8, pad=8)
# 次刻度：更密集的辅助刻度线
ax.yaxis.set_minor_locator(MultipleLocator(5)) # y 轴次刻度间隔 0.5
ax.tick_params(axis='y', which='minor', width=5, length=5)
# 关闭x轴次刻度（关键步骤）
ax.xaxis.set_minor_locator(plt.NullLocator())  # 移除x轴次刻度定位器

# 坐标轴刻度标签粗体
plt.xticks(rotation=90, fontsize=22, fontweight='bold')
plt.yticks(fontsize=28, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 5. 原有混淆矩阵（SVM、K-Means、HCA）
original_labels_mapped = np.array([np.where(unique_labels == label)[0][0] for label in labels])
hca_labels_idx = np.array([class_names.index(class_names[i]) for i in hca_labels_mapped])

plt.figure(figsize=(18, 5))

# 5. 原有混淆矩阵（修改部分）
# SVM混淆矩阵
plt.subplot(131)
svm_cm = confusion_matrix(y_test_1, y_pred_1)
svm_disp = ConfusionMatrixDisplay(confusion_matrix=svm_cm, display_labels=class_names)
# svm_disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), colorbar=False)
svm_disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), colorbar=False)
plt.title(f'SVM Confusion Matrix', fontsize=28, fontweight='bold')
# 设置XY轴标题
plt.xlabel('Prediction', fontsize=26, fontweight='bold')
plt.ylabel('Target', fontsize=26, fontweight='bold')
plt.gca().tick_params(axis='both', labelsize=26, width=4, length=0, pad=9)
plt.gca().set_xticklabels(class_names, rotation=90, ha='center', fontsize=22, fontweight='bold')  # 标签大小与标题一致
plt.gca().set_yticklabels(class_names, fontsize=22, fontweight='bold')  # 标签大小与标题一致

# 设置混淆矩阵中数值的字体大小和粗细
for text in svm_disp.text_.flatten():
    text.set_fontsize(22)
    text.set_weight('bold')


# K-Means混淆矩阵
plt.subplot(132)
kmeans_cm = confusion_matrix(original_labels_mapped, kmeans_labels_mapped)
kmeans_disp = ConfusionMatrixDisplay(confusion_matrix=kmeans_cm, display_labels=class_names)
# kmeans_disp.plot(cmap=plt.cm.Greens, ax=plt.gca(), colorbar=False)
kmeans_disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), colorbar=False)
kmeans_acc = accuracy_score(original_labels_mapped, kmeans_labels_mapped)
plt.title(f'K-means Confusion Matrix', fontsize=28, fontweight='bold')
# 设置XY轴标题
plt.xlabel('Prediction', fontsize=26, fontweight='bold')
plt.ylabel('Target', fontsize=26, fontweight='bold')
plt.gca().tick_params(axis='both', labelsize=28, width=4, length=0, pad=9)
plt.gca().set_xticklabels(class_names, rotation=90, ha='center', fontsize=22, fontweight='bold')  # 标签大小与标题一致
plt.gca().set_yticklabels(class_names, fontsize=22, fontweight='bold')  # 标签大小与标题一致
# 设置混淆矩阵中数值的字体大小和粗细
for text in kmeans_disp.text_.flatten():
    text.set_fontsize(22)
    text.set_weight('bold')

# HCA混淆矩阵
plt.subplot(133)
hca_cm = confusion_matrix(original_labels_mapped, hca_labels_idx)
hca_disp = ConfusionMatrixDisplay(confusion_matrix=hca_cm, display_labels=class_names)
# hca_disp.plot(cmap=plt.cm.Oranges, ax=plt.gca(), colorbar=False)
hca_disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), colorbar=False)
hca_acc = accuracy_score(original_labels_mapped, hca_labels_idx)
plt.title(f'HCA Confusion Matrix', fontsize=28, fontweight='bold')
# 设置XY轴标题
plt.xlabel('Prediction', fontsize=26, fontweight='bold')
plt.ylabel('Target', fontsize=26, fontweight='bold')
plt.gca().tick_params(axis='both', labelsize=28, width=4, length=0, pad=9)
plt.gca().set_xticklabels(class_names, rotation=90, ha='center', fontsize=22, fontweight='bold')  # 标签大小与标题一致
plt.gca().set_yticklabels(class_names, fontsize=22, fontweight='bold')  # 标签大小与标题一致
# 设置混淆矩阵中数值的字体大小和粗细
for text in hca_disp.text_.flatten():
    text.set_fontsize(22)
    text.set_weight('bold')

plt.tight_layout()

# 6. 新增DT和RF算法及混淆矩阵
# 决策树(DT)分类
dt_param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(class_weight='balanced', random_state=42),
    dt_param_grid,
    cv=cv_folds,
    scoring='accuracy',
    n_jobs=-1
)
dt_grid.fit(X_train_resampled, y_train_resampled)

print(f"DT Optimal parameters: {dt_grid.best_params_}")
print(f"DT Cross-validation accuracy: {dt_grid.best_score_:.4f}")
dt = dt_grid.best_estimator_
dt_pred = dt.predict(X_test_1)
dt_acc = accuracy_score(y_test_1, dt_pred)
print(f"DT Test set accuracy: {dt_acc:.2f}")

# 随机森林(RF)分类
# rf_param_grid = {
#     'n_estimators': [10, 20, 30, 50, 70, 100, 150, 200, 300],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2],
#     'class_weight': ['balanced', 'balanced_subsample']
# }
# 新增：提前执行RF网格搜索，获取全局统一的最优参数
rf_param_grid = {
    'n_estimators': [10, 20, 30, 50, 70, 100, 150, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', 'balanced_subsample']
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=cv_folds,  # 复用动态调整的交叉验证折数
    scoring='accuracy',
    n_jobs=-1
)
rf_grid.fit(X_train_resampled, y_train_resampled)  # 用重采样后的数据训练
print(f"RF Optimal parameters: {rf_grid.best_params_}")
print(f"RF Cross-validation accuracy: {rf_grid.best_score_:.4f}")

# 基于最优参数训练RF模型（用于混淆矩阵）
rf = rf_grid.best_estimator_  # 直接调用网格搜索得到的最优模型
rf_pred = rf.predict(X_test_1)
rf_acc = accuracy_score(y_test_1, rf_pred)
print(f"RF Test set accuracy: {rf_acc:.2f}")

# 对新数据进行预测
dt_pred_new = dt.predict(new_data_pca)
dt_pred_mapped = [class_names[int(p) - 1] for p in dt_pred_new]

rf_pred_new = rf.predict(new_data_pca)
rf_pred_mapped = [class_names[int(p) - 1] for p in rf_pred_new]

# 新增DT和RF混淆矩阵画布
plt.figure(figsize=(12, 5))

# DT混淆矩阵
plt.subplot(121)
dt_cm = confusion_matrix(y_test_1, dt_pred)
dt_disp = ConfusionMatrixDisplay(confusion_matrix=dt_cm, display_labels=class_names)
# dt_disp.plot(cmap=plt.cm.Purples, ax=plt.gca(), colorbar=False)
dt_disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), colorbar=False)
plt.title(f'DT Confusion Matrix', fontsize=28, fontweight='bold')
# 设置XY轴标题
plt.xlabel('Prediction', fontsize=26, fontweight='bold')
plt.ylabel('Target', fontsize=26, fontweight='bold')
plt.gca().tick_params(axis='both', labelsize=28, width=4, length=0, pad=9)
plt.gca().set_xticklabels(class_names, rotation=90, ha='center', fontsize=22, fontweight='bold')  # 标签大小与标题一致
plt.gca().set_yticklabels(class_names, fontsize=22, fontweight='bold')  # 标签大小与标题一致
# 设置混淆矩阵中数值的字体大小和粗细
for text in dt_disp.text_.flatten():
    text.set_fontsize(22)
    text.set_weight('bold')

# RF混淆矩阵
plt.subplot(122)
rf_cm = confusion_matrix(y_test_1, rf_pred)
rf_disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=class_names)
rf_disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), colorbar=False)
plt.title(f'RF Confusion Matrix', fontsize=28, fontweight='bold')
# 设置XY轴标题
plt.xlabel('Prediction', fontsize=26, fontweight='bold')
plt.ylabel('Target', fontsize=26, fontweight='bold')
plt.gca().tick_params(axis='both', labelsize=28, width=4, length=0, pad=9)
plt.gca().set_xticklabels(class_names, rotation=90, ha='center', fontsize=22, fontweight='bold')  # 标签大小与标题一致
plt.gca().set_yticklabels(class_names, fontsize=22, fontweight='bold')  # 标签大小与标题一致
# 设置混淆矩阵中数值的字体大小和粗细
for text in rf_disp.text_.flatten():
    text.set_fontsize(22)
    text.set_weight('bold')

plt.tight_layout()

# 打印所有预测结果（包含DT和RF）
print("\n=== Final Prediction Results ===")
print(f"Actual Category: ['{prediction}']")
print(f"SVM Prediction: {svm_pred_mapped}")
print(f"K-Means Prediction: {kmeans_pred_mapped}")
print(f"HCA Prediction: {hca_pred_mapped}")
print(f"Decision Tree Prediction: {dt_pred_mapped}")
print(f"Random Forest Prediction: {rf_pred_mapped}")


def plot_rf_tree_iteration_performance(X, y, class_names, cv_folds, rf_grid, X_train_resampled, y_train_resampled):
    # 1. 复用RF网格搜索中定义的n_estimators候选列表，避免参数差异
    n_estimators_list = rf_grid.param_grid['n_estimators']  # 直接从网格搜索对象获取树数量范围
    train_accuracies = []  # 训练集准确率列表
    cv_accuracies = []     # 交叉验证准确率列表
    std_errors = []        # 交叉验证标准差（用于误差线）

    # 2. 遍历不同树数量，训练模型并评估（参数与网格搜索保持一致）
    for n_estimators in n_estimators_list:
        # 构建RF模型：仅迭代n_estimators，其余参数固定为网格搜索最优值
        rf_temp = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=rf_grid.best_params_['max_depth'],
            min_samples_split=rf_grid.best_params_['min_samples_split'],
            min_samples_leaf=rf_grid.best_params_['min_samples_leaf'],
            class_weight=rf_grid.best_params_['class_weight'],
            random_state=42,  # 固定随机种子，确保结果可复现
            n_jobs=-1         # 并行计算加速
        )

        # 用与网格搜索相同的重采样数据训练（数据一致性保障）
        rf_temp.fit(X_train_resampled, y_train_resampled)
        # 计算训练集准确率（基于重采样数据）
        train_acc = rf_temp.score(X_train_resampled, y_train_resampled)
        train_accuracies.append(train_acc)

        # 用与网格搜索相同的交叉验证折数评估（评估标准一致性保障）
        cv_scores = cross_val_score(rf_temp, X, y, cv=cv_folds, scoring='accuracy', n_jobs=-1)
        cv_accuracies.append(cv_scores.mean())  # 交叉验证平均准确率
        std_errors.append(cv_scores.std())      # 交叉验证标准差

    # 3. 确定最优树数量（与网格搜索结果一致）
    # 原代码：仅从迭代图准确率选最优
    # best_cv_idx = np.argmax(cv_accuracies)
    # best_n_estimators = n_estimators_list[best_cv_idx]

    # 新代码：优先采用网格搜索的最优树数量，确保一致性
    grid_best_n = rf_grid.best_params_['n_estimators']
    # 检查网格搜索的最优树数量是否在迭代列表中
    if grid_best_n in n_estimators_list:
        best_n_estimators = grid_best_n
        # 找到对应网格搜索最优树数量的准确率
        best_cv_idx = n_estimators_list.index(grid_best_n)
        best_cv_accuracy = cv_accuracies[best_cv_idx]
    else:
        # 极端情况：网格搜索的树数量不在迭代列表，再用迭代图最优值
        best_cv_idx = np.argmax(cv_accuracies)
        best_n_estimators = n_estimators_list[best_cv_idx]
        best_cv_accuracy = cv_accuracies[best_cv_idx]
    # 验证：确保与网格搜索的最优n_estimators一致（可选，用于调试）
    assert best_n_estimators == rf_grid.best_params_['n_estimators'], \
        f"最优树数量不一致！迭代图：{best_n_estimators}，网格搜索：{rf_grid.best_params_['n_estimators']}"

    # 4. 绘制树数量-准确率关系图（保持原美化风格）
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # 绘制训练准确率曲线
    plt.plot(n_estimators_list, train_accuracies,
             label='Training Accuracy',
             marker='o', markersize=10,  # 圆形标记，大小10
             linewidth=4,                # 线宽4（匹配原图表风格）
             color=custom_colors[0])     # 复用自定义橙色（OTC类别色）

    # 绘制交叉验证准确率曲线（带误差线）
    plt.errorbar(n_estimators_list, cv_accuracies,
                 yerr=std_errors,        # 误差线：交叉验证标准差
                 fmt='-s',               # 方形标记+实线
                 markersize=10,          # 标记大小10
                 linewidth=4,            # 线宽4
                 color=custom_colors[3], # 复用自定义蓝色（DOX类别色）
                 ecolor='gray',          # 误差线颜色：灰色
                 capsize=5,              # 误差线帽宽5
                 label='CV Accuracy')

    # 标记最优树数量（红色圆点，突出显示）
    plt.scatter(best_n_estimators, best_cv_accuracy,
                color='red',      # 红色标记
                s=200,           # 标记大小200（大于曲线标记）
                zorder=5,        # 层级5（确保在曲线之上）
                edgecolor='black',# 黑色边框，增强辨识度
                label=f'Best: {best_n_estimators} trees\nAccuracy: {best_cv_accuracy:.4f}')

    # 5. 图表美化（与其他图保持风格统一）
    plt.title('Random Forest: Number of Trees vs Accuracy',
              fontsize=30, fontweight='bold', pad=10)  # 标题字体30号粗体
    plt.xlabel('Number of Trees', fontsize=28, fontweight='bold')  # X轴标签28号粗体
    plt.ylabel('Accuracy', fontsize=28, fontweight='bold')          # Y轴标签28号粗体

    # 刻度设置
    plt.xticks(n_estimators_list, fontsize=20, fontweight='bold')  # X轴刻度：树数量列表，20号粗体
    plt.yticks(np.arange(0, 1.5, 0.2), fontsize=20, fontweight='bold')  # Y轴0.5-1.0，间隔0.1
    plt.ylim(0, 1.5)  # Y轴范围：0.5-1.05，避免准确率超出视野

    # 网格与图例
    plt.grid(True, linestyle='--', alpha=0.6)  # 虚线网格，透明度0.6
    plt.legend(loc='best', frameon=False, fontsize=22)  # 图例22号粗体，无边框

    # 坐标轴细节（与其他图统一）
    ax.minorticks_on()  # 显示次刻度
    ax.tick_params(axis='both', which='major', width=5, length=8, pad=8)  # 主刻度线宽5，长8
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))  # Y轴次刻度间隔0.05
    ax.tick_params(axis='y', which='minor', width=5, length=5)  # 次刻度线宽5，长5
    ax.xaxis.set_minor_locator(plt.NullLocator())  # 关闭X轴次刻度（避免树数量刻度拥挤）

    plt.tight_layout()  # 自动调整布局，防止标签截断
    return plt  # 返回绘图对象，支持后续保存或显示



# 前提：rf_grid（RF网格搜索对象）、X_train_resampled/y_train_resampled（重采样数据）已定义
# 调用函数，传入所有必要参数
rf_tree_plot = plot_rf_tree_iteration_performance(
    X=score,  # PCA降维后的数据（用于交叉验证评估）
    y=labels,  # 原始标签
    class_names=class_names,  # 类别名称列表（如["OTC", "TC", "CTC", "DOX"]）
    cv_folds=cv_folds,  # 动态调整的交叉验证折数
    rf_grid=rf_grid,  # RF网格搜索对象（关键：提供最优参数和树数量范围）
    X_train_resampled=X_train_resampled,  # 重采样后的训练数据
    y_train_resampled=y_train_resampled   # 重采样后的训练标签
)
rf_tree_plot.show()  # 显示图表

plt.show()
