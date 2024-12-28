import os
import numpy as np
import pandas as pd
import librosa
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
warnings.filterwarnings('ignore')

# 设置数据路径
audio_path = r"CBU0521DD_stories"
csv_path = r"CBU0521DD_stories_attributes.csv"

# 读取标签数据
labels_df = pd.read_csv(csv_path)

# 特征提取函数
def extract_features(y, sr):
    features = []
    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    features.append(zcr)
    # Spectral Centroid
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features.append(spec_centroid)
    # Spectral Rolloff
    spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features.append(spec_rolloff)
    # Spectral Bandwidth
    spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features.append(spec_bandwidth)
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))
    return np.array(features)

# 准备数据集
print("准备数据集...")
features = []
labels = []

for index, row in labels_df.iterrows():
    file_path = os.path.join(audio_path, row['filename'])
    class_label = row['Story_type']
    y, sr = librosa.load(file_path, duration=300)
    feature = extract_features(y, sr)
    features.append(feature)
    labels.append(class_label)
    print(f'处理文件: {row["filename"]}')

X = np.array(features)
y = np.array(labels)

# 标签编码
print("标签编码...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 划分训练集和测试集
print("划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# 特征标准化
print("特征标准化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 特征选择
print("特征选择...")
selector = SelectFromModel(
    XGBClassifier(n_estimators=100, random_state=42),
    max_features=70,
    threshold='mean'
)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# XGBoost参数优化
print("优化XGBoost参数...")
xgb_params = {
    'n_estimators': [500],
    'max_depth': [4],
    'learning_rate': [0.01],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'min_child_weight': [5],
    'gamma': [0.2],
    'reg_alpha': [0.05],
    'reg_lambda': [0.8]
}

xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

grid_search = GridSearchCV(xgb, xgb_params, cv=5, scoring='accuracy')
grid_search.fit(X_train_selected, y_train)

print(f"最佳XGBoost参数: {grid_search.best_params_}")
best_xgb = grid_search.best_estimator_

# 定义基础模型
estimators = [
    ('xgb', best_xgb),
    ('gbc', GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.01,
        max_depth=4,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )),
    ('rf', RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced'
    )),
    ('svm', SVC(
        kernel='rbf',
        C=2.0,
        gamma='scale',
        probability=True,
        random_state=42,
        class_weight='balanced'
    ))
]

# 集成模型
voting_clf = VotingClassifier(
    estimators=estimators,
    voting='soft',
    weights=[4, 2, 2, 1]
)

print("进行交叉验证...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(voting_clf, X_train_selected, y_train, cv=skf, scoring='accuracy')
print(f'交叉验证平均准确率: {np.mean(scores) * 100:.2f}%')

print("训练最终模型...")
voting_clf.fit(X_train_selected, y_train)

print("评估模型...")
y_pred = voting_clf.predict(X_test_selected)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'测试集准确率: {test_accuracy * 100:.2f}%')

print("\n分类报告:")
print(classification_report(y_test, y_pred))

print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))