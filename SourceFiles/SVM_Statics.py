import os
import numpy as np
import pandas as pd
import librosa
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 设置数据路径
audio_path = "CBU0521DD_stories"
csv_path = "CBU0521DD_stories_attributes.csv"

# 读取标签数据
labels_df = pd.read_csv(csv_path)

def extract_features(file_name):
    y, sr = librosa.load(file_name, duration=300)
    return extract_features_from_array(y, sr)

def extract_features_from_array(y, sr):
    # 提取 MFCC 均值和标准差
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    # 提取 Chroma 特征
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    # 提取 Mel 频谱特征
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel, axis=1)
    mel_std = np.std(mel, axis=1)
    # 合并所有特征
    features = np.hstack([mfccs_mean, mfccs_std, chroma_mean, chroma_std, mel_mean, mel_std])
    return features

def augment_audio(y, sr):
    # 使用固定的sin波作为噪声
    t = np.arange(len(y))
    noise = 0.005 * np.sin(2 * np.pi * t / 100)  # 固定的sin波噪声
    y_noise = y + noise
    # 改变音调
    y_pitch = librosa.effects.pitch_shift(y, n_steps=2, sr=sr)
    # 改变速度
    y_speed = librosa.effects.time_stretch(y, rate=0.9)
    return [y_noise, y_pitch, y_speed]

# 准备数据集
print("开始处理原始数据...")
features = []
labels = []

# 首先处理原始数据
for index, row in labels_df.iterrows():
    file_path = os.path.join(audio_path, row['filename'])
    features.append(extract_features(file_path))
    labels.append(row['Story_type'])
    print(f'处理原始文件: {row["filename"]}')

# 然后处理增强数据
print("\n开始数据增强...")
for index, row in labels_df.iterrows():
    file_path = os.path.join(audio_path, row['filename'])
    y, sr = librosa.load(file_path, duration=300)
    augmented_audios = augment_audio(y, sr)
    for y_aug in augmented_audios:
        features.append(extract_features_from_array(y_aug, sr))
        labels.append(row['Story_type'])
    print(f'处理增强数据: {row["filename"]}')

# 转换为数组
X = np.array(features)
y = np.array(labels)

# 特征标准化
print("\n特征标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
print("划分数据集...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42, stratify=y
)

# 使用最佳参数训练SVM模型
print("训练SVM模型...")
svm = SVC(
    C=10,
    gamma='scale',
    kernel='rbf',
    probability=True,
    random_state=42,
    max_iter=10000,
    cache_size=2000,
    class_weight='balanced'
)

# 训练模型
svm.fit(X_train, y_train)

# 预测和评估
print("\n模型评估...")
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'模型准确率: {accuracy * 100:.2f}%')
print("\n分类报告:")
print(classification_report(y_test, y_pred))
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))