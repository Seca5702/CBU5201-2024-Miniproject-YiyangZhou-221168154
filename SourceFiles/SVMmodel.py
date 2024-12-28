import os
import numpy as np
import pandas as pd
import librosa
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 设置数据路径
audio_path = 'CBU0521DD_stories'  # 音频文件夹路径
csv_path = 'CBU0521DD_stories_attributes.csv'  # CSV 文件路径

# 读取标签数据
labels_df = pd.read_csv(csv_path)

# 提取特征函数
def extract_features(file_name):
    y, sr = librosa.load(file_name, duration=300)
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

# 数据增强函数
def augment_audio(y, sr):
    # 添加噪声
    noise = np.random.normal(0, 0.005, y.shape)
    y_noise = y + noise
    # 改变音调
    y_pitch = librosa.effects.pitch_shift(y, n_steps=2, sr=sr)
    # 改变速度
    y_speed = librosa.effects.time_stretch(y, rate=0.9)
    return [y_noise, y_pitch, y_speed]

# 准备数据集
features = []
labels = []

print("提取音频特征...")
for index, row in labels_df.iterrows():
    file_path = os.path.join(audio_path, row['filename'])
    class_label = row['Story_type']
    y, sr = librosa.load(file_path, duration=300)
    # 原始音频特征
    data = extract_features(file_path)
    features.append(data)
    labels.append(class_label)
    # 数据增强后的音频特征
    augmented_audios = augment_audio(y, sr)
    # 打印进度信息
    print(f'正在处理: {file_path}')
    for y_aug in augmented_audios:
        # 提取增强后的特征
        mfccs = librosa.feature.mfcc(y=y_aug, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        chroma = librosa.feature.chroma_stft(y=y_aug, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        mel = librosa.feature.melspectrogram(y=y_aug, sr=sr)
        mel_mean = np.mean(mel, axis=1)
        mel_std = np.std(mel, axis=1)
        data_aug = np.hstack([mfccs_mean, mfccs_std, chroma_mean, chroma_std, mel_mean, mel_std])
        features.append(data_aug)
        labels.append(class_label)

# 转换为数组
X = np.array(features)
y = np.array(labels)

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42)


# 使用网格搜索优化 SVM 模型
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5)
grid.fit(X_train, y_train)

# 最佳模型
best_clf = grid.best_estimator_

'''
# 在测试集上预测
y_pred = best_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'最佳参数: {grid.best_params_}')
print(f'模型准确率: {accuracy * 100:.2f}%')
print("\n分类报告:")
print(classification_report(y_test, y_pred))
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

'''

# 将网格搜索部分替换为直接使用最佳参数
print("训练SVM模型...")
svm = SVC(
    C=10,
    gamma='scale',
    kernel='rbf',
    probability=True,
    random_state=42,
    max_iter=10000,
    cache_size=2000
)

# 直接训练模型
svm.fit(X_train, y_train)

# 在测试集上预测
# y_pred = best_clf.predict(X_test)
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# print(f'最佳参数: {grid.best_params_}')
print(f'模型准确率: {accuracy * 100:.2f}%')
print("\n分类报告:")
print(classification_report(y_test, y_pred))
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))