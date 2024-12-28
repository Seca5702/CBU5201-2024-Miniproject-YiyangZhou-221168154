import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def extract_features(audio, sample_rate):
    """提取音频特征"""
    try:
        # 基础特征
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate).T, axis=0)

        # 额外特征
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate).T, axis=0)
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate).T, axis=0)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)

        # 合并所有特征
        return np.concatenate([mfccs, chroma, mel, contrast, tonnetz,
                               zcr, spectral_rolloff, spectral_bandwidth,
                               spectral_centroid, rms])
    except Exception as e:
        print(f"特征提取出错: {str(e)}")
        return None
def augment_audio(audio, sample_rate):
    """音频数据增强函数"""
    augmented_audios = []

    # 添加白噪声
    noise_factor = 0.005
    noise = np.random.normal(0, 1, len(audio))
    aug_noise = audio + noise_factor * noise
    augmented_audios.append(aug_noise)

    # 时间伸缩
    stretch_rates = [0.8, 1.2]
    for rate in stretch_rates:
        aug_stretch = librosa.effects.time_stretch(audio, rate=rate)
        augmented_audios.append(aug_stretch)

    # 音高偏移
    pitch_shifts = [-2, 2]
    for steps in pitch_shifts:
        aug_pitch = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=steps)
        augmented_audios.append(aug_pitch)

    # 速度变化
    speed_factors = [0.9, 1.1]
    for speed in speed_factors:
        aug_speed = librosa.effects.time_stretch(audio, rate=1 / speed)
        augmented_audios.append(aug_speed)

    return augmented_audios
def prepare_data():
    # 读取CSV文件
    data = pd.read_csv('CBU0521DD_stories_attributes.csv')

    # 标签编码
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Story_type'])
    print(label_encoder.classes_)  # 会输出 ['Deception' 'True']

    # 处理原始数据
    features_original = []
    labels_original = []

    # 提取原始特征
    for index, row in data.iterrows():
        file_path = os.path.join('CBU0521DD_stories', row['filename'])
        print(f"处理文件: {file_path}")

        try:
            audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
            features = extract_features(audio, sample_rate)

            if features is not None:
                features_original.append(features)
                labels_original.append(row['Label'])

        except Exception as e:
            print(f"处理文件出错 {file_path}: {str(e)}")
            continue

    # 转换为numpy数组
    X_original = np.array(features_original)
    y_original = np.array(labels_original)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_original,
        y_original,
        test_size=0.2,
        random_state=42,
        stratify=y_original
    )

    # 数据增强
    features_augmented = []
    labels_augmented = []

    # 添加原始训练数据
    features_augmented.extend(X_train)
    labels_augmented.extend(y_train)
    print(f"\n开始数据增强...")
    print(f"原始训练集样本数: {len(y_train)}")
    print(f"预期增强后样本数: {len(y_train) * (1 + len(augment_audio(np.zeros(1000), 22050)))}\n")

    # 对训练集进行数据增强
    for i, label in enumerate(y_train):
        file_path = os.path.join('CBU0521DD_stories', data.iloc[i]['filename'])
        try:
            audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
            augmented_audios = augment_audio(audio, sample_rate)

            print(f"  生成 {len(augmented_audios)} 个增强样本...")
            for j, aug_audio in enumerate(augmented_audios):
                features = extract_features(aug_audio, sample_rate)
                if features is not None:
                    features_augmented.append(features)
                    labels_augmented.append(label)
                    print(f"    第 {j+1}/{len(augmented_audios)} 个增强样本特征提取完成")

        except Exception as e:
            print(f"数据增强处理文件出错 {file_path}: {str(e)}")
            continue

    # 转换增强后的训练数据
    X_train_augmented = np.array(features_augmented)
    y_train_augmented = np.array(labels_augmented)

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_augmented)
    X_test_scaled = scaler.transform(X_test)

    # 保存处理后的数据
    np.save('X_train_scaled.npy', X_train_scaled)
    np.save('X_test_scaled.npy', X_test_scaled)
    np.save('y_train_augmented.npy', y_train_augmented)
    np.save('y_test.npy', y_test)
    joblib.dump(scaler, 'feature_scaler.joblib')
    joblib.dump(label_encoder, 'label_encoder.joblib')

    return X_train_scaled, X_test_scaled, y_train_augmented, y_test, label_encoder


if __name__ == "__main__":
    prepare_data()