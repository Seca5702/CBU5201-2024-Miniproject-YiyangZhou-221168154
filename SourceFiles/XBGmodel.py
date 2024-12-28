import numpy as np
import warnings
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
warnings.filterwarnings('ignore')

print("加载预处理后的数据...")
X_train = np.load('X_train_scaled.npy')
X_test = np.load('X_test_scaled.npy')
y_train = np.load('y_train_augmented.npy')
y_test = np.load('y_test.npy')

print("特征选择...")
selector = SelectFromModel(
    XGBClassifier(n_estimators=100, random_state=42),
    max_features=50,  # 增加特征数量
    threshold='mean'  # 使用平均值作为阈值
)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

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

# 简化集成模型，去掉AdaBoost
voting_clf = VotingClassifier(
    estimators=estimators,
    voting='soft',
    weights=[3, 2, 2, 1]  # 调整权重
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