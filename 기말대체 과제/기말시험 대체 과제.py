import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
import numpy as np  # NumPy import 추가
import warnings  # warnings 모듈 추가

# 경고 무시 설정
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.model_selection._split')

# 데이터 로드
try:
    data = pd.read_csv('./data/your_traffic_accident_data.csv', encoding='CP949')  # 인코딩 변경 시도
    print("데이터를 성공적으로 로드했습니다.")
except FileNotFoundError:
    print("파일을 찾을 수 없습니다. 파일 경로를 확인하세요.")
    exit()
except Exception as e:
    print(f"파일 로드 오류: {e}")
    exit()

# 데이터 확인
print(data.head())

# 컬럼 이름 수정 (특징 변수와 목표 변수 확인)
data.columns = data.columns.str.strip()  # 컬럼 이름의 공백 제거
print(f"수정된 열 이름: {data.columns}")

# 사용할 열들 설정 (data의 실제 열에 맞게 수정)
features = ['사고건수', '사망자수', '중상자수', '경상자수', '부상신고자수']  # 사고 건수와 관련된 특징 변수
target = '사고유형대분류'  # 목표 변수 설정

# 결측값 처리 (수치형 변수는 평균, 범주형 변수는 최빈값으로 처리)
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# 수치형 변수 결측값 처리
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())

# 범주형 변수 결측값 처리
data[categorical_cols] = data[categorical_cols].apply(lambda col: col.fillna(col.mode().iloc[0] if not col.mode().empty else 'Unknown'))

# 범주형 변수 변환 (Label Encoding)
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 특징(X)과 목표(y) 나누기
X = data[features]
y = data[target]

# 데이터셋을 훈련용(80%)과 테스트용(20%)으로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 표준화 (수치형 데이터만)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 교차 검증을 통해 모델 평가 (교차 검증의 n_splits 값 변경)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 교차 검증: n_splits=2로 변경
kf = StratifiedKFold(n_splits=2, random_state=42, shuffle=True)  # 최소 클래스 수에 맞춰 n_splits=2로 설정
cross_val_score_result = cross_val_score(model, X_train_scaled, y_train, cv=kf)  # 2-fold 교차 검증
print(f"교차 검증 정확도: {cross_val_score_result.mean():.4f}")

# 모델 학습
model.fit(X_train_scaled, y_train)

# 예측 수행
y_pred = model.predict(X_test_scaled)

# 모델 평가
print("모델 정확도 (Accuracy):", accuracy_score(y_test, y_pred))
print("\n혼동 행렬 (Confusion Matrix):")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 혼동 행렬 시각화
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data[target].unique(), yticklabels=data[target].unique())
plt.title("Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 분류 보고서
print("\n분류 보고서 (Classification Report):")
print(classification_report(y_test, y_pred, zero_division=1))

# 다중 클래스 ROC Curve 시각화 (One-vs-Rest 방식)
y_test_bin = label_binarize(y_test, classes=model.classes_)  # y_test를 이진화
n_classes = y_test_bin.shape[1]  # 클래스 수

fpr = dict()
tpr = dict()
roc_auc = dict()

# 각 클래스에 대해 ROC curve를 계산
for i in range(n_classes):
    if np.any(y_test_bin[:, i]):  # 클래스에 실제 양성 샘플이 있는 경우에만 ROC 계산
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], model.predict_proba(X_test_scaled)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

# 다중 클래스에 대한 ROC curve 시각화
plt.figure()
colors = ['blue', 'green', 'red', 'yellow']  # 색상은 클래스 수에 맞게 설정

for i in range(n_classes):
    if i in roc_auc:  # ROC 계산이 완료된 클래스만 시각화
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (Multiclass)')
plt.legend(loc='lower right')
plt.show()

# 하이퍼파라미터 튜닝 (예시: GridSearchCV)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# 최적 하이퍼파라미터 출력
print("최적 하이퍼파라미터:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# 최적 모델로 예측 수행
y_pred_best = best_model.predict(X_test_scaled)

# 최적 모델 평가
print("\n최적 모델 정확도 (Accuracy):", accuracy_score(y_test, y_pred_best))
