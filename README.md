# CNN-FRD: CNN 기반 가짜 리뷰 탐지 모델

리뷰 텍스트의 의미 표현(CNN)에 **언어적·리뷰·리뷰어 속성**을 결합해 가짜 리뷰를 탐지하는 모델입니다.
경희대학교 빅데이터응용학과 석사 학위논문 연구 코드입니다.

- **결과**: Yelp 리뷰 데이터에서 비교 모델 평균 대비 **Accuracy 0.77%, F1-Score 18.95% 향상**
- **논문**: Haeun Koo, Qinglong Li, Seok-kee Lee. (2025). *Enhancing Fake Review Detection with CNN and Linguistic-Semantic Feature Integration.* 빅데이터서비스학회 논문집, 3(2), 5-23.

![CNN-FRD 프레임워크](assets/cnn-frd-framework.jpg)

## 문제

기존 가짜 리뷰 탐지 연구는 딥러닝이 뽑은 텍스트 특징에만 기대거나, 선행연구에서 밝혀진 결정 요인만 사용해 둘을 함께 활용하지 못했습니다.

## 접근

| 모듈 | 입력 | 처리 |
| --- | --- | --- |
| Review Pattern | 리뷰 텍스트 | Embedding → Conv1D(256, kernel 3) → GlobalMaxPooling |
| Linguistic Feature | 정서·움직임·인지·지각·사회적 표현, 심리적 거리 | 속성별 Dense → Concatenate |
| Review-related Feature | 평점, 리뷰 깊이·구조·극단성, 사진 유무, 문체 | 속성별 Dense → Concatenate |
| Reviewer-related Feature | 리뷰어 활동 속성 | 속성별 Dense → Concatenate |
| Prediction | 네 모듈의 출력 | Concatenate → Dense → 가짜 여부 분류 |

비교 모델: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, MLP, LightGBM, CNN 단독 모델

## 파일 구성

| 순서 | 파일 | 내용 |
| --- | --- | --- |
| 1 | `preprocessing.ipynb` | 소문자화, 토큰화, 불용어 제거, 표제어 추출 |
| 2 | `variable_maker.ipynb` | 언어적·리뷰·리뷰어 속성 생성(감성 분석, 토픽 모델링 등) |
| 3 | `git_baseline.ipynb` | 머신러닝 비교 모델 학습·평가 |
| 4 | `git_model.ipynb` | CNN-FRD 및 변형 모델(CNN 단독, CNN 제외) 학습·평가 |

## 기술 스택

Python, TensorFlow/Keras, scikit-learn, LightGBM, gensim(Word2Vec·LDA), NLTK, TextBlob, pandas

> 데이터는 Yelp 리뷰를 사용했으며, 데이터 파일은 저장소에 포함하지 않았습니다.
