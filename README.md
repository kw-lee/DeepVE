# Variance Estimation using Deep Learning with PM25 Datasets 

## Data

2015-2016 데이터 사용 (PM, BPM)

| Data     | Description     | Source                                   |
| -------- | --------------- | ---------------------------------------- |
| PM25     | 일 평균 초미세먼지 농도   | 한국환경공단(www.airkorea.or.kr)               |
| BPM25    | 전일 베이징 초미세먼지 농도 | 미국 국무부 대기 질 모니터링 프로그램 (www.stateair.net) |
| meanTemp | 전일 평균기온         | 기상자료개방포털(data.kma.go.kr)                 |
| minTemp  | 전일 최저기온         | 기상자료개방포털(data.kma.go.kr)                 |
| maxTemp  | 전일 최고기온         | 기상자료개방포털(data.kma.go.kr)                 |
| meanWind | 전일 평균풍속         | 기상자료개방포털(data.kma.go.kr)                 |
| maxWind  | 전일 최대풍속         | 기상자료개방포털(data.kma.go.kr)                 |
| Rain     | 강우 여부           | 기상자료개방포털(data.kma.go.kr)                 |
| *lat*      | *위도*              | *TBA*                                      |
| *long*     | *경도*              | *TBA*                                      |

## Model 

### 변수 

- 서울 전체 평균: `PM25 ~ .-lat-long`
- *관측소별 예측으로,* `PM25 ~ .`

### 모형

#### 딥러닝 모형

1. ~~Base model~~
2. *RNN model*
3. LSTM
4. Classical models:
   1. Linear model
   2. SVM
   3. random forests

> todo: Bayesian deep learning?

#### 반응변수 가정

1. ~~정규분포~~
2. ~~코시 분포 (t분포, `df=1`~~
3. ~~라플라스 분포~~
4. t 분포 (자유도를 조율하는 형태로)
5. *log-normal 분포*

<!-- 1. Linear regression 
1. Deep Learning
2. Deep Learning w/ variance estimation
3. Spatial regression -->

## 결과 비교 

* measure: MSE
* confidence interval: proportion (ratio)

# To Do

* *기울임체*로 작성한 부분들 구현
* 최근 자료 사용 (2019년 관련 자료 이용)
* confidence interval이 실제로 맞나 확인하기 위해 이론적 모형 
  $$y = f(\mathbf{x}) + \epsilon, \quad \epsilon \sim D(0, \sigma(\mathbf{x}))$$
  에서 데이터를 생성하고, 여러 번의 신뢰구간 추정을 통해 실제로 추정이 가능한 지, 아닌 지를 확인