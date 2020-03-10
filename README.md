# PM25 Prediction with Confidence Interval Using Deep Learning

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
| lat      | 위도              | TBA                                      |
| long     | 경도              | TBA                                      |

> todo: 
>   1. 전 일 24시간 자료로부터 다음날 24시간 자료
>   2. 관측소 위치 정보 함께 반영 (단순하게)
>   3. 공간 정보 함께 반영  (근처 n개 정보 등): 공간회귀모형?

## Model 

### 변수 

서울 전체 평균: `PM25 ~ .-lat-long`

> todo: 관측소별 예측으로, `PM25 ~ .`

### 모형

#### 딥러닝 모형

$k$: 이전 자료 갯수

0. linear model ($k=1$)
1. Base Model ($k=1$)
2. Base model ($k=3$)
3. RNN ($k=3$)
4. LSTM ($k=3$)

> todo: Bayesian deep learning?

#### 반응변수 가정

1. 정규분포
2. t분포
3. 라플라스 분포
4. log-normal 분포

<!-- 1. Linear regression 
1. Deep Learning
2. Deep Learning w/ variance estimation
3. Spatial regression -->

## 결과 비교 

* measure: RMSE, MAE
* confidence interval: proportion