# PM25 Prediction with Confidence Interval Using Deep Learning

## Data

2015-2017 데이터 사용
  
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

## Model 

### 변수 

1. 서울 전체 평균
    $$PM25 \sim .-lat-long$$
2. 관측소 예측
    $$PM25 \sim .$$

### 모형

1. Linear regression 
2. Deep Learning
3. Deep Learning w/ variance estimation
4. Spatial regression