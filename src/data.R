library(tidyverse)
library(readr)
library(stringr)

# BPM
bpm = NULL
for (i in 2015:2016) {
    bpmname = paste0("data/seoul/BPM", i, ".csv")
    tmp_bpm = readr::read_csv(bpmname, skip = 3) %>%
        dplyr::select(Year, Month, Day, Hour, Value) %>%
        dplyr::group_by(Year, Month, Day) %>%
        summarise(PM25 = mean(Value))
    bpm = rbind(bpm, tmp_bpm)
}

# KMA
kma = NULL
for (i in 2015:2016) {
    kmaname = paste0("data/seoul/kma", i, ".csv")
    tmp_kma = readr::read_csv(kmaname, locale=locale('ko',encoding='euc-kr'))[,1:5]
    colnames(tmp_kma) = c("loc", "date", "temp", "rain", "wind")
    tmp_kma = tmp_kma %>%
        mutate(Year = as.numeric(format(date, "%Y")),
            Month = as.numeric(format(date, "%m")),
            Day = as.numeric(format(date, "%d"))) %>%
        group_by(Year, Month, Day) %>%
        summarise(meanTemp = mean(temp),
            minTemp = min(temp),
            maxTemp = max(temp),
            meanWind = mean(wind),
            maxWind = max(wind),
            Rain = (sum(!is.na(rain))>0))
    kma = rbind(kma, tmp_kma)
}
kma

# PM25
pm25 = NULL
i = 2015
for (j in 1:4) {
    cat(i, j, "\n")
    pm25name = paste0("data/seoul/PM", i, "/", i, "년 ", j, "분기.csv")
    tmp_pm25 = readr::read_csv(pm25name, 
        col_types = cols(측정일시 = "c")) %>%
        filter(지역 == "서울")
    tmp_pm25 = tmp_pm25 %>% 
        select(측정일시, SO2, CO, O3, NO2, PM25) %>%
        mutate(Year = as.numeric(substr(측정일시, 1, 4)),
            Month = as.numeric(substr(측정일시, 5, 6)),
            Day = as.numeric(substr(측정일시, 7, 8))) %>%
        group_by(Year, Month, Day) %>%
        summarise(
            SO2 = mean(SO2, na.rm = T),
            CO = mean(CO, na.rm = T),
            O3 = mean(O3, na.rm = T),
            NO2 = mean(NO2, na.rm = T),
            PM25 = mean(PM25, na.rm = T)
        )
    pm25 = rbind(pm25, tmp_pm25)
}
i = 2016
for (j in 1:4) {
    cat(i, j, "\n")
    pm25name = paste0("data/seoul/PM", i, "/", i, "년 ", j, "분기.csv")
    tmp_pm25 = readr::read_csv(pm25name, locale = locale('ko',             encoding = 'euc-kr'), 
        col_types = cols(측정일시 = "c")) %>%
        filter(지역 == "서울")
    tmp_pm25 = tmp_pm25 %>% 
        select(측정일시, SO2, CO, O3, NO2, PM25) %>%
        mutate(Year = as.numeric(substr(측정일시, 1, 4)),
            Month = as.numeric(substr(측정일시, 5, 6)),
            Day = as.numeric(substr(측정일시, 7, 8))) %>%
        group_by(Year, Month, Day) %>%
        summarise(
            SO2 = mean(SO2, na.rm = T),
            CO = mean(CO, na.rm = T),
            O3 = mean(O3, na.rm = T),
            NO2 = mean(NO2, na.rm = T),
            PM25 = mean(PM25, na.rm = T)
        )
    pm25 = rbind(pm25, tmp_pm25)
}

seoul_pm <- pm25 %>%
    right_join(kma, by = c("Year", "Month", "Day")) %>%
    right_join(bpm, by = c("Year", "Month", "Day"), suffix = c(".seoul", "beijing"))

save(list = c("seoul_pm"), file = "data/seoul/seoul.Rdata")