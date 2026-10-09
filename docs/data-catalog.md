# ML 학습 데이터 카탈로그 (GX 검증 기준)

> 추천/검색 모델이 학습에 사용하는 데이터 전수 목록. 스키마는 GX 검증이 매일 기록하는 실측 컬럼(`data-validation/_metrics/*.json`의 `__columns__`) 기준이며, 컬럼 구성이 바뀌면 스키마 변경 감지가 Slack으로 알린다.
> 검증 구성: row_count 7일 시계열(평균 ±20%) + 스키마 변경 감지 (시간 단위 DAG는 스키마 감지만). 알림 채널: #데이터검증(C05HDUSSB7T).
> **타입**: parquet/avro는 파일 스키마 실측, csv는 DuckDB 샘플 추론(정수/실수 구분은 근사). **row 추이**: `_metrics` 히스토리 최근 14일 row_count 스파크라인(`▁`최소~`█`최대)+최신값. 신규 배포분은 히스토리 누적 중이라 1~2점만 표시됨(2026-07-23 기준).

생산 DAG 31개 · 데이터셋 87개 (2026-07-23 기준 전 항목 운영 적용, relevance 2개는 상류 DAG 중단으로 미동작)

## `export_user2item_using_inhouse` (매일 01:45)

- **소비 모델**: cbf · cbf3 · bert · embedding_generator · complement · lda_onboarding · seasonality_predictor · sim_markets
- **생성 로직**: `event_log.user_event`(Iceberg) 최근 30일에서 CLICK_GOODS/CLICK_LIKE_GOODS/CLICK_ADD_TO_CART 이벤트를 click/like/cart로 매핑해 유저-상품 인터랙션 CSV 생성. ITEM_ID int 캐스팅·NULL 제거.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `user2item_inhouse` | `s3://ably-analytics-data/personalize/inhouse/{dt}` | csv | 실행일 | 4.8억 *(1d)* | `USER_ID`&nbsp;str · `ITEM_ID`&nbsp;i64 · `TIMESTAMP`&nbsp;i64 · `EVENT_TYPE`&nbsp;str |

## `export_today_ranker_train_dataset` (매일 04:30)

- **소비 모델**: ranker_din_v7 · ranker_din_time_decay · ranker_din_novelty
- **생성 로직**: `user_event` 기반 홈 투데이 랭커 학습셋. 학습 대상 users/items 추출 → df_ctr(노출/클릭 원천) → df_cvr(전환) → 파생 산출. 라벨은 CTR/CVR 멀티태스크(task 컬럼)로 합치고 '전체' 칩 상태만 필터. first_touch 버전은 전환 크레딧을 세션 첫 클릭에 부여(A/B). goods_stat은 datalake event_stat에서 복사.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `ranker_today_df_input` | `s3://ably-analytics-data/personalize/ranker_today/{dt}/df_input` | csv | 실행일 | 1.9억 *(1d)* | `USER_ID`&nbsp;str · `ITEM_ID`&nbsp;i64 · `TIMESTAMP`&nbsp;i64 · `event_code`&nbsp;i64 |
| `ranker_today_df_search` | `s3://ably-analytics-data/personalize/ranker_today/{dt}/df_search` | parquet | 실행일 | 2049만 *(1d)* | `USER_ID`&nbsp;str · `TIMESTAMP`&nbsp;i64 · `search_keyword`&nbsp;str |
| `ranker_today_df_labels_all_chip` | `s3://ably-analytics-data/personalize/ranker_today/{dt}/df_labels_all_chip` | csv | 실행일 | 4.4억 *(1d)* | `USER_ID`&nbsp;str · `ITEM_ID`&nbsp;i64 · `TIMESTAMP`&nbsp;i64 · `task`&nbsp;i64 · `label`&nbsp;i64 · `action_to_order_label`&nbsp;i64 · `ctcar_label`&nbsp;i64 · `ctcvr_label`&nbsp;i64 · `delivery_code`&nbsp;i64 · `is_ad`&nbsp;i64 · `category_chip`&nbsp;i64 · `today_delivery_chip`&nbsp;i64 · `seconds_since_open`&nbsp;i64 |
| `ranker_today_df_labels_all_chip_first_touch` | `s3://ably-analytics-data/personalize/ranker_today/{dt}/df_labels_all_chip_first_touch` | csv | 실행일 | 4.4억 *(1d)* | `USER_ID`&nbsp;str · `ITEM_ID`&nbsp;i64 · `TIMESTAMP`&nbsp;i64 · `task`&nbsp;i64 · `label`&nbsp;i64 · `action_to_order_label`&nbsp;i64 · `ctcar_label`&nbsp;i64 · `ctcvr_label`&nbsp;i64 · `delivery_code`&nbsp;i64 · `is_ad`&nbsp;i64 · `category_chip`&nbsp;i64 · `today_delivery_chip`&nbsp;i64 · `seconds_since_open`&nbsp;i64 |
| `ranker_today_goods_stat` | `s3://ably-analytics-data/personalize/ranker_today/{dt}/goods_stat` | csv | 실행일 | 7370만 *(1d)* | `sno`&nbsp;i64 · `click_goods_count_1d`&nbsp;i64 · `click_goods_count_7d`&nbsp;i64 · `click_goods_count_30d`&nbsp;i64 · `click_goods_count_1y`&nbsp;i64 · `total_click_goods_count`&nbsp;i64 · `display_goods_count_1d`&nbsp;i64 · `display_goods_count_7d`&nbsp;i64 · `display_goods_count_30d`&nbsp;i64 · `display_goods_count_1y`&nbsp;i64 · `total_display_goods_count`&nbsp;i64 |
| `ranker_today_goods_event_stat` | `s3://ably-analytics-data/personalize/ranker_today/{dt}/goods_event_stat` | csv | 실행일 | 246만 *(1d)* | `ITEM_ID`&nbsp;i64 · `CLICK_GOODS_COUNT_30d`&nbsp;i64 · `CLICK_LIKE_GOODS_COUNT_30d`&nbsp;i64 · `CLICK_ADD_TO_CART_COUNT_30d`&nbsp;i64 · `CLICK_BUY_GOODS_COUNT_30d`&nbsp;i64 · `CLICK_GOODS_COUNT_7d`&nbsp;i64 · `CLICK_LIKE_GOODS_COUNT_7d`&nbsp;i64 · `CLICK_ADD_TO_CART_COUNT_7d`&nbsp;i64 · `CLICK_BUY_GOODS_COUNT_7d`&nbsp;i64 · `CLICK_GOODS_COUNT_1d`&nbsp;i64 · `CLICK_LIKE_GOODS_COUNT_1d`&nbsp;i64 · `CLICK_ADD_TO_CART_COUNT_1d`&nbsp;i64 · `CLICK_BUY_GOODS_COUNT_1d`&nbsp;i64 |
| `ranker_today_df_order` | `s3://ably-analytics-data/personalize/ranker_today/{dt}/df_order` | parquet | 실행일 | 2797만 *(1d)* | `USER_ID`&nbsp;str · `ITEM_ID`&nbsp;str · `TIMESTAMP`&nbsp;i64 |

## `export_today_ranker_train_dataset_v2` (매일 04:45)

- **소비 모델**: ranker_din_att_v2 계열
- **생성 로직**: v1과 동일 파이프라인의 v2 root. 라벨 B 방식 = 구매 전환 크레딧을 구매 전 모든 클릭에 1/N 분배(export_df_labels_v2.py --label_mode B). df_labels_all_chip은 v1 방식 라벨을 v2 데이터로 재생산.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `ranker_today_v2_df_input` | `s3://ably-analytics-data/personalize/ranker_today_v2/{dt}/df_input` | csv | 실행일 | 1.7억 *(1d)* | `USER_ID`&nbsp;str · `ITEM_ID`&nbsp;i64 · `TIMESTAMP`&nbsp;i64 · `event_code`&nbsp;i64 |
| `ranker_today_v2_df_search` | `s3://ably-analytics-data/personalize/ranker_today_v2/{dt}/df_search` | parquet | 실행일 | 1759만 *(1d)* | `USER_ID`&nbsp;str · `TIMESTAMP`&nbsp;i64 · `search_keyword`&nbsp;str |
| `ranker_today_v2_df_labels_v2_B` | `s3://ably-analytics-data/personalize/ranker_today_v2/{dt}/df_labels_v2_B` | csv | 실행일 | 5.0억 *(1d)* | `USER_ID`&nbsp;str · `ITEM_ID`&nbsp;i64 · `TIMESTAMP`&nbsp;i64 · `task`&nbsp;i64 · `label`&nbsp;i64 · `action_to_order_label`&nbsp;i64 · `ctcar_label`&nbsp;f64 · `ctcvr_label`&nbsp;i64 · `delivery_code`&nbsp;i64 · `is_ad`&nbsp;i64 · `category_chip`&nbsp;i64 · `today_delivery_chip`&nbsp;i64 · `seconds_since_open`&nbsp;i64 |
| `ranker_today_v2_df_labels_all_chip` | `s3://ably-analytics-data/personalize/ranker_today_v2/{dt}/df_labels_all_chip` | csv | 실행일 | 4.3억 *(1d)* | `USER_ID`&nbsp;str · `ITEM_ID`&nbsp;i64 · `TIMESTAMP`&nbsp;i64 · `task`&nbsp;i64 · `label`&nbsp;i64 · `action_to_order_label`&nbsp;i64 · `ctcar_label`&nbsp;i64 · `ctcvr_label`&nbsp;i64 · `delivery_code`&nbsp;i64 · `is_ad`&nbsp;i64 · `category_chip`&nbsp;i64 · `today_delivery_chip`&nbsp;i64 · `seconds_since_open`&nbsp;i64 |

## `export_search_ranker_train_dataset` (매일 03:30)

- **소비 모델**: search_ranker v1 · v2
- **생성 로직**: 검색 세션 기반 랭커 학습셋. 검색 결과 노출/클릭(df_ctr) → user_code/goods_sno를 user_index/item_index로 매핑(preprocess) 후 {dt}/output/에 최종본 생성. df_ctr_v2는 세션 정의 보정판(build_df_ctr_v2). 전 산출물 dropna 후 저장.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `search_ranker_df_input` | `s3://ably-analytics-data/personalize/search_ranker/{dt}/output/df_input` | parquet | 실행일 | 1.2억 *(1d)* | `user_code`&nbsp;str · `goods_sno`&nbsp;str · `TIMESTAMP`&nbsp;str · `item_index`&nbsp;i32 · `is_train`&nbsp;i32 · `user_index`&nbsp;i32 |
| `search_ranker_df_search` | `s3://ably-analytics-data/personalize/search_ranker/{dt}/output/df_search` | parquet | 실행일 | 1713만 *(1d)* | `user_code`&nbsp;str · `search_keyword`&nbsp;str · `TIMESTAMP`&nbsp;i64 · `is_train`&nbsp;i32 · `user_index`&nbsp;i32 |
| `search_ranker_df_ctr` | `s3://ably-analytics-data/personalize/search_ranker/{dt}/output/df_ctr` | parquet | 실행일 | 5.0억 *(1d)* | `user_code`&nbsp;str · `goods_sno`&nbsp;str · `TIMESTAMP`&nbsp;i64 · `search_keyword`&nbsp;str · `ctr`&nbsp;i32 · `ctcar`&nbsp;i32 · `ctcvr`&nbsp;i32 · `is_train`&nbsp;i32 · `user_index`&nbsp;i32 · `delivery_code`&nbsp;i32 · `item_index`&nbsp;i32 · `session`&nbsp;i32 · `list_index`&nbsp;i32 · `is_ad`&nbsp;bool · `surface_type`&nbsp;str |
| `search_ranker_df_ctr_v2` | `s3://ably-analytics-data/personalize/search_ranker/{dt}/output/df_ctr_v2` | parquet | 실행일 | 5.0억 *(1d)* | `goods_sno`&nbsp;str · `user_code`&nbsp;str · `TIMESTAMP`&nbsp;i64 · `search_keyword`&nbsp;str · `ctr`&nbsp;i32 · `ctcar`&nbsp;i32 · `ctcvr`&nbsp;i32 · `is_train`&nbsp;i32 · `user_index`&nbsp;i32 · `delivery_code`&nbsp;i32 · `item_index`&nbsp;i32 · `session`&nbsp;i32 · `list_index`&nbsp;i32 · `is_ad`&nbsp;bool · `surface_type`&nbsp;str · `item_age_seconds`&nbsp;i64 |
| `search_ranker_df_goods` | `s3://ably-analytics-data/personalize/search_ranker/{dt}/output/df_goods` | parquet | 실행일 | 18만 *(1d)* | `goods_sno`&nbsp;str · `disp_cnt`&nbsp;i64 · `ctr_cnt`&nbsp;i64 · `item_index`&nbsp;i32 · `CLICK_GOODS_COUNT_30d`&nbsp;i64 · `CLICK_LIKE_GOODS_COUNT_30d`&nbsp;i64 · `CLICK_ADD_TO_CART_COUNT_30d`&nbsp;i64 · `CLICK_BUY_GOODS_COUNT_30d`&nbsp;i64 · `CLICK_GOODS_COUNT_7d`&nbsp;i64 · `CLICK_LIKE_GOODS_COUNT_7d`&nbsp;i64 · `CLICK_ADD_TO_CART_COUNT_7d`&nbsp;i64 · `CLICK_BUY_GOODS_COUNT_7d`&nbsp;i64 · `CLICK_GOODS_COUNT_1d`&nbsp;i64 · `CLICK_LIKE_GOODS_COUNT_1d`&nbsp;i64 · `CLICK_ADD_TO_CART_COUNT_1d`&nbsp;i64 · `CLICK_BUY_GOODS_COUNT_1d`&nbsp;i64 |

## `prod_long_term_train_data_v2` (매일 03:47)

- **소비 모델**: multi_interest 계열 7개
- **생성 로직**: `personalize/inhouse` 일별본을 1년치로 누적: 최근 1달은 전체, 이전 기간은 유저 50% 샘플링. order는 reco_raw 주문에서 대상 유저·상품으로 필터 후 order.csv로 rename 복사.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `prod_1yr_train_interaction` | `s3://ably-analytics-data/daily/prod_1yr_train_data_v2/interaction/{dt}` | csv | 실행일 | 6.3억 *(1d)* | `USER_ID`&nbsp;str · `ITEM_ID`&nbsp;i64 · `TIMESTAMP`&nbsp;i64 · `EVENT_TYPE`&nbsp;str |
| `prod_1yr_train_order` | `s3://ably-analytics-data/daily/prod_1yr_train_data_v2/order/{dt}` | csv | 실행일 | 1273만 *(1d)* | `member_sno`&nbsp;i64 · `goods_sno`&nbsp;i64 · `sno`&nbsp;i64 · `order_sno`&nbsp;i64 · `goods_option_sno`&nbsp;i64 · `ea`&nbsp;i64 · `price`&nbsp;i64 · `emoney`&nbsp;i64 · `coupon`&nbsp;i64 |

## `4910_train_data` (매일 02:12)

- **소비 모델**: cbf_4910 · multi_interest_4910 계열 5개 · sim_brands_4910
- **생성 로직**: 4910 앱 이벤트(`aglo_event_log.user_event_all`)에서 interaction 생성(click/like/cart 매핑). order는 reco_raw 주문을 4910 상품으로 필터 후 rename 복사(csv+parquet).

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `prod_4910_train_interaction` | `s3://ably-analytics-data/daily/prod_4910_train_data/interaction.csv/{dt}` | csv | 실행일 | 2076만 *(1d)* | `USER_ID`&nbsp;str · `ITEM_ID`&nbsp;i64 · `TIMESTAMP`&nbsp;i64 · `EVENT_TYPE`&nbsp;str · `EVENT_VALUE`&nbsp;i64 |
| `prod_4910_train_order` | `s3://ably-analytics-data/daily/prod_4910_train_data/order/{dt}` | csv | 실행일 | 491만 *(1d)* | `goods_sno`&nbsp;i64 · `sno`&nbsp;i64 · `order_sno`&nbsp;i64 · `goods_option_sno`&nbsp;i64 · `member_sno`&nbsp;i64 · `ea`&nbsp;i64 · `price`&nbsp;i64 · `emoney`&nbsp;i64 · `coupon`&nbsp;i64 |

## `prod_amood_train_data` (매일 10:00)

- **소비 모델**: pastel_cbf(amood) · multi_interest_amood_v3(400d)
- **생성 로직**: 파스텔 원본(`pastel-personalize/datasets/{interactions,items}`)을 ably 학습 포맷으로 변환. interactions(view/purchase→이벤트 매핑) + sql 5종(order/goods/category/standard_category/market — 파스텔 데이터에서 합성, coalesce(1) 후 고정 파일명으로 복사). base(기본)와 400d(400일치) 두 번 실행.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `amood_train_interactions` | `s3://ably-analytics-data/daily/amood_train_data/interactions/{dt}` | csv | 전일 | `██▇▇▆▆▅▄▃▂▂▂▁` 248만 *(13d)* | `USER_ID`&nbsp;str · `ITEM_ID`&nbsp;i64 · `TIMESTAMP`&nbsp;i64 · `EVENT_TYPE`&nbsp;str |
| `amood_train_order` | `s3://ably-analytics-data/daily/amood_train_data/sql/{dt}/order.csv` | csv | 전일 | `▅▄▇▇▇█▆▄▂▁▁▁▁` 12만 *(13d)* | `sno`&nbsp;i64 · `order_sno`&nbsp;i64 · `goods_sno`&nbsp;i64 · `option_sno`&nbsp;i64 · `member_sno`&nbsp;i64 |
| `amood_train_goods` | `s3://ably-analytics-data/daily/amood_train_data/sql/{dt}/goods.csv` | csv | 전일 | `█▁` 204만 *(2d)* | `market_sno`&nbsp;i64 · `sno`&nbsp;i64 · `name`&nbsp;str · `price`&nbsp;i64 · `category_sno`&nbsp;i64 · `standard_category_sno`&nbsp;i64 · `market_type_sno`&nbsp;i64 |
| `amood_train_category` | `s3://ably-analytics-data/daily/amood_train_data/sql/{dt}/category.csv` | csv | 전일 | `▄▄` 160 *(2d)* | `sno`&nbsp;i64 · `catnm`&nbsp;str · `depth`&nbsp;i64 · `parent_category__sno`&nbsp;i64 |
| `amood_train_standard_category` | `s3://ably-analytics-data/daily/amood_train_data/sql/{dt}/standard_category.csv` | csv | 전일 | `▄▄` 160 *(2d)* | `sno`&nbsp;i64 · `name`&nbsp;str · `depth`&nbsp;i64 · `parent_standard_category__sno`&nbsp;i64 · `catnm`&nbsp;str |
| `amood_train_market` | `s3://ably-analytics-data/daily/amood_train_data/sql/{dt}/market.csv` | csv | 전일 | `▁█` 3만 *(2d)* | `sno`&nbsp;i64 · `name`&nbsp;str |
| `amood_train_400d_interactions` | `s3://ably-analytics-data/daily/amood_train_data_400d/interactions/{dt}` | csv | 전일 | `█▁` 8462만 *(2d)* | `USER_ID`&nbsp;str · `ITEM_ID`&nbsp;i64 · `TIMESTAMP`&nbsp;i64 · `EVENT_TYPE`&nbsp;str |
| `amood_train_400d_order` | `s3://ably-analytics-data/daily/amood_train_data_400d/sql/{dt}/order.csv` | csv | 전일 | `█▁` 86만 *(2d)* | `sno`&nbsp;i64 · `order_sno`&nbsp;i64 · `goods_sno`&nbsp;i64 · `option_sno`&nbsp;i64 · `member_sno`&nbsp;i64 |
| `amood_train_400d_goods` | `s3://ably-analytics-data/daily/amood_train_data_400d/sql/{dt}/goods.csv` | csv | 전일 | `█▁` 204만 *(2d)* | `market_sno`&nbsp;i64 · `sno`&nbsp;i64 · `name`&nbsp;str · `price`&nbsp;i64 · `category_sno`&nbsp;i64 · `standard_category_sno`&nbsp;i64 · `market_type_sno`&nbsp;i64 |
| `amood_train_400d_category` | `s3://ably-analytics-data/daily/amood_train_data_400d/sql/{dt}/category.csv` | csv | 전일 | `▄▄` 160 *(2d)* | `sno`&nbsp;i64 · `catnm`&nbsp;str · `depth`&nbsp;i64 · `parent_category__sno`&nbsp;i64 |
| `amood_train_400d_standard_category` | `s3://ably-analytics-data/daily/amood_train_data_400d/sql/{dt}/standard_category.csv` | csv | 전일 | `▄▄` 160 *(2d)* | `sno`&nbsp;i64 · `name`&nbsp;str · `depth`&nbsp;i64 · `parent_standard_category__sno`&nbsp;i64 · `catnm`&nbsp;str |
| `amood_train_400d_market` | `s3://ably-analytics-data/daily/amood_train_data_400d/sql/{dt}/market.csv` | csv | 전일 | `▁█` 3만 *(2d)* | `sno`&nbsp;i64 · `name`&nbsp;str |

## `export_bandit_dataset` (매시 정각)

- **소비 모델**: bandit · bandit_v3 · bandit_v4 · bandit_two_tower(_v2) · bandit_v5 · bandit_two_tower_v5
- **생성 로직**: `user_event` 기반 시간 단위 밴딧 학습셋. df_input(최근 7일 클릭/찜), df_input_21d(8일+일별 user_history_v2 누적, two_tower·v5용), df_v3/df_v4(타겟 지면 노출/클릭 라벨, 기간 상이), goods_v4(상품 메타+노출/클릭 통계). 시간 파티션({dt}/{HH})이라 시계열 검증 없이 스키마 감지만 적용. ※ 신규 산출물 goods_v5(신상품 cold pool 포함)·df_input_v2(활성 유저 필터)는 검증 미적용.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `bandit_df_input` | `s3://ably-analytics-data/personalize/bandit/{dt}/df_input` | parquet | 실행일/시 | — *(시간단위·스키마검증만)* | `user_code`&nbsp;str · `goods_sno`&nbsp;str · `event_name`&nbsp;str · `timestamp`&nbsp;i64 |
| `bandit_df_input_21d` | `s3://ably-analytics-data/personalize/bandit/{dt}/df_input_21d` | parquet | 실행일/시 | — *(시간단위·스키마검증만)* | `user_code`&nbsp;str · `goods_sno`&nbsp;str · `event_name`&nbsp;str · `timestamp`&nbsp;i64 |
| `bandit_df_v3` | `s3://ably-analytics-data/personalize/bandit/{dt}/df_v3` | parquet | 실행일/시 | — *(시간단위·스키마검증만)* | `user_code`&nbsp;str · `goods_sno`&nbsp;str · `label`&nbsp;i32 · `like_label`&nbsp;i32 · `timestamp`&nbsp;i64 · `list_index`&nbsp;i32 · `session`&nbsp;i32 |
| `bandit_df_v4` | `s3://ably-analytics-data/personalize/bandit/{dt}/df_v4` | parquet | 실행일/시 | — *(시간단위·스키마검증만)* | `user_code`&nbsp;str · `goods_sno`&nbsp;str · `label`&nbsp;i32 · `like_label`&nbsp;i32 · `timestamp`&nbsp;i64 · `list_index`&nbsp;i32 · `session`&nbsp;i32 |
| `bandit_goods_v4` | `s3://ably-analytics-data/personalize/bandit/{dt}/goods_v4` | parquet | 실행일/시 | — *(시간단위·스키마검증만)* | `goods_sno`&nbsp;i32 · `name`&nbsp;str · `market_sno`&nbsp;i32 · `category_sno`&nbsp;i32 · `standard_category_sno`&nbsp;i32 · `delivery_type`&nbsp;i32 · `display_cnt`&nbsp;i64 · `click_cnt`&nbsp;i64 · `display_user_cnt`&nbsp;i64 · `click_user_cnt`&nbsp;i64 |

## `ml_daily_goods_display_click` (매시 14분)

- **소비 모델**: ctr_based_reranker · ctr_based_reranker_4910
- **생성 로직**: 당일 상품별 노출/클릭 누적 집계를 매시 갱신(같은 dt 경로 덮어쓰기). ably·4910 각각 생산. 시간마다 데이터가 자라므로 스키마 감지만 적용.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `goods_display_click` | `s3://ably-analytics-data/daily/goods_display_click/{dt}` | csv | interval_start(전일 표기 주의) | — *(시간단위·스키마검증만)* | `goods_sno`&nbsp;i64 · `DISPLAY_GOODS`&nbsp;i64 · `CLICK_GOODS`&nbsp;i64 |
| `goods_display_click_4910` | `s3://ably-analytics-data/daily/goods_display_click/4910/{dt}` | csv | interval_start | — *(시간단위·스키마검증만)* | `goods_sno`&nbsp;i64 · `DISPLAY_GOODS`&nbsp;i64 · `CLICK_GOODS`&nbsp;i64 |

## `ad_ctr_train_data_v3` (매일 13:13)

- **소비 모델**: ctr_predictor_v3 · ctr_predictor_v3_oov
- **생성 로직**: 광고 지면 노출/클릭 로그에서 CTR 학습셋 생성(list_id/list_index 포함, is_test 분리). oov 변형은 OOV 상품 대상 노출/클릭. user_goods_interaction_7d는 최근 7일 유저-상품 인터랙션.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `ad_ctr_v3_user_goods_display_click` | `s3://ably-analytics-data/daily/ad_ctr_train_data_v3/{dt}/user_goods_display_click` | csv | 실행일 | `▁█` 4582만 *(2d)* | `user_id`&nbsp;i64 · `goods_sno`&nbsp;i64 · `list_id`&nbsp;str · `list_index`&nbsp;i64 · `click`&nbsp;i64 · `is_test`&nbsp;i64 |
| `ad_ctr_v3_user_goods_interaction_7d` | `s3://ably-analytics-data/daily/ad_ctr_train_data_v3/{dt}/user_goods_interaction_7d` | csv | 실행일 | `█▁` 1.9억 *(2d)* | `user_id`&nbsp;i64 · `goods_sno`&nbsp;i64 |
| `ad_oov_ctr_user_goods_display_click` | `s3://ably-analytics-data/daily/ad_oov_ctr_train_data/{dt}/user_goods_display_click` | csv | 실행일 | `▁█` 3516만 *(2d)* | `user_id`&nbsp;i64 · `goods_sno`&nbsp;i64 · `list_id`&nbsp;str · `list_index`&nbsp;i64 · `click`&nbsp;i64 · `is_test`&nbsp;i64 |

## `ad_cvr_train_data_v3` (매일 10:55)

- **소비 모델**: cvr_predictor · cvr_predictor_v3
- **생성 로직**: 광고 전환(CVR) 학습셋: 클릭 후 주문 여부 라벨(event_order), 상품 정보(리뷰/클릭/찜/구매 수), 유저 기구매 상품 목록.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `ad_cvr_v3_user_goods_event_order` | `s3://ably-analytics-data/daily/prod_ad_cvr_train_data_v3/{dt}/user_goods_event_order` | csv | 실행일 | `▁█` 1357만 *(2d)* | `user_id`&nbsp;i64 · `goods_sno`&nbsp;i64 · `event_name`&nbsp;str · `event_ts`&nbsp;i64 · `list_id`&nbsp;str · `list_index`&nbsp;i64 · `is_ad`&nbsp;i64 · `label`&nbsp;i64 · `is_test`&nbsp;i64 |
| `ad_cvr_v3_goods_info` | `s3://ably-analytics-data/daily/prod_ad_cvr_train_data_v3/{dt}/goods_info` | csv | 실행일 | `▁█` 875만 *(2d)* | `goods_sno`&nbsp;i64 · `reg_ts`&nbsp;i64 · `price_group`&nbsp;i64 · `review_cnt`&nbsp;i64 · `positive_review_cnt`&nbsp;i64 · `item_click`&nbsp;i64 · `item_like`&nbsp;i64 · `item_purchase`&nbsp;i64 |
| `ad_cvr_v3_user_goods_purchased` | `s3://ably-analytics-data/daily/prod_ad_cvr_train_data_v3/{dt}/user_goods_purchased` | csv | 실행일 | `█▁` 819만 *(2d)* | `user_id`&nbsp;i64 · `goods_sno`&nbsp;i64 · `event_ts`&nbsp;i64 |

## `4910_ad_ctr_train_data_v3` (매일 14:30)

- **소비 모델**: ctr_predictor_v3_4910
- **생성 로직**: ad_ctr_train_data_v3의 4910 앱 버전 (oov 산출물은 없음).

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `4910_ad_ctr_v3_user_goods_display_click` | `s3://ably-analytics-data/daily/4910_ad_ctr_train_data_v3/{dt}/user_goods_display_click` | csv | 실행일 | 395만 *(1d)* | `user_id`&nbsp;i64 · `goods_sno`&nbsp;i64 · `list_id`&nbsp;str · `list_index`&nbsp;i64 · `click`&nbsp;i64 · `is_test`&nbsp;i64 |
| `4910_ad_ctr_v3_user_goods_interaction_7d` | `s3://ably-analytics-data/daily/4910_ad_ctr_train_data_v3/{dt}/user_goods_interaction_7d` | csv | 실행일 | 543만 *(1d)* | `user_id`&nbsp;i64 · `goods_sno`&nbsp;i64 |

## `4910_ad_cvr_train_data_v3` (매일 09:55)

- **소비 모델**: cvr_predictor_v3_4910
- **생성 로직**: ad_cvr_train_data_v3의 4910 앱 버전.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `4910_ad_cvr_v3_user_goods_event_order` | `s3://ably-analytics-data/daily/4910_ad_cvr_train_data_v3/{dt}/user_goods_event_order` | csv | 실행일 | `▁█` 14만 *(2d)* | `user_id`&nbsp;i64 · `goods_sno`&nbsp;i64 · `event_name`&nbsp;str · `event_ts`&nbsp;i64 · `list_id`&nbsp;str · `list_index`&nbsp;i64 · `is_ad`&nbsp;i64 · `label`&nbsp;i64 · `is_test`&nbsp;i64 |
| `4910_ad_cvr_v3_goods_info` | `s3://ably-analytics-data/daily/4910_ad_cvr_train_data_v3/{dt}/goods_info` | csv | 실행일 | `▁█` 875만 *(2d)* | `goods_sno`&nbsp;i64 · `reg_ts`&nbsp;i64 · `price_group`&nbsp;i64 · `review_cnt`&nbsp;i64 · `positive_review_cnt`&nbsp;i64 · `item_click`&nbsp;i64 · `item_like`&nbsp;i64 · `item_purchase`&nbsp;i64 |
| `4910_ad_cvr_v3_user_goods_purchased` | `s3://ably-analytics-data/daily/4910_ad_cvr_train_data_v3/{dt}/user_goods_purchased` | csv | 실행일 | `█▁` 24만 *(2d)* | `user_id`&nbsp;i64 · `goods_sno`&nbsp;i64 · `event_ts`&nbsp;i64 |

## `reco_click_etl` (매일 10:30)

- **소비 모델**: substitute (+ 다수 전처리 DAG의 상류)
- **생성 로직**: reco-etl JAR(`GoodsClickBySession`): user_event 클릭을 세션 단위로 묶어 goods/market/category/price/timestamp 로그 배열 컬럼으로 생성, Glue `reco_etl.goods_click_session_log`에 기록. ※ 같은 DAG의 GoodsClickV2(goods_click_v2 테이블)는 검증 미적용.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `reco_etl_goods_click_session_log` | `s3://ably-analytics-data/reco/etl/db/goods_click_session_log/dt={dt}` | parquet | 전일(logical_date) | `▁█` 6289만 *(2d)* | `user_code`&nbsp;str · `session_id`&nbsp;str · `goods_log`&nbsp;str[] · `market_log`&nbsp;str[] · `category_log`&nbsp;str[] · `price_log`&nbsp;i64[] · `event_timestamp_log`&nbsp;i64[] · `dt`&nbsp;i64 |

## `reco_metadata_etl` (매일 10:15)

- **소비 모델**: substitute
- **생성 로직**: reco-etl JAR(`GoodsMetadata`): 상품 메타(가격/마켓/카테고리) 스냅샷을 Glue `reco_etl.goods_metadata`에 기록.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `reco_etl_goods_metadata` | `s3://ably-analytics-data/reco/etl/db/goods_metadata/dt={dt}` | parquet | 전일(logical_date) | `▁█` 4977만 *(2d)* | `goods_sno`&nbsp;str · `price`&nbsp;i64 · `market_sno`&nbsp;i64 · `category_sno`&nbsp;i64 · `dt`&nbsp;i64 |

## `etl_reco_raw_daily_parquet` (매일 00:05)

- **소비 모델**: 공용 로더 전 모델(런타임 sql 테이블) · lightgcn · related_keywords_v2 등
- **생성 로직**: ably-etl JAR(`RecoRawStoreDailyIceberg`): CDC(`ably-datalake/cdc/db/ably/`)에서 서비스 DB 원본 테이블들의 일 스냅샷 parquet 생성. ably-reco의 reco_common이 학습 런타임에 goods/order/category/standard_category/member/market을 직접 다운로드.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `reco_raw_goods` | `s3://ably-datalake/reco/db/reco_raw_parquet/goods/dt={dt}` | parquet | 실행일 | 905만 *(1d)* | `market_sno`&nbsp;i32 · `sno`&nbsp;i32 · `name`&nbsp;str · `price`&nbsp;i64 · `category_sno`&nbsp;i32 · `standard_category_sno`&nbsp;i32 · `positive_review_count`&nbsp;i64 · `total_review_count`&nbsp;i64 · `market_type_sno`&nbsp;i32 · `brand_sno`&nbsp;i32 · `delivery_type`&nbsp;i32 · `dt`&nbsp;i64 |
| `reco_raw_order` | `s3://ably-datalake/reco/db/reco_raw_parquet/order/dt={dt}` | parquet | 실행일 | 4691만 *(1d)* | `sno`&nbsp;i64 · `order_sno`&nbsp;i64 · `goods_sno`&nbsp;i32 · `goods_option_sno`&nbsp;i32 · `member_sno`&nbsp;i32 · `ea`&nbsp;i64 · `price`&nbsp;i64 · `emoney`&nbsp;i32 · `coupon`&nbsp;i32 · `dt`&nbsp;i64 |
| `reco_raw_category` | `s3://ably-datalake/reco/db/reco_raw_parquet/category/dt={dt}` | parquet | 실행일 | 582 *(1d)* | `sno`&nbsp;i32 · `catnm`&nbsp;str · `depth`&nbsp;i64 · `parent_category__sno`&nbsp;i32 · `dt`&nbsp;i64 |
| `reco_raw_standard_category` | `s3://ably-datalake/reco/db/reco_raw_parquet/standard_category/dt={dt}` | parquet | 실행일 | 837 *(1d)* | `sno`&nbsp;i32 · `name`&nbsp;str · `depth`&nbsp;i64 · `parent_standard_category__sno`&nbsp;i32 · `dt`&nbsp;i64 |
| `reco_raw_member` | `s3://ably-datalake/reco/db/reco_raw_parquet/member/dt={dt}` | parquet | 실행일 | 1634만 *(1d)* | `m_no`&nbsp;i32 · `birth_date`&nbsp;date · `height`&nbsp;i32 · `weight`&nbsp;i32 · `top`&nbsp;i32 · `bottom`&nbsp;i32 · `shoes`&nbsp;i32 · `dt`&nbsp;i64 |
| `reco_raw_market` | `s3://ably-datalake/reco/db/reco_raw_parquet/market/dt={dt}` | parquet | 실행일 | 8만 *(1d)* | `sno`&nbsp;i32 · `name`&nbsp;str · `age`&nbsp;i32 · `age2`&nbsp;i32 · `style`&nbsp;str · `style2`&nbsp;str · `dt`&nbsp;i64 |

## `preprocess_search_data_inhouse` (매일 02:00)

- **소비 모델**: search_tower · search_tower_v2 · search_tower_py311 · search_tower_mask
- **생성 로직**: `user_event`에서 검색 쿼리 로그(search, avro)와 검색 경유 상품 인터랙션(interaction, csv) 생성. reco_raw goods로 유효 상품 필터.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `search_inhouse_search` | `s3://ably-analytics-data/search/inhouse/{dt}/search` | avro | 실행일 | `▁▃▆▅▆▅▄▄▆█▇▅▄` 1905만 *(13d)* | `user_id`&nbsp;str · `timestamp`&nbsp;i64 · `query`&nbsp;str |
| `search_inhouse_interaction` | `s3://ably-analytics-data/search/inhouse/{dt}/interaction` | csv | 실행일 | `▁▃▇▆▆▆▃▄▆█▆▃▂` 2.9억 *(13d)* | `ITEM_ID`&nbsp;i64 · `USER_ID`&nbsp;str · `TIMESTAMP`&nbsp;i64 · `event_name`&nbsp;str · `screen_name`&nbsp;str |

## `4910_search_train_data_v2` (매일 02:58)

- **소비 모델**: search_tower_4910
- **생성 로직**: preprocess_search_data_inhouse의 4910 버전.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `search_4910_v2_search` | `s3://ably-analytics-data/daily/4910_search_train_data_v2/{dt}/search` | avro | 실행일 | 224만 *(1d)* | `user_id`&nbsp;str · `timestamp`&nbsp;i64 · `query`&nbsp;str |
| `search_4910_v2_interaction` | `s3://ably-analytics-data/daily/4910_search_train_data_v2/{dt}/interaction` | csv | 실행일 | 902만 *(1d)* | `ITEM_ID`&nbsp;i64 · `USER_ID`&nbsp;str · `TIMESTAMP`&nbsp;i64 · `event_name`&nbsp;str · `screen_name`&nbsp;str |

## `pyspark_search_related_keyword_v2_preprocess_parquet` (매일 07:00)

- **소비 모델**: related_keywords_v2
- **생성 로직**: 검색 이벤트에서 키워드-클릭 히스토리(검색어, 결과 상품, 클릭 상품) parquet 생성.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `search_click_history_v2_parquet` | `s3://ably-analytics-data/daily/search_click_history_v2_parquet/{dt}` | parquet | 실행일 | 1765만 *(1d)* | `user_code`&nbsp;str · `app_platform`&nbsp;str · `analytics_event_name`&nbsp;str · `GOODS_SNO`&nbsp;str · `LIST_ID`&nbsp;str · `SEARCH_KEYWORD`&nbsp;str · `SEARCH_RESULT_GOODS`&nbsp;str · `occurred_time`&nbsp;str |

## `pyspark_complement_npmi_reco_preprocess` (매일 01:30)

- **소비 모델**: complement
- **생성 로직**: 함께 구매/클릭된 카테고리 쌍의 동시출현 통계로 NPMI 점수 집계.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `complement_npmi` | `s3://ably-analytics-data/daily/complement-npmi/{dt}` | csv | 실행일 | 2만 *(1d)* | `cate1_sno`&nbsp;i64 · `cate1`&nbsp;str · `cate2_sno`&nbsp;i64 · `cate2`&nbsp;str · `count_of_cate1`&nbsp;i64 · `count_of_cate2`&nbsp;i64 · `count_of_pair`&nbsp;i64 · `pmi`&nbsp;f64 · `npmi`&nbsp;f64 |

## `prod_amood_goods_click_by_session` (매일 11:00)

- **소비 모델**: substitute_amood
- **생성 로직**: amood 클릭 로그를 세션화(goods/market/category/price/ts 로그 배열) + 상품 메타. reco_etl 세션 로그와 동일 스키마의 amood 버전.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `amood_click_by_session` | `s3://ably-analytics-data/daily/amood_click_logs/goods_click_by_session/{dt}` | parquet | 전일 | `█▁` 60만 *(2d)* | `user_code`&nbsp;i32 · `session_id`&nbsp;str · `goods_log`&nbsp;str[] · `market_log`&nbsp;str[] · `category_log`&nbsp;str[] · `price_log`&nbsp;i32[] · `event_timestamp_log`&nbsp;i64[] · `dt`&nbsp;str |
| `amood_click_goods_metadata` | `s3://ably-analytics-data/daily/amood_click_logs/goods_metadata/{dt}` | parquet | 전일 | `█▁` 204만 *(2d)* | `market_sno`&nbsp;str · `category_sno`&nbsp;str · `goods_sno`&nbsp;str · `name`&nbsp;str · `price`&nbsp;str · `standard_category_sno`&nbsp;str · `market_type_sno`&nbsp;str · `catnm`&nbsp;str · `depth`&nbsp;str · `parent_category__sno`&nbsp;str · `market_name`&nbsp;str |

## `lightgcn_prepare_train_data` (매일 20:00)

- **소비 모델**: lightgcn_v2
- **생성 로직**: user_event에서 유저-상품 이벤트 시퀀스(user_history)와 인기 상품 풀(popular_goods) 생성. goods 메타는 학습 시 reco_raw_parquet에서 직접 사용.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `lightgcn_v2_user_history` | `s3://ably-analytics-data/personalize/lightgcn_v2/{dt}/user_history` | parquet | 실행일(수동 트리거 시 당일) | 1.2억 *(1d)* | `user_code`&nbsp;str · `goods_sno`&nbsp;str · `synced_occurred_ts`&nbsp;ts |
| `lightgcn_v2_popular_goods` | `s3://ably-analytics-data/personalize/lightgcn_v2/{dt}/popular_goods` | parquet | 실행일 | 9563 *(1d)* | `goods_sno`&nbsp;str |

## `review_ranker_train_data` (매일 10:16)

- **소비 모델**: review_ranker
- **생성 로직**: 리뷰 노출/도움돼요 데이터: review(리뷰·평가·좋아요), user(출생연도), user_goods_interaction(유저-상품), user_goods_order(리뷰 선택-주문 전환 라벨).

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `review_ranking_review` | `s3://ably-analytics-data/daily/review_ranking_train_data/{dt}/review` | csv | 실행일 | `▁█` 1.2억 *(2d)* | `review_sno`&nbsp;i64 · `user_id`&nbsp;i64 · `goods_sno`&nbsp;i64 · `regdt`&nbsp;timestamp with time zone · `eval`&nbsp;i64 · `like`&nbsp;i64 |
| `review_ranking_user` | `s3://ably-analytics-data/daily/review_ranking_train_data/{dt}/user` | csv | 실행일 | `▁█` 1634만 *(2d)* | `user_id`&nbsp;i64 · `birth_year`&nbsp;i64 |
| `review_ranking_user_goods_interaction` | `s3://ably-analytics-data/daily/review_ranking_train_data/{dt}/user_goods_interaction` | csv | 실행일 | `▁█` 4025만 *(2d)* | `user_id`&nbsp;i64 · `goods_sno`&nbsp;i64 |
| `review_ranking_user_goods_order` | `s3://ably-analytics-data/daily/review_ranking_train_data/{dt}/user_goods_order` | csv | 실행일 | `▁█` 705만 *(2d)* | `user_id`&nbsp;i64 · `goods_sno`&nbsp;i64 · `review_cnt`&nbsp;i64 · `eval_avg`&nbsp;f64 · `selected_review_sno`&nbsp;i64 · `writer_id`&nbsp;i64 · `selected_review_like`&nbsp;i64 · `selected_review_eval`&nbsp;i64 · `is_order`&nbsp;i64 |

## `prepare_outfit_contents_ranker_dataset` (매일 08:00)

- **소비 모델**: outfit_contents_ranker(+expand) · outfit_image_ranker
- **생성 로직**: 코디(outfit) 서비스 DB 덤프 5종: 콘텐츠 메타, 북마크, 좋아요, 코디-상품 매핑, 이미지 파일명.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `outfit_info_contents` | `s3://ably-analytics-data/daily/outfit_info/{dt}/outfit_contents` | csv | 실행일 | `▁█` 3334 *(2d)* | `sno`&nbsp;i64 · `member_sno`&nbsp;i64 · `created_at`&nbsp;timestamp with time zone · `updated_at`&nbsp;timestamp with time zone · `source`&nbsp;i64 · `status`&nbsp;i64 · `height_segment`&nbsp;i64 · `weight_segment`&nbsp;i64 · `age_group`&nbsp;i64 · `physical_appearance_group`&nbsp;i64 |
| `outfit_info_bookmarks` | `s3://ably-analytics-data/daily/outfit_info/{dt}/outfit_bookmarks` | csv | 실행일 | `▁█` 2만 *(2d)* | `sno`&nbsp;i64 · `member_sno`&nbsp;i64 · `outfit_sno`&nbsp;i64 · `created_at`&nbsp;timestamp with time zone |
| `outfit_info_likes` | `s3://ably-analytics-data/daily/outfit_info/{dt}/outfit_likes` | csv | 실행일 | `▁█` 4만 *(2d)* | `sno`&nbsp;i64 · `target_type_id`&nbsp;i64 · `member_sno`&nbsp;i64 · `target_sno`&nbsp;i64 · `created_at`&nbsp;timestamp with time zone |
| `outfit_info_goods_mapping` | `s3://ably-analytics-data/daily/outfit_info/{dt}/outfit_goods_mapping` | csv | 실행일 | `▁█` 72만 *(2d)* | `sno`&nbsp;i64 · `outfit_sno`&nbsp;i64 · `sequence`&nbsp;i64 · `goods_sno`&nbsp;i64 · `category_sno`&nbsp;i64 |
| `outfit_info_images` | `s3://ably-analytics-data/daily/outfit_info/{dt}/outfit_images` | csv | 실행일 | `▁█` 47만 *(2d)* | `outfit_sno`&nbsp;i64 · `file_names`&nbsp;str |

## `find_co_wearing_goods_daily_v2` (매일 12:00)

- **소비 모델**: co_wearing_goods
- **생성 로직**: 착장 이미지 비주얼 검색으로 '함께 입은 상품' 후보를 찾고 이전 결과와 병합(total_final_result). 구 DAG·os 버전은 비활성, v2가 활성 생산자.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `co_wearing_total_final_result` | `s3://ably-analytics-data/daily/co_wearing_goods/{dt}/total_final_result` | parquet | 실행일 | 267만 *(1d)* | `goodsno`&nbsp;i32 · `co_wearing_goods`&nbsp;i32[] · `co_wearing_goods_distances`&nbsp;f32[] |

## `export_gd_goods` (매일 01:00)

- **소비 모델**: lda_onboarding
- **생성 로직**: 상품 마스터(진열/품절/재고/가격/브랜드 등) 덤프.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `gd_goods` | `s3://ably-analytics-data/daily/gd_goods/{dt}` | csv | 실행일 | 4969만 *(1d)* | `goodsno`&nbsp;i64 · `img_s`&nbsp;str · `open`&nbsp;bool · `runout`&nbsp;bool · `totstock`&nbsp;i64 · `market_sno`&nbsp;i64 · `price`&nbsp;i64 · `brand_sno`&nbsp;i64 · `category_sno`&nbsp;i64 · `cate_cnt`&nbsp;i64 |

## `dump_brands_info_for_sim_brands` (매일 01:00)

- **소비 모델**: sim_brands_4910
- **생성 로직**: 브랜드-상품 매핑 정보(브랜드명/서브명) 덤프.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `brand_info` | `s3://ably-analytics-data/daily/brand_info/{dt}` | parquet | 실행일 | 1199만 *(1d)* | `brand_sno`&nbsp;i32 · `goodsno`&nbsp;i32 · `goodsnm`&nbsp;str · `brand_name`&nbsp;str · `brand_sub_name`&nbsp;str |

## `prod_custom_sql_goods` (매일 10:00)

- **소비 모델**: cbf_append · cbf_4910
- **생성 로직**: goods 스냅샷 커스텀 쿼리 결과를 ably/4910 각각 생성 후 goods.csv로 rename 복사.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `custom_sql_goods` | `s3://ably-analytics-data/daily/prod_custom_sql/goods/{dt}` | csv | 실행일 | `▁█` 1218만 *(2d)* | `market_sno`&nbsp;i64 · `sno`&nbsp;i64 · `name`&nbsp;str · `price`&nbsp;i64 · `category_sno`&nbsp;i64 · `positive_review_count`&nbsp;i64 · `total_review_count`&nbsp;i64 · `standard_category_sno`&nbsp;i64 · `market_type_sno`&nbsp;i64 |
| `custom_sql_goods_4910` | `s3://ably-analytics-data/daily/prod_custom_sql/goods_4910/{dt}` | csv | 실행일 | `▁█` 210만 *(2d)* | `market_sno`&nbsp;i64 · `sno`&nbsp;i64 · `name`&nbsp;str · `price`&nbsp;i64 · `category_sno`&nbsp;i64 · `positive_review_count`&nbsp;i64 · `total_review_count`&nbsp;i64 · `standard_category_sno`&nbsp;i64 · `market_type_sno`&nbsp;i64 |

## `daily_category_click_event` (매일 01:30)

- **소비 모델**: complementary_v2
- **생성 로직**: 유저별 카테고리 클릭 시퀀스로 word2vec 학습 → 카테고리 임베딩(word, vector) 생성.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `category_embedding` | `s3://ably-analytics-data/personalize/category_embedding/{dt}` | parquet | interval_start(전일) | 604 *(1d)* | `word`&nbsp;str · `vector`&nbsp;struct(sparse-vec) |

## `4910_sim_brands_train_data` (매일 02:30)

- **소비 모델**: sim-brands-4910 (브랜드 유사도 w2v)
- **생성 로직**: 4910 세션×브랜드 상호작용 집계(클릭/찜/장바구니/주문 수, 첫 이벤트 시각). 2026-07-23 신규 추가(정유석, #16656).

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `prod_4910_sim_brands_train_data` | `s3://ably-analytics-data/daily/prod_4910_sim_brands_train_data/session_brand/{dt}` | csv | 실행일 | 354만 *(1d)* | `USER_ID`&nbsp;str · `DT`&nbsp;str · `BRAND_SNO`&nbsp;i32 · `CLICK_CNT`&nbsp;i64 · `LIKE_CNT`&nbsp;i64 · `CART_CNT`&nbsp;i64 · `ORDER_CNT`&nbsp;i64 · `FIRST_TS`&nbsp;i64 |

## `ml_relevance_daily_events` (매일 01:00 — ⚠️ 2026-05-22부터 DAG 중단)

- **소비 모델**: relevance_embedding
- **생성 로직**: 검색 이벤트에서 쿼리-상품 클릭 학습셋(train/val parquet) 생성. 현재 DAG 자체가 실행되지 않아(마지막 파티션 dt=20260522) 검증도 미동작 — 파이프라인 상태 점검 필요.

| 데이터셋 | 경로 | 포맷 | dt 기준 | row 추이 (최근 14d) | 스키마 (실측: 컬럼·타입) |
| --- | --- | --- | --- | --- | --- |
| `relevance_train_dataset_train` | `s3://ably-analytics-data/daily/relevance_embedding/train_dataset/dt={dt}/train.parquet` | parquet | 실행일 | — *(DAG 중단)* | `query`&nbsp;str · `goods_sno`&nbsp;str · `clicks`&nbsp;i64 · `dt`&nbsp;i64 |
| `relevance_train_dataset_val` | `s3://ably-analytics-data/daily/relevance_embedding/train_dataset/dt={dt}/val.parquet` | parquet | 실행일 | — *(DAG 중단)* | `query`&nbsp;str · `goods_sno`&nbsp;str · `clicks`&nbsp;i64 · `dt`&nbsp;i64 |

---

## 부록 A. 검증 미적용 입력 (사유별)

- **airflow 내 생산, 추가 가능**: bandit `df_v5`(bandit_v5 소비)·`goods_v6`(bandit_v4/v5·two_tower_v5 소비)·`df_input_v2`(two_tower·two_tower_v5 소비)·`user_history_v2`(21d 상류) — bandit 계열은 개발 활발(주 단위 산출물 추가)하므로 주기적 재점검 필요, reco_etl `goods_click_v2`·purchase 3종(2차 상류), `reco_raw{,_parquet}/standard_display_category_relation`(tag_classifier·visual_search_es 소비 — etl_reco_raw_daily_parquet에 spec 추가 가능), category-classifier 3종(콘텐츠), outfit-image-ranker-score(모델 스코어), SE팀 perf_features 3종(타팀 소유)
- **생산 중이나 소비 소멸**: bandit `goods_v5` — 7/21 도입 후 1주 만에 goods_v6로 대체되어 현재 무소비 (생산 정리 후보)
- **생산자 없음/외부**: `models/pretrained_text/`(2021 정적), `info/`(정적), 파스텔 버킷(amood 원천), `pretrained-seasonality-fasttext/`
- ~~생산자 미상~~ **해소**: `ably-datalake/reco/db/event_stat/goods_stat/` — OM 리니지로 확인: ScyllaDB(`ably.goods`) → `dump_scylla_goods_statistics` DAG(`dags/de/`)가 매일 덤프 (`aggregate_event_to_scylla_total`이 트리거). 검증 추가 가능해짐
- **모델 아티팩트(비대상)**: `models/*/output/` 재사용(substitute 임베딩, warm-start 복사 등)

## 부록 B. 소비 정의는 있으나 생산 중단(stale) — 파이프라인 정리 필요

| 경로 | 소비 파이프라인 | 마지막 생산 |
| --- | --- | --- |
| `personalize/ranker_today/{dt}/df_labels/` | ranker_din_v7 | all_chip 전환 이후 중단 |
| `personalize/bandit/{dt}/{hour}/goods_v3` | bandit(v1) | goods_v4 전환 이후 중단 |
| `ably-datalake/reco/db/reco_raw/member/`(csv) | related_keywords_v2 | 2026-03-30 |
| `personalize/complementary_v2_train/*` | complementary_v2 | 파티션 없음 (생산 DAG no_use_dags) |
| `daily/goods_stat/28d/` | multi_interest_bam | 2024-09-19 |
| `daily/relevance_embedding/train_dataset/` | relevance_embedding | 2026-05-22 (DAG 중단) |
