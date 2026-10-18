# R335: qualify the source before collecting only the fixed target's support

최근 작업은 저장·복구와 반복 수집에 과도하게 치우쳤다. 이번에는 원래 연구 대상을
유지하면서 실제 평가에 필요한 시간 범위를 계산하고, 적합하지 않은 공급원에
전체 수집을 반복하지 않는 방향으로 전환했다. 새 수집기나 자동 실행 workflow는 없다.

## 실제로 수행한 연구·확인

- 기존 공개 원본 목록의 해시를 인증하고 **1,235개 대상 / 2,470개 스트림**을 모두
  유지한 별도 target-support 계획을 계산했다. 학습407 / 재사용 개발828은 같다.
- 기존 비공개 날짜 archive 10개의 현재 보관 기록을 찾았다. 이는 모든 원본이 현재
  재검증됐다는 뜻은 아니다. 남은 May19/20의 완료·보관을 확인한 기록은 없다.
- 수익을 기준으로 고르지 않고 첫 원래 날짜 May5의 실제 archive를 재획득했다.
  261개 응답의 바이트 해시가 내부 취득 기록과 일치하며, 204개 파티션의 페이지·행
  수가 게시된 Oct12 recovery ledger와 일치한다. 체결438,903 / 호가633,648행이다.
  현재 archive의 독립적인 과거 해시 pin은 없으므로 전체 acquisition replay나
  출처 인증 성공으로 표현하지 않는다. 이 점검은 파일 무결성·필드 감사다.
- 그 표본에는 별도 `sip_timestamp`, 수신 시각, sequence 필드가 없다. 단일 `t`를
  임의로 전략 수신 시각으로 바꾸지 않는다. 같은 API의 재수집은 이 구조적 공백을
  해결하지 못한다. 원래 경제 라벨·모델 학습·수익성 실험은 추가하지 않았다.

## 실험 범위를 보존하며 줄일 수 있는 부분

기존 원본 계획은 각 대상의 HOT−60초(정규장 시작 이전 제외)부터 **장 마감까지**다.
고정 R330/R331 ENTER 계약에서 결정 시각은 `min(HOT+60분, 장 마감)` 전이며,
모든 청산 제출은 이 terminal 이하다. 청산 reference는 제출 후 **3초 이내**까지
허용한다. 따라서 마지막 허용 reference의 **전체 1초 봉**을 포함해야 한다.

```text
start = original_start                         # 변경 없음
end_exclusive = min(original_session_end, HOT + 60min + 3s + 1s)
```

| 고정 목록의 시간 범위 합계(종목×스트림, 초) | 기존 | 별도 target-support 계획 |
|---|---:|---:|
| 전체2,470개 스트림 |38,748,960|8,724,720|
| 원래 대상 수 |1,235|1,235|
| 학습 / 재사용 개발 |407 / 828|407 / 828|

시간 범위 합계는 **77.483989% 감소**한다. 실제 요청 수·행 수·바이트·실행 시간의
감소율은 아직 측정하지 않았다. 장 마감에 HOT가 생긴 원래2개 대상도 제거하지 않는다.

기존 `enter_outcome`은 수정하지 않았다. 9개 테스트가 terminal 이후 청산, 정확히
+3초의 허용 reference, +4초의 미해결 reference, 정규장 끝, 희소·급락 경로 및
양쪽 스트림/모든 대상 보존을 검사한다. 384개의 합성 경로/결정 조합에서 전체와
범위 제한 입력의 결과 필드를 비교했다. 실제 시장 데이터의 경제 라벨은 계산하지 않았다.

**이 동등성의 적용 범위:** 고정된 cached-second R330/R331 즉시 ENTER reference
평가에 필요한 지원 범위다. 전체 장의 R332 missing-reference 원인 진단이나 기존
full-session R335 census를 대체하지 않는다. 새 정책, 120초 feature 재구축, 장시간
보유, 계좌 검증에 그대로 확대 적용하지 않는다. 이후 실험마다 필요한 의존 범위를
별도 산출한다. 궁극적인 연속 스캔/WAIT·ENTER·ADD·HOLD·REDUCE·EXIT 목표는 같다.

새 공급원의 조회 필터 시계가 이 reference 시계와 일치한다는 근거도 필요하다.
participant 시각으로 잘린 응답을 SIP 공개 시각의 완전한 이력으로 표현하지 않는다.
초기 NBBO 상태를 위한 이전 호가와 창 밖에서 시작한 거래의 정정·취소 연결은
별도 context 추출이 필요하다. 위77.48%를 이런 추가 자료까지 포함한 절감으로
주장하지 않는다. 이 계획만으로 collection/source_ready를 켜지 않는다.

## 새로운 확보 경로와 순서

**우선 확인할 후보는 기존 기관 구독을 통한 NYSE Daily TAQ 원자료의 서버 측 추출**이다.
연세대 공식 안내 자료에 WRDS가 기재되어 있다. 해당 자료는 오래됐으며 현재 구독,
학부생 접근, Daily TAQ 상품, May2026 접근·반출 권한은 아직 확인하지 못했다.
따라서 무료 이용 가능성이 있는 후보이며, 확보된 데이터 공급원은 아니다.

NYSE 명세에는 SIP 공개 시각, participant 시각, 메시지 번호, trade ID와 정정 코드,
NBBO·관리·LULD 파일이 설명되어 있다. 하지만 이 명세의 존재만으로 actual export의
충분성을 인증하지 않는다. 특히 Daily TAQ의 정정된 원래 행과 최종 정정 기록이
**중간 정정 전달 이력 전체**를 보존하는지는 별도 검증해야 한다. 채널별 sequence는
체결·호가 전체의 하나의 순서로 간주하지 않는다. 2026년 해당 시점의 단위/명세,
WRDS가 제공하는 실제 시간 정밀도와 원래 문자열 반출 가능 여부도 확인한다.

| 경로 | 이번 판단 | 다음 조치 |
|---|---|---|
| 보관된 Alpaca10일치 | 기존 관측·무결성 감사에 재사용. source qualification 미완료 | 재수집하지 않음. 검증된 맥락에서만 사용 |
| 동일 Alpaca 전체 재수집 | 알려진 정보 공백을 해소하지 못함 | 중단 |
| 기관 NYSE Daily TAQ | 누락 필드 확보 가능성이 있는 조건부 후보 | 이용권한 → 실제 표본의 정밀도·시계·정정 복원성 순으로 검사 |
| Databento EQUS.MINI | component-venue 합성 BBO이며 sequence=0. 기존 SIP/NBBO와 동등성 미확인 | 무료 credit만 보고 기존 원자료로 대체하지 않음 |
| Massive 유료 upgrade | 현재 미승인 | 결제·변경하지 않음 |

### 권한 확인 이후의 작은 사전 검사

1. 상품·개인 연구/반출 권한과 원래12일(2026-05-05~20 중 원래 거래일) 접근 여부를
   metadata에서 확인한다. 이 단계는 대량 시장 데이터 다운로드를 하지 않는다.
2. 기존12일 qualification 목록의 종목을 그대로 사용한다. 날짜별 첫1분의 trade /
   NBBO 표본과 필수 clock/unit/correction 명세만 확인한다. 새 종목/성과 좋은 날짜를
   고르지 않는다. 경계 호가·연결된 정정 이력이 없다면 부재를 명시한다.
3. 나노초 원래 값, SIP 공개 시각, 채널별 순서, 원래/정정/취소 연결, NBBO 단위와
   거래정지/LULD 상태를 복원할 수 있는지 검사한다. 불충분한 export는 확대하지 않는다.
   가격을 보고 적합성을 결정하지 않는다. 실제 수집 전 별도 bounded 계약을 게시한다.
4. 충분성을 확인한 공급원에서만 고정 목표 범위와 필요한 boundary context를 추출한다.
   원래12일/1,235개 전체 목록으로 되돌려 검증하며 실패/미해결도 분모에 유지한다.
   보관본과 해시가 같은 이미 확보된 파일은 재사용한다.
5. 원래90% execution-resolution 조건과 별도 출처 조건을 통과해야 경제 라벨과
   모델 학습으로 넘어간다. 원래34.8352% 실패는 그대로 보존한다.

권한이나 필수 정보가 부족하면 해당 후보의 검사를 끝내고 부족한 **상품/필드**를
명시한다. 저장 도구 추가 개발이나 같은 데이터의 전체 재수집으로 대체하지 않는다.
지원팀 메시지, 새 계정 생성, 구독·결제, 주문은 실행하지 않았다.

## 증거·재현

- [고정 전체 목록의 별도 범위 계획](request335-target-support-plan-20261018.json.gz)
- [현재 재획득한 첫 날짜의 필드·내부 해시 점검](request335-retained-probe-20261018.json)
- [기존 계약과 경계 동등성 검사](../tests/test_r335_target_support.py)
- [현재 진행·검증 기록](request335-efficient-validation-20261018.json)

```bash
python -m victory_trader.r335_target_support \
  --ledger research/request335-input-ledger.json.gz --output /tmp/r335-target-plan.json.gz
python research/analyze_request335_retained_probe.py \
  --archive <private-retained-first-date.tar.gz> \
  --reference research/request335-recovery-census-interrupted-ledger-20261012.json.gz \
  --output /tmp/r335-retained-probe.json
python -m pytest -q tests/test_r335_target_support.py
```

고정 계약 source SHA256:
`178548ba93fd19978fde79a1fe050b947ab2df89c162bc52bf3ac15d16ab2bd1`.

공식 근거(2026-10-18 열람):

- [NYSE Daily TAQ v4.1a](https://www.nyse.com/publicdocs/nyse/data/Daily_TAQ_Client_Spec_v4.1a.pdf),
  §§1,3,5,8,9. 2025년 명세이며 May2026의 유효 명세/단위는 확인 대상이다.
- [연세대 공식 안내 자료의 WRDS 목록](https://enf.yonsei.ac.kr/enf/notice2.do?articleNo=48933&attachNo=28865&mode=download),
  PDF p39. 현재 구독이나 TAQ 권한을 증명하지 않는다.
- [Databento EQUS.MINI 명세](https://databento.com/docs/venues-and-datasets/equs-mini),
  aggregated BBO / venues / timestamp 항목.
- [Alpaca 담당자의 Timestamp1 설명](https://forum.alpaca.markets/t/v2-api-what-is-the-timestamp-in-stream-trade-schema-and-how-to-filter-late-reported-trades/4839),
  2025-08-09 답변. stream 설명을 역사 REST의 수신 이력 증거로 확대하지 않는다.

추가 비용USD0 / 새 시장 요청0 / 새 경제 라벨0 / fit0 / source_ready=false.
June HOLD / July·August SEALED. 실행 중인 장기 수집은 없으며 다음 의존성은
**기관 원자료 접근과 실제 export 적합성 확인**이다.
