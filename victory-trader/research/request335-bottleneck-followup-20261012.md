# R335 — original-identity bottlenecks and current raw-evidence availability

기존 전체 수집은 중단되지 않았다. 최종 커밋
`15d2eec88038b77889fc90113dae5dd099ad2911`에서 원래 1,235개 대상 /2,470개
API 체인, 전 구간 감사와 실제 archive 복원 replay가 완료됐다. 수집 4,453.065초
(약74.2분), 후처리 1,547.850초(약25.8분)가 걸렸다. 요청 시작 간격과 실제
2.8GB 응답 처리·복원 검증에 든 시간이며, 백그라운드 연구가 이후까지 계속
실행되고 있었다는 뜻은 아니다. 마지막 응답 후 작업은 종료됐다.

이번 후속은 새 시장 요청과 추가 비용 없이, 고정된 공개 감사 기록을 원래
대상에 연결했다. 전체 실험을 다시 다운로드하거나 경제적 라벨을 만들지 않았다.

## 현재 원본 보존 상태

현 실행환경에서 이전 전체 census manifest, 자체 archive, 복원본 manifest와
성공한 full-window pilot manifest는 발견되지 않았다. 알려진 경로와 사용 가능한
작업 파일명 목록을 검사했다. 삭제 원인이나 모든 원격 저장소의 부재를 단정하지
않는다. 이전의 실제 복원 성공 기록도 변경하지 않는다.

이 때문에 **이 세션에서 전체 원본 replay를 다시 수행할 수는 없다.** 공개 결과의
해시·모든 항목은 GitHub에서 복원했지만, 집계 기록은 원자료가 아니다. 남아 있는
오래된 첫1분 표본도 다른 수집의 원본을 대체하지 않는다. 이번 결과의 증거 수준은
`hash_pinned_published_diagnostics_not_current_raw_replay`로 명시했다.
[현재 확인 기록](request335-raw-availability-20261012.json)을 참고한다.

## 원래 1,235개 대상별 결과

학습 407 /재사용 개발 828, 모든12일과 원래 구간을 보존했다. 아래는 구간 안에서
해당 상태가 한 번 이상 관측된 **대상 수**다. 상태끼리 겹치므로 열을 합산하지 않는다.

| 구간 내 관측 | 학습407 | 재사용 개발828 | 전체1,235 |
|---|---:|---:|---:|
| 같은 provider 시각의 복수 행 |74|151|225|
| 같은 시각의 가격 또는 수량 필드 차이 |74|151|225|
| 같은 시각 묶음이 페이지 경계를 통과 |10|9|19|
| 잠김 호가 |220|420|640|
| 역전 호가 |89|178|267|
| 비활성 호가 측면 |1|5|6|
| 현재 trade update 표시 |9|12|21|
| 점검한 상태 중 하나 이상 |250|490|740|
| 위 상태가 관측되지 않음 |157|338|495|
| 빈 체결/호가 응답 구간 |0|0|0|
| 점검한 필수 필드 누락/스키마 오류 |0|0|0|
| 응답 내 provider 시각 역전 |0|0|0|

36개의 현재 canceled trade 관측은 원래21개 대상에 걸쳐 있다. 동일 티커의
다른 날짜·구간을 섞지 않고 원래 identity index, 범위, stream, 행 수와 수집 상태를
모두 연결했다. 단위는 겹칠 수 있는 구간별 관측이며 유일한 시장 사건 수가 아니다.
가격·수량 차이는 scalar type과 Decimal text를 포함한 정확한 decoded-field
비교다. 숫자의 실제 가격 차이라고 확대 해석하지 않는다.

**740개를 일괄 제외하거나, 나머지495개를 검증된 표본으로 채택하지 않는다.**
구간 어딘가의 잠김/역전/동일 시각 행은 모든 판단 시점의 체결 불가능을 뜻하지
않는다. 반대로 이런 상태가 없다고 늦게 도착한 정보나 정정 이력이 없었던 것도
아니다. 따라서 이 비율은 execution-resolution이 아니며, 원래90% 문턱과 비교해
통과·실패를 판정하지 않는다. 실제 execution-resolution은 NOT_EVALUATED다.

## 시간·정정에 관한 공식 문서 재검토

[공식 Market Data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq)는 bar
집계가 trade의 SIP 시각을 사용한다고 설명한다. 이 내용은 trade clock에 대한
관련 근거이며 무시하지 않는다. 다만 bar 집계 설명만으로 내려받은 모든 historical
quote의 `t` 의미, 전략 수신 시각, 동일 시각의 총순서나 original/correction의
전달 이력을 확인했다고 할 수 없다. 그 확장은 아직 추론이며 adapter에서
provider clock을 SIP/strategy receipt로 자동 치환하지 않았다.

[공식 실시간 stock 문서](https://docs.alpaca.markets/us/docs/real-time-stock-pricing-data)는
이미 전송한 trade의 correction/cancel이 별도 메시지로 전달됨을 설명한다.
또 timestamp가 이전 분에 속하는 trade가 뒤늦게 도착해 bar가 갱신되는 사례를
제시한다. 따라서 과거 시각으로 정렬된 현재 REST 기록만으로 당시 전략이 알 수
있었던 내용을 재구성하는 것은 별도 검증이 필요하다. 이것은 문서에서 도출한
판단이며 이번 세션에서 실제 지연이나 정정 도착 시각을 측정한 결과가 아니다.

## 다음 실제 연구로 넘어가는 조건과 비용 판단

1. 기존 private census archive가 별도로 존재하면 먼저 exact archive hash를
   확인하고 안전하게 복원한 뒤, 고정된 collector replay로 manifest·전체 page·token
   chain·denominator를 검증한다. 공개 해시만으로 원본이 존재한다고 처리하지 않는다.
2. 원본 복원이 불가능하면 새 원본 수집은 독립 계약·provenance로 등록해야 한다.
   이전 결과와 이어 붙이지 않는다. 이번처럼 현재 기록만으로 해결할 수 있는
   분석을 위해 약74분의 전체 수집을 반복하는 것은 이 단계의 병목을 해결하지 않는다.
3. 새 장시간 수집 전에 적법하고 승인된 지속 보존 위치를 정하고, 작은 실제
   private checkpoint의 저장·재획득·hash 검증까지 확인한다. 동일 실행환경의
   archive/restore는 외부 지속 보존을 보장하지 않는다. 원본을 공개 repo/CI에 넣지 않는다.
4. 시간 의미·정보 도착·정정/취소의 as-observed 이력과 날짜별 condition/halt/OTC/
   inactive-symbol 완전성 근거를 확보한 뒤, causal availability·latency·quote age·
   routing/depth/fill/fees/capital 규칙으로 원래 모든 판단을 평가한다. 문서에 없는
   시각·순서·체결을 임의로 채워 넣는 방법은 연구 엄밀성을 보존하지 못한다.

새 유료 구독을 권하거나 실행하지 않았다. historical SIP 접근 권한 자체는 이전
무료 실제 수집으로 확인된 상태다. 단순 재다운로드나 요금제 변경이 누락된
arrival/correction 이력을 제공한다고 가정할 근거는 없다. 기존 Massive Starter
유지, 추가 비용USD0. 경제 라벨0 /fit0 /scientifically qualified0.

June HOLD /July·August SEALED, R331 pooled34.8352% 실패, 원래90% resolution,
all-phase/all-clock denominator,60초 hold,3초 boundary와 BASE−2.5% completed-close
stop을 보존했다. 수익성 실험은 위 의존성이 해결된 뒤 진행한다.

## 구현·재현·검증

`r335_census_bottlenecks.py`는 고정 source plan과 공개 ledger/acquisition/audit의
uncompressed SHA256을 확인한다. 원래 모든 stream과 identity/window 연결, audit와
acquisition의 hash 연결, 행 수·상태, nonnegative integer counts와 unknown economic
status를 검사하고, 원래 대상별/집단별/날짜별·상태 조합별 수치를 출력한다.
이번 분석에서 raw replay를 했다고 표시할 수 없으며, 상태가 없는 대상을 qualified로
올리지 않는다. frozen collector/adapter/기존 경제적 연구 코드는 변경하지 않았다.

- [전체 대상별 후속 분석](request335-bottlenecks-20261012.json.gz)
- [집단·날짜별 요약](request335-bottlenecks-summary-20261012.json)
- [현재 원자료 가용성](request335-raw-availability-20261012.json)
- [검증 기록](request335-bottlenecks-validation-20261012.json)

재현 명령:

```bash
python -m victory_trader.r335_census_bottlenecks --output research/request335-bottlenecks-20261012.json.gz
```

17개 회귀 테스트는 원래 집단/대상 보존, 다른 날짜의 같은 티커를 잘못 섞는 경우,
관측 상태가 없는 대상의 unknown 유지, 공개 원본 변경, 잘못된 수치·범위·연결과
거짓 source/economic 승격을 검증한다. 전체 suite와 CI 결과는 검증 기록 및 PR에
반영한다. 실행 테스트에는 새 시장 API 요청이나 credentials가 사용되지 않는다.
