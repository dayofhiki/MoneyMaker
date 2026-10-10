2026-10-10 전체 구간 후속: 고정 24개 구간의 무료 수집에서 14회 요청, 13개 HTTP 200, 체결 12,769건 / 호가 17,825건을 보존했다. 12개 API 종료 후 전송 오류와 실행환경의 네트워크 승인 취소로 중단했다. 추가 $0, source/local/CI 1,525 tests 통과, 전체 수익성 검증은 미완료다. [최신 전체 구간 결과와 예산](request335-full-window-followup-20261010.md)을 먼저 참고하며 아래 이전 기록은 보존한다.

2026-10-10 단위·페이지 후속: 공식 historical schema에 따라 May 호가 수량을
shares로 확인했다. 별도 사전등록 후 AIRS 새 조회 1회에서 367행·terminal 및
원래 200행 prefix 일치를 관측했다. 기존 미완결 기록은 변경하지 않았다. 추가 $0,
local/해당 source CI 1,492 tests 통과. 시계·동일 시각 순서·정정 전달 이력과
full census/체결·수익 검증은 미완료다.
[최신 단위·순서 감사와 페이지 결과](request335-semantics-followup-20261010.md)를
우선 참고하며 아래 기록은 이전 취득과 조사 상태를 보존한다.

2026-10-10 실제 접근 후속: Paper 키 연결 후 등록된 표본 31 요청 모두 200,
체결 674행/호가 812행, 23 terminal pair/1 미완결 호가 pair를 확보했다. 추가 $0.
[실제 접근 결과](request335-free-access-result-20261010.md)를 우선 참고한다.
무료 접근은 확인했지만 full census/NBBO·clock·단위·순서·체결/계좌 검증은 미완료다.
이후 문서는 이전 단계의 조사와 실행 상태를 보존한 것이다.

2026-10-10 이전 후속: 고정 무료 SIP 접근 표본 수집기·오프라인 replay와 사용자 실행
안내를 추가했다. 실제 시장 요청 0, full census와 경제적 연구는 계속 blocked다.
[최신 실행 안내](request335-free-qualification-guide.md)와
[후속 검증 기록](request335-free-qualification-validation.json)을 참고한다.
같은 날 계정 후속: 실제 Paper Basic 무료 플랜을 확인했다. 최신 Basic 안내와
historical SIP FAQ 및 개인/비상업 Terms를 근거로 등록된 접근 표본을 진행한다.
지원팀의 별도 답변은 필수 조건에서 제외했다. 아래 문서 충돌 검토는 이전 기록이며,
actual access 및 NBBO/clock/units/order/completeness 미확인 상태는 유지한다.
[계정 확인 및 결정 수정](request335-alpaca-account-review.md)을 참고한다.
기존 결과와 아래 이전 단계 기록은 보존한다.

# R335: 비용을 낮추면서 원자료·실험 무결성을 보존하는 경로

확인일: 2026-10-09. 출발점은 PR108의 `afc106e77c36c219482641c2114f441fbd2e2b93`.
사용자가 현재 Massive Stocks Starter($29/월)라고 확인했다. 추가 구독, 구매,
유료 API 호출 승인 없음. 이번 단계에서 새 시장 API 요청·원자료 수집·경제적
라벨·모델 학습은 모두 0이다. 공개 문서 조회와 로컬 코드 검증만 수행한다.

**권장 결정: Advanced를 바로 구매하지 않는다. 적법하게 보유한 원본이나
기관 Daily TAQ 접근을 확인하고, 무료 Alpaca historical SIP를 먼저 기술적으로
자격 검증한다. 원래 데이터 조건을 충족하지 못하면 결측 상태를 유지하고
정확한 12일/1,235개 대상의 별도 견적·학술 지원으로 넘어간다.** 무료라는
이유로 비동등한 피드나 불명확한 시계를 실제 NBBO로 인정하지 않는다.

## 1. 최신 연구와 정확한 막힘

| 연구 | 확인된 결과 | 다음 연구에 남긴 조건 |
|---|---|---|
| R331 | primary May6–8: 156개 compatible complete episode, positive 59개; pooled 해소율 34.8352%; 등록된 10개 점검 중 90% 해소 기준 실패 | 150/30 episode floor와 원래 denominator 유지; positive 사례가 있다고 성공 아님 |
| R332 | 1,235개 전체의 원래 second 경로 존재. primary unresolved의 98.6936%는 경계 안에 다음 print가 있지만 3초보다 늦음 | no-trade, 조건 제외, halt, provider 손실, 실제 미체결을 구분할 독립 원자료 부족 |
| R333 | 기존 계정 REST qualification: trades와 quotes 첫 요청 각각 403 NOT_AUTHORIZED | 12개 60초 표본은 접근 점검일 뿐 전체 실험 아님 |
| R334 | 12일 × trades/quotes = 24개 Flat File HEAD 모두 403, 본문 다운로드 없음 | 정확한 S3 거부 원인과 원자료 크기는 미확인; 65-byte 오류 길이는 데이터 크기 아님 |
| R335 / PR108 | 오프라인 export inventory 준비 완료; 407 training + 828 reused development의 2,470 stream entry 전부 NOT_SUPPLIED | hashes/schema 통과만으로 license, completeness, causal execution을 인증하지 않음 |

현재 막힘은 **고정 시간창 전체의 원시 체결 및 공식 consolidated NBBO,
그 데이터의 획득·완결성·시계·정정 근거**다. 더 많은 aggregate나 다른 value
fit으로 이를 대신할 수 없다. 최신 PR108은 draft이고 모델 promotion은 없다.

원래 목표는 지속적인 시장 scan과 종목별 WAIT / ENTER / ADD / HOLD / REDUCE /
EXIT, 거래비용·자본과 기회비용을 함께 관리하는 after-cost 계좌 성장이다.
R335는 그 목표를 위한 관측/실행 검증 단계이며 수익성 실험을 완료한 단계가 아니다.

## 2. 바꾸지 않는 범위와 데이터 조건

전체 날짜는 May5,6,7,8,11,12,13,14,15,18,19,20, 2026.
407 training / 828 reused development와 모든 phase/미관측 identity를 유지한다.
primary 305개 May6–8을 별도로 보고한다. 각 원래 identity의 acquisition window는
`[max(regular_open, HOT−60s), regular_close)`다. 12개 qualification anchor는
원래 R333의 날짜별 lexical-first(day,ticker,HOT)와 첫 60초를 유지한다.
전체 성공은 이 24개 trade/quote sample pair의 성공으로 대체하지 않는다.
June HOLD와 July/August 봉인은 유지하며 해당 월의 무료 시장 표본도 열지 않는다.

| 항목 | 필요한 조건 | 금지하는 대체/추론 |
|---|---|---|
| 시간 정밀도 | 원본 nanosecond integer 또는 9자리 RFC3339 문자열을 정확히 보존; 원본 해상도와 정확도를 구분 | 초/ms를 ns로 곱한다고 원래 정밀도가 생기지 않음; float/datetime microsecond 왕복 금지 |
| 시계 의미 | SIP 처리/수신, participant/event, TRF clock을 구분하고 있으면 모두 유지. strategy/broker receipt는 별도 근거 또는 등록된 latency scenario | provider `t`를 의미 확인 없이 SIP/실제 주문 수신으로 alias하지 않음 |
| NBBO | 공식 CTA/UTP consolidated NBBO의 bid/ask, venue, displayed size, condition과 날짜별 단위 | IEX/Nasdaq Basic/부분 거래소 composite/추정 spread는 동등하지 않음 |
| Trades | 필터 전 price, shares/decimal size, exchange/TRF, conditions, correction/cancel, ID/sequence와 명세; 존재하지 않는 필드는 결측 | bar에 없는 체결을 없었던 거래로 취급하거나 corrected final history를 as-received로 취급하지 않음 |
| 시간 순서 | acquisition order와 clock/sequence 의미, late/out-of-order/동일 clock 충돌을 보존; causal availability prefix로 재생 | participant clock 재정렬로 미래 정보 유입, 임의 tie-break, 무단 dedup 금지 |
| 완결성 | 정확한 bounds, 모든 page/file 연결과 terminal marker, status/count, 원본 byte hash, manifest 및 정규화 버전 | 처음/마지막 tick, 200 응답, count 또는 빈 export만으로 complete/no-trade 인증 금지 |
| 유동성/체결 | stale/locked/crossed/zero book, eligibility, symbol-date round-lot units, displayed-size cap, partial/미체결/terminal uncertainty를 명시 | NBBO top-of-book은 depth, routing, queue priority, 실제 체결의 증거 아님 |
| 비용/슬리피지 | 관측 ask→bid crossing과 추가 adverse slippage/impact, broker commission와 날짜별 sell/regulatory fees를 분리해 새 proxy 전에 등록 | 임의 fill price, synthetic quote, spread의 이중 부과/누락으로 수익을 입증하지 않음 |
| 계좌/기회비용 | 같은 chronological cash/position/slots, unresolved capital, censoring/drawdown, 대기와 다른 종목의 기회비용 보존 | 미해결 포지션을 cash/loss로 강제 종료하거나 known subset만 재가중하지 않음 |

기존 R330/R331 계약은 **60초 hold, BASE after-cost −2.5% completed-close stop,
entry/exit reference 최대 3초**, 기존 HOT/session 경계와 동일한 시나리오다.
BASE는 편도 half-spread 25bps(최소 1cent), adverse slippage 25bps이고 기존
`sell_fee_bps=0`이다. LIGHT/STRESS도 그대로 보존한다. 이들은 시장에서 측정한
체결/실제 all-in fee가 아니라 기존 가정이다. 원자료가 생겨도 기존 failed90%
결과를 새 NBBO proxy로 소급해서 통과시키지 않는다. 실측 crossing·추가 비용,
주문 종류/limit/cancel, availability·latency·quote age/size, stop·terminal 처리는
**별도 사전 등록된 이후 실험**에만 사용한다.

## 3. 저비용 데이터 경로 비교

USD 공시 가격은 checkout/세금/환율/일할 계산/맞춤 견적이 아니다. 미공개 가격은
추정 숫자를 만들지 않았다. 모든 공급자의 실제 1,235개 symbol-date, rename/delist/
OTC 여부, 페이지 완결성과 사용·보관·재현·공유 권리는 별도로 확인해야 한다.

| 경로 | 추가 비용 / 공시 가격 | 범위·품질·시간 | 누락/라이선스/재현성 및 R335 적합성 |
|---|---|---|---|
| 현재 Massive Starter | 추가 $0; 현재 $29/월 | 5년 second/minute aggregate, 15분 delayed, unlimited REST; tick trades/quotes 제외 | 원래 aggregates 재검증에 사용 가능. 독립 Trades/NBBO는 확보 불가. S3 access 자체가 tick entitlement 아님 |
| Massive Advanced (비용 비교 기준) | $199/월; Starter를 대체하는 정가 차이 $170/월, 실제 청구/일할 계산 미확인 | 원시 trades+NBBO, ns, all history since2003, REST/Flat Files | 필요한 dataset은 공시 포함. completeness/causal fill은 별도 검증 필요. 현재 예산 때문에 구매 권장하지 않으며 승인 없음 |
| 이미 적법하게 확보한 원본 재사용 | 추가 $0 | 계약에 따라 다름; 동일 12일 전체창+원본 clock/fields 필요 | 이번 authorized catalog에는 0개. 기존 R332 seconds와 R333 aggregate sample은 NBBO 대체 아님. 타인 구독 export/공개 GitHub 파일은 자동 합법 아님; license·보관 권리·page provenance 필요 |
| Alpaca Basic historical `feed=sip` | 후보 추가 $0; Basic Free | 공식 FAQ: end가 15분 이상 과거이면 SIP historical를 무구독 query 가능; since2016, 200req/min. historical quotes는 NBBO 후보; raw RFC3339 ns | 아래 충돌·field/ordering/correction·OTC·round-lot 문제가 미확인. actual account/license와 bounded same-date sample 성공 전에는 동등성 인증 불가. IEX fallback 금지 |
| Alpaca Algo Trader Plus | $99/월 별도; Starter 유지 시 총 $128/월 | historical since2016 동일, 10,000req/min 및 realtime SIP | 역사 데이터의 필드 문제를 $99로 해결한다고 가정하지 않음. 무료 historical 접근이 되면 R335 때문에 throughput만 구매할 근거 없음 |
| 기관 NYSE Daily TAQ / WRDS | 이미 해당 TAQ 모듈에 적법한 권한이 있으면 개인 추가 $0; 신규 기관/직접 구매 견적 미확인 | Daily TAQ all CTA/UTP trades, quotes, NBBO, master/admin; 1993–present. 현재 원본 명세 9자리 ns, sequence/다중 clocks | 유력 full-source 경로. WRDS 계정만으로 TAQ subscription을 추정하지 않음; 기관 재배포/상업 목적 제한, update lag, WRDS에서 timestamp 저장 정밀도 점검. monthly/초 자료나 derived intraday indicators는 대체 아님 |
| Massive 학술 할인/연구 지원/12일 별도 export | 할인율·subset 구매 총액 공개 확정 없음; 서면 견적 필요 | 학생/교수의 대학 이메일을 검증하면 individual plan 할인, tier 데이터는 동일 | 할인은 자동 적용되지 않음. 개인 plan 무료 trial 없음; business evaluation은 sales가 scope 결정. 정액 $199보다 저렴한 subset 공급은 아직 확인되지 않은 협상 후보, 보장 아님 |
| Massive x402 요청별 과금 | route별 USDC 가격; R335 tick route 총액 산정 불가 | 2026-09 공개 범위: bars, snapshots/FMV, indicators, reference/news/filings | 공식 출시/문서 목록에 원시 historical Trades/NBBO가 없음. 알려진 경로로는 부적합. wallet 생성·결제·유료 endpoint 호출하지 않음 |
| ThetaData historical UTP/CTA | stock tick tier의 현재 확정 USD 견적 미확인; 무료 tier EOD만 | Pro tick, UTP2012; CTA 문서2020 vs 상품페이지2017 불일치. v3 quote 출력은 `...SSS` ms; `venue` 기본 `nqb` | `utp_cta` 및 raw tick interval 명시 필요. `nqb`/resampled quote 비동등. ns ordering, correction, units 원본 제공을 증명하기 전에는 R335 대체 승인 불가. pricing 화면의 $40/$80/$160은 **options**이므로 stock 가격으로 옮기지 않음 |
| Databento historical | GB별 usage estimate, 신규 $125 credit; 고정 범위 총액 미확인 | direct feeds ns·receipt clocks; venue별 archive. EQUS.MINI는 부분 시장 synthetic mini-NBBO | 모든 prop-feed BBO를 합친 synthetic NBBO도 SIP와 odd-lot/clock/venue 규칙이 다르며 LTSE 누락 문서 있음. $125 credit을 같은 SIP 자료로 간주하지 않음. 공식 SIP 원본 계약 없으면 non-equivalent; paid API는 credit이 있어도 실행 안 함 |
| algoseek Equity TAQ / subset / academic / sandbox | 현재 열람 pricing 본문 $1,500/월; 검색 색인 $1,800/월과 불일치, 서면 확인 필요. subset/academic 총액 미확인 | CTA/UTP raw trades, NBBO/top-of-book since2007; 명세·맞춤 전달 검증 필요 | technical/academic/sandbox 프로그램과 별도 outright purchase 상담 경로 존재. archive 전체 lease는 비용 효율 낮음. exact May subset·correction/order/ns·외부 보관/재현 권리와 기간별 요금 확인 후만 판단 |

**Alpaca 조건부 후보의 중요한 제약.** 최신 plan 문서와 FAQ는 무료 historical
SIP를 명시하지만, 8개월 전 historical-source 문서는 IEX만 무구독 사용 가능하다고
쓴다. 이 충돌을 숨기지 않는다. 세부 FAQ의 historical 예외가 더 구체적이라는
해석이며 실제 계정의 $0 접근은 아직 시험하지 않았다. Public plan은 entitlement
증명이 아니다. 과거 자료를 받는 15분 제한은 2026년 5월 backtest의 event clock을
15분 늦추라는 뜻도, 미래 live-SIP 비용이 없어졌다는 뜻도 아니다.

현재 공개 row 명세는 `t`, 가격/수량/venue/conditions/tape, trade ID를 보여준다.
participant/TRF clock, proven ordering sequence, 원시 cancel/correction 전달 시각을
복원할 수 있다는 근거는 확인하지 못했다. realtime c/x 메시지가 있다는 사실은
historical endpoint가 이 이력을 준다는 증거가 아니다. Quote size의 round lots와
Massive REST shares를 혼동하지 않고, history endpoint의 실제 단위와 symbol-date
round-lot mapping을 확인한다. OTC는 일반 Basic에서 특수 broker entitlement를
요구한다고 FAQ에 명시되어 있다. 범위 밖 종목을 제거해 coverage를 높이지 않는다.

**구독을 끝내도 원본을 영구 재사용할 수 있다고 가정하지 않는다.** Massive Market
Data Terms 8은 agreement/account의 terminated/restricted/suspended 상태에서 데이터
사용 중단과 삭제를 요구한다. Advanced 한 달 수집 뒤 Starter로 downgrade하는
경우의 원본 계속 보관·재생 권리는 서면 확인 없이 보장할 수 없다. 개인 약관은
계정/키 공유도 금지한다. 보유 파일 재사용과 모든 subset 견적은 실제 허용되는
사용·보관·재현 권리를 함께 확인해야 경제적으로 유효한 경로가 된다.

algoseek Sandbox는 $0/no-card, up-to-one-year 조회·CSV 다운로드를 공시한다.
그러나 예시는 2023/2025 minute bars이고 raw TAQ의 May2026 availability, 허용
symbol/query 범위, ns clocks와 demo license의 연구/보관 권리는 확인되지 않았다.
따라서 무료 기술 sample 후보를 추가할 수 있으나 full R335 source로 확정하지
않는다. NYSE 공개 sample도 다른 날짜로 원래 census를 바꾸지 않으며 sealed
July/August 파일은 다운로드하지 않는다.
algoseek TAQ 명세는 2016 이후 ns event timestamp, shares, cancel/correct event를
포함한다. 그러나 이 clock을 SIP/strategy receipt라고 단정하지 않고, `EST`라고
쓴 시간대의 May DST 처리와 별도 bid/ask NB 행의 동일 시각 원자성·ordering을
확인해야 한다. 원본 명세에 없는 clock/sequence를 생성하지 않는다.

## 4. 비용 $0으로 수행하는 순서와 중단 조건

1. 기존 authorized export/기관 Daily TAQ의 권한·원본 명세·보관 권리를 확인한다.
   확인된 파일만 explicit catalog에 넣고 원본 byte/provenance hashes를 고정한다.
   이번 실행은 credentials/다른 사람의 계정/무허가 공유물을 찾지 않았다.
2. 사용 가능한 Alpaca Basic 계정이 적법하게 제공되면, 무료 entitlement와 above
   source semantics를 확인하고 collection contract를 먼저 게시한다. 새 계정의
   약관 동의/유료 upgrade는 이번 코드에 포함하지 않았다. API key를 보고서나
   공개 repo에 넣지 않는다.
3. 기존 12개 anchor × trades/quotes의 첫 60초를 `feed=sip, asof=-, currency=USD`,
   정확한 `[start,end)`로 qualification한다. Alpaca end는 inclusive라 `end_ns−1`
   문자열을 보낸다. limit100, 2page/pair, 최대48attempt/2MiB-body/32MiB-total,
   12.5초 간격, timeout30, retry0. 401/429는 전부 중단, 403은 stream 중단,
   IEX/다른 종목/다른 날짜 fallback 없음. HTTP/body/page token/hash/status/
   received time과 모든 미요청 pair를 먼저 보존한다. 예산이 끊은 chain은 PARTIAL.
   이 제한은 제안이며 **현재 collection_enabled=false**다.
4. 기술 qualification은 schema, ns/units, provenance, pagination, source consistency
   검증이다. 성공해도 R331 해소율·수익·full census 완결성을 주장하지 않는다.
   동일 clock의 순서나 정정/availability 복원이 안 되면 해당 비교는 UNKNOWN.
5. 적합성이 확인되면 원래 모든 1,235개 full window를 수집할 별도 요청·byte/
   storage/time budget을 등록한다. 이번 원래 census는 ticker-day가 전부 달라
   partition dedup에 따른 실제 절감은 0이다(2,470 unique partition-stream).
   큰 whole-market daily 파일을 먼저 내려받지 않는다. per-ticker/window를 사용해
   irrelevant 날짜/종목 구매를 줄이며 어느 원래 identity도 제외하지 않는다.
6. 무료 source가 필수 조건을 못 충족하면 R335 전체와 결측 ledger를 보존한다.
   같은 May 범위/원시 fields/원본 acquisition chronology/재현 보관 권리/총액
   상한을 지정해 Massive/TAQ/algoseek의 subset·academic 견적을 비교한다.
   승인 없는 구매/paid API/retry 없음. 새로운 돈이 없으면 full-source 단계는
   미완료로 남는다. 90% 기준이나 3초 경계를 바꾸지 않는다.

샘플 적합성 확인 다음에도 R335 source reconciliation과 별도로 causal execution
proxy를 사전 등록해야 한다. 원시 NBBO 자체는 actual fill proof가 아니다.
그 다음 원래 ALL-phase continuous ENTER/WAIT와 chronological after-cost account
learning을 재개하고, HOLD/EXIT, 다른 종목 기회비용, ADD/REDUCE allocation을
검증한다. 표본 기술 검증으로 장기 계좌 성장이나 모델 완성을 주장하지 않는다.

## 5. 실제 코드와 재현 준비

- `tick_execution_readiness.py`: float 없는 RFC3339 ↔ integer ns 변환;
  Alpaca row를 unverified provider clock으로 보존하는 adapter. missing SIP/
  participant/TRF/sequence/correction을 만들어 채우지 않고 quote lots를 shares로
  바꾸지 않는다. 기존 inventory의 source-ready 판정에 연결하지 않았다.
  역사 endpoint 단위는 UNVERIFIED로 남기고 realtime 명세의 shares/round-lots는
  별도 hint로만 둔다. 두 명세가 같다고 코드가 가정하지 않는다.
- 같은 모듈의 quote-at-arrival diagnostic: explicit latency/availability/freshness/
  participation/eligibility 규칙, buy ask / sell bid의 displayed ceiling만 검사.
  미래 fallback, stale receipt 기반 freshness, crossed/zero/missing book, 잘못된
  단위/피드, 동일 clock 순서 불명은 해소로 인정하지 않는다. 최신 invalid book을
  건너뛰어 과거 valid book을 구하는 동작도 금지한다. `fill_confirmed=false`,
  `economic_label=null`은 항상 유지한다. commission/slippage/account P&L simulator를
  완료했다고 주장하지 않는다. 실제 실행 proxy의 입력 검사 준비다.
- `r335_free_source_plan.py`: 기존 3개 source bytes와 R333 pins/plan을 인증하고
  원래 모든 identity/stream과 24개 sample descriptor를 고정한다. network/secret/
  billing API가 없는 offline planner; exact inclusive-end 변환과 rename mapping
  방지(`asof=-`), full/sample 분리. 실제 수집 executor는 활성화하지 않았다.
- 기존 full unit/lint CI는 유지하고 R335 branch에서만 `flatfiles-access`
  metadata network job을 생략했다. 이번 offline 준비에 시장 credentials/API를
  사용하지 않도록 하는 제한이며 다른 branch의 접근 검사는 바꾸지 않는다.
- `request335-free-source-plan.json.gz`: 원래 946,999-byte JSON을 deterministic
  gzip으로 보존한 plan. 압축 해제한 원본 SHA256
  `0ad2603a2a62ef49a975f21c5951e55fa6708fbe0d062bf5df5c832bb88c235e`.
  원래 windows hash `69f1ba97c5edd16f8b819a36be710a47e1c5488af2df0378466968fd0794b5dc`.
- 테스트와 재현 결과는 `request335-cost-validation.json`에 기록한다. 모든 신규
  호가/가격 fixture는 synthetic code test만이며 시장 데이터/P&L output이 아니다.
  신규 50개, 기존 inventory 포함 targeted89개와 full1,390개가 통과했다(local
  Python3.12.14, 기존243 warnings). Critical Ruff와 workflow YAML parse도 통과했다.
  Plan의 두 독립 로컬 실행은 바이트 일치했고 기존 report/ledger의7,423 numeric
  fields, identity, states, missingness도 전부 동일했다. 초기 fresh runtime에는
  pytest/ruff가 없어 실행되지 않았으며 dev dependencies 설치 후 실제 검증했다.
  게시된 source `07849b83313c9dab75bc61082bf14fb1f219b562`의 push/PR CI
  [37940378756](https://github.com/dayofhiki/MoneyMaker/actions/runs/37940378756),
  [37940383056](https://github.com/dayofhiki/MoneyMaker/actions/runs/37940383056)도
  critical lint와 full1,390 tests(기존160 warnings, Python3.11)를 통과했다.
  두 실행 모두 credentialed metadata job은 skipped이며 시장 수집 실행이 아니다.

추가 결제 없이 가능: 명세·가격/라이선스 조사, pinned census 보존, 정확한 시간/
unit/missingness 검사, future-prefix·order ambiguity 테스트, source/plan byte replay,
실행 proxy 준비. 현재 추가 결제 없이 **완료되지 않은 것**: 권한이 아직 제공되지
않은 alternative 원자료 획득, full page-complete Trades/NBBO reconciliation,
causal fill/계좌 수익성 검증. 무료 source의 기술적·계약상 적합성 성공을 전제로
미래 지출 $0 가능성이 있으나 이를 보장하지 않는다.

## 6. 다음 실험의 최소 조건

적법한 $0 계정 또는 원본 export, exact same-date entitlement, license/보관·재현
권리, raw NBBO/clock/unit/correction/ordering 명세, 사전 등록된 bounded acquisition
contract가 있어야 기술 qualification을 시작할 수 있다. Full reconciliation에는
모든 원래 identity의 page/file completion ledger와 인증된 원본이 추가로 필요하다.
Economic run에는 따로 고정된 causal order/quote/cost/censoring/account contract가
필요하다. unresolved 상태는 성공 또는 cash로 바꾸지 않는다.

## 확인한 1차 자료

- Massive plan: https://massive.com/stocks
- Massive fields: https://massive.com/docs/rest/stocks/trades-quotes/trades
  및 https://massive.com/docs/rest/stocks/trades-quotes/quotes
- Massive flat-file sample: https://massive.com/docs/flat-files/quickstart
- Massive education: https://massive.com/knowledge-base/article/does-massive-offer-any-education-discounts
- Massive trial: https://massive.com/knowledge-base/article/does-massive-offer-free-trials
- Massive pay-per-request: https://massive.com/blog/x402-payments-for-ai-agents
- Massive license: https://massive.com/legal/terms
  및 https://massive.com/legal/market-data-terms-of-service
- Alpaca plan: https://docs.alpaca.markets/us/docs/about-market-data-api
- Alpaca FAQ: https://docs.alpaca.markets/us/docs/market-data-faq
- Alpaca conflicting older overview: https://docs.alpaca.markets/us/docs/historical-stock-data-1
- Alpaca quotes: https://docs.alpaca.markets/us/reference/stockquotes-1
  및 https://docs.alpaca.markets/us/reference/stocktrades-1
- Alpaca NBBO: https://alpaca.markets/learn/fetch-historical-data
- Alpaca realtime schema/corrections: https://docs.alpaca.markets/us/docs/real-time-stock-pricing-data
- Alpaca agreements: https://alpaca.markets/disclosures
- NYSE Daily TAQ: https://www.nyse.com/data-products/catalog/daily-taq
- Daily TAQ ns/sequence specification: https://wrds-www.wharton.upenn.edu/documents/1967/Daily_TAQ_Client_Spec_v4.0.pdf
- WRDS TAQ module: https://wrds-www.wharton.upenn.edu/pages/grid-items/wrds-intraday-indicators/
- WRDS fees: https://wrds-www.wharton.upenn.edu/pages/about/what-wrds/
- Theta schema: https://thetadata.net/docs/operations/stock_history_quote.html
- Theta tiers: https://thetadata.net/docs/Articles/Getting-Started/Subscriptions.html
- Theta CTA discrepancy: https://www.thetadata.net/stocks-data
- Theta pricing: https://www.thetadata.net/pricing
- Databento cost/credit: https://databento.com/docs/faqs/usage-pricing-and-data-credits
- Databento non-equivalence: https://databento.com/docs/examples/equities/consolidated-bbo
  및 https://databento.com/equities
- algoseek prices/programs: https://algoseek.com/pricing/
- algoseek sandbox: https://algoseek.com/sandbox/
- algoseek academic: https://algoseek.com/academic/
- algoseek schema: https://us-equity-market-data-docs.s3.amazonaws.com/algoseek.US.Equity.TAQ.pdf

위 문서는 공개 capability 근거이지 사용자 권한, 견적 승인 또는 실제 데이터
완결성 증거가 아니다. 원자료나 계정을 제공받지 않은 점을 후속 실행에서 보완한다.
