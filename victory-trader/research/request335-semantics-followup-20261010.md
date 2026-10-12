2026-10-10 전체 구간 후속: 고정 24개 구간의 무료 수집에서 14회 요청, 13개 HTTP 200, 체결 12,769건 / 호가 17,825건을 보존했다. 12개 API 종료 후 전송 오류와 실행환경의 네트워크 승인 취소로 중단했다. 추가 $0, source/local/CI 1,525 tests 통과, 전체 수익성 검증은 미완료다. [최신 전체 구간 결과와 예산](request335-full-window-followup-20261010.md)을 먼저 참고하며 아래 이전 기록은 보존한다.

# R335 historical 단위·순서 감사와 별도 페이지 검증

**후속 완료:** 공식 May quote shares 단위를 반영했고, 사전등록 후 AIRS 새 조회
1회에서 367행·terminal 및 원래 200행 prefix 일치를 확인했다. 추가 비용 $0,
local 전체 1,492 tests와 해당 source의 push/PR CI가 통과했다. 수익 검증은 미완료다.

원래 R335의 연구 대상·성공 기준과 R331의 실패 결과를 보존하면서 무료 SIP
표본의 해석을 구체화했다. 기존 31개 응답은 변경하지 않았다. 이번 원본 감사는
추가 시장 API 요청 0회, 추가 비용 $0, 경제적 라벨 0개다.

## 공식 명세로 해결한 부분

Alpaca의 2025-10-30 변경 공지는 CTA/UTP 호가 수량이 2025-11-03부터 주식 수로
표시된다고 설명한다. 최신 공식 CLI의 OpenAPI에서도 historical quote 응답이
참조하는 `stock_quote`의 `bs/as`에 같은 날짜 구분이 있다. 따라서 2026년 5월
원래 표본의 수량에 100이나 symbol lot size를 곱하지 않는다.

공식 schema 원본은 476,591 bytes, SHA256
`56b0278e0eca3dde122304b5dd6b74abfab6848123141a7947bc15614d8843b7`이며,
upstream commit `53606273aa230a40c64b783425dcb3f4423ede30`에 고정했다.
단위 adapter는 New York 시장 날짜로 적용 시점을 구분하고 원래 수량을 보존한다.
이전 날짜의 round-lot 수량도 추정 100주로 변환하지 않는다.

UTP/CTA 명세는 100/40/10/1주 lot tier와 반기별 재지정을 설명하며, UTP 공지는
2026-05-01 첫 재지정을 확인한다. May의 **주식 수 변환**에 lot mapping을 요구할
필요는 없다. 실제 lot eligibility/rounding의 역사적 감사에는 date-specific
reference가 필요하며 현재 symbol 파일이나 HOT 가격으로 이를 추정하지 않는다.

최신 schema의 선택적 trade `u`는 현재 레코드의 canceled/incorrect/corrected
상태를 나타낸다. 없어도 현재 API 레코드 기준 유효하다는 명세가 있다. 그러나
이것은 정정 전달 시점·원래 기록 연결·전략이 당시 본 순서를 제공하지 않는다.
adapter는 관측된 상태를 그대로 보존하고 `correction` 시각이나 sequence를 만들지
않는다. 원래 체결 674행에는 `u`가 없었다.

timestamp schema는 ns 정밀도의 RFC3339 표현을 명시한다. SIP/participant/전략
receipt 중 어떤 clock인지는 명시하지 않는다. symbol과 timestamp 정렬 명세를
동일 clock의 실제 전달 순서로 확대하지 않는다. Trade ID도 범용 sequence가 아니다.

## 원래 private 표본의 재현 가능한 감사

`r335_tick_semantics_audit`는 먼저 원래 수집기의 replay로 계획·계약·raw hash·페이지
chain·bounds·분모를 검증한다. 이후 페이지를 합친 동일 provider timestamp 그룹을
감사한다. 행 정렬·제거·중복 제거·보간·가격 추정·SIP clock alias는 하지 않는다.
출력은 값 없는 집계와 해시다. 원본은 공개 저장소 밖에 그대로 둔다.

| 항목 | 체결 | 호가 |
|---|---:|---:|
| 관측 행 | 674 | 812 |
| 동일 provider clock 그룹 | 27 | 10 |
| 그 그룹의 행 | 60 | 21 |
| 페이지 경계를 가로지르는 동일 clock 그룹 | 0 | 0 |
| 동일한 decoded 행 | 0 | 0 |
| 검사한 schema 오류·필수 필드 누락 | 0 | 0 |
| locked / crossed / inactive-side 행 | 해당 없음 | 94 / 0 / 0 |

호가 10개 그룹 모두 수량이 달랐다. 가격/venue는 같았다. 어떤 수량이 마지막
시장 상태인지 선택하지 않는다. Trades/Quotes 사이 exact provider clock 공통
그룹은 이 작은 표본에서 0개다. 이것도 전체 시장 순서/clock 의미의 증명이 아니다.
호가 단위 812행은 공식 날짜 규칙에 따라 shares로 기록되며 source_ready는 false다.

감사는 독립 2회 byte-equal이며 원래 replay hash도 보존한다. 기계 판독 결과:
[request335-tick-semantics-audit-20261010.json](request335-tick-semantics-audit-20261010.json).
공식 명세의 범위·남은 UNKNOWN:
[request335-official-semantics-20261010.json](request335-official-semantics-20261010.json).

## 다음 무료 페이지 검증의 사전등록

이 절은 새 조회 **전** 공개 PR에 등록한다. 기존 sample의 PAGE_CAP_INCOMPLETE를
사후 완결로 바꾸지 않는다. 원본 manifest SHA256
`44b5f124d1ad2533dc42a800d9032c64fbffae3d7b7c5f1938db0b88e8a20507`에 고정된
유일한 capped pair, 2026-05-05 AIRS quotes의 원래 첫 60초만 새 chain으로 조회한다.
가격·수익·사후 결과로 종목을 새로 선택하지 않는다.

- Endpoint/window/feed/asof/currency/sort는 같고 새 limit은 공식 최대 10,000이다.
- 최대 2회/2페이지/2 MiB per page/4 MiB total, 최소 12.5초 간격, timeout30,
  retry0. 어떤 오류든 종료하며 다른 feed·유료 경로·추가 종목으로 전환하지 않는다.
- 새 root(0700), wire·계약·manifest·확인서(0600)를 만들며 old token을 재개하거나
  새 응답을 기존 chain에 붙이지 않는다.
- 새 chain의 첫 200행과 원래 retained prefix 전체를 원래 순서대로 비교한다.
  차이는 변경 관측으로 기록한다. 일치해도 나머지 시장 누락이나 정정 이력을
  인증하지 않는다. Terminal marker는 이 API chain의 종료만 나타낸다.
- 추가 비용 권한 $0, full collection=false, economic labels=false.

고정 계약: [request335-page-check-contract.json](request335-page-check-contract.json),
SHA256 `45733ee0319e325100b01c4ace64ba2d9d8a1765459724b41dfa4b15b03d99c7`.
기존 수집기 구현 hash, 48/2/32 MiB 계약, source plan과 과거 결과는 변경하지 않는다.
이 후속 조회의 수량/순서 검사도 기술 검증이며 R335 수익성 성공이 아니다.

## 별도 조회 실제 결과

조회 전에 코드·계약·검증을 commit `9fc4fc578d0fa987e17786890531cb04da8ffa06`
(tree `c006f1b69723e16a0cf417c60f47290ca087c545`)에 공개했다. 원래 고정 collector를
변경하지 않은 새 코드로 2026-10-10 15:03:42~15:03:53 UTC에 실행했다.

| 항목 | 실제 관측 |
|---|---|
| 요청/HTTP/비용 | 1회 / 200 / 추가 $0 |
| 응답 | 41,490 bytes, byte-complete, terminal marker |
| AIRS quotes | 367행, documented shares |
| 원래 prefix 비교 | 첫 200행 모두 원래 순서와 decoded 값 일치 |
| 검사한 schema 누락/오류·역순·동일 clock·동일 decoded 행 | 각 0 |
| locked / crossed / inactive-side | 38 / 0 / 0행 |
| 독립 replay | 2회 byte-equal |

원래 prefix 이후 167행을 관측했다. 새로 받은 367행을 원래 chain에 붙이거나
이전 PAGE_CAP_INCOMPLETE 상태를 덮어쓰지 않았다. 다른 23개 pair의 기존 terminal
관측과 이번 별도 chain의 terminal을 모두 사용할 수 있지만, 서로 다른 취득 시점의
evidence다. prefix 일치는 비교한 200행의 일치이며 전체 vendor 수정/시장 완결성
보증이 아니다. 이 60초 표본을 원래 full-window나 수익 결과로 일반화하지 않는다.

새 manifest SHA256:
`bf1b1ece92fecde6910a0cdc5fc9f60874df663dd9730249ab1498aaec30d98d`.
새 replay SHA256:
`dfd6db662b2532e055bab0c85aba7573464201ec283daf8cdcf0ee1451b055ea`.
root0700 / 모든 파일0600을 확인했고 원본·키·account ID를 공개하지 않았다.
기계 판독 결과:
[request335-page-check-result-20261010.json](request335-page-check-result-20261010.json).

## 검증과 남은 최소 조건

별도 조회 코드와 단위/감사 테스트를 포함한 관련 147개 synthetic cases가 통과했다
(Python3.12.14, 0.39초). 전체 local suite는 **1,492 passed**, 기존 warnings243,
43.05초다. 저장소 CI와 같은 Ruff E4/E7/E9/F 검사도 통과했다. 모의 fixture는
코드 검증에만 사용한다. 같은 source commit의 push CI38062043122 / PR CI38062047672에서
필수 lint와 test 단계가 모두 성공했다. credentialed metadata job은 모두 skipped다.
Decoded job log에서 양쪽 모두 1,492 passed / 기존 warnings160을 확인했다.
Python3.11에서 push86.47초 / PR56.37초이며 local Python3.12 결과와 구분한다.
[request335-semantics-validation-20261010.json](request335-semantics-validation-20261010.json)에
검증 대상 코드/테스트 hash, local 결과, CI job, 실제 replay 결과를 구분해 기록했다.

현재 가장 경제적인 경로는 Basic historical SIP를 개인 비상업 연구의 private
acquisition에 계속 사용하는 것이다. 무료 접근·May quote 단위는 해결되었다.
전체 1,235 identity / 2,470 stream의 full-window chain은 아직 수집하지 않았다.
전체 acquisition의 요청/페이지/byte/storage/time 예산과 중단·누락 ledger를 별도로
등록해야 한다. 작은 lexical sample에서 전체 데이터량·coverage를 일반화하지 않는다.

체결/수익 검증을 시작하려면 historical clock·동일 시각 순서·as-observed 정정/취소,
조건/거래정지·OTC/비활성 종목 coverage 및 시장 완결성을 확보해야 한다.
그 위에 주문 availability/latency/age/participation/cancel/가격 충격/실제 비용과
시간 순서 계좌·자본/기회비용 계약이 필요하다. NBBO displayed size는 fill 확정이 아니다.
없는 clock을 latency 시나리오로 대체하더라도 시나리오 근거·범위를 별도 고정하고
실제 역사적 관측/체결의 검증과 분리해야 하며 원래 성공 기준은 그대로다.

지원팀 문의·업그레이드·새 키·새 로그인은 현재 필요하지 않다. 추가 결제는 없다.
원래 407 training / 828 development, 3초 reference, 60초 hold, BASE −2.5%
completed-close stop, 90% 해소 기준 및 R331 34.8352% 실패를 보존한다.
June HOLD / July·August sealed는 열지 않는다.

| 경로 | 예상 추가 비용 | 이번 판단 |
|---|---|---|
| 현재 Alpaca Basic historical SIP | $0 | 무료 raw acquisition 우선; 과거 clock·순서·정정 전달 이력과 시장 완결성은 별도 검증 |
| 현재 Massive Starter | $0, 기존 $29/월 | 보유 aggregates 재사용 가능; Trades/NBBO 제공 범위는 해결하지 못함 |
| Massive Advanced | $199/월 플랜 | 새 결제 미승인; 라이선스·보관 조건과 execution 계약까지 해결되는 것은 아님 |
| 기관 Daily TAQ/WRDS | 이미 해당 모듈을 허가받았으면 증분 $0 가능; 그 외 견적 | 사용자에게 실제 module 권한이 있는지 미확인 |
| vendor exact May subset / 연구 지원 | 개별 견적, 가격 미확정 | 무료 경로가 제공하지 않는 필드를 확인하고 필요한 부분만 문의·구매 검토 |
| Databento / ThetaData / algoseek | 상품·범위·권리에 따라 다름 | credit/부분 BBO/ms/synthetic 자료를 원래 NBBO와 동등하다고 간주하지 않음 |

전체 비용·범위·누락·라이선스·재현성 비교는
[request335-cost-integrity-review.md](request335-cost-integrity-review.md)의 기존 조사를
참조한다. 이번 변경은 May quote 수량 단위와 실제 capped-pair 접근 검증을 해결한
것이며 그 비교의 미확정 견적을 새 확정 가격으로 바꾸지 않는다. 현재 사용자에게
요구할 결제·문의·로그인 행동은 없다.

공식 근거(2026-10-10 열람):
- https://docs.alpaca.markets/us/v1.1/changelog/marketdata-bid-and-ask-size-display-change
- https://github.com/alpacahq/cli/blob/53606273aa230a40c64b783425dcb3f4423ede30/api/specs/market-data-api.json
- https://www.nasdaqtrader.com/TraderNews.aspx?id=UTP2025-10
- https://www.nasdaqtrader.com/TraderNews.aspx?id=UTP2026-12
- https://www.ctaplan.com/publicdocs/ctaplan/CTA_Round_Lot_Changes_FAQ.pdf
- https://docs.alpaca.markets/us/reference/stockquotes-1
