# R335: 무료 Alpaca SIP 접근 표본 결과 — 2026-10-10

사용자가 Paper API 키 생성과 비공개 연구환경 연결을 승인했다. 실제 Paper 계정의
새 키를 생성하여 공개 저장소 밖 private 파일에 연결했고 secret을 화면에서 숨겼다.
계정 화면의 Subscription Status는 Basic이다. 추가 구독·체험·구매·주문·지원팀
문의는 없다. 이번 추가 비용은 $0이며 기존 Massive Starter는 변경하지 않았다.

**무료 historical SIP의 실제 계정 접근을 확인했다. 전체 R335 데이터의 적합성이나
수익성 검증을 완료한 결과는 아니다.** 아래 수치는 등록된 작은 표본의 관측이다.

## 등록된 접근 표본

- 기존 12개 2026년 5월 날짜의 lexical-first 원래 anchor × Trades/Quotes, 총 24 pair.
  각 원래 window의 첫 60초이며 다른 종목·날짜·sealed 기간으로 바꾸지 않았다.
- 고정 계획 SHA256 `0ad2603a2a62ef49a975f21c5951e55fa6708fbe0d062bf5df5c832bb88c235e`,
  계약 SHA256 `993074e68131160345d33758ea44907493e37ab234bb92b9fb93d518bc03909f`.
- 원래 수집기를 변경하지 않았다. 구현 SHA256
  `b1e3bb1dd6909c8e83f92adaa2070f60dbe2dcbb5ef1569a4af7aad0daec18d1`.
- `data.alpaca.markets`의 historical endpoint만 사용했다. `feed=sip`, `asof=-`, USD,
  오름차순, `end_ns−1`, limit100. 48 attempts/2 pages/2 MiB/32 MiB/12.5초/
  timeout30/retry0 계약을 유지했다. IEX·유료·다른 데이터 경로로 전환하지 않았다.
- 실행: 2026-10-10 13:14:41~13:21:04 UTC (22:14:41~22:21:04 KST).

| 항목 | 관측 결과 |
|---|---|
| HTTP 요청 | 31, 모두 200; redirect/retry/denial 없음 |
| 원본 응답 바이트 | 158,327; 31개 응답 본문은 전부 byte-complete |
| 체결 | 15페이지, 674행; 12 pair 모두 terminal marker 관측 |
| 호가 | 16페이지, 812행; 11 pair terminal, 1 pair PAGE_CAP_INCOMPLETE |
| 필수 필드 누락 및 검사한 비양수/잘못된 숫자 | 각 0 |
| page 내부 또는 page 사이 역순 | 0 |
| page 내부 완전히 동일한 행 | 0 |
| 인접한 동일 clock | 체결 33, 호가 11; cross-page 0 |
| timestamp 소수부 자릿수 | 체결 7~9, 호가 6~9; 정확히 파싱하여 보존 |
| 오프라인 replay | 독립 2회 결과와 최초 replay가 byte-equal |
| 현재 환경의 수집기 synthetic tests | 51 passed, 0.65초 |

23개 pair에서 terminal marker를 관측했다는 것은 요청한 API chain의 종료를
확인했다는 뜻이다. 공급자가 시장 원본을 빠짐없이 전달했다는 보장은 아니다.
미완결 호가 pair는 다음 토큰이 남아 있으나 등록된 2페이지 한도를 적용했다.
추가 요청을 하지 않았고, 이를 완결로 표시하거나 분석에서 제외하지 않았다.

## 원본을 변경하지 않은 형식·순서 감사

예정된 필드/clock/숫자 검사 후 추가로 수행한 아래 감사는 exploratory 관측이다.
경제적 라벨, 성공 기준, 표본 선정에 사용하지 않는다.

- Quote의 관측 필드는 `t,bp,ap,bs,as,bx,ax,c,z`이며 sequence나 별도
  participant/TRF/strategy receipt clock은 관측되지 않았다.
- Trade의 관측 필드는 `t,p,s,x,c,i,z`이다. Trade ID를 시간 순서나 correction
  전달 순서로 해석하지 않는다.
- 체결 venue 코드 14종, bid venue 13종, ask venue 9종을 관측했다. 다수 코드와
  `feed=sip` 요청만으로 원래 요구한 official NBBO 동등성을 인증하지 않는다.
- 10개 동일 timestamp 호가 그룹의 21행에서 수량이 서로 달랐다. 가격/venue는
  그룹 안에서 같았다. 동일 clock의 마지막 수량을 임의로 고르지 않는다.
- locked quote 94행, crossed quote 0행. Locked 행을 사후에 제거하지 않았다.
- 9자리 timestamp 문자열은 정확한 정수 ns 파싱 근거다. 실제 측정 정확도나
  SIP clock 의미, 역사적 strategy availability를 증명하는 근거는 아니다.
- share/round-lot 단위·symbol-date lot size를 추정하지 않았고 정정·취소·누락
  이벤트를 만들지 않았다. 응답을 정렬하거나 중복 제거하거나 보간하지 않았다.

공식 historical quote 안내는 selected feed를 통해 NBBO가 제공될 수 있다고
설명한다. endpoint 명세는 `sip`을 전체 미국 거래소 피드로 설명하며 timestamp로
정렬한다고 설명한다. 이를 문서 근거로 보존하되, 동일 clock의 실제 순서나
historical 수량 단위·정정 이력까지 확인된 것으로 확대하지 않는다.

## 무결성·보안·검증 범위

원본 페이지·manifest·replay·확인서는 private 로컬 디렉터리(0700), 파일(0600)에
있다. 원본 시장데이터, key/secret, 계정 ID 또는 이메일을 공개 PR·저장소·Actions
artifact에 올리지 않았다. 공개 기록은 값 없는 집계 및 해시뿐이다.
키는 수집 프로세스의 명시된 두 환경변수에만 주입했고 종료 시 제거했다.
private 키 파일이 이 작업 환경에 존재한다는 것은 영구 외부 credential vault에
등록되었다는 뜻은 아니다. 공개 원본 공유나 추가 보관 권리를 주장하지 않는다.

- private manifest SHA256:
  `44b5f124d1ad2533dc42a800d9032c64fbffae3d7b7c5f1938db0b88e8a20507`
- private replay SHA256:
  `f4f15426cb81ffee03755c44f1963dcec540951a62281833cb5c68fea78e84dd`
- runtime: Python3.12.14, requests2.34.2, urllib3 2.8.0, certifi2026.7.22,
  pytest9.1.1. 이번 환경에서 필요한 requests와 pytest를 설치하고 51 synthetic
  tests를 다시 통과했다. 이전 application source의 전체 1,441 local/CI 결과는
  이전 검증 기록이며, 이번에 전체 suite를 재실행했다고 주장하지 않는다.
- 키 표시 화면 캡처는 자동 승인 검토가 credential 노출 위험으로 거부했다.
  키가 없는 Basic 화면 캡처도 브라우저 timeout으로 저장되지 않았다.
  안전한 실제 응답/replay/파일 권한 검증 기록을 남기며 이미지 링크를 만들지 않는다.

기계 판독 집계: [request335-free-access-result-20261010.json](request335-free-access-result-20261010.json).
프로그램·테스트·원래 계획과 계약에는 이번 결과 기록으로 인한 변경이 없다.

## 다음 단계의 최소 조건

사용자가 지원팀 문의·유료 플랜·새 로그인·키 전달을 할 필요는 없다. 현재 비공개
키로 무료 접근할 수 있다는 근거를 확보했다. 다음 연구의 남은 조건은 다음과 같다.

1. official NBBO의 구성·조건, historical clock 의미, 수량 단위/lot mapping,
   correction/cancel 처리와 동일 clock 순서에 대한 명세를 근거로 정리한다.
   없는 정보는 UNKNOWN으로 보존하며 실제 전략 관측/체결을 발명하지 않는다.
2. 원래 1,235 identity의 2,470 full-window stream 전체를 위한 acquisition 계약과
   request/page/byte/storage/time 예산을 별도로 등록한다. 현재 한도를 늘리거나
   미완결 표본을 추가 조회하지 않았다. 무료라는 이유로 full census를 임의 수집하지 않는다.
3. OTC/비활성 종목·정정·거래정지·실제 시장 완결성 ledger와 causal execution /
   계좌 계약을 확보한다. Top-of-book 수량만으로 체결·slippage를 확정하지 않는다.

`source_ready=false`, `new_economic_labels=0`, full census=false,
profitability_validation_complete=false다. 원래 407 training / 828 development,
3초 reference bound, 60초 hold, BASE −2.5% completed-close stop, 90% 해소 기준과
R331의 34.8352% 실패 결과를 모두 유지한다. 무료 접근 확인을 전체 실험 성공으로
일반화하지 않는다.

공식 근거(2026-10-10 열람):
- https://docs.alpaca.markets/us/docs/about-market-data-api
- https://docs.alpaca.markets/us/docs/market-data-faq
- https://files.alpaca.markets/disclosures/library/TermsAndConditions.pdf
- https://docs.alpaca.markets/us/reference/stockquotes-1
- https://docs.alpaca.markets/us/reference/stocktrades-1
- https://alpaca.markets/learn/fetch-historical-data
