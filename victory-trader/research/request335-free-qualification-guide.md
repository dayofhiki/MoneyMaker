# R335 무료 표본 검증 실행 안내

2026-10-10 단위·페이지 후속: 공식 historical schema에 따라 May 호가 수량을
shares로 확인했다. 별도 사전등록 후 AIRS 새 조회 1회에서 367행·terminal 및
원래 200행 prefix 일치를 관측했다. 기존 미완결 기록은 변경하지 않았다. 추가 $0,
local/해당 source CI 1,492 tests 통과. 시계·동일 시각 순서·정정 전달 이력과
full census/체결·수익 검증은 미완료다.
[최신 단위·순서 감사와 페이지 결과](request335-semantics-followup-20261010.md)를
우선 참고하며 아래 기록은 이전 취득과 조사 상태를 보존한다.

**같은 날 실제 연결·접근 완료:** 사용자 승인 후 Paper 키를 비공개 수집기에
연결했다. 등록된 24 pair에 대해 31 HTTP 요청 모두 200, 체결 674행/호가 812행을
받았다. 23 pair terminal, 호가 1 pair는 2페이지 한도로 미완결이다. replay 2회가
동일하며 추가 지출은 $0이다. 아래 자격증명 없는 실행 기록은 이전 단계다.
[실제 표본 결과 및 다음 데이터 조건](request335-free-access-result-20261010.md)을
우선 참고한다. 전체 source/execution qualification은 계속 미완료다.

2026-10-10 후속 작업. 추가 지출 허용액은 $0이며, PR108은 draft 상태를 유지한다.
이 단계는 **접근권한·응답 형식·페이지 연결의 기술 검증**이다. 전체 1,235개
identity의 원자료 확보, 실제 체결 검증, 모델 수익성 검증을 완료하지 않는다.

## 공식 근거와 다음 단계

사용자가 보안 로그인을 완료한 실제 Paper 계정의 Basic / Included / Current Plan을
확인했다. 계정 작업은 사용자가 위임했으며 지원팀 문의는 진행 조건이 아니다.
최신 Basic 안내는 Paper/Live 기본 플랜을 무료로 설명하고 실시간 주식은 IEX로
제한한다. FAQ는 historical `end`가 15분 이상 과거이면 SIP를 구독 없이 조회할 수
있다고 명시한다. 일반 Terms의 개인/비상업 이용 범위와 이 공식 근거를 기록하여
등록된 2026년 5월 접근 표본을 진행한다. 별도의 공급자 답변을 기다리지 않는다.

확인서의 권리 근거에는 공식 URL, 확인일, 비공개 개인 연구 범위를 적는다.
이는 영구 보관·공개 배포·상업 이용 권리의 확인이나 전체 source qualification이
아니다. Paper Only 및 일반 Terms의 범위 차이는
[계정 확인 기록](request335-alpaca-account-review.md)에 보존한다. 기술적 200 응답도
별도의 계약 권리를 만들지 않는다. 구체적인 제한·추가 계약 요구·접근 거부가
나타나면 해당 문제를 확인하며, IEX를 동등한 NBBO로 대체하거나 유료 경로로
전환하지 않는다. 기관 Daily TAQ와 적법한 보유 원본 경로도 유지한다.

API credential은 실제 비공개 수집기에 연결했다. API key/secret은
대화, PR, 코드, 로그에 보내지 않는다. 현재 PR의 자동 CI에는 Alpaca 자격증명을
주입하거나 수집하는 job이 없다. GitHub Secrets에 키를 넣는 것만으로 수집이
시작되지 않는다. AlgoTrader Plus 결제, 입금, 실거래 계좌 개설 또는 Massive
업그레이드는 진행하지 않는다.

- Basic 플랜: https://docs.alpaca.markets/us/docs/about-market-data-api
- FAQ: https://docs.alpaca.markets/us/docs/market-data-faq
- Paper Only 안내: https://docs.alpaca.markets/us/docs/paper-trading
- 계정별 적용 약관 확인: https://alpaca.markets/disclosures

## 이번에 실제 수행한 작업

- 고정 계획 SHA256 `0ad2603a2a62ef49a975f21c5951e55fa6708fbe0d062bf5df5c832bb88c235e`
  를 인증하는 CLI를 작성했다. 다른 표본·sealed date·전체 수집 계획은 CLI 입력으로
  받아들이지 않는다. 기존 계획이나 R331~R334의 결과는 변경하지 않았다.
- `request335-free-qualification-contract.json`을 사전 등록용으로 고정했다.
  기존 계획에서 요구한 시계/NBBO/단위/정정/순서 명세의 **미확인 항목을 명시**하고,
  이 항목을 면제하지 않는다. 표본으로 접근/형식에 대한 근거만 얻을 수 있으며,
  모든 실제 데이터와 경제적 실행 검증은 계속 미자격 상태다.
- 원래 12일의 lexical-first 60초 Trades/Quotes 24개 pair만 수집 대상으로 한다.
  `feed=sip`, `asof=-`, USD, 오름차순, inclusive end에 `end_ns−1`을 명시한다.
- 최대 48 HTTP 시도, pair당 2페이지, 응답당 2 MiB, 총 32 MiB,
  시작 간격 최소 12.5초, timeout 30초, retry 0이다. 401/429는 전체 중지,
  403은 해당 stream 중지다. redirect·IEX·다른 공급자·유료 경로로 전환하지 않는다.
- JSON 숫자는 Decimal로 검사하고 RFC3339 ns를 정확히 보존한다. 원본 바이트,
  SHA256, 전체/접두부 여부, 요청·응답 UTC, opaque pagination token의 해시,
  페이지 순서, 누락 필드, 잘못된 가격/수량, 중복·역순·같은 clock을 기록한다.
  정렬·중복 제거·시계 alias·round-lot 환산·가격 보간은 수행하지 않는다.
- key/secret 또는 민감 필드가 포함된 응답은 원본을 저장하지 않고 전체 수집을
  중지한다. 오류 문자열이나 응답 내용을 stdout으로 출력하지 않는다.
- 원본은 로컬 private 디렉터리(0700), 파일(0600)에 보관한다. 이 권한은 로컬
  접근 제한이며 암호화는 아니다. 백업·동기화 위치도 적용 라이선스를 따라야 한다.
  공개 PR/저장소/Actions artifact에 원본 시장데이터를 올리는 기능은 없다.
- 오프라인 replay는 계획·계약·구현 파일·확인서·원본 해시, bounds, terminal marker,
  page token 연결, 시도 수와 용량을 재검증한다. 같은 timestamp의 실제 순서는
  여전히 미확인이다. cross-page 같은 clock/역순은 별도로 보고한다.
- 어떤 200/terminal page도 전체 시장 완결성, SIP clock 의미, official NBBO 동등성,
  체결 가능성 또는 계좌 수익성을 인증하지 않는다. 출력에는 항상
  `source_ready=false`, `new_economic_labels=0`이 남는다.

**이전 실행 결과(API 연결 전):** 자격증명 없는 원래 24 pair의 사전 점검과 replay를 각각 두 번
수행했다. 계약·manifest·replay 결과가 바이트 단위로 일치했다. 모든 pair는
`NOT_REQUESTED_MISSING_CREDENTIAL`, 실제 시장 HTTP 시도 0이다. 상세 해시는
`request335-free-qualification-reproduction.json`에 고정했다.

검증: 추가 synthetic 테스트 51개, 관련 테스트 140개 통과. 전체 로컬 suite는
Python 3.12.14에서 **1,441 passed, 기존 warning 243개**다. Critical Ruff와
`git diff --check`도 통과했다. 모의 응답은 코드 검증에만 사용했다.
게시된 source `216f0df45e331c2f1b401f9a01934ed64b82f491`의 push CI
[38040701948](https://github.com/dayofhiki/MoneyMaker/actions/runs/38040701948)와
PR CI [38040704653](https://github.com/dayofhiki/MoneyMaker/actions/runs/38040704653)
모두 lint 및 **1,441 tests**를 통과했다(Python 3.11, 기존 warning 160개).
자격증명을 사용하는 Flat Files metadata job은 두 실행에서 모두 skipped였다.
CI는 모의 코드 검증이며 시장 원자료나 수익성의 검증이 아니다.

## 로컬 실행 절차

계정 권한을 아직 확인하지 않아도 아래 `preflight`와 `replay`는 실행 가능하다.
MoneyMaker 체크아웃의 `victory-trader` 디렉터리에서 실행한다. Python 3.11 이상,
프로젝트 dependencies가 필요하다. 출력 디렉터리는 기존에 존재하면 안 된다.

```bash
python -m pip install -e '.[dev]'
python -m victory_trader.r335_alpaca_qualification preflight --output ../r335-free-preflight
python -m victory_trader.r335_alpaca_qualification replay --output ../r335-free-preflight
```

실제 `collect`는 위 공식 근거와 무료 계정 및 비공개 개인 연구 범위를 기록한 뒤
실행한다. 별도 지원팀 답변은 필요하지 않다. 다음 형식의
`attestation.json`을 저장한다. 각 false는 사실을 확인한 뒤에만 true로 변경하고,
`rights_evidence_reference`에는 적용 공식 문서와 범위의 참조를 넣는다. 이 파일은 private
원본 디렉터리에 복사되며 manifest에는 내용 대신 해시만 기록된다.

```json
{
  "account_is_free_Basic": false,
  "no_incremental_charge": false,
  "historical_SIP_research_and_local_retention_permitted": false,
  "no_redistribution": true,
  "only_registered_technical_sample": true,
  "rights_evidence_reference": ""
}
```

확인된 계정의 키는 Bash에서 아래처럼 숨겨 입력할 수 있다. 대화에 키를 붙여
넣지 않는다. Python은 명시된 두 변수만 읽으며 다른 자격증명을 탐색하지 않는다.
키 문자열을 명령행 인자로 넣는 기능은 없다.

```bash
read -r -s -p 'Alpaca key ID: ' APCA_API_KEY_ID
read -r -s -p 'Alpaca secret: ' APCA_API_SECRET_KEY
export APCA_API_KEY_ID APCA_API_SECRET_KEY
python -m victory_trader.r335_alpaca_qualification collect --attestation attestation.json --output ../r335-free-access-sample
unset APCA_API_KEY_ID APCA_API_SECRET_KEY
python -m victory_trader.r335_alpaca_qualification replay --output ../r335-free-access-sample
```

중단된 실행은 같은 디렉터리에 이어쓰거나 덮어쓰지 않는다. 수집 도중 프로세스가
종료되어 manifest가 없으면 해당 실행은 미완결 증거다. 원본을 유지하고 이유와
별도 실행 ID를 기록한다. 임의 재시도·예산 증가·샘플 변경을 하지 않는다.

CLI 출력의 시도 수·pair 수와 manifest의 상태 집계부터 공유할 수 있다. 원본 시장데이터를
업로드하기 전에는 적용 라이선스와 비공개 저장·처리 환경을 확인한다. 키는
원본/manifest에도 포함되면 안 된다. 원본을 공유하지 않은 요약만으로 연구팀이
데이터 무결성이나 경제적 결론을 인증할 수는 없다.

## 다음 연구를 시작할 최소 조건

접근 표본을 시작하려면 공식 문서를 근거로 한 무료 이용·비공개 개인 연구 확인서,
원본 계획과 이번 계약의
등록된 코드, 자격증명, 적법한 private 저장 환경이 필요하다. 미확인 명세는 위
목록 그대로 남겨 접근/형식만 조사한다. 실제 source qualification은 공식 NBBO,
clock, share/lot units, correction/cancel chronology, same-clock ordering,
원래 ticker coverage 및 license에 대한 근거를 별도로 확보해야 한다.

전체 실험에는 추가로 모든 1,235 identity의 고정 full window와 2,470 stream의
완결성 ledger 및 별도 full acquisition 예산이 필요하다. 경제적 run에는 causal
availability, 주문/미체결/partial fill, 비용/slippage, censoring, 계좌·기회비용 계약을
사전 고정해야 한다. 원래 3초 reference bound, 60초 hold, BASE −2.5% completed-close
stop, 90% 해소 기준은 유지한다. R331의 34.8352%는 여전히 실패한 기준이다.
