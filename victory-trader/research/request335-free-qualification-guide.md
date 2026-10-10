# R335 무료 표본 검증 실행 안내

2026-10-10 후속 작업. 추가 지출 허용액은 $0이며, PR108은 draft 상태를 유지한다.
이 단계는 **접근권한·응답 형식·페이지 연결의 기술 검증**이다. 전체 1,235개
identity의 원자료 확보, 실제 체결 검증, 모델 수익성 검증을 완료하지 않는다.

## 지금 사용자가 확인할 사항

1. 이미 무료 Alpaca Basic 계정이 있으면 그 계정을 우선 확인한다. 없으면
   본인이 무료 계정의 가입 조건을 검토한다. AlgoTrader Plus 결제, 입금,
   실거래 계좌 개설 또는 Massive 업그레이드를 이 작업을 위해 진행하지 않는다.
2. 해당 계정에서 2026년 5월 과거 **SIP Trades/Quotes 조회, 개인 연구 사용,
   원본의 로컬 보관과 재현**이 무료로 허용되는지 확인한다. 적용 약관이나
   권한 설명의 URL·확인일 또는 별도 확인의 참조를 보관한다. 기술적 200 응답은
   라이선스의 증거가 아니며, 코드의 확인서도 법적 권리를 만들어 주지 않는다.
3. 우선 확인 결과만 알려준다. API key/secret은 대화, PR, 코드, 로그에 보내지
   않는다. 현재 PR의 자동 CI에는 Alpaca 자격증명을 주입하거나 수집하는 job이
   없다. 지금 GitHub Secrets에 키를 넣어도 실제 수집이 시작되는 구조가 아니다.

Alpaca FAQ는 historical `end`가 15분보다 오래된 SIP 조회를 구독 없이 허용한다고
설명한다. 반면 Paper Trading 문서는 Paper Only 계정의 데이터 이용권을 IEX로
설명한다. 따라서 무료 계정 생성 가능성이 곧 이 연구용 SIP 권리를 뜻하지
않는다. 문서가 충돌하면 IEX를 동등한 NBBO로 대체하거나 유료 플랜을 결제하지
않는다. 권한 확인이 어려우면 기관 Daily TAQ 또는 적법한 보유 원본 경로를
유지한다. 대학/WRDS 로그인 자체는 TAQ 모듈 사용권의 증거가 아니다.

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

**실행 결과:** 자격증명 없는 원래 24 pair의 사전 점검과 replay를 각각 두 번
수행했다. 계약·manifest·replay 결과가 바이트 단위로 일치했다. 모든 pair는
`NOT_REQUESTED_MISSING_CREDENTIAL`, 실제 시장 HTTP 시도 0이다. 상세 해시는
`request335-free-qualification-reproduction.json`에 고정했다.

검증: 추가 synthetic 테스트 51개, 관련 테스트 140개 통과. 전체 로컬 suite는
Python 3.12.14에서 **1,441 passed, 기존 warning 243개**다. Critical Ruff와
`git diff --check`도 통과했다. 모의 응답은 코드 검증에만 사용했다.

## 로컬 실행 절차

계정 권한을 아직 확인하지 않아도 아래 `preflight`와 `replay`는 실행 가능하다.
MoneyMaker 체크아웃의 `victory-trader` 디렉터리에서 실행한다. Python 3.11 이상,
프로젝트 dependencies가 필요하다. 출력 디렉터리는 기존에 존재하면 안 된다.

```bash
python -m pip install -e '.[dev]'
python -m victory_trader.r335_alpaca_qualification preflight --output ../r335-free-preflight
python -m victory_trader.r335_alpaca_qualification replay --output ../r335-free-preflight
```

실제 `collect`는 위 무료 사용·보관 권한이 확인된 경우에만 실행한다. 다음 형식의
`attestation.json`을 저장한다. 각 false는 사실을 확인한 뒤에만 true로 변경하고,
`rights_evidence_reference`에는 적용 권리 확인의 참조를 넣는다. 이 파일은 private
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
python -m victory_trader.r335_alpaca_qualification collect --attestation attestation.json --output ../r335-free-qualified-access
unset APCA_API_KEY_ID APCA_API_SECRET_KEY
python -m victory_trader.r335_alpaca_qualification replay --output ../r335-free-qualified-access
```

중단된 실행은 같은 디렉터리에 이어쓰거나 덮어쓰지 않는다. 수집 도중 프로세스가
종료되어 manifest가 없으면 해당 실행은 미완결 증거다. 원본을 유지하고 이유와
별도 실행 ID를 기록한다. 임의 재시도·예산 증가·샘플 변경을 하지 않는다.

CLI 출력의 시도 수·pair 수·상태 요약부터 공유할 수 있다. 원본 시장데이터를
업로드하기 전에는 적용 라이선스와 비공개 저장·처리 환경을 확인한다. 키는
원본/manifest에도 포함되면 안 된다. 원본을 공유하지 않은 요약만으로 연구팀이
데이터 무결성이나 경제적 결론을 인증할 수는 없다.

## 다음 연구를 시작할 최소 조건

접근 표본을 시작하려면 무료 계정의 권한·보관 확인서, 원본 계획과 이번 계약의
등록된 코드, 자격증명, 적법한 private 저장 환경이 필요하다. 미확인 명세는 위
목록 그대로 남겨 접근/형식만 조사한다. 실제 source qualification은 공식 NBBO,
clock, share/lot units, correction/cancel chronology, same-clock ordering,
원래 ticker coverage 및 license에 대한 근거를 별도로 확보해야 한다.

전체 실험에는 추가로 모든 1,235 identity의 고정 full window와 2,470 stream의
완결성 ledger 및 별도 full acquisition 예산이 필요하다. 경제적 run에는 causal
availability, 주문/미체결/partial fill, 비용/slippage, censoring, 계좌·기회비용 계약을
사전 고정해야 한다. 원래 3초 reference bound, 60초 hold, BASE −2.5% completed-close
stop, 90% 해소 기준은 유지한다. R331의 34.8352%는 여전히 실패한 기준이다.
