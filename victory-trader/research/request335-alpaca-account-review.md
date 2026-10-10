# R335 Alpaca 계정 확인 — 2026-10-10

사용자가 Alpaca 계정에서 무료 연구에 필요한 확인 작업을 위임했고 직접 보안
로그인을 완료했다. 계정 화면에서 Paper Trading 및 Plans & Features의
`Subscription Status: Basic`, Basic `Included / Current Plan`을 확인했다.
계정 ID, 이메일, API key/secret, 인증 쿠키는 이 기록에 포함하지 않는다.

이번 단계의 추가 지출, 시장 데이터 HTTP 요청, 새 경제적 라벨은 모두 0이다.
구독 변경, 무료 체험 시작, live 계좌 개설, 주문, 보안 설정 변경은 없다.
기존 407 training / 828 development의 원자료 및 전체 수익성 검증은 계속 막혀 있다.

## 권한 확인 상태

| 항목 | 확인 상태 |
|---|---|
| 사용자가 로그인한 실제 Paper 계정 | 확인 |
| 실제 현재 데이터 플랜 Basic | 확인 |
| Basic 플랜의 추가 구독료 없음 | 계정의 Included 표시 및 공식 Terms의 no-cost 설명 확인 |
| 2026년 5월 historical SIP의 실제 계정 접근 | 미실행 / 미확인 |
| 해외 Paper Only 계정의 historical SIP entitlement | 공개 FAQ와 Paper Trading 안내의 범위 차이 미해소 |
| 개인 연구를 위한 SIP 원본의 비공개 보관·재현 권리 | 개인/비상업 용도는 일반 Terms에 명시. 현재 계정의 SIP 및 보관 범위에 대한 확정 답변 미확보 |
| official NBBO, clock, units, corrections, chronological order | 기존 미확인 상태 유지 |

공식 FAQ는 15분보다 오래된 historical SIP 요청을 구독 없이 허용한다고 설명한다.
Paper Trading 안내는 전 세계 Paper Only 가입을 허용하면서 IEX 이용권으로 설명한다.
일반 Terms는 Basic 무료 및 개인/비상업 이용을 설명하지만 별도의 미국 거주자 대상
문구가 있다. 특정 문구를 골라 현재 계정의 모든 권리를 인증하지 않는다.
원본의 공개 배포·상업 서비스 이용은 현재 위임과 데이터 계약의 범위에 포함하지 않는다.

근거:
- https://docs.alpaca.markets/us/docs/market-data-faq
- https://docs.alpaca.markets/us/docs/paper-trading
- https://alpaca.markets/disclosures
- https://files.alpaca.markets/disclosures/library/TermsAndConditions.pdf

NASDAQ 및 NYSE subscriber agreement 링크는 공개 조회에서 403이 반환되어 해당
파일 내용을 확인하지 못했다. 이 403은 계정의 시장 API 거부 증거가 아니다.

## 공급자 문의 초안 — 아직 발송하지 않음

수신 대상: Alpaca Support. 공식 https://alpaca.markets/support/request 양식에
Subject와 Description만 작성했다. Name/Email은 비워 두었고 Submit은 누르지 않았다.
검토 화면은 아래와 같다.

![미발송 문의 초안](request335-alpaca-support-draft-20261010.jpg)

 용도: 기존 무료 Basic Paper Only 계정의 원자료 권한과
재현 가능한 비공개 보관 확인. API key/secret, 계정 ID, 금융정보, 연구 결과,
원본 시장데이터를 보내지 않는다. 구독이나 구매를 요청하지 않는다.

Subject: Free Basic Paper Only — historical SIP access and private research retention

Hello Alpaca Support,

I have a free Basic Paper Only account and want to evaluate historical US stock
trades and NBBO quotes for private, non-commercial research. Could you confirm:

1. Whether historical /v2/stocks/trades and /v2/stocks/quotes with feed=sip for
   May 5–20, 2026 are available to this account with no additional charge,
   subscription upgrade, or paid trial.
2. Whether the historical SIP permission also applies to non-US Paper Only
   users. The Market Data FAQ describes SIP queries ending over 15 minutes ago
   without a subscription, while the Paper Trading page describes IEX-only rights.
3. Whether original API responses can be retained privately while this account
   remains active and replayed for reproducible personal research, without any
   redistribution. Please identify the applicable data-use/retention terms and
   whether private cloud research processing requires separate permission.

Please do not activate any subscription, trial, purchase, or paid service. This
request is for clarification only.

Thank you.

## 후속 조건

지원팀 문의를 발송하는 것은 사용자 명의의 외부 메시지이므로 명시적인 발송
승인을 받기 전에는 보내지 않는다. 로그인 및 화면 확인의 완료를 라이선스
확인서의 true 값으로 바꾸지 않는다. 현재 추가 계약이나 권리 확인서에 서명하지 않는다.

무료 이용 및 비공개 보관 범위가 확인되면 필요한 API credential 연결 단계로
진행한다. 새 key 생성/재발급은 새 접근권한 또는 기존 credential 변경이므로
해당 동작 직전에 보안 확인 또는 사용자 직접 수행을 거친다. Key를 채팅이나
공개 저장소에 기록하지 않는다. 실제 수집은 이미 고정된 24개 technical pair,
48 HTTP 시도, page/byte/time/denial stop 규칙을 그대로 적용한다.
