# Page snapshot

```yaml
- generic [active] [ref=e1]:
  - region "여행지 선택 화면" [ref=e2]:
    - generic [ref=e4]:
      - heading "✈️ 유럽 여행 2024" [level=1] [ref=e5]
      - paragraph [ref=e6]: 어느 도시의 추억을 보시겠어요?
      - navigation "여행지 선택" [ref=e7]:
        - button "런던 여행 보기" [ref=e8] [cursor=pointer]:
          - generic [ref=e9]: GB
          - generic [ref=e10]: 런던
          - generic [ref=e11]: London
        - button "파리,브뤼셀 여행 보기" [ref=e12] [cursor=pointer]:
          - generic [ref=e13]:
            - text: FR,BE
            - generic [ref=e14]: 파리,브뤼셀 Paris,Brussel
  - complementary "환율 정보" [ref=e15]:
    - generic [ref=e16]: 💰 환율 정보 로딩 중...
  - contentinfo [ref=e17]:
    - paragraph [ref=e18]: © 2024 유럽 여행 회고. All rights reserved.
```