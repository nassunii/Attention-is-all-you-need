# Attention is all you need: A Pytorch Implementation
forked from https://github.com/jadore801120/attention-is-all-you-need-pytorch



### 스터디 진행 과정
스터디는 회차별 분량을 지정해 본인이 맡은 파트(사다리 타기로 결정)를 공부해오고 줌미팅에서 본인 파트를 설명하고 q&a하는 방식으로 진행하였음
공통으로 학습한 부분 주석은 가장 마지막에 커밋한 사람으로 하였음


1회차: 2/5
2회차: 2/8 
2-2회자: 2/11 (못한 부분만 잠깐 진행)
3회차: 2/12


modules -> sublayers -> layers -> models
- moduels
- sublayers
- layers는 모두 이해가 필요하다고 생각되어 공통학습
  * 주석
- models에서 encoder: 예선
            decoder: 다연
            transformer: 예원
  (encoder, decoder, transformer 외 다른 함수는 본인 클래스에서 필요한 경우 주석달기)
---

2회차
learn_bpe -> apply_bpe -> preprocess
- learn_bpe: 다연
- apply_bpe: 예원
- preprocess: 예선
----

3회차
optim-> train -> translator -> translate
- optim: 예선
- train: 예선
- translator: 다연
- translate: 예원

  
