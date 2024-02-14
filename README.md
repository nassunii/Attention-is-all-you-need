# Attention is all you need Study (Add comments)
forked from https://github.com/jadore801120/attention-is-all-you-need-pytorch  
paper: https://arxiv.org/abs/1706.03762
<br/><br/>

## 스터디원
- [Yeseon Hong](https://github.com/nassunii)
- [Dayeon Seo](https://github.com/dayeon-seo)
- [Yewon Kim](https://github.com/yewonkim01)

<br/>

## 스터디 진행 과정
스터디는 회차별 분량을 지정해 본인이 맡은 파트(사다리 타기로 결정)를 공부해온 후  
줌미팅에서 본인 파트를 설명하고 q&a하는 방식으로 진행하였음  
공통으로 학습한 부분 주석은 가장 마지막에 커밋한 사람으로 설정됨

```
<줌스터디 일정>
1회차: 2/5  
2회차: 2/8  
2-2회차: 2/11 (못한 부분만 잠깐 진행)  
3회차: 2/12
```


---
**[1회차]**  

modules -> sublayers -> layers -> models
- modules
- sublayers
- layers

modules/ sublayers/ layers는 모두 이해가 필요하다고 생각되어 공통학습<br/>
models.py만 파트 분배
<br/>

*<models.py주석>*
```
- encoder: 예선
- decoder: 다연
- transforemr: 예원

(models.py 내 encoder, decoder, transformer 외 다른 함수는 본인 클래스에서 필요한 경우 주석달기)
```
---

**[2회차]**  

learn_bpe -> apply_bpe -> preprocess

*<주석>*
```
- learn_bpe: 다연
- apply_bpe: 예원
- preprocess: 예선
```
----

**[3회차]**  

optim-> train -> translator -> translate

*<주석>*
```
- optim: 예선
- train: 예선
- translator: 다연
- translate: 예원
```

  
