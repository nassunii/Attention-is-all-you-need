#공통

# 시퀀스에서 계속해서 사용되고, 특수한 의미를 가진 토큰들을 정의한다.

PAD_WORD = '<blank>' # 패딩을 나타내는 토큰이다. 문장의 길이가 다를 때 패딩을 추가하여 같은 길이로 맞춰줄 수 있다.
UNK_WORD = '<unk>' # unknown 단어를 unk 토큰으로 바꾼다.
BOS_WORD = '<s>' # 문장의 시작을 나타내는 토큰이다.
EOS_WORD = '</s>' # 문장의 끝을 나타내는 토큰이다.
