''' Define the Layers '''
import torch.nn as nn # torch.nn은 PyTorch에서 신경망을 구축할 수 있게 해주는 패키지다.
import torch # PyTorch는 미분을 자동으로 해줘서 구한 기울기로 가중치를 이용할 수 있게 해준다.
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward # MultiHeadAttention을 이용한다. 위치별로 FeedForward 신경망을 정의한다.


__author__ = "Yu-Hsiang Huang" # 논문 저자~


class EncoderLayer(nn.Module): # 인코더 레이어를 정의한다. nn.Module을 상속 받는다. nn.Module은 모든 신경망 모듈의 기본 클래스다. 자체 가중치와 학습 가능한 매개변수를 관리할 수 있게 해준다. 
    ''' Compose with two layers ''' # 두 개의 레이어로 정의되어 있다. self-attention과 위치별 FeedForward다.

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1): # 인코더 레이어의 가중치를 적절하게 만든다. 필요한 하이퍼 파라미터들을 알 수 있다.
        # d_model은 차원의 크기를 결정한다. n_head는 헤드의 개수다. d_v와 d_k는 각각 값과 키의 차원이다. dropout은 일반화를 위하여 일부 유닛을 비활성화하는 것이고, 그 비율을 나타낸다.
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout) # 인코더의 self-attention 레이어 정의.
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout) # 인코더의 위치별 FeedForward 레이어 정의.

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask) # self-attention 레이어다. 입력 시퀀스가 들어간다.
        enc_output = self.pos_ffn(enc_output) # 위치별 FeedForward 레이더다.
        return enc_output, enc_slf_attn 


class DecoderLayer(nn.Module): # 디코더 레이어를 정의한다.
    ''' Compose with three layers ''' # 세 개의 레이어로 정의되어 있다. self-attention과 encoder-decoder-attention과 위치별 FeedForward다.

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1): # 인코더와 동일하다.
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout) # 디코더의 self-attention 레이어 정의.
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout) # 디코더의 encoder-decoder-attention 레이어 정의
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout) # 디코더의 위치별 FeedForawrd 레이어 정의.

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask) # self-attention 레이어다.
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask) # encoder-decoder 레이어다.
        dec_output = self.pos_ffn(dec_output) # 위치별 FeedForward 레이어다.
        return dec_output, dec_slf_attn, dec_enc_attn
