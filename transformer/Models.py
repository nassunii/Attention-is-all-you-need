''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer


__author__ = "Yu-Hsiang Huang"

"""
get_pad_mask: 예원
get_subsequent_mask: 예원
positional encoding: 예선
encoder: 예선
decoder: 다연
transformer: 예원
"""

'''
[pad] 토큰은 모델이 입력으로 받는 <최대길이>보다 길이가 짧은 문장들에 한해 부여되는 토큰
[pad] 토큰은 실질적인 의미가 없으므로, attention 연산에 [pad]토큰 반영 안해주기 위해서 

나중에 scaledotproduct에서 쓰임!!!
'''
def get_pad_mask(seq, pad_idx):  #패딩된 부분을 마스킹 -> 패딩된 위치에 0을, 패딩되지 않은 위치에 1을 -> 패딩된 곳 연산 안하도록
    return (seq != pad_idx).unsqueeze(-2)  #마스크에 차원추가: 뒤에서 두번째 차원에 1추가 -> sequence 차원이 (배치크기, sequence 길이)라면 (배치크기, 1, sequence 길이)

"""
다음 단어에 대한 마스킹 생성하는 것-> cheating 예방
attention 연산 수행할 때 현재 위치 이후의 정보는 사용하지 않도록 하기 위해 사용
"""
def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    #torch.triu: 행렬의 upper triangular matrix 만드는 함수
    # '1-*' 로 현재있는 0과 1을 뒤집어서 현재 위치 이후를 0으로 만들어주고 bool()로 마스크를 불리언 형태로 변환
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


#input으로 들어갈 vector 안에 positional encoding(단어의 위치 정보) 포함
class PositionalEncoding(nn.Module):

    #d_hid : 주어진 차원
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        
        #sin과 cos를 이용해 위치(Posisiton) encoding table형성
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
        # x(input vector) + encoding vector(위치 정보)
    

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        #input 단어 embedding
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        #위치(Position) encoding
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        #이게 residual Dropout?
        self.dropout = nn.Dropout(p=dropout)
        #여러 개의 Encoder layer를 쌓아올림
        self.layer_stack = nn.ModuleList([
            #하나의 Encoder layer를 정의
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)]) #n_layers = Encoder layer의 총 개수
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb #scale할지말지 여부 결정
        self.d_model = d_model #model의 hidden state 차원 나타냄

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            #True인 경우, 각 layer의 Self Attention 가중치(W)를 list에 저장하여 return
            return enc_output, enc_slf_attn_list #Self-Attention 가중치를 포함하는 list
        return enc_output,



class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. ''' # self-attention 메커니즘을 갖는 디코더 모델이다.

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx) # nn.Embedding은 n_trg_vocab과 d_word_vec에 따라서 단어를 벡터로 매핑해주는 파라미터화된 층이다.
                # 텍스트 데이터의 단어를 연속적인 벡터 공간으로 매핑하는 것에 사용된다.
                # n_trg_vocab은 목효하는 언어의 어휘의 크기다. 모델이 학습할 수 있는 고유한 단어의 수다.
                # d_word_vec은 단어를 임베딩하는 데 사용되는 벡터의 차원이다. (임베딩은 문장 또는 개체를 고차원 벡터 공간에 매핑하는 기법이다.)
                # padding_idx는 패딩을 위한 인덱스다.
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position) # 위치 인코딩을 해준다. (위에 PositionalEncoding 나옴.)
        self.dropout = nn.Dropout(p=dropout) # 드롭아웃을 적용한다.
        self.layer_stack = nn.ModuleList([ # nn.ModuleList를 사용해서 디코더 레이어를 쌓을 수 있다.
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) # DecoderLayer라는 클래스를 정의하고, 클래스의 인스턴트들을 nn.ModuleList에 담아서 쌓아 놓는 것이다.
            # d_model은 디코더 레이어의 입력 및 출력 차원의 수다. d_inner는 디코더 레이어 내부의 FeedForward 신경망에서 중간에 있는 차원의 수다.
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6) # 레이어 정규화를 수행한다.
        self.scale_emb = scale_emb # 임베딩 스케일 조절 여부다.
        self.d_model = d_model # 모델의 임베딩 차원이다.

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False): # 디코더의 순전파 연산을 정의한다.
        # trg_seqp는 목표 언어 시퀀스다. 디코더의 입력으로 사용된다.
        # trg_mask는 목표 언어 시퀀스에서 패딩을 마스킹한 텐서다.
        # enc_output은 인코더의 출력이다.
        # return_attns는 어텐션 결과를 반환할지 여부를 결정한다.

        dec_slf_attn_list, dec_enc_attn_list = [], [] # 어텐션 메커니즘의 결과를 저장하기 위한 빈 리스트다.

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq) # 목표 언어의 시퀀스를 단어 임베딩으로 매핑한다.
        # nn.Embedding 층을 사용해서 단어를 임베딩 벡터로 바꾼다.
        if self.scale_emb: # scale_emb가 true면 임베딩 출력에 self.d_model에 아래의 연산을 해서 임베딩을 스케일링한다.
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output)) # 임베딩에 드롭아웃을 적용한다.
        dec_output = self.layer_norm(dec_output) # 레이어 정규화를 수행한다.

        for dec_layer in self.layer_stack: # 디코더의 레이어 스택을 반복한다.
            # 반복문을 동태서 디코더의 각 레이어에 입력을 전달하고 각 레이어에서의 어텐션 결과를 기록할 수 있다.
            # 디코더의 순전파가 이루어지게 된다.
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer( # 각 디코더 레이어에 dec_output과 enc_output을 전달해서 디코더 레이어의 순전파 연산을 수행한다.
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else [] # return_attns가 True면 self-attention 결과인 dec_slf_attn을 어텐션 리스트인 dec_slf_attn_list에 추가한다.
            dec_enc_attn_list += [dec_enc_attn] if return_attns else [] # return_attns가 True면 enc-dec 결과인 dec_enc_attn을 어텐션 리스트인 dec_enc_attn_list에 추가한다.

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

"""예원 작성"""
class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        
        #assert : assert 구문이 들어가는 코드부분에서 어떤 조건이 참임을 확고히하는 것
        # 'emb', 'prj', 'none' 중 하나를 선택해서 scaling 적용할지 말지
        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        #d_model: decoder출력의 차원/ target의 vocab size 크기로 차원 맞춰줌
        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        #모든 가중치에 대해 초기화 수행
        for p in self.parameters():
            #텐서 차원이 1보다 클 경우(bias 편향은 초기화 안함)
            if p.dim() > 1:
                #가중치 초기화는 Xavier uniform 사용
                nn.init.xavier_uniform_(p) 

        #d_model == d_word_vec이어야 한다!
        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'
        
        #타겟 언어와 linear projection layer간 가중치 고려할지 여부
        #가중치 공유: 학습 parameter 수 줄이고, 일반화 성능 향상
        if trg_emb_prj_weight_sharing: 
            # Share the weight between target word embedding & last dense layer
            #출력단어의 단어 임베딩 가중치와 디코더의 last layer의 가중치 공유 -> 모델이 효과적으로 문장생성한다고 함
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            #이번에는 인코더의 input source임베딩과 타겟 단어 임베딩 가중치 공유 -> 인코더와 디코더 사이의 임베딩 공유를 하면 일반화된다고 함
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq, trg_seq):
        #패딩된 idx에 따라서 마스킹(패딩된 위치는 연산 안하게)
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        #target 디코더에서는 두 개 다 고려해야함
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        #encoder/decoder output 가져와서 계산
        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)

        #scale하는 거면 scaling
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5 #(루트 d_k랑 똑같음)

        #벡터로 변환
        return seq_logit.view(-1, seq_logit.size(2))
