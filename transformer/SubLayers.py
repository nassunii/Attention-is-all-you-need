''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang" # 저자~

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module ''' # Multi-Head Attention 모듈 클래스다.
    # head는 병렬로 작동하는 어텐션 메커니즘을 의미한다.
    # 가중치를 계산하는 하나의 부분이다. 모델이 서로 다른 부분에 주의를 기울일 수 있도록 한다.

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head # Multi-Head Attention에서 사용할 head의 개수다.
        self.d_k = d_k # 각 head마다 사용되는 key 벡터의 차원이다.
        self.d_v = d_v # 각 head마다 사용되는 value 벡터의 차원이다.

        # 아래의 코드들은 각 head가 입력을 서로 다른 부분 공간으로 매핑하고, 다양한 정보를 추출할 수 있도록 한다.
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False) # query에 대한 선형 변환 레이어다. 편향을 사용하지 않는다.
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False) # key에 대한 선형 변환 레이어다. 편향을 사용하지 않는다.
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False) # value에 대한 선형 변환 레이어다. 편향을 사용하지 않는다.
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False) # 선형 변환 레이어의 정의다. 편향을 사용하지 않는다.

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5) # Scaled Dot-Product 어텐션의 정의다.

        self.dropout = nn.Dropout(dropout) # Multi-Head Attention에서 이용되는 드롭아웃을 정의한다.
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6) # Multi-Head Attention에서 이용되는 레이어 정규화 정의한다.


    def forward(self, q, k, v, mask=None): # Multi-Head Attention 클래스 내부의 순전파 메서드다.
        # 입력으로 주어진 query, key, value에 대한 연산을 수행한다.

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head # 클래스 초기화에서 정의된 것들이다.
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1) # 입력 query, key, value의 크기와 관련된 정보들이다.
        # 배치 크기, query의 길이, key의 길이, value의 길이를 나타낸다.

        residual = q # 입력 쿼리인 q를 residual에 저장한다.

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        
        # .view(sz_b, len_q, n_head, d_k)는 텐서를 새로운 모양으로 변환한다.
        # Multi-Head Attention을 위하여 head로 나누어진 4차원 텐서(다차원 배열)로 변환한다. 
        # 결과적으로 크기가 sz_b * len_q * n_head * d_k인 텐서가 된다.
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        # 어텐션 연산을 위하여 차원을 바꾸는 부분이다.
        # q.transpose(1,2)에서는 텐서 q의 차원을 바꾼다. 두 번째와 세 번째 차원을 서로 교한한다. 순서가 sz_b, n_head, lem_1, d_k로 바뀐다.
        # k와 v에 대해서도 같은 연산이 수행된다.
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None: # Mask가 주어진 경우에 수행하는 코드다. 브로드캐스팅을 위한 과정이다.
            # 브로드캐스팅은 모양이 다른 두 배열 간의 연산을 가능하게 하는 기능이다.
            # mask.unsqueeze(1)은 mask 텐서에 차원을 추가한다. unsqueeze(1)은 head 축에 대한 자원을 추가하는 것이다.
            # 예를 들어, mask의 크기가 (sz_b, len_q)였다면, (sz_b, 1, len_q)가 된다.
            # 각 헤드에 대한 마스킹을 수행할 수 있도록 헤드 축을 브로드캐스팅할 수 있게 만들어준 것이다.
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask) # attention 연산을 수행한다. mask는 각 head에 적용되어 특정 위치에 대한 attention을 제한하는 역할을 한다.

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # Multi-Head Attention을 수행한 후에 각 head에서 얻은 결과를 다시 원래 형태로 조합하고 최종적인 출력을 생성한다.

        # .contiquous()는 텐서를 메모리에 연속적으로 배치하는 연산을 수행한다.
        # 일부 PyTorch 연산에서 연속 메모리가 필요하기 때문에 이를 보장하기 위하여 사용한다.
        # .view(sz_b, len_q,-1)은 텐서의 크기를 (sz_b, len_1, n*dv)로 변환한다.
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q)) # 선형 변환과 드롭아웃을 수행한다.
        q += residual # 이전에 저장한 residual 값과 어텐션 결과를 더한다. 기울기 소실 문제를 해결하고, 모델의 학습 안정성을 향상하는 효과가 있다.

        q = self.layer_norm(q) # 레이어 정규화를 수행한다. 모델의 안정성을 높이고 학습을 돕는다.

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module ''' # 두 개의 FeedForward 레이어를 가진 모듈이다.

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()

        # 두 개의 선형 변환 레이어를 정의한다.
        self.w_1 = nn.Linear(d_in, d_hid) # 위치에 따라서 다른 가중치를 적용하여 위치 정보를 학습하는 것에 사용된다.
        self.w_2 = nn.Linear(d_hid, d_in) # hidden state를 다시 원래의 입력 차원으로 변환하는 레이어다.
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6) # 입력에 대한 레이어 정규화를 수행한다. 안정성을 높인다.
        self.dropout = nn.Dropout(dropout) # 드롭아웃 레이어다. 일부 뉴런을 무작위로 비활성화해서 과적합을 방지한다.

    def forward(self, x):

        residual = x # 입력을 residual로 지정한다. residual은 네트워크의 깊이가 깊어질 때 기울기 소실 문제를 해결하는 것에 도움을 준다.
        # 원본 x의 값을 저장해둔다.

        x = self.w_2(F.relu(self.w_1(x))) # 입력 x에 첫 번째 선형 레이어인 self.w_1을 적용하고, 그 결과에ReLU 활성화 함수를 적용한다.
        # ReLU는 x>0일 때 x=y, x<0일 때 0이다.
        # 두 번째 선형 레이어인 self.w_2를 적용한다. 이 과정으로 비선형성이 추가되고, 위치 정보를 학습할 수 있다.
        x = self.dropout(x) # 드롭아웃을 수행한다.
        x += residual # 레이어를 거친 x 값에 원본 x의 값을 더한다.

        x = self.layer_norm(x) # 레이어 정규화를 수행한다. 평균이 0, 분산이 1이 되게 만들어준다.

        return x
