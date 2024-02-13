''' This module will handle the text generation with beam search. '''
# beam search를 사용해서 텍스트를 생성한다. 
# beam search는 시퀀스 생성 작업에서 사용되는 탐색 알고리즘 중 하나다. 가장 가능성 높은 시퀀스를 찾는 것에 이용된다.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Models import Transformer, get_pad_mask, get_subsequent_mask


class Translator(nn.Module): # 번역 작업을 수행하는 Translator를 정의한다.
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):
        # model은 훈련된 모델이다. beam_size는 beam search에서 사용할 beam width다. 각 시점에서 고려할 후보의 수를 결정한다.
                # max_seq_len은 생성할 번역 시퀀스의 최대 길이다. 이 길이를 넘으면 작업이 종료된다.
                # src_pad_idxdhk trg_pad_idx는 각각 입력,출력 시퀀스에서 사용되는 패딩 토큰의 인덱스다.
                # trg_bos_idx와 trg_eos_idx는 출력 시퀀스에서 사용되는 시작, 종료 토큰이다.

        super(Translator, self).__init__()

        # 이 아래는 값을 초기화하는 부분이다. 설명은 위에 있는 것과 같다.
        self.alpha = 0.7 # 번역하는 후보에 대한 길이 보정을 한다. (짧은 것이 선호된다.)
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval() # .eval()을 하면 평가모드로 설정할 수 있다.
        # 모델이 학습 중에 사용되었던 드롭아웃, 정규화 같은 것들이 비활성화된다. 일관된 결과를 얻을 수 있다.

        # 아래는 추가적인 버퍼를 등록하는 부분이다. beam search가 효율적이고 정확하게 실행되게 해준다.
        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]])) # init_seq는 빔 서치를 시작할 때 사용되는 초기 시퀀스다.
                # 이것을 번역 시퀀스의 시작을 나타내는 토큰인 trg_bos_idx 하나만 포함된 텐서로 초기화한다.
        self.register_buffer(
            'blank_seqs', 
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long)) # blank_seqs는 빔 서치 중에 채워지는 빈 시퀀스다.
                # trg_pad_idx로 초기화된 (beam_size, max_seq_len) 크기의 텐서를 생성한다.
        self.blank_seqs[:, 0] = self.trg_bos_idx # blank_seqs의 각각 시퀀스에 있는 첫 번째 원소를 trg_bos_idx로 설정한다.
        self.register_buffer(
            'len_map', 
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0)) # len_map은 생성된 각 시퀀스의 길이를 나타낸다.
                # 1에서 max_seq_len까지 정수를 포함한다. 차원을 추가해서 (1, max_seq_len) 모양으로 바꿔준다.


    def _model_decode(self, trg_seq, enc_output, src_mask): # 모델의 디코딩 결과를 확률 분포로 반환하는 역할을 한다.
        trg_mask = get_subsequent_mask(trg_seq) # 마스크를 생성하는 함수를 호출해서 목표 시퀀스 이후에 나오는 토큰에 대한 마스크를 생성한다.
        dec_output, *_ = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask) # 트랜스포머 디코더 모델에 인자를 전달해서 디코딩을 수행한다.
        return F.softmax(self.model.trg_word_prj(dec_output), dim=-1) # trg_word_prj()은 디코더 출력을 받아서 번역된 단어를 확률 분포로 변환하는 선형 변환을 수행한다.
        # F.softmax()는 소프트맥스 함수를 사용해서 디코더 출력을 확률 분포로 변환한다.
        # 다음에 생성될 단어의 확률을 나타낸다.


    def _get_init_state(self, src_seq, src_mask): # 빔 서치의 초기 상태를 설정하고, 빔 서치를 수행할 준비를 한다.
        beam_size = self.beam_size

        enc_output, *_ = self.model.encoder(src_seq, src_mask) # 인코더에 소스 시퀀스와 소스 마스크를 전달해서 인코더 출력을 얻는다.
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask) # 디코더의 초기 상태를 얻는다.
        
        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(beam_size) # 디코더 출력에서 각 빔마다 최상위 k개의 시퀀스를 가져온다.

        #아래는 각 빔에 대한 로그 확률을 계산하고 빈 시퀀스를 초기화해서 생성 시퀀스를 얻는 과정이다.
        scores = torch.log(best_k_probs).view(beam_size) # 각 빔에 대한 로그 확률을 계산하고 빔의 크기에 맞게 형태를 조정한다.
        gen_seq = self.blank_seqs.clone().detach() # 빈 시퀀스를 복제해서 초기화한다.
        gen_seq[:, 1] = best_k_idx[0] # 초기화된 시퀀스의 두 번째 위치에 위에서 구한 k개의 인덱스를 설정한다.
        enc_output = enc_output.repeat(beam_size, 1, 1) # 인코더 출력을 빔 크기만큼 반복한다.
        return enc_output, gen_seq, scores
    


    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        # 빔 서치 과정 중에 현재까지의 시퀀스인 gen_seq와 디코더의 출력인 dec_output, 현재까지의 누적 로그 확률인 scores를 사용한다.
        # 현재까지의 상태를 고려해서 다음 단계의 빔 서치를 수행하는 함수다.
        # 누적 확률이 가장 높은 후보를 선택하고, 새로운 확률 및 시퀀스를 업데이트한다.
        
        assert len(scores.size()) == 1 # 현재까지의 scores가 1차원이어야 한다.
        
        beam_size = self.beam_size # beam_size를 확인해서 변수에 저장한다.

        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size) # 위에 있었던 최상위 k개의 후보를 선택하는 과정이다. 

        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1) # 현재까지의 누적 확률에 새로운 후보의 로그 확률을 더해서 새로운 누적 확률을 계산한다.

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size) # 새롭게 구한 누적 확률에서 상위 k개의 후보를 선택한다.
 
        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size # 선택된 후보의 인덱스를 반환한다.
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step] # 현재까지의 시퀀스를 선택된 후보들로 갱신한다.
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores # 새로운 누적 확률을 반환한다.


    def translate_sentence(self, src_seq): # 소스 시퀀스인 src_seq를 이용해서 번역을 수행한다.
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        assert src_seq.size(0) == 1 # batch size가 1인 경우만 허용한다. 단일 문장에 대한 번역만 처리하는 것이다.

        # 아래는 하이퍼파라미터를 설정하는 코드다.
        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx 
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha 

        with torch.no_grad(): # 모델이 평가 중인 것을 나타낸다. 기울기 계산이나 역전파가 일어나지 않게 한다.
            src_mask = get_pad_mask(src_seq, src_pad_idx) # 소스코드에서 패딩을 마스킹한다.
            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask) # 빔 서치의 초기 상태를 생성한다.

            ans_idx = 0   # default
            for step in range(2, max_seq_len):    # decode up to max length
            # 빔 서치 단계를 반복하면서 step을 늘린다. 디코딩 과정이다.
            # max_seq_len까지 디코딩한다.
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask) # _model_decode 함수를 사용해서 디코더 출력을 계산한다.
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step) # _get_the_best_score_and_idx 함수를 사용해서 누적된 확률과 시퀀스를 업데이트한다.

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx   # 종료 토큰의 위치를 확인한다.
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1) # 모든 빔이 종료되었는지 확인한다.
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0) # 누적 확률과 길이를 고려해서 최종 최상위 후보의 인덱스를 선택한다.
                    ans_idx = ans_idx.item()
                    break
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()
