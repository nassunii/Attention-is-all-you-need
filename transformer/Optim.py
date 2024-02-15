

# [[예선 주석]]

'''A wrapper class for scheduled optimizer ''' # 학습률을 동적으로 스케줄하기 위한 클래스다.
# 옵티마이저는 모델의 가중치를 조절하여 손실 함수를 최소화하는 방향으로 학습하는 알고리즘이다.
import numpy as np

#학습률 scaling -> 학습의 안정성과 성능 향상
class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''
    # 기본 data type을 Object로 변환하기 위해 수행 -> 외부에서 기본 type 값 변경 불가

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul # 초기 학습률을 나타내는 매개변수다.
        self.d_model = d_model # 모델의 차원이다.
        self.n_warmup_steps = n_warmup_steps # 모델의 학습률을 증가시키는 것에 필요한 매개변수다.
        self.n_steps = 0 # 현재 몇 단계까지 진행됐는지 알려주는 매개변수다.


    def step_and_update_lr(self):
        "Step with the inner optimizer" # 내부 옵티마이저로 단계를 진행하고 학습률을 업데이트한다.
        self._update_learning_rate() # 학습률 업데이트.
        #gradient 초기화
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer" # 내부 옵티마이저의 기울기를 초기화한다.
        self._optimizer.zero_grad()


    def _get_lr_scale(self): # 현재 단계 카운트를 기반으로 학습률 속도 계산한다. 학습률이 시간에 따라서 어떻게 변할지 결정할 수 있게 한다.
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5)) # 계산 결과다.


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step ''' # 각 단계마다 학습률을 스케줄링한다.

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale() #학습률 계산

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr #각 매개변수(parameter) group에 대한 학습률 update

