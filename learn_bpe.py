#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich

"""Use byte pair encoding (BPE) to learn a variable-length encoding of the vocabulary in a text.
Unlike the original BPE, it does not compress the plain text, but can be used to reduce the vocabulary
of a text to a configurable number of symbols, with only a small increase in the number of tokens.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

# BPE를 이용하여 주어진 텍스트의 어휘를 가변 길이로 인코딩하는 것을 목적으로 한다.
# BPE는 텍스트 어휘를 구성하는 단어 또는 문자열의 쌍을 찾아서 합치는 과정을 반복하면서 어휘의 크기를 줄이는 방법이다.
# 이 코드에서는 원본 텍스트를 압축하지 않고 심볼 수로 어휘의의 크기를 줄이는 것에 사용된다.

from __future__ import unicode_literals

import os # 운영체제와 상호작용하기 위한 모듈이다.
import sys # 파이썬 인터프리터와 관련된 기능을 제공하는 모듈이다.
import inspect # 현재 실행 중인 코드의 정보를 검사하기 위한 모듈이다.
import codecs # 파일을 읽고 쓰는 것에 이용되는 문자 인코딩을 처리하기 위한 모듈이다.
import re # 정규 표현식을 지원하는 모듈이다. 텍스트에서 패턴을 검색하거나 변형하는 것에 사용된다.
import copy # 객체의 복사본을 생성하기 위한 모듈이다.
import warnings # 경고 메세지를 처리하기 위한 모듈을 가져온다.
from collections import defaultdict, Counter # 기본 값을 가진 딕셔너리를 생성하는 클래스와 요소들의 개수를 세는 것에 사용되는 클래스다.


def update_vocabulary(vocab, file_name, is_dict=False): # is_dict는 파일이 단어 빈도 정보를 포함하는 딕셔너리 형식인지 여부를 나타낸다.
    """Read text and return dictionary that encodes vocabulary # 주어진 파일에서 텍스트를 읽고 업데이트한다.
    """

    #vocab = Counter()
    with codecs.open(file_name, encoding='utf-8') as fobj: # codecs.open을 사용하여 주어진 파일을 utf-8 인코딩으로 연다.
        # utf-8은 유니코드 문자 집합을 지원한다. 다양한 언어의 문자를 효과적으로 다루기 위하여 자주 사용된다.
        for i, line in enumerate(fobj): # 파일에서 한 줄씩 읽어오며 각 줄에 대한 반복을 수행한다.
            # enumerate 함수를 사용하면 현재의 줄 번호인 i와 해당 줄의 내용인 line을 얻을 수 있다.
            if is_dict: # is_dict가 true일 때. (단어 빈도 정보를 포함한다.)
                try:
                    word, count = line.strip('\r\n ').split(' ') # 공백으로 구분된 단어와 빈도를 추출한다.
                except: # 에러 발생
                    print('Failed reading vocabulary file at line {0}: {1}'.format(i, line))
                    sys.exit(1)
                vocab[word] += int(count) # 추출한 단어와 빈도를 어휘에 추가한다.
            else: # is_dict가 false일 때 (단어 빈도 정보를 포함하지 않는다.)
                for word in line.strip('\r\n ').split(' '): # 공백으로 구분된 단어를 추출한다.
                    if word:
                        vocab[word] += 1 # 빈 문자열이 아니면 해당 단어를 vocab에 추가하고 빈도를 1 증가시킨다.
    return vocab # 업데이트된 vocab을 반환한다.


def update_pair_statistics(pair, changed, stats, indices): # 특정 바이트 쌍을 합치면 해당 쌍이 포함된 기존 단어들의 빈도 및 인덱스를 최소한으로 업데이트한다.
    # 기존에 등장한 쌍에 대한 빈도 및 인덱스를 효율적으로 조정하는 것이 목표다.
    # pari = 바이트 쌍
    """Minimally update the indices and frequency of symbol pairs

    if we merge a pair of symbols, only pairs that overlap with occurrences
    of this pair are affected, and need to be updated.
    """
    stats[pair] = 0
    indices[pair] = defaultdict(int)
    first, second = pair
    new_pair = first+second
    for j, word, old_word, freq in changed: # j는 변경된 단어의 인덱스다. word는 새로운 단어다. old_word는 기존 단어다. freq는 해당 단어의 빈도다.

        # find all instances of pair, and update frequency/indices around it
        # old_word에서 new_pair로 합쳐진 pair에 대해서 주어진 pair가 등장하는 모든 위치를 찾아서 해당 위치 주변의 빈도 및 인덱스를 업데이트한다.
        i = 0 # 현재 위치인 i를 0으로 초기화한다.
        while True: # 바이트 쌍이 등장하는 위치를 찾는다.
            # find first symbol
            try:
                i = old_word.index(first, i) # 기존 단어에서 first 심볼을 찾는다. i를 해당 위치로 업데이트한다.
            except ValueError:
                break
            # if first symbol is followed by second symbol, we've found an occurrence of pair (old_word[i:i+2])
            if i < len(old_word)-1 and old_word[i+1] == second: # first 심볼 뒤에 second 심볼이 나오면 쌍이 등장한 것으로 간주한다.
                # assuming a symbol sequence "A B C", if "B C" is merged, reduce the frequency of "A B"
                if i: # i가 0이 아니라면 쌍 앞에 다른 심볼이 있는 경우다. 이전 심볼 쌍의 빈도를 감소시키고 해당 인덱스를 업데이트 한다.
                    prev = old_word[i-1:i+1] # 현재 위치 i를 기준으로 바이트 쌍 앞에 있는 심볼을 포함한 새로운 쌍을 변수 prev에 할당한다.
                    stats[prev] -= freq # 이전에 찾은 심볼 쌍의 빈도를 감소시킨다. A B C에서 B C가 합쳐졌으면 A B의 빈도를 감소시킨다.
                    indices[prev][j] -= 1 # 해당 인덱스를 업데이트 한다.
                if i < len(old_word)-2: # 현재 위치가 마지막에서 두 번째인 경우, 뒤에 다른 심볼이 있는지 확인한다.
                    # assuming a symbol sequence "A B C B", if "B C" is merged, reduce the frequency of "C B".
                    # however, skip this if the sequence is A B C B C, because the frequency of "C B" will be reduced by the previous code block
                    if old_word[i+2] != first or i >= len(old_word)-3 or old_word[i+3] != second: # 이미 처리된 쌍이 뒤에 다시 오는 경우를 처리한다.
                        nex = old_word[i+1:i+3] # 이미 처리된 쌍이 뒤에 나오면 새로운 쌍을 변수 nex에 할당한다.
                        stats[nex] -= freq # 새로운 쌍의 빈도를 감소시킨다.
                        indices[nex][j] -= 1 # 해당 인덱스를 업데이트한다.
                i += 2
            else:
                i += 1

        i = 0
        while True: # 새로 합쳐진 바이트 쌍이 등장하는 부분에서 해당 위치 주변의 빈도 및 인덱스를 업데이트한다. 
            try:
                # find new pair
                i = word.index(new_pair, i) # 현재 위치 i부터 new_pair가 등장하는 위치를 찾는다.
            except ValueError: # 등장하지 않으면 에러를 반환하고 루프를 종료한다.
                break
            # assuming a symbol sequence "A BC D", if "B C" is merged, increase the frequency of "A BC"
            if i: # i가 0이 아니라면 new_pair 앞에 다른 심볼이 있는 경우다.
                prev = word[i-1:i+1] # 현재 위치 i를 기준으로 바이트 쌍 앞에 있는 심볼을 포함한 새로운 쌍을 변수 prev에 할당한다.
                stats[prev] += freq # new_pair의 빈도를 증가한다.
                indices[prev][j] += 1 # 해당 인덱스를 업데이트한다.
            # assuming a symbol sequence "A BC B", if "B C" is merged, increase the frequency of "BC B"
            # however, if the sequence is A BC BC, skip this step because the count of "BC BC" will be incremented by the previous code block
            if i < len(word)-1 and word[i+1] != new_pair: # 해당 위치가 단어의 마지막 바이트가 아니고, 다음 바이트가 new_pair와 다르면, new_pair 뒤에 다른 심볼이 있는 경우다.
                nex = word[i:i+2] # new_pair 뒤에 있는 다음 바이트 쌍을 변수 nex에 할당한다.
                stats[nex] += freq # 다음에 나오는 바이트 쌍의 빈도를 증가시킨다.
                indices[nex][j] += 1 # 해당 인덱스를 업데이트한다.
            i += 1 # 현재 위치를 한 칸 증가시킨다. (다음 검색을 위하여)


def get_pair_statistics(vocab): # 어휘에서 모든 바이트 쌍의 빈도를 계산하고 인덱스를 생성한다.
    """Count frequency of all symbol pairs, and create index"""

    # data structure of pair frequencies
    stats = defaultdict(int) # stats는 바이트 쌍의 빈도를 저장하는 딕셔너리다.
    # defaultdict(int)는 기본값(default)를 갖는 딕셔너리를 생성하는 것이다.
    # 여기서는 기본값을 정수 0으로 가지는 딕셔너리를 생성한다.

    #index from pairs to words
    indices = defaultdict(lambda: defaultdict(int)) # 바이트 쌍에 대한 인덱스를 저장하는 딕셔너리다.
    # 각 바이트 쌍이 어떤 단어에서 등장했는지 기록한다.
    # lambda:를 쓰면 간단한 함수를 생성할 수 있다. 여기서는 인자를 받지 않는 함수를 정의한다.

    for i, (word, freq) in enumerate(vocab): # enumerate(vocab)은 vocab에서 각 단어와 그 빈도를 순회한다.
        # i에는 인덱스, (word, freq)에는 단어와 빈도가 할당된다.
        prev_char = word[0] # 각 단어의 첫 번째 문자를 prev_char에 할당한다.
        for char in word[1:]: # 첫 문자 이후에 반복한다.
            stats[prev_char, char] += freq # 현재 문자인 char과 이전 문자인 prev_char로 이루어진 바이트 쌍의 빈도를 업데이트한다.
            indices[prev_char, char][i] += 1 # 현재의 바이트쌍이 어떤 단어에서 등장했는지 인덱스를 기록한다.
            prev_char = char

    return stats, indices

# 하나의 바이트 쌍에 대하여 아래의 코드를 수행한다.
def replace_pair(pair, vocab, indices): # 주어진 바이트 쌍을 새로운 심볼로 교체한다.
    """Replace all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'"""
    first, second = pair
    pair_str = ''.join(pair) # 바이트 쌍을 문자열로 결합해서 pair_str 변수에 저장한다.
    pair_str = pair_str.replace('\\','\\\\') # 백슬래시를 추가한다. AB-> A\B
    changes = []
    pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)') # 바이트 쌍을 대체할 때 사용할 패턴을 정의한다.
    # 바이트 쌍의 양쪽에 공백이 있을 때만 대체하도록 정의되어 있다.
    if sys.version_info < (3, 0): # 여기는 파이썬 버전 맞춰주는 파트다.
        iterator = indices[pair].iteritems()
    else:
        iterator = indices[pair].items()
    for j, freq in iterator: # 바이트 쌍이 등장한 단어의 인덱스와 빈도에 대하여 반복한다.
        if freq < 1: # 바이트 쌍의 빈도가 1 미만이면 반복문을 계속 진행한다.
            continue
        word, freq = vocab[j] # vocab에서 해당 인덱스에 대응하는 단어와 빈도를 가져온다.
        new_word = ' '.join(word) # 가져온 단어에 공백을 붙이고 문자열로 만든다.
        new_word = pattern.sub(pair_str, new_word) # 위에서 만든 패턴을 사용한다. 바이트 쌍을 만든 심볼로 바꿔준다.
        new_word = tuple(new_word.split(' ')) # 심볼로 바꾸었으니 공백을 기준으로 다시 분할하고 튜플로 바꾼다.

        vocab[j] = (new_word, freq) # 변경된 단어와 기존의 빈도를 사용해서 vocab을 업데이트한다.
        changes.append((j, new_word, word, freq)) # 여기도 정보를 맞게 업데이트한다.

    return changes

def prune_stats(stats, big_stats, threshold): # 빈도가 낮은 바이트 쌍을 제거하여 계산 효율성을 향상시킨다.
    # max() 함수의 효율성을 향상시키기 위하여 빈도가 낮은 바이트 쌍을 제거한다.
    # stats는 딕셔너리를 순회하여 각 바이트 쌍의 빈도를 확인한다.
    
    """Prune statistics dict for efficiency of max()

    The frequency of a symbol pair never increases, so pruning is generally safe
    (until we the most frequent pair is less frequent than a pair we previously pruned)
    big_stats keeps full statistics for when we need to access pruned items
    """

    for item,freq in list(stats.items()): 
        if freq < threshold: # 임계값인 threshold보다 작으면 바이트 쌍을 stats 딕셔너리에서 제거한다.
            # 빈도가 임계값보다 크거나 같으면 그대로 유지한다.
            del stats[item]
            if freq < 0: # 빈도인 freq가 음수면 big_stats 딕셔너리에 더해준다.
                # 전체 통계 정보를 유지하기 위해서 big_stats에 보존하는 것이다.
                big_stats[item] += freq
            else: # 양수면 그대로 big_stats에 할당한다.
                big_stats[item] = freq


def learn_bpe(infile_names, outfile_name, num_symbols, min_frequency=2, verbose=False, is_dict=False, total_symbols=False):
    # infile_names 파일에서 어휘를 수집하고, 주어진 바이트 쌍의 수에 도달할 때까지 가장 빈도가 높은 바이트 쌍을 학습하여 파일에 저장한다.
    
    """Learn num_symbols BPE operations from vocabulary, and write to outfile.
    """
    # 아래의 코드 3줄 파이썬 버전 호환을 위하여 작성된 코드다.
    sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer) # sys.stderr은 표준 오류 출력 스트림이다. 인코딩을 설정한다.
    sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer) # sys.stdout은 표준 출력 스트림이다. 인코딩을 설정한다.
    sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer) # sys.stdin은 표준 입력 스트림이다. 디코딩을 설정한다.

    #vocab = get_vocabulary(infile, is_dict)
    vocab = Counter() # Counter 객체를 이용해서 빈도수를 세기 위한 vocab을 초기화한다.
    for f in infile_names: # infile_names를 순회한다.
        sys.stderr.write(f'Collecting vocab from {f}\n') # 현재 처리 중인 파일의 이름을 출력한다. 과정을 추적하기 위한 메세지다.
        vocab = update_vocabulary(vocab, f, is_dict) # update_vocabulary를 이용해서 각 파일의 어휘를 업데이트한다.
        # update_vocabulary는 위에서 나왔던 것처럼 파일에서 텍스트를 읽어와서 어휘를 추출하고 빈도수를 계산해서 Counter 객체에 더한다.

    vocab = dict([(tuple(x[:-1])+(x[-1]+'</w>',) ,y) for (x,y) in vocab.items()]) # 각 단어를 BPE 알고리즘에 맞게 가공한다.
    # 마지막 문자 뒤에 </w>를 추가하고 단어를 튜플로 변환해서 딕셔너리 형태로 저장한다.
    # 이 과정을 거치면 BPE 알고리즘이 더 효과적으로 단어를 합병할 수 있게 된다.
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True) # 어휘를 빈도수에 따라서 내림차순으로 정렬한다.
    # BPE 알고리즘에서 높은 빈도수를 갖는 바이트 쌍을 먼저 학습하게 된다.

    stats, indices = get_pair_statistics(sorted_vocab) # sorted_vocab을 이용해서 바이트 쌍의 빈도와 인덱스를 계산한다.
    big_stats = copy.deepcopy(stats) # 초기의 바이트 쌍 빈도와 인덱스를 big_stats에 복사한다. (초기 자료 보존)

    if total_symbols: # total_symbols가 True면 단어 내부와 단어 마지막에 있는 고유한 문자의 수를 계산한다.
        # 그 후 BPE 알고리즘에서 합치는 작업 수를 감소시킨다.
        uniq_char_internal = set() # 단어 내부에 있는 고유한 문자를 저장하기 위한 집합을 초기화한다.
        uniq_char_final = set() # 단어 마지막에 있는 고유한 문자를 저장하기 위한 집합을 초기화한다.
        for word in vocab: # vocab에 있는 각 단어에 대해서 반복한다.
            for char in word[:-1]: # 각단어의 마지막 문자를 제외한 문자에 대해서 반복한다. 
                uniq_char_internal.add(char) # 각 문자를 uniq_char_internal 집합에 추가한다. 단어 내부에 있는 모든 고유한 문자가 추가된다.
            uniq_char_final.add(word[-1]) # 단어의 마지막 문자를 uniq_char_final 집합에 추가한다.
        sys.stderr.write('Number of word-internal characters: {0}\n'.format(len(uniq_char_internal))) # 단어 내부에 있는 고유한 문자의 수를 출력한다.
        sys.stderr.write('Number of word-final characters: {0}\n'.format(len(uniq_char_final))) # 단어 마지막에 있는 고유한 문자의 수를 출력한다.
        sys.stderr.write('Reducing number of merge operations by {0}\n'.format(len(uniq_char_internal) + len(uniq_char_final)))
        # BPE 알고리즘에서 합치는 작업 수를 감소시키기 위하여 단어 내부와 마지막에 있는 모든 고유한 문자의 수를 더한 값을 출력한다.
        num_symbols -= len(uniq_char_internal) + len(uniq_char_final) # BPE 알고리즘에서 합치는 작업의 수를 감소시킨다.
        # 이전에 계산된 합을 뺀다. (합쳐진만큼 빼서 수를 줄인다.)


    sys.stderr.write(f'Write vocab file to {outfile_name}') # 오류 출력 스트림에 vocab file을 쓸 것이라는 것을 메세지로 출력하게 한다.
    with codecs.open(outfile_name, 'w', encoding='utf-8') as outfile: # UTF-8 인코딩으로 어휘 파일을 쓰기 위해서 codecs.oen()을 이용하여 파일을 연다.
        # version 0.2 changes the handling of the end-of-word token ('</w>');
        # version numbering allows bckward compatibility

        outfile.write('#version: 0.2\n') # vocab 파일의 첫 줄에 버전 정보를 쓰기 위하여 outfile.write()를 사용한다.
        # 버전이 바뀔 때 호환성을 유지하기 위해서 하는 과정이다.
        # threshold is inspired by Zipfian assumption, but should only affect speed
        threshold = max(stats.values()) / 10 # 임계값인 threshold를 설정한다. Zipf의 법칙을 참고해서 만들어진 값이다.
        # stats.values()는 바이트 쌍의 빈도를 나타내는 딕셔너리에서 빈도값들을 추출한 리스트다.
        # 최대값을 찾아서 10으로 나눈 값을 thresold로 설정한다.
        # thresold를 설정해서 BPE 학습의 속도를 조절할 수 있다.
        # Zipf의 법칙은 어떤 자연어에서 상위 빈도의 단어가 전체 단어 빈도의 상당 부분을 차지한다는 법칙이다.
        
        # 가장 빈도가 높은 바이트 쌍을 학습하는 과정이다.
        for i in range(num_symbols): # 위에서 구한 num_symbols만큼 반복한다.
            if stats: # stats 딕셔너리에 바이트 쌍의 정보가 있는지 확인한다.
                most_frequent = max(stats, key=lambda x: (stats[x], x)) # 현재 정보에서가장 빈도가 높은 바이트 쌍을 찾는다.
                # stats[x],x를 기준으로 최댓값을 찾는다. stats[x]는 빈도수고, x는 바이트 쌍 자체를 의미한다.

            # we probably missed the best pair because of pruning; go back to full statistics
            if not stats or (i and stats[most_frequent] < threshold):
                # stats가 비어있거나(학습 초기다.) 이전에 찾은 가장 빈도가 높은 바이트 쌍의 빈도가 현재의 threshold보다 낮은 경우에 실행한다.
                # 이전에 제거한 바이트 쌍이 있다고 가정하고 전체 통계 정보로 돌아가서 다시 계산한다.
                prune_stats(stats, big_stats, threshold) # prune_stats 함수를 호출해서 빈도가 낮은 바이트 쌍을 제거한다.
                stats = copy.deepcopy(big_stats) # big_stats를 복사해서 stats에 할당한다.
                most_frequent = max(stats, key=lambda x: (stats[x], x)) # 새로 업데이트된 stats에서 가장 빈도가 높은 바이트를 찾는다.
                # threshold is inspired by Zipfian assumption, but should only affect speed
                threshold = stats[most_frequent] * i/(i+10000.0) # 새로운 임계값을 설정한다. 현재까지의 학습 횟수인 i에 따라서 조정된다.
                prune_stats(stats, big_stats, threshold) # 새로 계산된 임계값을 기반으로 다시 빈도가 낮은 바이트 쌍을 제거한다.

            # BPE 알고리즘에서 학습된 가장 빈도가 높은 바이트 쌍을 이용해서 합병 작업을 수행하고, 통계를 업데이트하는 부분이다.
            if stats[most_frequent] < min_frequency: # 학습된 가장 빈도가 높은 바이트 쌍의 빈도가 사용자가 설정한 최소 빈도인 min_frequency보다 작은 경우에 실행된다.
                sys.stderr.write(f'no pair has frequency >= {min_frequency}. Stopping\n') # 이 경우에 빈도가 충분히 높은 바이트 쌍이 없다고 판단되어 학습을 중단하고 프로그램이 종료된다.
                # 오류 출력 메세지를 출력한다.
                break

            if verbose: # verbose 모드가 활성화된 경우에 실행한다.
                # verbos 모드는 각 합병 작업의 진행 상황을 자세히 출력하도록 하는 옵션이다.
                sys.stderr.write('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format( 
                    i, most_frequent[0], most_frequent[1], stats[most_frequent])) # 현재까지의 합병 작업 횟수와 해당 작업에서 합병되는 두 바이트 쌍 빈도를 출력한다.
                # 학습 진행 상황을 확인하는 것에 사용된다.
            outfile.write('{0} {1}\n'.format(*most_frequent)) # 학습된 바이트 쌍을 출력 파일에 기록한다.
            changes = replace_pair(most_frequent, sorted_vocab, indices) # replace_pair 함수를 호출해서 합병된 바이트 쌍으로 대체된 단어들의 변화를 불러온다.
            update_pair_statistics(most_frequent, changes, stats, indices) # update_pair_statistics 함수를 호출해서 합병 작업에 따른 통계 정보를 업데이트한다.
            stats[most_frequent] = 0 # 합병된 바이트 쌍의 빈도를 0으로 설정해서 해당 쌍을 다시 선택하지 않도록 한다.
            if not i % 100: # 100번째 합병 작업마다 prune_stats 함수를 호출해서 빈도가 낮은 바이트 쌍을 정리해서 메모리를 관리한다.
                prune_stats(stats, big_stats, threshold)

