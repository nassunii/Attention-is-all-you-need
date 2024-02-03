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
    stats = defaultdict(int)

    #index from pairs to words
    indices = defaultdict(lambda: defaultdict(int))

    for i, (word, freq) in enumerate(vocab):
        prev_char = word[0]
        for char in word[1:]:
            stats[prev_char, char] += freq
            indices[prev_char, char][i] += 1
            prev_char = char

    return stats, indices


def replace_pair(pair, vocab, indices): # 주어진 바이트 쌍을 새로운 심볼로 교체한다.
    """Replace all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'"""
    first, second = pair
    pair_str = ''.join(pair)
    pair_str = pair_str.replace('\\','\\\\')
    changes = []
    pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')
    if sys.version_info < (3, 0):
        iterator = indices[pair].iteritems()
    else:
        iterator = indices[pair].items()
    for j, freq in iterator:
        if freq < 1:
            continue
        word, freq = vocab[j]
        new_word = ' '.join(word)
        new_word = pattern.sub(pair_str, new_word)
        new_word = tuple(new_word.split(' '))

        vocab[j] = (new_word, freq)
        changes.append((j, new_word, word, freq))

    return changes

def prune_stats(stats, big_stats, threshold): # 빈도가 낮은 바이트 쌍을 제거하여 계산 효율성을 향상시킨다.
    """Prune statistics dict for efficiency of max()

    The frequency of a symbol pair never increases, so pruning is generally safe
    (until we the most frequent pair is less frequent than a pair we previously pruned)
    big_stats keeps full statistics for when we need to access pruned items
    """
    for item,freq in list(stats.items()):
        if freq < threshold:
            del stats[item]
            if freq < 0:
                big_stats[item] += freq
            else:
                big_stats[item] = freq


def learn_bpe(infile_names, outfile_name, num_symbols, min_frequency=2, verbose=False, is_dict=False, total_symbols=False):
    # 여러 파일에서 어휘를 수집하고, 주어진 바이트 쌍의 수에 도달할 때까지 가장 빈도가 높은 바이트 쌍을 학습하여 파일에 저장한다.
    """Learn num_symbols BPE operations from vocabulary, and write to outfile.
    """
    sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
    sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
    sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)

    #vocab = get_vocabulary(infile, is_dict)
    vocab = Counter()
    for f in infile_names:
        sys.stderr.write(f'Collecting vocab from {f}\n')
        vocab = update_vocabulary(vocab, f, is_dict)

    vocab = dict([(tuple(x[:-1])+(x[-1]+'</w>',) ,y) for (x,y) in vocab.items()])
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

    stats, indices = get_pair_statistics(sorted_vocab)
    big_stats = copy.deepcopy(stats)

    if total_symbols:
        uniq_char_internal = set()
        uniq_char_final = set()
        for word in vocab:
            for char in word[:-1]:
                uniq_char_internal.add(char)
            uniq_char_final.add(word[-1])
        sys.stderr.write('Number of word-internal characters: {0}\n'.format(len(uniq_char_internal)))
        sys.stderr.write('Number of word-final characters: {0}\n'.format(len(uniq_char_final)))
        sys.stderr.write('Reducing number of merge operations by {0}\n'.format(len(uniq_char_internal) + len(uniq_char_final)))
        num_symbols -= len(uniq_char_internal) + len(uniq_char_final)


    sys.stderr.write(f'Write vocab file to {outfile_name}')
    with codecs.open(outfile_name, 'w', encoding='utf-8') as outfile:
        # version 0.2 changes the handling of the end-of-word token ('</w>');
        # version numbering allows bckward compatibility

        outfile.write('#version: 0.2\n')
        # threshold is inspired by Zipfian assumption, but should only affect speed
        threshold = max(stats.values()) / 10
        for i in range(num_symbols):
            if stats:
                most_frequent = max(stats, key=lambda x: (stats[x], x))

            # we probably missed the best pair because of pruning; go back to full statistics
            if not stats or (i and stats[most_frequent] < threshold):
                prune_stats(stats, big_stats, threshold)
                stats = copy.deepcopy(big_stats)
                most_frequent = max(stats, key=lambda x: (stats[x], x))
                # threshold is inspired by Zipfian assumption, but should only affect speed
                threshold = stats[most_frequent] * i/(i+10000.0)
                prune_stats(stats, big_stats, threshold)

            if stats[most_frequent] < min_frequency:
                sys.stderr.write(f'no pair has frequency >= {min_frequency}. Stopping\n')
                break

            if verbose:
                sys.stderr.write('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(
                    i, most_frequent[0], most_frequent[1], stats[most_frequent]))
            outfile.write('{0} {1}\n'.format(*most_frequent))
            changes = replace_pair(most_frequent, sorted_vocab, indices)
            update_pair_statistics(most_frequent, changes, stats, indices)
            stats[most_frequent] = 0
            if not i % 100:
                prune_stats(stats, big_stats, threshold)

