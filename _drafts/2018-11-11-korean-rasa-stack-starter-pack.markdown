---
layout: post
title:  "RASA Tutorial - RASA 설치 및 Starter-pack으로 간단 chatbot 만들기"
date:   2017-11-29 00:00:00
author: y-rok
categories: rasa
tags: chatbot rasa
---


## Intro 

이 번 Post에서는 인사를 나누고 간단한 농담을 할 수 있는 `한국어 Chatbot`을 만들어 보고자 한다. Rasa에서 제공하는 [Rasa Stack Starter pack](https://github.com/RasaHQ/starter-pack-rasa-stack)와 대부분 유사하지만 `영어`가 아닌 `한국어`를 이해하기 위해 다음 사항을 수정 하였다.

- 영어 학습데이터를 한국어로 수정
- Mecab을 활용하여 형태소 분석 단위로 Tokenization
- tensorflow_embedding을 활용하여 NLU 모듈 학습

수정한 소스 코드는 [Korean RASA Stack starter-pack](https://github.com/y-rok/Korean_starter-pack-rasa-stack.git)에서 확인할 수 있다. 

Post는 다음 순서로 설명할 것이다.

- [Korean RASA Stack starter-pack 다운로드 및 설치](#Korean-RASA-Stack-starter-pack-다운로드-및-설치)
- [RASA NLU 모델 학습](#RASA-NLU-모델-학습)
- [RASA Core 모델 학습](#RASA-Core-모델-학습)
- [RASA Chatbot 서버 실행 및 간단한 대화 테스트](#RASA-Chatbot-서버-실행-및-간단한-대화-테스트)

## Korean RASA Stack starter-pack 다운로드 및 설치

`git`을 활용하여 [RASA Stack starter-pack](https://github.com/y-rok/Korean_starter-pack-rasa-stack.git)를 다운로드 한다.

```bash
❯❯❯ git clone https://github.com/y-rok/Korean_starter-pack-rasa-stack.git
```

이 후 프로젝트에 필요한 pacakge(rasa_nlu, rasa_core,konlpy)를 설치 하기 위해 프로젝트 폴더에서 다음 명령어를 수행한다.  

```bash
❯❯❯ pip install -r requirements.txt
```

## RASA NLU 모델 학습

RASA NLU 모델 학습을 위해서는 다음 2가지 데이터가 필요하다.

###  학습을 위한 Training Data(각 Intent의 Training Examples) 정의

[./data/nlu_data.md](https://github.com/RasaHQ/starter-pack-rasa-stack/blob/master/data/nlu_data.md)

  - 6개의 Intents (goodbye, greet, thanks, affirm, name, joke)
  - 1개의 Entity (name)
  - [Training Data Format 설명 참고](https://rasa.com/docs/nlu/dataformat/)

### 학습에 사용되는 Component, 언어 정의

[./nlu_config.yml](https://github.com/RasaHQ/starter-pack-rasa-stack/blob/master/nlu_config.yml)


[spacy_sklearn](https://rasa.com/docs/nlu/choosing_pipeline/#spacy-sklearn)은 `한국어`를 지원하지 않으므로 [tensorflow_embedding](https://rasa.com/docs/nlu/choosing_pipeline/#tensorflow-embedding)에서 toeknization 를 Custom Component인 [KoreanTokenizer](https://github.com/y-rok/Korean_starter-pack-rasa-stack/blob/master/component/korean_tokenizer.py)로 수정하여 사용한다. 

```yml
language: "kr"
pipeline:
- name: "component.KoreanTokenizer"
- name: "ner_crf"
- name: "ner_synonyms"
- name: "intent_featurizer_count_vectors"
- name: "intent_classifier_tensorflow_embedding"
```

#### KoreanTokenizer를 따로 정의해서 사용하는 이유는?

기존 `tokenizer_whitespace`는 `공백`을 기준으로 `tokenization`을 수행한다. `영어`에서는 이가 잘 동작하지만 `한국어`에서는 다음과 같은 경우 문제가 생길 수 있다.

- ≈
`KoreanTokenizer`는 Mecab을 활용하여 형태소 분석 단위로 Tokenization을 수행한다. 
```python
import re

from konlpy.tag import Mecab

from rasa_nlu.components import Component
from rasa_nlu.tokenizers import Tokenizer, Token

class KoreanTokenizer(Tokenizer, Component):

    name = "tokenizer_whitespace"
    provides = ["tokens"]

    def __init__(self, component_config=None):
        self.mecab=Mecab()
        super(KoreanTokenizer, self).__init__(component_config)


    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text):
        # type: (Text) -> List[Token]

        # there is space or end of string after punctuation
        # because we do not want to replace 10.000 with 10 000
        words = re.sub(r'[.,!?]+(\s|$)', ' ', text)
        words=self.mecab.morphs(words)

        running_offset = 0
        tokens = []
        for word in words:
            word_offset = text.index(word, running_offset)
            word_len = len(word)
            running_offset = word_offset + word_len
            tokens.append(Token(word, word_offset))
        return tokens
```

![](/assets/img/korean-rasa-stack-starter-pack/nlu_config.png)

  - Starter-pack에서는 spacy_sklearn Component 및 영어 사용 
  - [Component Configuration 설명 참고](https://rasa.com/docs/nlu/components/)

NLU 모델의 학습을 위해 다음 명령어를 수행한다.

```bash
❯❯❯ make train-nlu
```

Makefile을 보면 이 명령어는 다음과 동일하다.

```bash
python -m rasa_nlu.train -c nlu_config.yml --data data/nlu_data.md -o models --fixed_model_name nlu --project current --verbose
```

여기서 `-o models --fixed_model_name nlu --project current`은 `/models/current/nlu`에 학습결과(모델,학습 데이터 등)를 저장하도록 한다.

학습된 NLU 모델을 테스트 하기 위해 NLU Server를 실행시키고 이에 Message를 보내보도록 하자.
다음과 같이 models 폴더를 기반으로 NLU Server를 실행하자.

```bash
 python -m rasa_nlu.server --path models
```

 이 후, curl 명령어를 이용하여 "bye"에 대한 요청을 Server에 보내면 다음과 같이 "goodBye" Intent로 분류되는 것을 확인할 수 있다.

```bash
 ❯❯❯ curl 'localhost:5000/parse?q=bye&project=current&model=nlu'
 {
  "intent": {
    "name": "goodbye",
    "confidence": 0.660794739911686
  },
  "entities": [],
  "intent_ranking": [
    {
      "name": "goodbye",
      "confidence": 0.660794739911686
    },
    {
      "name": "greet",
      "confidence": 0.23335467238107127
    },
    {
      "name": "affirm",
      "confidence": 0.06674510183302036
    },
    {
      "name": "thanks",
      "confidence": 0.017740955672121344
    },
    {
      "name": "joke",
      "confidence": 0.01235896254029361
    },
    {
      "name": "name",
      "confidence": 0.00900556766180776
    }
  ],
  "text": "bye",
  "project": "current",
  "model": "nlu"
}

```

## RASA Core 모델 학습

## RASA Chatbot 서버 실행 및 간단한 대화 테스트

## TroubleShooting

#### tensorflow package 설치가 안될 떄

python version이 [tensorflow 지원 버전](https://www.tensorflow.org/install/pip?hl=ko)인지 확인 하자. 

#### sklearn_crfuite package 설치가 안될 때

Error log 예시 

```
Could not find a version that satisfies the requirement sklearn_crfuite==0.3.6 (from -r requirements.txt (line 36)) (from versions: )
No matching distribution found for sklearn_crfuite==0.3.6 (from -r requirements.txt (line 36))
``` 

(2018.12.02 기준) requirements.txt 파일에 오타로 인해 발생하는 문제....

requirements.txt의 마지막 line의 sklearn_crfuite을 sklearn_crfsuite로 수정한다. 


#### spacy.load('en') 도중 Error 발생

Error log 예시

```
rasa_nlu.utils.spacy_utils  - Trying to load spacy model with name 'en'
Traceback (most recent call last):

...

  File "pipeline.pyx", line 627, in spacy.pipeline.Tagger.from_disk.load_model
  File "/Users/y-rok/.pyenv/versions/rasa/lib/python3.6/site-packages/thinc/neural/_classes/model.py", line 335, in from_bytes
    data = msgpack.loads(bytes_data, encoding='utf8')
  File "/Users/y-rok/.pyenv/versions/rasa/lib/python3.6/site-packages/msgpack_numpy.py", line 214, in unpackb
    return _unpackb(packed, **kwargs)
  File "msgpack/_unpacker.pyx", line 187, in msgpack._cmsgpack.unpackb
ValueError: 1792000 exceeds max_bin_len(1048576)
make: *** [train-nlu] Error 1
```

정확한 이유는 파악하지 못했지만... spacy의 Version을 Upgrade하니 잘 동작한다. 이를 위해 다음 명령어를 수행하자.

```bash
 ❯❯❯ pip install -U spacy
```



