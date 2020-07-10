---
layout: post
title:  "Pytorch에서 대용량 데이터 활용 딥러닝 모델 학습시 전처리 방법 (AI Hub News Q&A 데이터셋으로 BERT 학습하기)"
date:   2020-07-10 00:00:00
author: y-rok
categories: 
- pytorch
tags: 
- dataset
- dataloader
- training
- pytorch
- bert
---



[인라이플에서 pytorch로 작성한 BERT 코드](https://github.com/enlipleai/kor_pretrain_LM)와 BERT Large를 공개하였습니다.이를 활용하여  [KorQuAD 데이터셋](https://korquad.github.io/)으로 Fine-Tuning 시 문제가 없었지만 [AI Hub의 News Q&A 데이터셋](http://www.aihub.or.kr/aidata/86)을 학습 시에는 문제가 발생했습니다. KorQuAD 데이터셋은 약 6만개(v1.0 기준)의 학습 데이터지만 News Q&A는 약 45만개로 학습 데이터가 너무 큰 경우 발생하는 문제였습니다.

이번 포스트에서는 **대용량 데이터를 활용하여 AI 모델을 학습하는 경우 데이터를 불러오고 전처리 하는 과정에서 발생 가능한 문제점과 해결책**을 적고자 합니다. 예시로 인라이플에서 공개한 BERT 코드를 수정하여 대용량 데이터 News Q&A를 학습할 수 있도록 합니다.

## 대용량 데이터를 활용하여 AI 모델 학습시 문제점

- 대용량 학습 데이터를 Input Tensor로 변환하여  RAM 혹은 GPU Memory에 모두 올릴 경우 Out-of-Memory가 발생하거나 매우 느려짐
- 학습을 위한 Input Tensor를 미리 Memory에 올리지 않고 모델 학습 시 Iteration 마다 데이터를  전처리할 경우 학습 시간이 오래 걸림 ([데이터 전처리에 오랜 시간이 소요될 경우 GPU가 Iteration 마다 이를 기다리게 됨](https://jybaek.tistory.com/799))

### 인라이플 공개 BERT Code의 문제점

Q&A를 학습하는 [run_qa.py](train_dataset = load_and_cache_examples(args, tokenizer)) 코드를 보면 모델 학습 전  load_and_cache_examples 함수에서 **학습 데이터를 모두 불러와 Input Tensor로 변환하는데 이 과정에서 RAM(32 gb 사용) 용량이 부족**하여 매우 느려지는 문제가 발생하였습니다.

 

```python
def load_and_cache_examples(args, tokenizer):
   
	.....
	
	# 학습 데이터를 읽음
  examples = read_squad_examples(input_file=args.train_file, is_training=True, version_2_with_negative=False)
  
	# 읽은 학습 데이터를 활용하여 Input Tensor 생성
  # Input Tensor의 크기가 너무 크기 떄문에 문제 발생!! 
  features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=True)
```

## 대용량 데이터 활용 시 데이터 전처리 방법

따라서, 대용량 데이터를 활용하여 학습 시 Input Tensor를 모두 RAM에 올리지 못하니 AI 모델 학습 시 Iteration 마다 필요한 학습 데이터를 Input Tensor로 전처리 하도록 수정합니다. 즉, 다음의 방식으로 BERT 코드를 수정해야합니다.

- (RAM에 올리기 너무 크지 않으므로) 학습 데이터를 미리 RAM에 읽어옴
- BERT 학습 시 DataLoader를 활용하여 Iteration 마다 Input Tensor를 생성. 이 때, Multi-processing을 활용하여 전처리에 의한 학습 속도 저하 문제를 해결.

### 인라이플 공개 BERT Code 수정

코드를 수정하기 위해서는 먼저 Dataset과 DataLoader에 대해 알아야합니다. 

Dataset은 getitem 함수를 통해 특정 index의 학습 데이터를 Input Tensor형태로 전달해주는 역할을 하고 **DataLoader는 학습시 Iteration 마다 Dataset으로 부터 Input Tensor를 불러오는 역할**을 합니다. 

(Dataset, DataLoader에 대한 자세한 설명은 [이 곳](https://pytorch.org/docs/stable/data.html) 참고)

기존 코드의 동작 방식은 다음과 같습니다.

- 학습 데이터를 읽어 Input Tensor를 생성한 후 이를 pickle 파일로 생성
- 학습 시 pickle file을 읽어 메모리에 들고 있는 상태에서 Dataset의 getitem 함수는 해당 index의 Input Tensor를 Return

이를 다음과 같이 수정합니다.

- 학습 데이터를 읽음 (미리 Input Tensor를 생성하지 않음)
- Dataset의 getitem 함수 호출 시 해당 index의 Input Tensor를 읽은 학습 데이터로 부터 동적으로 생성하여 Return

이를 위해 Custom Dataset을 생성하였고 __init__, __getitem__, __len__ 함수를 구현하였습니다. (자세한 코드 설명은 생략합니다.)

```python
class KorQuADDataset(Dataset) :
    def __init__(self,examples,data_index_dict,tokenizer,args):

				... (중략) ...

    def __getitem__(self, index):

        ... (중략) ...

        # Training Data로 부터 해당 index의 Input Tensor 생성
        input_feature = convert_examples_to_features(examples=self.examples,
                                     tokenizer=self.tokenizer,
                                     max_seq_length=self.max_seq_length,
                                     doc_stride=self.doc_stride,
                                     max_query_length=self.max_query_length,
                                     is_training=True,ex_doc_index=ex_doc_index)

        all_input_ids = torch.tensor(input_feature.input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(input_feature.input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(input_feature.segment_ids, dtype=torch.long)
        all_start_positions = torch.tensor(input_feature.start_position, dtype=torch.long)
        all_end_positions = torch.tensor(input_feature.end_position, dtype=torch.long)

        return tuple([all_input_ids,all_input_mask,all_segment_ids,all_start_positions,all_end_positions])

    def __len__(self):
        ... (중략) ...
```

구현한 소스는 아래에 공개 했으니 참고 바랍니다.

[y-rok/BERT-KorQuAD-dynamic-training](https://github.com/y-rok/BERT-KorQuAD-dynamic-training)

 

다만, 이렇게 구현을 수정하였을 때 Memory 부족 문제는 해결되었지만 다소 학습 속도가 느려지는 문제가 발생하였습니다. 이는 iteration 마다 학습 데이터를 Input Tensor로 바꾸는 CPU Job이 생겼기 때문입니다. ([이 곳](https://jybaek.tistory.com/799) 참고)

이는 dataloader에 num_workers argument를 통해 Input Tensor를 생성하는 작업의 Worker 수를 설정하여 속도를 개선할 수 있습니다.  다음은 worker 수에 따른 학습 속도 비교입니다. 

- KorQuAD Dataset 일부에 대해 188 iteration 3 epoch에 대해 평가
- CPU - i7-9700k 3.6ghz / GPU - 2080ti 사용
- worker 수가 3개 이상일 때는 2개일 때와 속도 유사


|  모델     | 초당 iteration |  소요시간   |
| ------   | ------  |------|
| pkl 사용  |   6.36it | 29.58 |
| dynamic (num workers = 1) | 5.29 | 35.66 | 
| dynamic (num workers = 2) | 6.26 | 30.59 |

## References

[DataLoader num_workers에 대한 고찰](https://jybaek.tistory.com/799)

- Pytorch 공식 Document에서 Dataset, DataLoader에 대한 설명

[torch.utils.data - PyTorch master documentation](https://pytorch.org/docs/stable/data.html#map-style-datasets)