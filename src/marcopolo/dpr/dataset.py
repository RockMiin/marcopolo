import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Preprocess:
    def __init__(self, train, valid, tokenizer, args):
        self.args = args
        self.tokenizer = tokenizer
        self.train = train
        self.valid = valid

    def make_dataset(self):
        if self.args.loss_option == "con":
            trainset = self.make_contrastive_dataset(
                self.train, self.args.samples, self.args.samples_batch, self.args.max_length
            )
            validset = self.make_contrastive_dataset(
                self.valid, int(0.2 * self.args.samples), self.args.samples_batch, self.args.max_length
            )

        elif self.args.loss_option == "bce":
            trainset = self.make_bce_dataset(
                self.train, self.args.samples, self.args.samples_batch, self.args.max_length
            )
            validset = self.make_bce_dataset(
                self.valid, int(0.2 * self.args.samples), self.args.samples_batch, self.args.max_length
            )

        return trainset, validset

    """
    batch sample : [1, 0, 0, 0, 0] (1 pos + TOTAL_SAMPLES neg)
    """

    def make_contrastive_dataset(self, data, SAMPLES_CNT, TOTAL_SAMPLES, MAX_LENGTH) -> TensorDataset:
        query_list = []
        passage_list = []
        target_list = []

        # passage가 10개가 아닌 경우도 있음 -> padding
        for idx in tqdm(range(SAMPLES_CNT)):
            if sum(data[idx]["passages"]["is_selected"]) == 0:
                continue # No answer 사용 X

            else:
                if sum(data[idx]["passages"]["is_selected"]) == 1:  # positive sample이 한 개일 경우
                    pos_idx = []

                    for i in range(len(list(data[idx]["passages"]["passage_text"]))):
                        if data[idx]["passages"]["is_selected"][i] == 1:
                            pos_idx.append(i)  # positive index

                    loop_cnt = 0
                    while True:
                        neg_idxs = np.random.randint(
                            len(list(data[idx]["passages"]["passage_text"])), size=TOTAL_SAMPLES - 1
                        )
                        if loop_cnt >= 10:
                            break

                        flag = True
                        for n in neg_idxs:
                            if n in pos_idx:
                                flag = False
                                break

                        if flag:
                            break
                        loop_cnt += 1

                    if loop_cnt >= 10:
                        continue

                    add_passages = [data[idx]["passages"]["passage_text"][pos_idx[0]]]
                    add_targets = [data[idx]["passages"]["is_selected"][pos_idx[0]]]

                    for n in neg_idxs:
                        add_passages.append(data[idx]["passages"]["passage_text"][n])
                        add_targets.append(data[idx]["passages"]["is_selected"][n])

                    query_list.append(data[idx]["query"])
                    passage_list.extend(add_passages)
                    target_list.extend(add_targets)

                elif sum(data[idx]["passages"]["is_selected"]) > 1:  # positive sample이 여러개 일 경우
                    pos_idx = []

                    for i in range(len(list(data[idx]["passages"]["passage_text"]))):
                        if data[idx]["passages"]["is_selected"][i] == 1:
                            pos_idx.append(i)  # positive index

                    for p in pos_idx:
                        loop_cnt = 0

                        while True:
                            neg_idxs = np.random.randint(
                                len(list(data[idx]["passages"]["passage_text"])), size=TOTAL_SAMPLES - 1
                            )
                            if loop_cnt >= 10:
                                break
                                # loop 횟수가 너무 많아지면 skip

                            flag = True
                            for n in neg_idxs:
                                if n in pos_idx:
                                    flag = False
                                    break

                            if flag:
                                break
                            loop_cnt += 1

                        if loop_cnt >= 10:
                            continue

                        add_passages = [data[idx]["passages"]["passage_text"][p]]
                        add_targets = [data[idx]["passages"]["is_selected"][p]]

                        for n in neg_idxs:
                            add_passages.append(data[idx]["passages"]["passage_text"][n])
                            add_targets.append(data[idx]["passages"]["is_selected"][n])

                        query_list.append(data[idx]["query"])
                        passage_list.extend(add_passages)
                        target_list.extend(add_targets)

                else:  # pos idx가 없는 경우 (no answer인 경우) -> 이 때 BATCH COUNT보다 긴 경우가 있음
                    # print('Positive idx is None & Check passage length (e.g. passage length > BATCH_SIZE)')
                    continue

        tokenized_query = self.tokenizer(
            query_list, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )

        tokenized_passage = self.tokenizer(
            passage_list, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )

        targets = torch.tensor(target_list)

        print(f'tokenized query length: {len(tokenized_query["input_ids"])}')
        print(f"tokenized passage length: {len(tokenized_passage['input_ids'])}")
        print(f"target length: {len(targets)}")
        print(tokenized_passage["input_ids"].shape)

        # tokenize할 때에는 3차원으로 들어가면 안되서 바꿨다가 다시 shape을 맞춰주는 느낌
        tokenized_passage["input_ids"] = tokenized_passage["input_ids"].view(
            -1, TOTAL_SAMPLES, tokenized_passage["input_ids"].shape[1]
        )
        tokenized_passage["attention_mask"] = tokenized_passage["attention_mask"].view(
            -1, TOTAL_SAMPLES, tokenized_passage["attention_mask"].shape[1]
        )
        tokenized_passage["token_type_ids"] = tokenized_passage["token_type_ids"].view(
            -1, TOTAL_SAMPLES, tokenized_passage["token_type_ids"].shape[1]
        )

        targets = targets.view(-1, TOTAL_SAMPLES)

        train_dataset = TensorDataset(
            tokenized_query["input_ids"],
            tokenized_query["attention_mask"],
            tokenized_query["token_type_ids"],
            tokenized_passage["input_ids"],
            tokenized_passage["attention_mask"],
            tokenized_passage["token_type_ids"],
            targets,
        )

        return train_dataset

    def make_bce_dataset(self, data, SAMPLES_CNT, TOTAL_SAMPLES, MAX_LENGTH) -> TensorDataset:
        query_list = []
        passage_list = []
        target_list = []

        # passage가 10개가 아닌 경우도 있음 -> padding해줌
        for idx in tqdm(range(SAMPLES_CNT)):
            if sum(data[idx]["passages"]["is_selected"]) == 0:
                continue
                # No answer 제거

            if len(list(data[idx]["passages"]["passage_text"])) != TOTAL_SAMPLES:
                pos_idx = []
                need_cnt = TOTAL_SAMPLES - len(list(data[idx]["passages"]["passage_text"]))

                for i in range(len(list(data[idx]["passages"]["passage_text"]))):
                    if data[idx]["passages"]["is_selected"][i] == 1:
                        pos_idx.append(i)

                if pos_idx != [] and need_cnt > 0:
                    loop_cnt = 0
                    while True:
                        neg_idxs = np.random.randint(len(list(data[idx]["passages"]["passage_text"])), size=need_cnt)
                        if loop_cnt >= 10:
                            break

                        flag = True
                        for n in neg_idxs:
                            if n in pos_idx:
                                flag = False
                                break

                        if flag:
                            break
                        loop_cnt += 1

                    if loop_cnt >= 10:
                        continue

                    add_passages = []
                    add_targets = []
                    for n in neg_idxs:
                        add_passages.append(data[idx]["passages"]["passage_text"][n])
                        add_targets.append(data[idx]["passages"]["is_selected"][n])

                    query_list.append(data[idx]["query"])
                    passage_list.extend((list(data[idx]["passages"]["passage_text"]) + add_passages))
                    target_list.extend((list(data[idx]["passages"]["is_selected"]) + add_targets))

                    # print(len(list(train[idx]['passages']['passage_text'])))

                else:  # pos idx가 없는 경우 (no answer인 경우) -> 이 때 BATCH COUNT보다 긴 경우가 있음
                    # print('Positive idx is None & Check passage length (e.g. passage length > BATCH_SIZE)')
                    continue

            else:
                query_list.append(data[idx]["query"])
                passage_list.extend(list(data[idx]["passages"]["passage_text"]))
                target_list.extend(list(data[idx]["passages"]["is_selected"]))

        tokenized_query = self.tokenizer(
            query_list, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )

        tokenized_passage = self.tokenizer(
            passage_list, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )

        targets = torch.tensor(target_list)

        print(f'tokenized query length: {len(tokenized_query["input_ids"])}')
        print(f"tokenized passage length: {len(tokenized_passage['input_ids'])}")
        print(f"target length: {len(targets)}")
        print(tokenized_passage["input_ids"].shape)

        # tokenize할 때에는 3차원으로 들어가면 안되서 바꿨다가 다시 shape을 맞춰주는 느낌
        tokenized_passage["input_ids"] = tokenized_passage["input_ids"].view(-1, TOTAL_SAMPLES, MAX_LENGTH)
        tokenized_passage["attention_mask"] = tokenized_passage["attention_mask"].view(-1, TOTAL_SAMPLES, MAX_LENGTH)
        tokenized_passage["token_type_ids"] = tokenized_passage["token_type_ids"].view(-1, TOTAL_SAMPLES, MAX_LENGTH)

        targets = targets.view(-1, TOTAL_SAMPLES)

        train_dataset = TensorDataset(
            tokenized_query["input_ids"],
            tokenized_query["attention_mask"],
            tokenized_query["token_type_ids"],
            tokenized_passage["input_ids"],
            tokenized_passage["attention_mask"],
            tokenized_passage["token_type_ids"],
            targets,
        )

        return train_dataset
