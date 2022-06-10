import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim
# import wandb
from dataset import Preprocess
from datasets import load_dataset
from model import Encoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


def get_config():
    parser = argparse.ArgumentParser()

    """basic, model option"""
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--qmodel", type=str, default="bert-base-uncased")
    parser.add_argument("--pmodel", type=str, default="bert-base-uncased")
    # parser.add_argument('--qmodel', type=str, default= 'facebook/dpr-question_encoder-single-nq-base')
    # parser.add_argument('--pmodel', type=str, default= 'facebook/dpr-ctx_encoder-single-nq-base')
    parser.add_argument("--loss_option", type=str, default="con", help="con: using nll loss bce: bce loss")

    """hyperparameter"""
    parser.add_argument("--samples", type=int, default=100000)  # choose data sample size
    parser.add_argument("--samples_batch", type=int, default=10)  # pos + neg sample
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accum", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--max_length", type=int, default=256)

    args = parser.parse_args()

    return args


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_epoch(train_loader, valid_loader, q_model, p_model, optimizer, scheduler, criterion, epoch, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q_model.zero_grad()
    p_model.zero_grad()

    torch.cuda.empty_cache()

    train_loss, train_acc = 0, 0

    q_model.train()
    p_model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, batch in pbar:

        if torch.cuda.is_available():
            batch = tuple(t.to(device) for t in batch)
            q_model.to(device)
            p_model.to(device)

        q_inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}

        # batch * samples, max_len
        p_inputs = {
            "input_ids": batch[3].view(batch[3].shape[0] * args.samples_batch, -1),
            "attention_mask": batch[4].view(batch[3].shape[0] * args.samples_batch, -1),
            "token_type_ids": batch[5].view(batch[3].shape[0] * args.samples_batch, -1),
        }

        # targets= batch[6].cpu() # batch
        targets = torch.tensor([0] * batch[3].shape[0])  # 0번째로 고정시켜 놓음
        # targets= targets.type(torch.FloatTensor)

        q_outputs = q_model(**q_inputs)  # batch * emb
        p_outputs = p_model(**p_inputs)  # batch * samples * emb

        q_outputs = q_outputs.view(batch[0].shape[0], 1, -1)
        p_outputs = torch.transpose(
            p_outputs.view(batch[3].shape[0], args.samples_batch, -1), 1, 2
        )  # batch * emb * samples

        sim_scores = torch.bmm(q_outputs, p_outputs).view(batch[0].shape[0], -1)  # batch * samples
        sim_scores = F.log_softmax(sim_scores, dim=1).cpu()

        loss = criterion(sim_scores, targets)

        preds = torch.argmax(sim_scores.cpu(), axis=1)

        train_loss += loss.item()
        train_acc += torch.sum(preds == targets)

        loss.backward()
        optimizer.step()

        q_model.zero_grad()
        p_model.zero_grad()

    train_loss_ = train_loss / len(train_loader)
    train_acc_ = train_acc / len(train_loader.dataset)
    # train_acc_= train_acc / (len(train_loader.dataset) *args.samples_batch) # using BCE

    valid_loss_, valid_acc_ = validation_epoch(valid_loader, q_model, p_model, criterion, args)

    print(
        f"epoch: {epoch}, train loss: {train_loss_}, train acc: {train_acc_}, valid loss: {valid_loss_}, valid acc: {valid_acc_} "
    )
    # wandb.log({"train acc": train_acc_, "train loss": train_loss_, "val loss": valid_loss_, "val acc": valid_acc_})

    # scheduler
    scheduler.step()

    # model save
    if not os.path.exists("./model/q_model") and not os.path.exists("./model/p_model"):
        os.makedirs("./model/q_model")
        os.makedirs("./model/p_model")

    torch.save(q_model.state_dict(), f"./model/q_model/q_model_{epoch}.pt")
    torch.save(p_model.state_dict(), f"./model/p_model/p_model_{epoch}.pt")


def validation_epoch(valid_loader, q_model, p_model, criterion, args):
    valid_loss, valid_acc = 0, 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q_model.eval()
    p_model.eval()

    with torch.no_grad():
        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for step, batch in pbar:

            if torch.cuda.is_available():
                batch = tuple(t.to(device) for t in batch)
                q_model.to(device)
                p_model.to(device)

            q_inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}

            p_inputs = {
                "input_ids": batch[3].view(batch[3].shape[0] * args.samples_batch, -1),
                "attention_mask": batch[4].view(batch[3].shape[0] * args.samples_batch, -1),
                "token_type_ids": batch[5].view(batch[3].shape[0] * args.samples_batch, -1),
            }

            # targets= batch[6].cpu()
            targets = torch.tensor([0] * batch[3].shape[0])
            # targets= targets.type(torch.FloatTensor)

            q_outputs = q_model(**q_inputs)  # batch * emb
            p_outputs = p_model(**p_inputs)  # (batch * samples, * emb)

            q_outputs = q_outputs.view(batch[0].shape[0], 1, -1)
            p_outputs = torch.transpose(
                p_outputs.view(batch[3].shape[0], args.samples_batch, -1), 1, 2
            )  # batch * emb * samples

            sim_scores = torch.bmm(q_outputs, p_outputs).view(batch[0].shape[0], -1)  # batch * samples
            sim_scores = F.log_softmax(sim_scores, dim=1).cpu()
            loss = criterion(sim_scores, targets)
            preds = torch.argmax(sim_scores.cpu(), axis=1)
            # print(preds, targets)
            valid_loss += loss.item()
            valid_acc += torch.sum(preds == targets)

        valid_loss_ = valid_loss / len(valid_loader)
        valid_acc_ = valid_acc / len(valid_loader.dataset)
        # valid_acc_= valid_acc / (len(valid_loader.dataset) * args.samples_batch)

        return valid_loss_, valid_acc_


if __name__ == "__main__":

    args = get_config()
    seed_everything(args.seed)

    # load model
    q_model = Encoder(args.qmodel)
    p_model = Encoder(args.pmodel)
    tokenizer = AutoTokenizer.from_pretrained(args.qmodel)

    # load dataset
    dataset = load_dataset("ms_marco", "v2.1")
    train, valid = dataset["train"], dataset["validation"]

    print(f"train detail: {train}")
    print(f"valid detail: {valid}")

    # make dataset
    Preprocess = Preprocess(train, valid, tokenizer, args)
    trainset, validset = Preprocess.make_dataset()

    # dataloader
    train_loader = DataLoader(trainset, shuffle=True, batch_size=args.batch_size)
    valid_loader = DataLoader(validset, shuffle=True, batch_size=args.batch_size)

    # optimizer & criterion
    optimizer_grouped_parameters = [{"params": q_model.parameters()}, {"params": p_model.parameters()}]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32, eta_min=args.lr * 0.1)

    if args.loss_option == "con":
        criterion = torch.nn.NLLLoss()
    elif args.loss_option == "bce":
        criterion = torch.nn.BCELoss()

    # run = wandb.init(project="ms-marco", entity="rockmiin", name="BERT-CON", group="BASIC",)

    # training
    for epoch in range(args.epochs):
        train_epoch(train_loader, valid_loader, q_model, p_model, optimizer, scheduler, criterion, epoch, args)

    # run.finish()
