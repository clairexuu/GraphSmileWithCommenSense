"""
python -u run.py --gpu 0 --port 1532 --classify emotion \
--dataset MELD --epochs 50 --textf_mode graphsmile_cosmic \
--loss_type emo_sen_sft --lr 7e-05 --batch_size 16 --hidden_dim 384 \
--win 3 3 --heter_n_layers 5 5 5 --drop 0.2 --shift_win 3 --lambd 1.0 0.5 0.2 \
--balance_strategy subsample

Path to feature pickle files might need to be modified.
"""
import logging
import os
import numpy as np
import pickle as pk
import datetime
import torch.nn as nn
import torch.optim as optim
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import time
from utils import AutomaticWeightedLoss

# from hybrid_model import ModifiedGraphSmile
from GraphSmile_COSMIC_model import GraphSmile_COSMIC
# from model import GraphSmile 

from sklearn.metrics import confusion_matrix, classification_report
from trainer import train_or_eval_model, seed_everything
from dataloader import MELDDataset_BERT
from torch.utils.data import DataLoader
from model_utils import create_model_saver
import argparse
import platform

parser = argparse.ArgumentParser()

parser.add_argument("--no_cuda",
                    action="store_true",
                    default=False,
                    help="does not use GPU")
parser.add_argument("--gpu", default="2", type=str, help="GPU ids")
parser.add_argument("--port", default="15301", help="MASTER_PORT")
parser.add_argument("--classify", default="emotion", help="sentiment, emotion")
parser.add_argument("--lr",
                    type=float,
                    default=0.00001,
                    metavar="LR",
                    help="learning rate")
parser.add_argument("--l2",
                    type=float,
                    default=0.0001,
                    metavar="L2",
                    help="L2 regularization weight")
parser.add_argument("--batch_size",
                    type=int,
                    default=32,
                    metavar="BS",
                    help="batch size")
parser.add_argument("--epochs",
                    type=int,
                    default=100,
                    metavar="E",
                    help="number of epochs")
parser.add_argument("--tensorboard",
                    action="store_true",
                    default=False,
                    help="Enables tensorboard log")
parser.add_argument("--modals", default="avl", help="modals")
parser.add_argument(
    "--dataset",
    default="MELD",
    help="dataset to train and test.MELD/IEMOCAP/IEMOCAP4/CMUMOSEI7",
)
parser.add_argument(
    "--textf_mode",
    default="concat13",
    help="text feature mode: textf0/textf1/textf2/textf3/concat2/concat4/concat13/sum2/sum4/gated_fusion/graphsimile_cosmic/dual_encoder",
)

parser.add_argument(
    "--conv_fpo",
    nargs="+",
    type=int,
    default=[3, 1, 1],
    help="n_filter,n_padding; n_out = (n_in + 2*n_padding -n_filter)/stride +1",
)

parser.add_argument("--hidden_dim", type=int, default=256, help="hidden_dim")
parser.add_argument(
    "--win",
    nargs="+",
    type=int,
    default=[17, 17],
    help="[win_p, win_f], -1 denotes all nodes",
)
parser.add_argument("--heter_n_layers",
                    nargs="+",
                    type=int,
                    default=[6, 6, 6],
                    help="heter_n_layers")

parser.add_argument("--drop",
                    type=float,
                    default=0.3,
                    metavar="dropout",
                    help="dropout rate")

parser.add_argument("--shift_win",
                    type=int,
                    default=12,
                    help="windows of sentiment shift")

parser.add_argument(
    "--loss_type",
    default="emo_sen_sft",
    help="auto/epoch/emo_sen_sft/emo_sen/emo_sft/emo/sen_sft/sen",
)
parser.add_argument(
    "--lambd",
    nargs="+",
    type=float,
    default=[1.0, 1.0, 1.0],
    help="[loss_emotion, loss_sentiment, loss_shift]",
)

parser.add_argument('--balance_strategy', default='None', help='oversample / subsample / None')
parser.add_argument('--balance_target', default='emotion', help='Target for class balancing')
parser.add_argument('--weight', default='false', help='apply COSMIC weight or not')
parser.add_argument('--role_dim', type=int, default=150, help='dimension of speaker / listener')

args = parser.parse_args()

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = args.port
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
world_size = torch.cuda.device_count()
os.environ["WORLD_SIZE"] = str(world_size)

MELD_path = "features/meld_multi_features.pkl"
COMET_path = "features/meld_features_comet.pkl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_ddp(local_rank):
    try:
        if not dist.is_initialized():
            torch.cuda.set_device(local_rank)
            os.environ["RANK"] = str(local_rank)
            # dist.init_process_group(backend="nccl", init_method="env://")
            backend = "gloo" if platform.system() == "Windows" else "nccl"
            dist.init_process_group(backend=backend, init_method="env://")
        else:
            logger.info("Distributed process group already initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize distributed process group: {e}")
        raise


def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()

    return rt


def get_ddp_generator(seed=3407):
    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)

    return g


def get_train_valid_sampler(trainset, valid_ratio):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid_ratio * size)

    return DistributedSampler(idx[split:]), DistributedSampler(idx[:split])


def get_data_loaders(comet_path, meld_path, dataset_class, batch_size, valid_ratio, num_workers,
                     pin_memory, balance_strategy=None, balance_target='emotion'):
    trainset = dataset_class(comet_path, meld_path, train=True,
                             balance_strategy=balance_strategy, balance_target=balance_target)
    train_sampler, valid_sampler = get_train_valid_sampler(
        trainset, valid_ratio)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=valid_sampler,
        collate_fn=trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    testset = dataset_class(comet_path, meld_path, train=False)
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        collate_fn=testset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader


def setup_samplers(trainset, valid_ratio, epoch):
    train_sampler, valid_sampler = get_train_valid_sampler(
        trainset, valid_ratio=valid_ratio)
    train_sampler.set_epoch(epoch)
    valid_sampler.set_epoch(epoch)


def main(local_rank):
    print(f"Running main(**args) on rank {local_rank}.")
    init_ddp(local_rank)  # 初始化

    today = datetime.datetime.now()
    name_ = args.modals + "_" + args.dataset

    # Create model saver for this experiment
    balance_suffix = f"_{args.balance_strategy}_{args.balance_target}" if args.balance_strategy else ""
    weight_suffix = f"_weighted" if args.weight == "true" else ""
    experiment_name = f"{name_}_{args.textf_mode}_{args.classify}{balance_suffix}{weight_suffix}"
    model_saver = create_model_saver(experiment_name) if local_rank == 0 else None

    if local_rank == 0:
        model_saver.save_experiment_config(args)

    cuda = torch.cuda.is_available() and not args.no_cuda
    if args.tensorboard:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter()

    n_epochs = args.epochs
    batch_size = args.batch_size
    modals = args.modals

    embedding_dims = [1024, 342, 300]
    n_classes_emo = 7

    seed_everything()
    model = GraphSmile_COSMIC(args, embedding_dims, n_classes_emo)

    model = model.to(local_rank)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    if args.weight == "true": 
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        COSMIC_weights = torch.tensor(
            [0.30427062, 1.19699616, 5.47007183, 1.95437696, 0.84847735, 5.42461417, 1.21859721],
            dtype=torch.float32,
            device=device
        )
        loss_function_emo = nn.NLLLoss(weight=COSMIC_weights)
    else: 
        loss_function_emo = nn.NLLLoss()

    loss_function_sen = nn.NLLLoss()
    loss_function_shift = nn.NLLLoss()

    if args.loss_type == "auto_loss":
        awl = AutomaticWeightedLoss(3)
        optimizer = optim.AdamW(
            [
                {
                    "params": model.parameters()
                },
                {
                    "params": awl.parameters(),
                    "weight_decay": 0
                },
            ],
            lr=args.lr,
            weight_decay=args.l2,
            amsgrad=True,
        )
    else:
        optimizer = optim.AdamW(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.l2,
                                amsgrad=True)

    train_loader, valid_loader, test_loader = get_data_loaders(
        comet_path=COMET_path,
        meld_path=MELD_path,
        dataset_class=MELDDataset_BERT,
        valid_ratio=0.1,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        balance_strategy=args.balance_strategy,
        balance_target=args.balance_target
    )

    best_f1_emo, best_f1_sen, best_loss = None, None, None
    best_label_emo, best_pred_emo = None, None
    best_label_sen, best_pred_sen = None, None
    best_extracted_feats = None
    all_f1_emo, all_acc_emo, all_loss = [], [], []
    all_f1_sen, all_acc_sen = [], []
    all_f1_sft, all_acc_sft = [], []

    for epoch in range(n_epochs):
        trainset = MELDDataset_BERT(COMET_path, MELD_path, train=True)

        setup_samplers(trainset, valid_ratio=0.1, epoch=epoch)

        start_time = time.time()

        train_loss, _, _, train_acc_emo, train_f1_emo, _, _, train_acc_sen, train_f1_sen, train_acc_sft, train_f1_sft, _, _, _ = train_or_eval_model(
            model,
            loss_function_emo,
            loss_function_sen,
            loss_function_shift,
            train_loader,
            epoch,
            cuda,
            args.modals,
            optimizer,
            True,
            args.dataset,
            args.loss_type,
            args.lambd,
            args.epochs,
            args.classify,
            args.shift_win,
        )

        valid_loss, _, _, valid_acc_emo, valid_f1_emo, _, _, valid_acc_sen, valid_f1_sen, valid_acc_sft, valid_f1_sft, _, _, _ = train_or_eval_model(
            model,
            loss_function_emo,
            loss_function_sen,
            loss_function_shift,
            valid_loader,
            epoch,
            cuda,
            args.modals,
            None,
            False,
            args.dataset,
            args.loss_type,
            args.lambd,
            args.epochs,
            args.classify,
            args.shift_win,
        )

        print(
            "epoch: {}, train_loss: {}, train_acc_emo: {}, train_f1_emo: {}, valid_loss: {}, valid_acc_emo: {}, valid_f1_emo: {}"
            .format(
                epoch + 1,
                train_loss,
                train_acc_emo,
                train_f1_emo,
                valid_loss,
                valid_acc_emo,
                valid_f1_emo,
            ))

        # Update metrics in model saver
        if local_rank == 0 and model_saver:
            model_saver.update_metrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_acc_emo=train_acc_emo,
                train_f1_emo=train_f1_emo,
                train_acc_sen=train_acc_sen,
                train_f1_sen=train_f1_sen,
                train_acc_sft=train_acc_sft,
                train_f1_sft=train_f1_sft,
                valid_loss=valid_loss,
                valid_acc_emo=valid_acc_emo,
                valid_f1_emo=valid_f1_emo,
                valid_acc_sen=valid_acc_sen,
                valid_f1_sen=valid_f1_sen,
                valid_acc_sft=valid_acc_sft,
                valid_f1_sft=valid_f1_sft
            )

        if local_rank == 0:
            test_loss, test_label_emo, test_pred_emo, test_acc_emo, test_f1_emo, test_label_sen, test_pred_sen, test_acc_sen, test_f1_sen, test_acc_sft, test_f1_sft, _, test_initial_feats, test_extracted_feats = train_or_eval_model(
                model,
                loss_function_emo,
                loss_function_sen,
                loss_function_shift,
                test_loader,
                epoch,
                cuda,
                args.modals,
                None,
                False,
                args.dataset,
                args.loss_type,
                args.lambd,
                args.epochs,
                args.classify,
                args.shift_win,
            )

            all_f1_emo.append(test_f1_emo)
            all_acc_emo.append(test_acc_emo)
            all_f1_sft.append(test_f1_sft)
            all_acc_sft.append(test_acc_sft)

            # Update test metrics in model saver
            if model_saver:
                model_saver.update_metrics(
                    test_loss=test_loss,
                    test_acc_emo=test_acc_emo,
                    test_f1_emo=test_f1_emo,
                    test_acc_sen=test_acc_sen,
                    test_f1_sen=test_f1_sen,
                    test_acc_sft=test_acc_sft,
                    test_f1_sft=test_f1_sft
                )

            print(
                "test_loss: {}, test_acc_emo: {}, test_f1_emo: {}, test_acc_sen: {}, test_f1_sen: {}, test_acc_sft: {}, test_f1_sft: {}, total time: {} sec, {}"
                .format(
                    test_loss,
                    test_acc_emo,
                    test_f1_emo,
                    test_acc_sen,
                    test_f1_sen,
                    test_acc_sft,
                    test_f1_sft,
                    round(time.time() - start_time, 2),
                    time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime(time.time())),
                ))
            print("-" * 100)

            is_best = False
            if args.classify == "emotion":
                if best_f1_emo == None or best_f1_emo < test_f1_emo:
                    best_f1_emo = test_f1_emo
                    best_f1_sen = test_f1_sen
                    best_label_emo, best_pred_emo = test_label_emo, test_pred_emo
                    best_label_sen, best_pred_sen = test_label_sen, test_pred_sen
                    is_best = True

            elif args.classify == "sentiment":
                if best_f1_sen == None or best_f1_sen < test_f1_sen:
                    best_f1_emo = test_f1_emo
                    best_f1_sen = test_f1_sen
                    best_label_emo, best_pred_emo = test_label_emo, test_pred_emo
                    best_label_sen, best_pred_sen = test_label_sen, test_pred_sen
                    is_best = True

            # Save model checkpoint
            if model_saver:
                model_saver.save_model(model, optimizer, epoch + 1, is_best=is_best,
                                       extra_info={'best_f1_emo': best_f1_emo, 'best_f1_sen': best_f1_sen})

            if (epoch + 1) % 10 == 0:
                np.set_printoptions(suppress=True)
                print(
                    classification_report(best_label_emo,
                                          best_pred_emo,
                                          digits=4,
                                          zero_division=0))
                print(confusion_matrix(best_label_emo, best_pred_emo))
                print(
                    classification_report(best_label_sen,
                                          best_pred_sen,
                                          digits=4,
                                          zero_division=0))
                print(confusion_matrix(best_label_sen, best_pred_sen))
                print("-" * 100)

        dist.barrier()

        if args.tensorboard:
            writer.add_scalar("test: accuracy", test_acc_emo, epoch)
            writer.add_scalar("test: fscore", test_f1_emo, epoch)
            writer.add_scalar("train: accuracy", train_acc_emo, epoch)
            writer.add_scalar("train: fscore", train_f1_emo, epoch)

        if epoch == 1:
            allocated_memory = torch.cuda.memory_allocated()
            reserved_memory = torch.cuda.memory_reserved()
            print(f"Allocated Memory: {allocated_memory / 1024 ** 2:.2f} MB")
            print(f"Reserved Memory: {reserved_memory / 1024 ** 2:.2f} MB")
            print(
                f"All Memory: {(allocated_memory + reserved_memory) / 1024 ** 2:.2f} MB"
            )

    if args.tensorboard:
        writer.close()
    if local_rank == 0:
        print("Test performance..")
        print("Acc: {}, F-Score: {}".format(max(all_acc_emo), max(all_f1_emo)))

        # Finalize experiment with model saver
        if model_saver:
            emotion_labels = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
            sentiment_labels = ['negative', 'neutral', 'positive']
            model_saver.finalize_experiment(
                best_label_emo, best_pred_emo,
                best_label_sen, best_pred_sen,
                emotion_labels, sentiment_labels
            )

        # Keep original results saving for backward compatibility
        if not os.path.exists("results"):
            os.makedirs("results")

        if not os.path.exists("results/record_{}_{}_{}.pk".format(
                today.year, today.month, today.day)):
            with open(
                    "results/record_{}_{}_{}.pk".format(
                        today.year, today.month, today.day),
                    "wb",
            ) as f:
                pk.dump({}, f)
        with open(
                "results/record_{}_{}_{}.pk".format(today.year, today.month,
                                                    today.day),
                "rb",
        ) as f:
            record = pk.load(f)
        key_ = name_
        if record.get(key_, False):
            record[key_].append(max(all_f1_emo))
        else:
            record[key_] = [max(all_f1_emo)]
        if record.get(key_ + "record", False):
            record[key_ + "record"].append(
                classification_report(best_label_emo,
                                      best_pred_emo,
                                      digits=4,
                                      zero_division=0))
        else:
            record[key_ + "record"] = [
                classification_report(best_label_emo,
                                      best_pred_emo,
                                      digits=4,
                                      zero_division=0)
            ]
        with open(
                "results/record_{}_{}_{}.pk".format(today.year, today.month,
                                                    today.day),
                "wb",
        ) as f:
            pk.dump(record, f)

        # Save the trained model (legacy)
        if not os.path.exists("models"):
            os.makedirs("models")

        torch.save(model.state_dict(), f"models/model_{name_}.pt")
        print(f"Model saved to: models/model_{name_}.pt")

        print(
            classification_report(best_label_emo,
                                  best_pred_emo,
                                  digits=4,
                                  zero_division=0))
        print(confusion_matrix(best_label_emo, best_pred_emo))

    dist.destroy_process_group()


if __name__ == "__main__":
    print(args)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("not args.no_cuda:", not args.no_cuda)
    n_gpus = torch.cuda.device_count()
    print(f"Use {n_gpus} GPUs")
    mp.spawn(fn=main, args=(), nprocs=n_gpus)