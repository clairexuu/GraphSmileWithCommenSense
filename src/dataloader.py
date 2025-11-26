import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy as np
from collections import Counter
import random


class MELDDataset_BERT(Dataset):

    def __init__(self, comet_path, graph_smile_path, train=True, balance_strategy=None, balance_target='emotion',
                 seed=2024):
        """
        label index mapping = {0:neutral, 1:surprise, 2:fear, 3:sadness, 4:joy, 5:disgust, 6:anger}
        train: Whether to load train or test set
        balance_strategy:
            - None
            - oversample
            - subsample
        balance_target: emotion or sentiment
        """
        random.seed(seed)
        np.random.seed(seed)

        (
            self.videoIDs,
            self.videoSpeakers,
            self.videoLabels,
            self.videoSentiments,
            self.videoText0,
            self.videoText1,
            self.videoText2,
            self.videoText3,
            self.videoAudio,
            self.videoVisual,
            self.videoSentence,
            self.trainVid,
            self.testVid,
            _,
        ) = pickle.load(open(graph_smile_path, "rb"))

        # Load COMET commonsense features
        (
            self.xIntent,
            self.xAttr,
            self.xNeed,
            self.xWant,
            self.xEffect,
            self.xReact,
            self.oWant,
            self.oEffect,
            self.oReact
        ) = pickle.load(open(comet_path, "rb"), encoding='latin1')

        self.train = train

        # Get keys
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.balance_strategy = balance_strategy
        self.balance_target = balance_target

        self.len = len(self.keys)

        self.labels_emotion = self.videoLabels
        self.labels_sentiment = self.videoSentiments

        # Count dominant label per dialogue
        label_to_vids = {i: [] for i in range(7)}  # 7 emotion classes

        for vid in self.keys:
            emo_labels = self.videoLabels[vid]
            dominant_label = Counter(emo_labels).most_common(1)[0][0]
            label_to_vids[dominant_label].append(vid)

        if self.train and self.balance_strategy == 'oversample' and self.balance_target == 'emotion':
            # Choose target size: 2x minimum class size, or 30% of full size (cap)
            min_class_size = min(len(v) for v in label_to_vids.values())
            max_target = min(2 * min_class_size, int(0.3 * len(self.keys)))

            oversampled_keys = []
            for emo, vids in label_to_vids.items():
                if len(vids) < max_target:
                    multiplier = max_target // len(vids)
                    remainder = max_target % len(vids)
                    sampled = vids * multiplier + random.sample(vids, min(remainder, len(vids)))
                else:
                    sampled = vids
                oversampled_keys.extend(sampled)

            self.keys = list(set(self.keys + oversampled_keys))
            print(f"[Data Balance] Oversampled minority dominant emotion classes to {len(self.keys)} dialogues total.")

        elif self.train and self.balance_strategy == 'subsample' and self.balance_target == 'emotion':
            # Subsample each dominant class to the minimum available size
            min_class_size = min(len(v) for v in label_to_vids.values())
            target_size = int(min_class_size * 1.5)

            balanced_keys = []
            for emo, vids in label_to_vids.items():
                if len(vids) > target_size:
                    subsampled = random.sample(vids, target_size)
                else:
                    subsampled = vids
                balanced_keys.extend(subsampled)

            self.keys = balanced_keys
            print(f"[Data Balance] Subsampled dominant emotion classes to {len(self.keys)} dialogues total.")

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            torch.FloatTensor(np.array(self.videoText0[vid])),
            torch.FloatTensor(np.array(self.videoText1[vid])),
            torch.FloatTensor(np.array(self.videoText2[vid])),
            torch.FloatTensor(np.array(self.videoText3[vid])),
            # 9 COMET commonsense features
            torch.FloatTensor(np.array(self.xIntent[vid])),
            torch.FloatTensor(np.array(self.xAttr[vid])),
            torch.FloatTensor(np.array(self.xNeed[vid])),
            torch.FloatTensor(np.array(self.xWant[vid])),
            torch.FloatTensor(np.array(self.xEffect[vid])),
            torch.FloatTensor(np.array(self.xReact[vid])),
            torch.FloatTensor(np.array(self.oWant[vid])),
            torch.FloatTensor(np.array(self.oEffect[vid])),
            torch.FloatTensor(np.array(self.oReact[vid])),
            # Video features
            torch.FloatTensor(np.array(self.videoVisual[vid])),
            torch.FloatTensor(np.array(self.videoAudio[vid])),
            torch.FloatTensor(
                [[1, 0] if spk == "M" else [0, 1] for spk in self.videoSpeakers[vid]]
            ),
            torch.FloatTensor([1] * len(np.array(self.labels_emotion[vid]))),
            torch.LongTensor(np.array(self.labels_emotion[vid])),
            torch.LongTensor(np.array(self.labels_sentiment[vid])),
            vid,
        )

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label += self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [
            (
                pad_sequence(dat[i]) if i < 16
                else pad_sequence(dat[i]) if i < 19 else dat[i].tolist()
            )
            for i in dat
        ]
