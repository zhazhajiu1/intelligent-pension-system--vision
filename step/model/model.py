import math
import os
import os.path as osp
import pickle
import random
import sys
from datetime import datetime
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata

from .network import TripletLoss, SetNet
from .utils import TripletSampler

from PIL import Image

class Model:
    def __init__(self,
                 hidden_dim,
                 lr,
                 hard_or_full_trip,
                 margin,
                 num_workers,
                 batch_size,
                 restore_iter,
                 total_iter,
                 save_name,
                 train_pid_num,
                 frame_num,
                 model_name,
                 img_size=64):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.save_name = save_name
        self.train_pid_num = train_pid_num

        self.hidden_dim = hidden_dim
        self.lr = lr
        self.hard_or_full_trip = hard_or_full_trip
        self.margin = margin
        self.frame_num = frame_num
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.P, self.M = batch_size

        self.restore_iter = restore_iter
        self.total_iter = total_iter

        self.img_size = img_size

        self.encoder = SetNet(self.hidden_dim).float().to(self.device)
        self.encoder = nn.DataParallel(self.encoder)
        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float().to(self.device)
        self.triplet_loss = nn.DataParallel(self.triplet_loss)

        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters()},
        ], lr=self.lr)

        self.hard_loss_metric = []
        self.full_loss_metric = []
        self.full_loss_num = []
        self.dist_list = []
        self.mean_dist = 0.01

        self.sample_type = 'all'

    def collate_fn(self, batch):
        batch_size = len(batch)
        feature_num = len(batch[0][0])
        seqs = [batch[i][0] for i in range(batch_size)]
        frame_sets = [batch[i][1] for i in range(batch_size)]
        view = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        label = [batch[i][4] for i in range(batch_size)]
        batch = [seqs, view, seq_type, label, None]

        def select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                frame_id_list = random.choices(frame_set, k=self.frame_num)
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _

        seqs = list(map(select_frame, range(len(seqs))))

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)
            batch_frames = [[
                len(frame_sets[i])
                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                if i < batch_size
            ] for _ in range(gpu_num)]
            if len(batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)
            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
            seqs = [[
                np.concatenate([
                    seqs[i][j]
                    for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                    if i < batch_size
                ], 0) for _ in range(gpu_num)]
                for j in range(feature_num)]
            seqs = [np.asarray([
                np.pad(seqs[j][_],
                       ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                       'constant',
                       constant_values=0)
                for _ in range(gpu_num)])
                for j in range(feature_num)]
            batch[4] = np.asarray(batch_frames)

        batch[0] = seqs
        return batch

    def save_train_features(self):
        train_features, _, _, train_labels = self.transform('train')
        with open('train_features.pkl', 'wb') as f:
            pickle.dump((train_features, train_labels), f)


    def fit(self):
        if self.restore_iter != 0:
            self.load(self.restore_iter)

        self.encoder.train()
        self.sample_type = 'random'
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        triplet_sampler = TripletSampler(self.train_source, self.batch_size)
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        train_label_set = list(self.train_source.label_set)
        train_label_set.sort()

        _time1 = datetime.now()
        for seq, view, seq_type, label, batch_frame in train_loader:
            self.restore_iter += 1
            self.optimizer.zero_grad()

            for i in range(len(seq)):
                seq[i] = self.np2var(seq[i]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            feature, label_prob = self.encoder(*seq, batch_frame)

            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.array(target_label)).long()

            triplet_feature = feature.permute(1, 0, 2).contiguous()
            triplet_label = target_label.unsqueeze(0).repeat(triplet_feature.size(0), 1)
            (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num
             ) = self.triplet_loss(triplet_feature, triplet_label)
            if self.hard_or_full_trip == 'hard':
                loss = hard_loss_metric.mean()
            elif self.hard_or_full_trip == 'full':
                loss = full_loss_metric.mean()

            self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())
            self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
            self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
            self.dist_list.append(mean_dist.mean().data.cpu().numpy())

            if loss > 1e-9:
                loss.backward()
                self.optimizer.step()

            if self.restore_iter % 1000 == 0:
                print(datetime.now() - _time1)
                _time1 = datetime.now()

            if self.restore_iter % 100 == 0:
                self.save()
                print('iter {}:'.format(self.restore_iter), end='')
                print(', hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric)), end='')
                print(', full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)), end='')
                print(', full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)), end='')
                self.mean_dist = np.mean(self.dist_list)
                print(', mean_dist={0:.8f}'.format(self.mean_dist), end='')
                print(', lr=%f' % self.optimizer.param_groups[0]['lr'], end='')
                print(', hard or full=%r' % self.hard_or_full_trip)
                sys.stdout.flush()
                self.hard_loss_metric = []
                self.full_loss_metric = []
                self.full_loss_num = []
                self.dist_list = []

            if self.restore_iter == self.total_iter:
                self.save_train_features()
                break

    def ts2var(self, x):
        return autograd.Variable(x).to(self.device)

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def transform(self, flag, batch_size=1):
        self.encoder.eval()
        source = self.test_source if flag == 'test' else self.train_source
        print(source)
        self.sample_type = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        for i, x in enumerate(data_loader):
            seq, view, seq_type, label, batch_frame = x
            for j in range(len(seq)):
                seq[j] = self.np2var(seq[j]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()
            # print(batch_frame, np.sum(batch_frame))

            feature, _ = self.encoder(*seq, batch_frame)
            n, num_bin, _ = feature.size()
            feature_list.append(feature.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list += label

        return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list

    def save(self):
        os.makedirs(osp.join('checkpoint', self.model_name), exist_ok=True)
        torch.save(self.encoder.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-encoder.ptm'.format(
                                self.save_name, self.restore_iter)))
        torch.save(self.optimizer.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-optimizer.ptm'.format(
                                self.save_name, self.restore_iter)))

    # 修改后通过路径读取模型
    def load(self, encoder_path, optimizer_path):
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=torch.device('cpu')))
        self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=torch.device('cpu')))


    def preprocess_image(self, image):
        """
        预处理
        """
        # Convert the image to a numpy array if it is a PIL image
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Resize the image to the required input size
        image = cv2.resize(image, (self.img_size, self.img_size))
        # Convert the image to float32 and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        # Transpose the image dimensions to fit the model input (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        # Add a batch dimension
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image, batch_size=1):
        """
        提取特征
        """
        # Set the model to evaluation mode
        self.encoder.eval()
        # Preprocess the image
        image = self.preprocess_image(image)
        # Convert to a PyTorch tensor and move to GPU if available
        image = self.np2var(image).float()

        with torch.no_grad():  # Disable gradient computation
            # Run the model inference
            feature, _ = self.encoder(image)

        # Return the feature vector (62, 256)
        return feature.squeeze().cpu().numpy()

    def predict_person(self, image):
        """
        具体判断哪个人
        """
        # Get the predicted feature vector (62, 256)
        feature = self.predict(image)

        # Load the precomputed training features and labels
        data = np.load('train_features_labels.npz')
        train_features = data['train_features']
        train_labels = data['train_labels']

        # 计算每个标签的平均特征
        unique_labels = np.unique(train_labels)
        mean_features = []
        mean_labels = []

        for label in unique_labels:
            label_features = train_features[train_labels == label]
            mean_feature = np.mean(label_features, axis=0)
            mean_features.append(mean_feature)
            mean_labels.append(label)

        mean_features = np.array(mean_features)
        mean_labels = np.array(mean_labels)

        print("len:", len(mean_features))
        print("data:", mean_features, mean_labels)

        # 确保特征向量的形状兼容
        feature = feature.reshape(1, -1)  # 将特征向量展平成 (1, 15872)

        # 计算特征与每个标签的平均特征的距离，并找到最小距离对应的标签
        distances = np.linalg.norm(mean_features - feature, axis=1)
        print(distances)
        predicted_index = np.argmin(distances)
        predicted_person = mean_labels[predicted_index]

        return predicted_person