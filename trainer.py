import torch
import numpy as np
from sklearn.metrics import r2_score

from model import EyeTrackingPredictor
from settings import USE_CUDA


class Trainer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.mse_loss = torch.nn.MSELoss(reduction='sum')

    def init_model(self, args):
        if args.finetune_elmo != 'False' or args.static_embedding:
            model = EyeTrackingPredictor(
                finetune_elmo=eval(args.finetune_elmo),
                static_embedding=args.static_embedding)
        else:
            model = EyeTrackingPredictor(
                self.dataset.vocabulary.word_embeddings.clone())

        return (
            model,
            torch.optim.Adam(model.parameters(), lr=eval(args.lr))
        )

    def iterate(self, model, optimizer, dataloader):
        epoch_loss = 0.0
        # loss calculated on the real/original values (not scaled)
        epoch_loss_ = torch.Tensor([0, 0, 0, 0, 0])
        r2_scores = torch.Tensor([0, 0, 0, 0, 0])
        for i, (sentences, et_targets,
                et_targets_orig, indices) in enumerate(dataloader):

            # if not args.finetune_elmo_embeddings:
            # sentences = sentences.type(torch.LongTensor)
            if USE_CUDA:
                sentences = sentences.cuda()
                et_targets = et_targets.cuda()

            et_preds = model(sentences)

            et_preds_inverse = torch.Tensor([
                self.dataset.inverse_transform(idx, value)
                for (idx, value) in zip(indices, et_preds.detach().cpu())])

            # starting from the padding index, make the prediction values 0
            for sent, et_pred, et_pred_inverse in zip(
                    sentences, et_preds, et_preds_inverse):
                try:
                    pad_start_idx = np.where(sent.cpu().numpy() == 0)[0][0]
                except IndexError:
                    pad_start_idx = None
                et_pred[pad_start_idx:] = 0
                et_pred_inverse[pad_start_idx:] = 0

            num_data_points = et_targets_orig[et_targets_orig > 0].shape[0]
            # mse loss divided by the actual number of data points
            # (have to disregard the padding!)
            loss = torch.sqrt(
                self.mse_loss(et_preds, et_targets) / num_data_points)

            # calculate the loss PER FEATURE
            loss_ = torch.Tensor(
                [self.mse_loss(et_preds_inverse[:, :, i],
                               et_targets_orig[:, :, i]).item()
                 for i in range(5)]) / (num_data_points / 5)
            loss_ = torch.sqrt(loss_)
            r2 = torch.Tensor(r2_score(et_targets_orig.reshape(-1, 5),
                                       et_preds_inverse.reshape(-1, 5),
                                       multioutput='raw_values'))

            if model.training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_ += loss_
            r2_scores += r2

        return epoch_loss / (i + 1), epoch_loss_ / (i + 1), r2_scores / (i + 1)
