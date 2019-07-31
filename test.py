from argparse import ArgumentParser

from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import L1Loss

from model import load_pretrained_et_predictor
from data import CORPUS_CLASSES
from settings import USE_CUDA

parser = ArgumentParser()
parser.add_argument('--model-path', default='')
parser.add_argument('--test-dataset', default='')
args = parser.parse_args()

mae_loss = L1Loss(reduction='sum')
model, vocab, corpus_aggregator = load_pretrained_et_predictor()
test_dataset = CORPUS_CLASSES[args.test_dataset]()
test_loader = DataLoader(test_dataset)

print('===== Testing on {} ====='.format(args.test_dataset))
print('Model weights:', args.model_path)
print('\nFound {} samples. Will now run test...'.format(len(test_dataset)))

# do this per sample, to avoid complicated code because of padding and
# loss calculation...
loss = Tensor([0, 0, 0, 0, 0])
for (sentence, et_target, et_target_orig, index) in test_loader:
    if USE_CUDA:
        sentence = sentence.cuda()
        et_target = et_target.cuda()

    et_prediction = model(sentence)
    et_prediction_inverse = corpus_aggregator.inverse_transform(
        index[0], et_prediction.detach().cpu())
    _loss = Tensor([mae_loss(et_prediction_inverse[0][:, i],
                            et_target_orig[0][:, i]).item()
                   for i in range(5)])
    loss += _loss


print('Testing done. MAE Loss per feature:', torch.mean(loss, 0))

"""
for (sentences, et_targets, et_targets_orig, indices) in test_loader:
    if USE_CUDA:
        sentences = sentences.cuda()
        et_targets = et_targets.cuda()

    et_preds = model(sentences)
    et_preds_inverse = torch.Tensor([
        corpus_aggregator.inverse_transform(idx, value)
        for (idx, value) in zip(indices, et_preds.detach().cpu())])
"""
