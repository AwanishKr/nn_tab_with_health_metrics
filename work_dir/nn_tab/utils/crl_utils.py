import numpy as np
import torch
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# rank target entropy
def negative_entropy(data, normalize=False, max_value=None):
    softmax = F.softmax(data, dim=1)
    log_softmax = F.log_softmax(data, dim=1)
    entropy = softmax * log_softmax
    entropy = -1.0 * entropy.sum(dim=1)
    # normalize [0 ~ 1]
    if normalize:
        normalized_entropy = entropy / max_value
        return -normalized_entropy
    return -entropy


# forgetting history class
class ForgettingTracker:
    def __init__(self, num_examples):
        self.num_examples = num_examples
        self.previous_correct = [False] * num_examples
        self.first_learned_epoch = [None] * num_examples
        self.forgetting_counts = np.zeros((num_examples))
        self.max_forgetting = 1

    def forgetting_update(self, logits, labels, indices, epoch):
        preds = logits.argmax(dim=1)
        for pred, label, idx in zip(preds, labels, indices):
            was_correct = self.previous_correct[idx]
            now_correct = (pred.item() == label.item())

            # If previously correct and now incorrect: forgetting event
            if was_correct and not now_correct:
                self.forgetting_counts[idx] += 1

            # If learned for the first time
            if not was_correct and now_correct:
                if self.first_learned_epoch[idx] is None:
                    self.first_learned_epoch[idx] = epoch

            self.previous_correct[idx] = now_correct

    def get_stats(self):
        return {'first_learned_epoch': self.first_learned_epoch, 'forgetting_counts': self.forgetting_counts}
    
    def forgetting_normalize(self, data):
        data_min = self.forgetting_counts.min()
        data_max = float(self.max_forgetting)

        return (data - data_min) / (data_max - data_min)
    
    def get_forgetting_target_margin(self, data_idx1, data_idx2):
        data_idx1 = data_idx1.cpu().numpy()
        cum_forgetting1 = self.forgetting_counts[data_idx1]
        cum_forgetting2 = self.forgetting_counts[data_idx2]
        
        # normalize correctness values
        cum_forgetting1 = self.forgetting_normalize(cum_forgetting1)
        cum_forgetting2 = self.forgetting_normalize(cum_forgetting2)
        
        # make target pair
        n_pair = len(data_idx1)
        target1 = cum_forgetting1[:n_pair]
        target2 = cum_forgetting2[:n_pair]
        
        # calc target
        greater = np.array(target1 > target2, dtype='float') * (-1)
        less = np.array(target1 < target2, dtype='float') 

        target = greater + less
        target = torch.from_numpy(target).float().to(device)
        
        # calc margin
        margin = abs(target1 - target2)
        margin = torch.from_numpy(margin).float().to(device)

        return target, margin
    
    def max_forgetting_update(self, epoch):
        if epoch > 1:
            self.max_forgetting += 1
    

# correctness history class
class History(object):
    def __init__(self, n_data):
        self.correctness = np.zeros((n_data))
        self.confidence = np.zeros((n_data))
        self.max_correctness = 1

    # correctness update
    def correctness_update(self, data_idx, correctness, output):
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, _ = probs.max(dim=1)
        data_idx = data_idx.cpu().numpy()

        self.correctness[data_idx] += correctness.cpu().numpy()
        self.confidence[data_idx] = confidence.cpu().detach().numpy()

    # max correctness update
    def max_correctness_update(self, epoch):
        if epoch > 1:
            self.max_correctness += 1

    # correctness normalize (0 ~ 1) range
    def correctness_normalize(self, data):
        data_min = self.correctness.min()
        data_max = float(self.max_correctness)

        return (data - data_min) / (data_max - data_min)

    # get target & margin
    def get_target_margin(self, data_idx1, data_idx2):
        data_idx1 = data_idx1.cpu().numpy()
        cum_correctness1 = self.correctness[data_idx1]
        cum_correctness2 = self.correctness[data_idx2]
        # normalize correctness values
        cum_correctness1 = self.correctness_normalize(cum_correctness1)
        cum_correctness2 = self.correctness_normalize(cum_correctness2)
        # make target pair
        n_pair = len(data_idx1)
        target1 = cum_correctness1[:n_pair]
        target2 = cum_correctness2[:n_pair]
        # calc target
        greater = np.array(target1 > target2, dtype='float')
        less = np.array(target1 < target2, dtype='float') * (-1)

        target = greater + less
        target = torch.from_numpy(target).float().to(device)
        # calc margin
        margin = abs(target1 - target2)
        margin = torch.from_numpy(margin).float().to(device)

        return target, margin
    
