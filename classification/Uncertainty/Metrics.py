import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from scipy.special import psi, gammaln


# Returns the precision value based on received logits values
# Input:    logits outputs of a neural network
def precision(outputs):
    outputs = outputs.astype('float64')
    return np.sum(np.exp(outputs), axis=-1)


# Returns largest probability value based on received logits
# Input:    logits outputs of a neural network
def max_probability(outputs):
    return np.max(np.softmax(outputs, axis=-1), axis=-1)


# Returns mutual information based on received logits
# Input:    logits outputs of a neural network
def mutual_information(outputs):
    outputs = outputs.astype('float64')
    alpha_c = np.exp(outputs)
    alpha_0 = np.sum(alpha_c, axis=-1)

    gammaln_alpha_c = gammaln(alpha_c)
    gammaln_alpha_0 = gammaln(alpha_0)

    psi_alpha_c = psi(alpha_c)
    psi_alpha_0 = psi(alpha_0)
    psi_alpha_0 = np.expand_dims(psi_alpha_0, axis=1)

    temp_mat = np.sum((alpha_c - 1) * (psi_alpha_c - psi_alpha_0), axis=1)

    metric = np.sum(gammaln_alpha_c, axis=-1) - gammaln_alpha_0 - temp_mat
    return metric


# Returns probability vector based on received logits
# Input:    logits outputs of a neural network
def _get_prob(outputs):
    logits = outputs.astype('float64')
    alpha_c = np.exp(logits)
    alpha_c = np.clip(alpha_c, 10e-25, 10e25)
    alpha_0 = np.sum(alpha_c, axis=-1)
    alpha_0 = np.expand_dims(alpha_0, axis=-1)

    return (alpha_c / alpha_0)

# Returns entropy based on received logits
# Input:    logits outputs of a neural network
def entropy(outputs):
    prob = _get_prob(outputs)
    exp_prob = np.log(prob)

    ent = -np.sum(prob*exp_prob, axis=-1)
    return ent


# AUROC values for a binary setting
# Input:    scores of a computed metric
#           binary labels (0 or 1)
def auroc_values(scores, binary_label):
    return roc_auc_score(binary_label, scores)


# Get dictionary with several auroc scores
# Input:    logit outputs of a neural network
#           binary labels (0 or 1)
def auroc_dict(outputs, binary_label):
    res_dict = {}

    # precision metric
    res_dict["roc_precision"] = 100 * auroc_values(scores=precision(outputs=outputs), binary_label=binary_label)
    res_dict["pr_precision"] = 100 * precision_score(scores=precision(outputs=outputs), binary_label=binary_label)
    res_dict["rec_precision"] = 100 * recall_score(scores=precision(outputs=outputs), binary_label=binary_label)

    # max probability metric
    res_dict["roc_max_probability"] = 100 * auroc_values(scores=max_probability(outputs=outputs), binary_label=binary_label)
    res_dict["pr_max_probability"] = 100 * precision_score(scores=max_probability(outputs=outputs), binary_label=binary_label)
    res_dict["rec_max_probability"] = 100 * recall_score(scores=max_probability(outputs=outputs), binary_label=binary_label)
    
    # mutual information metric
    res_dict["roc_mutual_information"] = 100 * auroc_values(scores=mutual_information(outputs=outputs), binary_label=binary_label)
    res_dict["pr_mutual_information"] = 100 * precision_score(scores=mutual_information(outputs=outputs), binary_label=binary_label)
    res_dict["rec_mutual_information"] = 100 * recall_score(scores=mutual_information(outputs=outputs), binary_label=binary_label)
    
    # entropy metric
    res_dict["roc_entropy"] = 100 * auroc_values(scores=entropy(outputs=outputs), binary_label=binary_label)
    res_dict["pr_entropy"] = 100 * precision_score(scores=entropy(outputs=outputs), binary_label=binary_label)
    res_dict["rec_entropy"] = 100 * recall_score(scores=entropy(outputs=outputs), binary_label=binary_label)

    return res_dict