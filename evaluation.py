import torch

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum((SR == GT).float())
    tensor_size = float(SR.numel())
    acc = float(corr) / tensor_size

    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative

    TP = (SR & GT).float()
    FN = ((~SR) & GT).float()

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)
    
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((~SR) & (~GT)).float()
    FP = (SR & (~GT)).float()

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = (SR & GT).float()
    FP = (SR & (~GT)).float()

    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)
    
    Inter = torch.sum((SR & GT).float())
    Union = torch.sum((SR | GT).float())

    JS = float(Inter) / (float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR & GT).float())
    DC = float(2 * Inter) / (float(torch.sum(SR).float() + torch.sum(GT).float()) + 1e-6)

    return DC



