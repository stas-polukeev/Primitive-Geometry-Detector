import torch
import torch.nn as nn

def yolo_loss(y_hat, y, lambda_cord = 5, lambda_noobj = 0.5):
    # y_hat - prediction, y - true label
    B = y_hat.shape[0]
    eps = 1e-10
    a, b, c = torch.where(y[:, 0, ...] == 1)
    a_n, b_n, c_n = torch.where(y[:, 0, ...] != 1)
    
    bbox_loss_xy = torch.sum((y_hat[a, 1 : 3, b, c] - y[a, 1 : 3, b, c]) ** 2)  #x, y loss
    bbox_loss_wh = (torch.sum((torch.sqrt(eps + y_hat[a, 3 : 5, b, c]) - torch.sqrt(eps + y[a, 3 : 5, b, c])) ** 2))
    object_confidence_loss = torch.sum((y_hat[a, 0 , b, c] - y[a, 0 , b, c]) ** 2)
    class_loss = torch.sum((y_hat[a, 5 : , b, c] - y[a, 5 : , b, c]) ** 2)
    
    no_object_confidence_loss = torch.sum((y_hat[a_n, 0 , b_n,  c_n] - y[a_n, 0 , b_n,  c_n]) ** 2)
    coord_loss = bbox_loss_wh + bbox_loss_xy
    
    loss = lambda_cord * coord_loss + lambda_noobj * no_object_confidence_loss + object_confidence_loss + object_confidence_loss

    return loss / B