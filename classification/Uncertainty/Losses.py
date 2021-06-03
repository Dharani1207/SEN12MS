import torch

ce_loss = torch.nn.CrossEntropyLoss()

# input:    raw network output (logits) with two heads
#           the ouputs from the different parts are stacked along axis 0
def evidential_fusion_cross_entropy(output, target):

    output = torch.exp(output)
    output1, output2 = torch.squeeze(torch.split(output, 2, dim=0))
    input = torch.log_softmax(output1 + output2)
    loss_value = ce_loss(input, target)

    # Believe regulation could be activate to enfore the network to learn equivalent magnitudes of believe values
    #believe_reg = torch.mean(((output1 - output2)/(output1 + output2))**2)

    return loss_value #* (1 + believe_reg)


# input:    raw network output (logits) with two heads
#           the ouputs from the different parts are stacked along axis 0
def decision_fusion_cross_entropy(output, target):

    output1, output2 = torch.squeeze(torch.split(output, 2, dim=0))
    input = 0.5 * (torch.softmax(output1) + torch.softmax(output2))

    loss_value = ce_loss(input, target)

    return loss_value


