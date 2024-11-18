def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    
    return loss

def accuracy_Expval(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)
    return loss

def cost_squareloss(circuit,weights, features, labels): 
    loss = square_loss
    predictions = [circuit(weights, f) for f in features]
    return loss(labels,predictions)

def learnedmeasure_squareloss(circuit,weights,bias,features,labels):
    loss = square_loss
    predictions=[]
    for f in features:
        outcome=circuit(weights,f)
        predictions.append(bias[0]*outcome[0]+bias[1]*outcome[1]+bias[2]*outcome[2]+bias[3]*outcome[3])
    return loss(labels,predictions)

def biased_onemeasured_squareloss(circuit,weights,bias,features,labels):
    loss = square_loss
    predictions=[]
    for f in features:
        outcome=circuit(weights,f)
        predictions.append(bias[0]*outcome[0]+bias[1]*outcome[1])
    return loss(labels,predictions)