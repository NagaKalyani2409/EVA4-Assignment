import torch

import torchvision
import torchvision.transforms as transforms
def test(net, device, testloader, criterion): 
    correct = 0
    total = 0
    epoch_test_loss = 0.0
    epoch_test_accuracy = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)

            epoch_test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_test_accuracy = (100 * correct / total)
    epoch_test_loss /= len(testloader)
    # print("Test loss=",epoch_test_loss)
    # print('Accuracy of the network on the 10000 test images: %d %%' %epoch_test_accuracy)
    print('Epoch Test loss:',epoch_test_loss)        
    print('Epoch Test Accuracy:',epoch_test_accuracy)
    return epoch_test_accuracy, epoch_test_loss
  
def train(net, device, trainloader, optimizer, criterion, epoch):
    net.train()  
    running_loss = 0.0
    epoch_train_loss = 0.0
    epoch_train_accuracy = 0
    correct = 0
    processed = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs. Data is not on cuda. Move it. Number of images, labels per iteration = batch size in transform
        inputs, labels = data
              
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad() 

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels) #Loss calculation is on cuda
        loss.backward()
        optimizer.step()

        #Accuracy calculation
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        processed += len(inputs)
        

        # print statistics
        running_loss += loss.item()
        epoch_train_loss += loss.item()
        if i % 300 == 299:    # print every 300 mini-batches
            print('[%d, %5d] loss: %.3f' % 
                  (epoch + 1, i + 1, running_loss / 300))
            running_loss = 0.0

    # print("Number of correct",correct,"\nTotal:",processed)    
    epoch_train_accuracy=(100*correct/processed)
    # print("Accuracy=",epoch_train_accuracy)
    epoch_train_loss /= i
    # print("Total loss for epoch: ",i, "is", epoch_train_loss)

    print('Epoch Train loss:',epoch_train_loss)
    print('Epoch Train Accuracy:',epoch_train_accuracy)    
    return epoch_train_accuracy, epoch_train_loss
  
def predict(model, device, test_loader):
	pred_all=[]
	model.eval()

	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)

			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			#incorrect_pred.append(pred.eq(target.view_as(pred)))
			pred_all +=list(pred.squeeze().cpu().numpy())

	return pred_all

def get_misclassified(pred,labels):
	misclassified = []
	correct = []
	for i in (range(len(pred))):
		if pred[i] != labels[i] : misclassified.append((i,pred[i],labels[i]))
		else : correct.append((i,pred[i],labels[i]))
	return correct,	misclassified