import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from burnoutCNN import *

'''
Run this after ./data/normal/ and ./data/takedown have been successfully populated. 
This will generate a model file (.pth) at the end, don't forget to enter a name for it.
'''

print("Creating tensor and loading dataset.")

#normalize (R, G, B) values of pixels from [0-255] to [-1, 1], helps train the model better
transform_list = [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
transform = transforms.Compose(transform_list)
dataset = datasets.ImageFolder(DATA_FOLDER, transform=transform)
dataset.class_to_idx = {"normal": NORMAL_CLASSIFIER, "takedown": TAKEDOWN_CLASSIFIER} #so far we are just using binary classification, so video frames are just either normal gameplay or takedown camera

#split the data into training and validation
training_data_amount = int(len(dataset) * TRAINING_DATA_PERCENT)
validation_data_amount = len(dataset) - training_data_amount
training_dataset, validation_dataset = random_split(dataset, [training_data_amount, validation_data_amount]) #no seed so the loop can split data differently each time
training_data_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True) #dataloaders are of the form [(image tensor, label (actual value) tensor),..]
validation_data_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
#tensor example format (only 2 classifiers) : (-0.5, 0.8) -> takedown predicted to be more likely

print("Beginning training.")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model = BurnoutCNN().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
model.train()

for epoch in range(EPOCHS):
    current_epoch_loss = 0.0
    for images, labels in training_data_loader: #this loops one batch at a time
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        #feed this batch to the model and calculate how accurate the model is compared to the actual data's values
        model_output = model(images)
        loss = criterion(model_output, labels)

        #calculate how much the weights contributed to any errors (gradients -> delta loss over delta weight for every weight, positive/negative indicates correlation to use for next epoch)
        loss.backward()
        optimizer.step() #update weights using calculations and settings from optimizer

        current_epoch_loss += loss.item()

    print("Loss for epoch #" + str(epoch + 1) + ": " + str(current_epoch_loss / len(training_data_loader)))

print("Training complete! Now evaluating.")

model.eval()
correct_frames = 0
total_frames = 0

with torch.no_grad():
    for images, labels in validation_data_loader:
        #feed this batch and do the same thing to receive tensors 
        images, labels = images.to(device), labels.to(device)
        model_output = model(images)

        #compare the classes predictions to the actual values and add up the correct ones
        _, class_predictions = torch.max(model_output, 1) #this gets the most likely class for each frame in the batch
        total_frames += labels.size(0)
        for predicted_value, actual_value in zip(class_predictions, labels):
            if predicted_value.item() == actual_value.item():
                correct_frames += 1

print(f"Validation is " + str(100 * (correct_frames / total_frames)) + " percent correct.")

name = input("Enter name for .pth (enter nothing to not save): ")
if name:
    torch.save(model.state_dict(), name + ".pth")