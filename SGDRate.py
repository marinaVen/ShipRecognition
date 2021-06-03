import torch
import random
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

pic_height, pic_width = 224, 224
batch_size = 8

ratio = (1. * pic_width) / pic_height
train_dir = "train"
val_dir = "validation"

class_names = ['DDG Bristol (GB)', 'DDG Duquesne (FR)', 'DDG Indomptable (FR)', 'DDG Roosevelt (US)', 'DDG York (GB)',
               'DDG Zumwalt (US)', 'FFG Aquitaine (DEN)', 'FFG Chatham (GB)', 'FFG Oliver Hazard Perry (US)',
               'FFG Thetis (DEN)', 'FFG Vaedderen (DEN) 2016', 'Pr 11661']

trans = [
    transforms.Grayscale(1),
    transforms.RandomResizedCrop((pic_height, pic_width), scale=(0.4, 1.0), ratio=(ratio, ratio)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]

train_transforms = transforms.Compose(trans)
val_transforms = transforms.Compose(trans)

train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transforms)
val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transforms)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)

act = torch.nn.LeakyReLU()


class AlexNet3(torch.nn.Module):
    def __init__(self, act):
        super(AlexNet3, self).__init__()
        self.batch_norm0 = torch.nn.BatchNorm2d(1)

        self.conv1 = torch.nn.Conv2d(1, 96, 7, stride=3, padding=1)
        self.act1 = act
        self.batch_norm1 = torch.nn.BatchNorm2d(96)
        self.pool1 = torch.nn.MaxPool2d(3, 2)

        self.conv2 = torch.nn.Conv2d(96, 256, 5, stride=2, padding=2)
        self.act2 = act
        self.batch_norm2 = torch.nn.BatchNorm2d(256)
        self.pool2 = torch.nn.MaxPool2d(3, 2)

        self.conv3 = torch.nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.act3 = act
        self.batch_norm3 = torch.nn.BatchNorm2d(128)
        self.pool3 = torch.nn.MaxPool2d(3, 2)

        self.fc1 = torch.nn.Linear(1152, 1024)
        self.act4 = act
        self.batch_norm4 = torch.nn.BatchNorm1d(1024)

        self.fc2 = torch.nn.Linear(1024, 1024)
        self.act5 = act
        self.batch_norm5 = torch.nn.BatchNorm1d(1024)

        self.fc3 = torch.nn.Linear(1024, 12)

    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.batch_norm1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.batch_norm2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.act3(x)
        x = self.batch_norm3(x)
        x = self.pool3(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc1(x)
        x = self.act4(x)
        x = self.batch_norm4(x)
        x = self.fc2(x)
        x = self.act5(x)
        x = self.batch_norm5(x)
        x = self.fc3(x)

        return x


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
loss = torch.nn.CrossEntropyLoss()


def train_model(model, loss, optimizer, scheduler, num_epochs):
    train_loss = []
    train_accuracy = []
    validation_loss = []
    validation_accuracy = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                dataloader = val_dataloader
                model.eval()  # Set model to evaluate mode

            running_loss = 0.
            running_acc = 0.

            # Iterate over data.
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    loss_value = loss(preds, labels)
                    preds_class = preds.argmax(dim=1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                # statistics
                running_loss += loss_value.item()
                running_acc += (preds_class == labels.data).float().mean().item()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_accuracy.append(epoch_acc)
            else:
                validation_loss.append(epoch_loss)
                validation_accuracy.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)

    return validation_loss, validation_accuracy, train_loss, train_accuracy


fig0, ax0 = plt.subplots()
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

for learningrate in [5.0e-4, 1.0e-3, 5.0e-3, 1.0e-2]:
    net = AlexNet3(act)
    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learningrate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    val_loss, val_acc, train_loss, train_acc = train_model(net, loss, optimizer, scheduler, num_epochs=25)

    ax1.plot(val_loss, label=str(learningrate))
    ax0.plot(val_acc, label=str(learningrate))
    ax2.plot(train_loss, label=str(learningrate))
    ax3.plot(train_acc, label=str(learningrate))

ax1.legend()
ax1.set_title('Функция потерь')
fig1.show()
fig1.savefig("{} Validation Loss".format(str(learningrate)))

ax0.legend()
ax0.set_title('Точность')
fig0.show()
fig0.savefig("{} Validation Accuracy".format(str(learningrate)))

ax2.legend()
ax2.set_title('Train Loss')
fig2.show()
fig2.savefig("{} Train Loss".format(str(learningrate)))

ax3.legend()
ax3.set_title('Train Accuracy')
fig3.show()
fig3.savefig("{} Train Accuracy".format(str(learningrate)))
