import torch
import torchvision
import torchvision.transforms as transforms
from model import *
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import random


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def save_model(model):
    PATH = './cifar_net.pth'
    torch.save(model.state_dict(), PATH)
    return PATH


def ex2_loss(criterion, lamda):
    return lambda predicted_labels, true_labels, recon_imgs, true_imgs: criterion(predicted_labels,
                                                                                  true_labels) + lamda * F.mse_loss(
        recon_imgs, true_imgs, reduction='sum')


def train(ex_name, train_loader, optimizer, criterion, model, device, num_of_epochs):
    train_loss_list = []
    for epoch in range(num_of_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        running_loss_for_graph = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            if ex_name == "EX1":
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            elif ex_name == "EX2":
                outputs, recon_inputs = model(inputs)

                mse_loss = nn.MSELoss()
                recon_inputs = recon_inputs.float()  # Convert boolean tensor to float tensor
                inputs = inputs.float()  # Convert boolean tensor to float tensor
                mse_loss = mse_loss(recon_inputs, inputs)
                cross_loss = criterion(outputs, labels)
                loss = cross_loss + 2 * mse_loss

                # loss = criterion(outputs, labels, recon_inputs, inputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        if epoch % 5 == 0 : test_reconstruction(classes, model, test_loader, "Epoch " + str(epoch))
        train_loss_list.append(running_loss_for_graph / len(train_loader))
    print('Finished Training')
    return train_loss_list


def load_data(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    return train_set, test_set, train_loader, test_loader


def define_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_images_and_labels_from_loader(loader):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    return images.to(device), labels.to(device)


def plot_images(images, labels, batch_size):
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


def plot_model_predictions(model, images, classes):
    with torch.no_grad():
        model.eval()
        outputs, _ = model(images)
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                      for j in range(4)))


def test_model(device, test_loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs, _ = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    return correct, total

def plot_train_loss(train_loss_list, num_epochs):
    plt.plot(range(1, num_epochs + 1), train_loss_list, label='Train Loss')
    plt.title("Train loss as a function of Epochs")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc='upper right')  # Add legend to upper right corner
    plt.show()


def ex_1(device, classes, batch_size, model, criterion, optimizer, train_loader, test_loader, num_of_epochs):
    # Load and plot train data
    images, labels = get_images_and_labels_from_loader(train_loader)
    plot_images(images, labels, batch_size)
    # Train model
    train_loss_list = train("EX1", train_loader, optimizer, criterion, model, device, num_of_epochs)
    plot_train_loss(train_loss_list, num_of_epochs)
    saved_model_path = save_model(model)
    # Load and plot test data
    images, labels = get_images_and_labels_from_loader(test_loader)
    plot_images(images, labels, batch_size)
    # Load Model
    model = Net()
    model.to(device)
    model.load_state_dict(torch.load(saved_model_path))
    plot_model_predictions(model, images, classes)
    # since we're not training, we don't need to calculate the gradients for our outputs
    correct, total = test_model(device, test_loader, model)
    # prepare to count predictions for each class
    test_model_acc(classes, device, model, test_loader)



def plot_reconstructed_images(original_images, reconstructed_images, image_titles, title):
    # Create a 2x4 grid of subplots
    print(image_titles)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    # Plot the original images
    for i, ax in enumerate(axes[0]):
        original_images[i] = original_images[i] / 2 + 0.5   # unnormalize
        npimg = original_images[i].detach().cpu().numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(image_titles[i])

    # Plot the reconstructed images
    for i, ax in enumerate(axes[1]):
        min_value = reconstructed_images[i].min()
        max_value = reconstructed_images[i].max()
        normalized_image_tensor = torch.div(reconstructed_images[i] - min_value, max_value - min_value)
        # reconstructed_images[i] = reconstructed_images[i] / 2 + 0.5   # unnormalize
        npimg = normalized_image_tensor.detach().cpu().numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(image_titles[i + 4])

    # Set the main title
    fig.suptitle("Original Images vs Reconstructed Images\n" + title, fontsize=16)

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()


def ex_2(device, classes, batch_size, model, criterion, optimizer, train_loader, test_loader, num_of_epochs):
    # Load and plot train data
    images, labels = get_images_and_labels_from_loader(train_loader)
    # Train model
    train("EX2", train_loader, optimizer, criterion, model, device, num_of_epochs)

    # plot_images(recon_inputs, labels, batch_size)
    #
    saved_model_path = save_model(model)
    # # Load and plot test data
    images, labels = get_images_and_labels_from_loader(test_loader)
    plot_images(images, labels, batch_size)
    # Load Model
    model = Recon_Net()
    model.to(device)
    model.load_state_dict(torch.load(saved_model_path))
    plot_model_predictions(model, images, classes)
    # since we're not training, we don't need to calculate the gradients for our outputs
    correct, total = test_model(device, test_loader, model)
    # prepare to count predictions for each class
    test_model_acc(classes, device, model, test_loader)


def test_model_acc(classes, device, model, test_loader):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    # again no gradients needed
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs, _ = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


def test_reconstruction(classes, model, test_loader, title):
    with torch.no_grad():
        model.eval()
        images, labels = get_images_and_labels_from_loader(test_loader)
        labels = labels.tolist()
        labels = [str(classes[label]) for label in labels] + ["Recon " + str(classes[label]) for label in labels]
        _, recon_inputs = model(images)
        plot_reconstructed_images(images, recon_inputs, labels, title)


def plot_features(images, labels, title, zi):
    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 8 if zi == "2" else 3, figsize=(8, 8))
    # Iterate over the images and labels, and plot them in the grid
    for i, ax in enumerate(axes.flat):
        # Plot the image
        ax.imshow(images[i])
        ax.axis('off')

        # Set the label as the title of the subplot
        ax.set_title(labels[i])
    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.suptitle("Latent Features\n" + title)
    # Show the plot
    plt.show()


def plot_latent_features(model, image, zi, loader_type):
    feature_images = []
    labels = []
    for i in range(16 if zi == "2" else 6):
        _, recon_x = model(image, zi, i + 1)
        min_value = recon_x.min()
        max_value = recon_x.max()
        normalized_image_tensor = torch.div(recon_x - min_value, max_value - min_value)

        # recon_x = recon_x / 2 + 0.5
        npimg = normalized_image_tensor.squeeze().detach().cpu().numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        feature_images.append(npimg)
        labels.append("Feature #" + str(i))
    plot_features(images=feature_images, labels=labels, title= "Z" + zi+ ", " + loader_type + " Image", zi=zi)


def ex_3():
    model = Recon_Net()
    model.to(device)
    model.load_state_dict(torch.load('./cifar_net.pth'))
    # test_reconstruction(classes, model, test_loader, "Epoch 20")
    train_image, train_label = get_images_and_labels_from_loader(train_loader)
    test_image, test_label = get_images_and_labels_from_loader(test_loader)
    train_image = train_image[1].unsqueeze(0)
    test_image = test_image[3].unsqueeze(0)
    plot_images(train_image, train_label[3].unsqueeze(0), 1)
    plot_images(test_image, test_label[0].unsqueeze(0), 1)
    with torch.no_grad():
        model.eval()
        plot_latent_features(model = model, image=train_image, zi="1", loader_type="Train")
        plot_latent_features(model = model, image=test_image, zi="1", loader_type="Test")
        plot_latent_features(model = model, image=train_image, zi="2", loader_type="Train")
        plot_latent_features(model = model, image=test_image, zi="2", loader_type="Test")


if __name__ == '__main__':
    # Hyperparameters
    define_seed(1234)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    batch_size = 4
    num_of_epochs = 20
    model = Recon_Net()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # criterion = ex2_loss(criterion, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_set, test_set, train_loader, test_loader = load_data(batch_size)

    # EX1
    ex_1(device, classes, batch_size, model, criterion, optimizer, train_loader, test_loader, num_of_epochs)

    # EX2
    ex_2(device, classes, batch_size, model, criterion, optimizer, train_loader, test_loader, num_of_epochs)

    #EX3
    ex_3()


