import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

class Net(nn.Module):
    def __init__(self):
       super(Net, self).__init__()
       self.fc1 = nn.Linear(4, 6)
       self.fc2 = nn.Linear(6, 3)
       self.fc3 = nn.Linear(3, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MyDataset(Dataset):

    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

def show_neural_network_graph(layers=(4, 6, 3, 3)):
    """
    Отрисовывает "схематичный" граф сети (7 -> 6 -> 3 -> 1),
    где каждый нейрон соединён со всеми нейронами следующего слоя.

    layers — кортеж, описывающий число нейронов в каждом слое.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Граф топологии сети: {}-{}-{}-{}".format(*layers), fontsize=14)
    ax.set_aspect('equal')

    plt.xlim(-1, len(layers))
    plt.ylim(-1, max(layers) + 1)

    neuron_positions = []

    for i, n_neurons in enumerate(layers):

        y_positions = np.linspace(0, n_neurons - 1, n_neurons)

        max_neurons = max(layers)
        offset = (max_neurons - n_neurons) / 2

        layer_coords = []
        for y in y_positions:
            x_coord = i
            y_coord = y + offset
            layer_coords.append((x_coord, y_coord))
        neuron_positions.append(layer_coords)

    for layer_idx in range(len(layers) - 1):
        current_layer = neuron_positions[layer_idx]
        next_layer = neuron_positions[layer_idx + 1]

        for (x1, y1) in current_layer:
            for (x2, y2) in next_layer:
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3)

    for layer_idx, layer_coords in enumerate(neuron_positions):
        for j, (x, y) in enumerate(layer_coords):
            circle = plt.Circle((x, y), radius=0.2, fill=True, color='skyblue', ec='black')
            ax.add_patch(circle)
            ax.text(x, y, f"{j}", ha='center', va='center', fontsize=8)

    plt.axis('off')
    plt.show()

def sh_table(max_loss, avg_loss, acuracy):
    columns_name = ["Топология НС", "Максимальная ошибка (train/test)", "Средняя ошбка (train/test)", "% Распознавания (train/test)"]
    fig, ax = plt.subplots(figsize=(10, 8))
    cell_data = [["4-4-6-3",max_loss, avg_loss, acuracy]]
    ax.set_axis_off()
    table = ax.table(
        colLabels=columns_name,
        cellText=cell_data,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)

    ax.set_title('Результаты', fontsize=14, fontweight='bold')
    plt.show()

def create_nn(learning_rate=0.05, epochs=80, batch_sise=16, log_interval=1):

    df = pd.read_csv('train_data/Iris.csv', delimiter='\t')
    df["Класс"] = df["Класс"].map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
    X_train = df.drop(["Класс"], axis=1).values.astype('float32')
    Y_train = df["Класс"].values.astype('int64')

    df = pd.read_csv('train_data/Iris_test.csv', delimiter=', ')
    df["Класс"] = df["Класс"].map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
    print(df)
    X_test = df.drop(["Класс"], axis=1).values.astype('float32')
    Y_test = df["Класс"].values.astype('int64')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_data = MyDataset(X_train_scaled, Y_train)
    test_data = MyDataset(X_test_scaled, Y_test)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_sise, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_sise, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    '''
    CrossEntropyLoss уже содержит softmax-функцию (которая превращает наборы чисел в вероятности). Обе используются для многоклассовой классификации (то есть кол-во всевозможных меток > 2).
    Еще немного про softmax-функцию, ее используют для того, чтобы нейросеть определяла наиболее вероятный класс из всех потенциальных классов 
    Например, в прошлом примере 2-мя возможными метками были "человек" и "бот"
    там мы использовали sigmoid (изменящийся в пределах [0,1] int), и соответствующую функцию потерь
    которая этот sigmoid употребляла. То есть в этом примере мы давали НС возможность
    сделать вывод либо 1, либо 0, на основании признаков.
    Здесь же каждый объект из тестовых данных будет иметь ту или иную вероятнось
    отношения ко все классам, а сам класс определяется по максимальной из этих
    вероятностей
    '''
    print("Результаты на обучающей выборке...")

    train_losses_by_epoch = []
    all_train_losses = []
    total_train_correct = 0
    total_train_samples = 0

    for epoch in range(epochs):
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_samples = 0
        for batch_idx, (data_in, target) in enumerate(train_loader):
            data_in, target = data_in.to(device), target.to(device)

            optimizer.zero_grad()
            net_out = net(data_in)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_loss_sum += loss_val
            all_train_losses.append(loss_val)
            preds = torch.argmax(net_out, dim=1)
            correct_tensor = preds.eq(target).sum().item()
            epoch_correct += correct_tensor
            epoch_samples += len(data_in)

            if batch_idx % log_interval == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.6f}')

        avg_epoch_loss = epoch_loss_sum / len(train_loader)
        train_losses_by_epoch.append(avg_epoch_loss)

        total_train_correct += epoch_correct
        total_train_samples += epoch_samples
        print(f'Epoch {epoch}: Avg Loss: {epoch_loss_sum / len(train_loader):.6f}')

    avg_loss_train = sum(all_train_losses) / len(all_train_losses) if all_train_losses else 0
    max_loss_train = max(all_train_losses) if all_train_losses else 0
    accuracy_train = 100.0 * total_train_correct / total_train_samples
    print(f"Train set: Average loss: {avg_loss_train:.4f}, "
          f"Max loss: {max_loss_train:.4f}, "
          f"Accuracy: {accuracy_train:.2f}%")

    print("\nРезультаты на тестовой выборке")
    all_preds = []
    all_labels = []
    all_test_losses = []

    with torch.no_grad():
        for data_in, target in test_loader:
            data_in, target = data_in.to(device), target.to(device)
            net_out = net(data_in)
            loss = criterion(net_out, target)
            all_test_losses.append(loss.item())


            preds = torch.argmax(net_out, dim=1)


            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_labels.extend(target.cpu().numpy().flatten().tolist())
    print(all_test_losses)
    avg_loss_test = sum(all_test_losses) / len(all_test_losses) if all_test_losses else 0
    max_loss_test = max(all_test_losses) if all_test_losses else 0
    correct_test = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    accuracy_test = 100.0 * correct_test / len(all_labels) if len(all_labels) else 0

    print(f"Test set: Average loss: {avg_loss_test:.4f}")
    print(f"Test set: Max loss: {max_loss_test:.4f}")
    print(f"Accuracy: {correct_test}/{len(all_labels)} ({accuracy_test:.2f}%)")

    final_max_loss = f"{round(max_loss_train, 4)} / {round(max_loss_test, 4)}"
    final_avg_loss = f"{round(avg_loss_train, 4)} / {round(avg_loss_test, 4)}"
    final_accuracy = f"{round(accuracy_train, 2)} / {round(accuracy_test, 2)}"
    sh_table(final_max_loss, final_avg_loss, final_accuracy)
    show_neural_network_graph()
if __name__ == '__main__':
    create_nn()