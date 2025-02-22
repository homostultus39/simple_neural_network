import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

class Net(nn.Module):
    def __init__(self):
       super(Net, self).__init__()
       self.fc1 = nn.Linear(7, 6)
       self.fc2 = nn.Linear(6, 3)
       self.fc3 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MyDataset(Dataset):

    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

def to_df(data):
    df = pd.DataFrame(data)
    df = df.replace({"Да": 1, "Нет": 0, "Не_ясно": 2})
    df["Результат"] = df["Результат"].map({"Человек": 1, "Бот": 0})
    # X - признаки (исходный массив данных, разделенный по отличительным особенностям)
    # Y - метки (предмет/вещь, которая предсказывается, в нашем случае - это значения "Человек" или "Бот")

    X = df.drop(["Результат"], axis=1).values
    Y = df["Результат"].values

    return X, Y

def show_train_loss_chart(train_losses_by_epoch):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses_by_epoch) + 1),
             train_losses_by_epoch, marker='o', label='Train Loss')
    plt.title('Динамика ошибки на обучении')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def show_neural_network_graph(layers=(7, 6, 3, 1)):
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

def show_test_scatter_plot(all_labels, all_preds):
    """
    Диаграмма рассеяния (scatter plot): реальный класс vs предсказанный класс.
    """
    plt.figure(figsize=(8, 5))
    x_points = range(len(all_labels))
    # Реальные метки (синие кружки)
    plt.scatter(x_points, all_labels, c='b', marker='o', label='Реальные метки')
    # Предсказания (красные крестики)
    plt.scatter(x_points, all_preds, c='r', marker='x', label='Предсказания')
    plt.title('Диаграмма рассеяния: предсказания vs реальные метки')
    plt.xlabel('Индекс образца')
    plt.ylabel('Класс (0=Бот, 1=Человек)')
    plt.legend()
    plt.show()

def show_confusion_matrix(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Бот', 'Человек'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Матрица ошибок (Confusion Matrix)")
    plt.show()

def sh_table(max_loss, avg_loss, acuracy):
    columns_name = ["Топология НС", "Максимальная ошибка (train/test)", "Средняя ошбка (train/test)", "% Распознавания (train/test)"]
    fig, ax = plt.subplots(figsize=(10, 8))
    cell_data = [["7-6-3-1",max_loss, avg_loss, acuracy]]
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

def create_nn(learning_rate=0.02, epochs=50, batch_sise=4, log_interval=1):
    '''
    :param learning_rate: скорость обучения (параметр, отражающий, насколько сильно меняет веса нейронка во время обучения)
    :param epochs: количество эпох обучения (не всегда, но в моем случае, с маленьким набором признаков и меток, чем больше эпох - тем лучше резульат на train_data
    :param batch_sise: размер батча (батч - это набор исходных данных некоторой длины, наборы батчей используются в эпоху обучения)
    :param log_interval: интервал "времени", через которое мы выводим информацию об обучении в эпохе
    :return:
    '''
    source_data = {
        "Неформальное обращение": ["Да", "Да", "Нет", "Да", "Нет", "Да", "Нет", "Нет", "Нет", "Нет", "Нет", "Да", "Да", "Нет", "Нет", "Да", "Нет", "Нет", "Да", "Да", "Нет", "Да", "Да", "Нет", "Да", "Нет", "Нет", "Нет", "Нет", "Нет", "Да"],
        "Грамотность": ["Нет", "Да", "Да", "Нет", "Да", "Да", "Да", "Да", "Да", "Нет", "Да", "Нет", "Да", "Да", "Нет", "Нет", "Нет", "Да", "Нет", "Да", "Да", "Нет", "Да", "Да", "Нет", "Да", "Не_ясно", "Да", "Нет", "Да", "Нет"],
        "Ненормативная лексика": ["Да", "Да", "Нет", "Да", "Нет", "Нет", "Нет", "Да", "Да", "Нет", "Да", "Да", "Нет", "Нет", "Да", "Да", "Нет", "Нет", "Нет", "Нет", "Нет", "Да", "Не_ясно", "Нет", "Да", "Нет", "Да", "Да", "Нет", "Да", "Да"],
        "Отсылки к прошлому": ["Да", "Да", "Нет", "Нет", "Нет", "Нет", "Нет", "Нет", "Нет", "Нет", "Да", "Да", "Да", "Нет", "Нет", "Да", "Да", "Да", "Нет", "Да", "Нет", "Нет", "Да", "Нет", "Да", "Не_ясно", "Нет", "Нет", "Нет", "Да", "Не_ясно"],
        "Использование смайликов": ["Нет", "Да", "Да", "Да", "Нет", "Да", "Нет", "Да", "Да", "Да", "Да", "Нет", "Да", "Да", "Нет", "Нет", "Да", "Нет", "Да", "Нет", "Да", "Да", "Нет", "Да", "Нет", "Нет", "Да", "Да", "Не_ясно", "Да", "Нет"],
        "Оскорбления": ["Нет", "Нет", "Нет", "Да", "Нет", "Нет", "Нет", "Да", "Нет", "Нет", "Нет", "Да", "Нет", "Нет", "Да", "Да", "Нет", "Нет", "Нет", "Нет", "Нет", "Да", "Нет", "Не_ясно", "Да", "Нет", "Да", "Нет", "Нет", "Нет", "Да"],
        "Юмор": ["Да", "Да", "Нет", "Да", "Нет", "Нет", "Нет", "Нет", "Да", "Нет", "Да", "Да", "Нет", "Нет", "Нет", "Да", "Да", "Да", "Нет", "Да", "Нет", "Да", "Да", "Нет", "Да", "Не_ясно", "Нет", "Да", "Нет", "Да", "Да"],
        "Результат": ["Человек", "Человек", "Бот", "Человек", "Бот", "Бот", "Бот", "Человек", "Бот", "Бот", "Человек", "Человек", "Бот", "Бот", "Человек", "Человек", "Человек", "Бот", "Бот", "Человек", "Бот", "Человек", "Человек", "Бот", "Человек", "Бот", "Человек", "Бот", "Бот", "Человек", "Человек"]
    }
    test_data = {
    "Неформальное обращение": ["Нет", "Нет", "Нет", "Нет", "Нет", "Нет"],
    "Грамотность": ["Не_ясно", "Да", "Да", "Да", "Не_ясно", "Да"],
    "Ненормативная лексика": ["Не_ясно", "Да", "Нет", "Нет", "Нет", "Да"],
    "Отсылки к прошлому": ["Нет", "Не_ясно", "Да", "Нет", "Нет", "Да"],
    "Использование смайликов": ["Не_ясно", "Не_ясно", "Нет", "Не_ясно", "Нет", "Да"],
    "Оскорбления": ["Да", "Нет", "Не_ясно", "Нет", "Да", "Да"],
    "Юмор": ["Нет", "Да", "Не_ясно", "Нет", "Не_ясно", "Нет"],
    "Результат": ["Бот", "Человек", "Бот", "Бот", "Бот", "Человек"]
    }
    X_source, Y_source = to_df(source_data)
    X_test, Y_test = to_df(test_data)

    training_data = MyDataset(X_source, Y_source)
    test_data = MyDataset(X_test, Y_test)

    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_sise, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_sise, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

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
            target = target.unsqueeze(1)

            optimizer.zero_grad()
            net_out = net(data_in)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_loss_sum += loss_val
            all_train_losses.append(loss_val)

            preds = (torch.sigmoid(net_out) >= 0.5).float()
            correct_tensor = preds.eq(target).float().sum().item()
            epoch_correct += correct_tensor
            epoch_samples += len(data_in)

            if batch_idx % log_interval == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss_val:.6f}')

        avg_epoch_loss = epoch_loss_sum / len(train_loader)
        train_losses_by_epoch.append(avg_epoch_loss)

        total_train_correct += epoch_correct
        total_train_samples += epoch_samples

    avg_loss_train = sum(all_train_losses) / len(all_train_losses) if all_train_losses else 0
    max_loss_train = max(all_train_losses) if all_train_losses else 0
    accuracy_train = 100.0 * total_train_correct / total_train_samples
    print(f"Train set: Average loss: {avg_loss_train:.4f}, "
          f"Max loss: {max_loss_train:.4f}, "
          f"Accuracy: {accuracy_train:.2f}%")

    show_train_loss_chart(train_losses_by_epoch)

    print("\nРезультаты на тестовой выборке")

    all_test_losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data_in, target in test_loader:
            data_in, target = data_in.to(device), target.to(device)
            target = target.unsqueeze(1)
            net_out = net(data_in)
            loss = criterion(net_out, target)
            all_test_losses.append(loss.item())

            probs = torch.sigmoid(net_out)
            preds = (probs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy().flatten().tolist())
            all_labels.extend(target.cpu().numpy().flatten().tolist())

    avg_loss_test = sum(all_test_losses) / len(all_test_losses) if all_test_losses else 0
    max_loss_test = max(all_test_losses) if all_test_losses else 0

    correct_test = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    accuracy_test = 100.0 * correct_test / len(all_labels) if len(all_labels) else 0

    print(f"Test set: Average loss: {avg_loss_test:.4f}")
    print(f"Test set: Max loss: {max_loss_test:.4f}")
    print(f"Accuracy: {correct_test}/{len(all_labels)} ({accuracy_test:.2f}%)")

    show_test_scatter_plot(all_labels, all_preds)
    show_confusion_matrix(all_labels, all_preds)

    final_max_loss = f"{round(max_loss_train, 4)} / {round(max_loss_test, 4)}"
    final_avg_loss = f"{round(avg_loss_train, 4)} / {round(avg_loss_test, 4)}"
    final_accuracy = f"{round(accuracy_train, 2)} / {round(accuracy_test, 2)}"
    sh_table(final_max_loss, final_avg_loss, final_accuracy)
    show_neural_network_graph()
if __name__=="__main__":
    create_nn()