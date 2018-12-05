import numpy as np
import operator as op
import matplotlib
import matplotlib.pyplot as plt
DATA_FILE = "datingTestSet2.txt"

def data_normalize(train_data):
     data_max = train_data.max(0)
     data_min = train_data.min(0)
     data_range = data_max - data_min
     data_norm = train_data / data_range
     return data_norm, data_range

def load_data(path):
    with open(path, "r") as df:
        list_line = df.readlines()
        train_data = np.zeros((len(list_line), 3))
        labels = []
        index = 0
        for line in list_line:
            list_line_data = line.strip().split('\t')
            train_data[index, : ] = list_line_data[0:3]
            labels.append(int(list_line_data[-1]))
            index += 1
        return (train_data, labels)

def knn_classify(input_data, train_data, labels, k):
    train_data_num = train_data.shape[0]
    expand_input_data = np.tile(input_data, (train_data_num, 1))
    distance = ((expand_input_data - train_data) ** 2).sum(axis = 1) ** 0.5
    distance_sort_indies = distance.argsort()
    label_count = {}
    for step in range(k):
        label = labels[distance_sort_indies[step]]
        label_count[label] = label_count.get(label, 0) + 1
    list_label = sorted(label_count.items(), key = op.itemgetter(1), reverse = True)
    return list_label[0][0]

def process_data_labels(train_data, labels):
    data1 = []
    data2 = []
    data3 = []
    for step in range(len(labels)):
       if labels[step] == 1:
           data1.append(train_data[step])
       elif labels[step] == 2:
           data2.append(train_data[step])
       elif labels[step] == 3:
           data3.append(train_data[step])
    return np.array(data1), np.array(data2), np.array(data3)

def test_accuracy(path):
    train_data, labels = load_data(path)
    data_norm, data_range = data_normalize(train_data)
    error_count = 0.0
    for step in range(len(data_norm)):
        prediction = knn_classify(data_norm[step, :], data_norm, labels, 3)
        if prediction != labels[step]:
            error_count += 1
    error_rate = error_count / len(train_data)
    print("total error rate is : %f" %(error_rate))

def prediction(path):
    train_data, labels = load_data(path)
    data_norm, data_range = data_normalize(train_data)
    result_matrix = ['not at all', 'in small doses', 'in large doses']

    game_time = float(input("percentage of time spent playing video times: "))
    flying_miles = float(input("flying miles per miles: "))
    ice_cream = float(input("liters of ice cream consumed per year: ")) 
    new_date_man = [flying_miles, game_time, ice_cream] / data_range
    prediction = knn_classify(new_date_man, data_norm, labels, 3)
    
    print("you will probably like this person: ", result_matrix[prediction - 1])


def save_image(train_data, labels):
    data1, data2, data3 = process_data_labels(train_data, labels)
    plt.switch_backend("agg")
    fig = plt.figure(figsize = [19.2, 4.8])

    ax1 = fig.add_subplot(131)
    ax1.scatter(data1[:, 1], data1[:, 2], s = 30, c = "red", marker = "o", alpha = 0.5, label = "don'tLike")
    ax1.scatter(data2[:, 1], data2[:, 2], s = 30, c = "blue", marker = "*", alpha = 0.5, label = "smallDoes")
    ax1.scatter(data3[:, 1], data3[:, 2], s = 30, c = "yellow", marker = "+", alpha = 0.5, label = "largeDoes")
    ax1.set_xlabel("Game time percent")
    ax1.set_ylabel("Liter of icecream") 
    plt.legend()    
    
    ax2 = fig.add_subplot(132) 
    ax2.scatter(data1[:, 0], data1[:, 2], s = 30, c = "red", marker = "o", alpha = 0.5, label = "don'tLike")
    ax2.scatter(data2[:, 0], data2[:, 2], s = 30, c = "blue", marker = "*", alpha = 0.5, label = "smallDoes")
    ax2.scatter(data3[:, 0], data3[:, 2], s = 30, c = "yellow", marker = "+", alpha = 0.5, label = "largeDoes")
    ax2.set_xlabel("Miles of flying")
    ax2.set_ylabel("Liter of icecream")
    plt.legend()

    ax3 = fig.add_subplot(133)
    ax3.scatter(data1[:, 0], data1[:, 1], s = 30, c = "red", marker = "o", alpha = 0.5, label = "don'tLike")
    ax3.scatter(data2[:, 0], data2[:, 1], s = 30, c = "blue", marker = "*", alpha = 0.5, label = "smallDoes")
    ax3.scatter(data3[:, 0], data3[:, 1], s = 30, c = "yellow", marker = "+", alpha = 0.5, label = "largeDoes")
    ax3.set_xlabel("Miles of flying")
    ax3.set_ylabel("Game time percent")
    plt.legend()

    plt.savefig('data_visualize.png')
    plt.close()



if __name__ == "__main__":
    test_accuracy(DATA_FILE)
    prediction(DATA_FILE)
