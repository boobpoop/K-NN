import numpy as np
import operator as op

def load_data():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = np.array(['A', 'A', 'B', 'B'])
    return (group, labels)

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
    
if __name__ == "__main__":
    x, y = load_data()
    prediction = knn_classify([0.2, 0.4], x, y, 1)
    print(prediction)
