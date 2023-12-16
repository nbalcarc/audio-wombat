import main

labels, frames = main.load_file("data_shuffled/train1_shuffled.csv")
w = main.Wombat(labels, frames, 1)
labels1, frames1 = main.load_file("data_shuffled/train2_shuffled.csv")
w.train(labels1, frames1)
labels2, frames2 = main.load_file("data_shuffled/test_shuffled.csv")
acc = 0

for label, frame in zip(labels2, frames2):
    if label in w.predict(frame):
        acc += 1

print(acc / len(labels2))


