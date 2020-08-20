import numpy as np
import pandas as pd
from pandas_ml import ConfusionMatrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple
import sys

# task = str(sys.argv[1])
# embeddingPath = str(sys.argv[2])
# preprocessedDatafile = str(sys.argv[3])
task = "test"
embeddingPath = "numberbatch-en.txt"
preprocessedDatafile = "preProcessed_data.csv"

preprocessedData = pd.read_csv(preprocessedDatafile)

hyperparameters = {
    'layersCnt': 2,
    'batchSize': 64,
    'epochs': 5,
    'hiddenUnits': 64,
    'dropoutProb': 0.8,
    'dimEmbeddings': 300,
    'validationData': 0.2,
    'learningRateDecay': 0.95,
    'learningRate': 0.005,
    'checkUpdatesize': 500
}


embeddingDict = dict()
embeddingFile = open(embeddingPath, encoding='utf-8')
for f in embeddingFile:
    data = f.split(' ')
    embeddingDict[data[0]] = np.asarray(data[1:], dtype='float32')


textData = preprocessedData.text

wordCnt = {}
for line in textData:
    words = line.split()
    for w in words:
        if w in wordCnt:
            wordCnt[w] += 1
        else:
            wordCnt[w] = 1

cnt = 0
indexes = dict()

for w in wordCnt:
    if wordCnt[w] >= 20 or w in embeddingDict:
        indexes[w] = cnt
        cnt += 1

indexes["<unk>"] = cnt
indexes["<pad>"] = cnt+1

embeddingMatrix = np.zeros((len(indexes), hyperparameters['dimEmbeddings']), dtype=np.float32)

for w in indexes:
    if w in embeddingDict:
        embeddingMatrix[cnt] = embeddingDict[w]
    else:
        tempembedding = np.array(np.random.uniform(-1.0, 1.0, hyperparameters['dimEmbeddings']))
        embeddingDict[w] = tempembedding
        embeddingMatrix[cnt] = tempembedding

print(embeddingMatrix)

numSequence = []

for l in textData:
    words = l.split()
    s = []
    for w in words:
        if w in indexes:
            s.append(indexes[w])
        else:
            s.append(indexes["<unk>"])
    numSequence.append(s)


categories = 6

tf.reset_default_graph()
train_graph = tf.Graph()

input_data = tf.placeholder(tf.int32, [None, None], name='input')
labels = tf.placeholder(tf.int32, [None, None], name='labels')
lr = tf.placeholder(tf.float32, name='learning_rate')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

with train_graph.as_default():
    with tf.name_scope("inputs"):

        weight = tf.Variable(
            tf.truncated_normal([hyperparameters['hiddenUnits'], categories], stddev=(1 / np.sqrt(hyperparameters['hiddenUnits'] * categories))))
        bias = tf.Variable(tf.constant(0.1, shape=[categories]))

    embeddings = embeddingMatrix
    embs = tf.nn.embedding_lookup(embeddings, input_data)

    with tf.name_scope("RNN_Layers"):
        stacked_rnn = []
        for layer in range(hyperparameters['layersCnt']):
            cell_fw = tf.contrib.rnn.GRUCell(hyperparameters['hiddenUnits'])
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                    output_keep_prob=keep_prob)
            stacked_rnn.append(cell_fw)
        multilayer_cell = tf.contrib.rnn.MultiRNNCell(stacked_rnn, state_is_tuple=True)

    with tf.name_scope("init_state"):
        initial_state = multilayer_cell.zero_state(hyperparameters['batchSize'], tf.float32)

    with tf.name_scope("Forward_Pass"):
        output, final_state = tf.nn.dynamic_rnn(multilayer_cell,
                                                embs,
                                                dtype=tf.float32)

    with tf.name_scope("Predictions"):
        last = output[:, -1, :]
        predictions = tf.exp(tf.matmul(last, weight) + bias)
        tf.summary.histogram('predictions', predictions)

    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=labels))
        tf.summary.scalar('cost', cost)

    # Optimizer
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

    # Predictions comes out as 6 output layer, so need to "change" to one hot
    with tf.name_scope("accuracy"):
        correctPred = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    export_nodes = ['input_data', 'labels', 'keep_prob', 'lr', 'initial_state', 'final_state',
                    'accuracy', 'predictions', 'cost', 'optimizer', 'merged']

    merged = tf.summary.merge_all()

rating = preprocessedData.stars.values.astype(int)
categorizedRating = tf.keras.utils.to_categorical(rating)

trainData, testData, trainRatings, testRatings = train_test_split(numSequence, categorizedRating, test_size=0.2, random_state=9)

print("Graph is built.")
graph_location = "./graph"

Graph = namedtuple('train_graph', export_nodes)
local_dict = locals()
graph = Graph(*[local_dict[each] for each in export_nodes])

print(graph_location)
trainModelWriter = tf.summary.FileWriter(graph_location)
trainModelWriter.add_graph(train_graph)

savepoint = "./saves/best_model2.ckpt"


def getBatches(traind, trainl, batchSize):

    for i in range(0, len(traind)//batchSize):
        start = i * batchSize
        end = start + batchSize
        pad_batch_x = np.asarray(getPadding(traind[start:end], indexes))
        yield pad_batch_x, trainl[start:end]


def getPadding(batch):
    temp = []
    for b in batch:
        temp.append(len(b))
    maxLength = max(temp)
    return tf.keras.preprocessing.sequence.pad_sequences(batch,
                                                             maxlen=maxLength,
                                                             padding='post',
                                                             value=indexes['<pad>'])


def getBatchesTest(testd, batchSize):
    for i in range(0, len(testd)//batchSize):
        start = i * batchSize
        end = start + batchSize
        pad_batch_x = np.asarray(getPadding(testd[start:end], indexes))
        yield pad_batch_x


if task == "Train":

    resultSummary = []

    min_learning_rate, stop_early, stop = 0.0005, 0, 3

    session = tf.Session(graph=train_graph)
    session.run(tf.global_variables_initializer())

    trainModelWriter = tf.summary.FileWriter('trainSummary', session.graph)

    for e in range(1, hyperparameters['epochs'] + 1):
        state = session.run(graph.initial_state)
        updatedloss = 0

        for batch, (labelx, labely) in enumerate(getBatches(trainData, trainRatings, hyperparameters['batchSize'])):
            feed = {graph.input_data: labelx,
                    graph.labels: labely,
                    graph.keep_prob: keep_prob,
                    graph.initial_state: state,
                    graph.lr: learning_rate}

            summarytrain, loss, accura, state, _ = session.run([graph.merged,
                                                     graph.cost,
                                                     graph.accuracy,
                                                     graph.final_state,
                                                     graph.optimizer],
                                                    feed_dict=feed)

            trainModelWriter.add_summary(summarytrain, e * batch+ batch)

            updatedloss += loss

            if batch % hyperparameters['checkUpdatesize'] == 0 and batch > 0:
                #print("Average loss for this update:", round(update_loss / hyperparameters['checkUpdatesize'], 3))
                resultSummary.append(update_loss)

                # If the update loss is at a new minimum, save the model
                if update_loss <= min(resultSummary):
                    #print('New Record!')
                    stop_early = 0
                    saver = tf.train.Saver()
                    saver.save(session, savepoint)

                else:
                    #print("No Improvement.")
                    stop_early += 1
                    if stop_early == stop:
                        break
                update_loss = 0

                learning_rate *= hyperparameters['learningRate']
                if learning_rate < min_learning_rate:
                    learning_rate = min_learning_rate
                if stop_early == stop:
                    break

        session.close()

finalPredictions = []

saver = tf.train.Saver()
saver.restore(session, savepoint)
state = session.run(graph.initial_state)

for batch, labelx in enumerate(getBatches(testData, hyperparameters['batchSize']), 1):
    feedDict = {graph.input_data: labelx,
            graph.keep_prob: hyperparameters['dropoutProb'],
            graph.initial_state: state}

    getPredictions = session.run(graph.predictions, feed_dict = feedDict)

    for i in range(len(getPredictions)):
        finalPredictions.append(getPredictions[i, :])


finalPredictions = np.asarray(finalPredictions)
predictions = np.argmax(finalPredictions, axis=1)
actualPredictions = testRatings.argmax(axis=1)[:predictions.shape[0]]

cm = ConfusionMatrix(actualPredictions, predictions)
cm.plot(backend='seaborn', normalized=True)
plt.title('Confusion Matrix Stars prediction')
plt.figure(figsize=(12, 10))

test_correct_pred = np.equal(predictions, testRatings)
test_accuracy = np.mean(test_correct_pred.astype(float))

print("Test accuracy is: " + str(test_accuracy))