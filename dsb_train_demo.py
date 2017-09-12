import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
import warnings
import sys

IMG_SIZE_PX = 50
SLICE_COUNT = 20

TRAINING_DATA_DIR = 'demo-data-50-50-20.npy'
# TEST_DATA_DIR = 'lucas-test-data-50-50-20.npy'

# SAVED_MODEL_DIR = '/home/andy/LucasHu/models/lucas-model-5-xavier.ckpt'
# PREDICTIONS_FILE_DIR = 'id_probabilities.npy'

n_classes = 2
batch_size = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

sess = tf.Session()

KEEP_RATE = 0.8

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')


#                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
weights = {'W_conv1':tf.get_variable('W_conv1',shape=[3,3,3,1,32],initializer=tf.contrib.layers.xavier_initializer()),
           #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
           'W_conv2':tf.get_variable('W_conv2',shape=[3,3,3,32,64],initializer=tf.contrib.layers.xavier_initializer()),
           #                                  64 features
           'W_fc':tf.get_variable('W_fc', shape=[54080,1024], initializer=tf.contrib.layers.xavier_initializer()),
           'out':tf.get_variable('out', shape=[1024, n_classes], initializer=tf.contrib.layers.xavier_initializer())}

biases = {'b_conv1':tf.Variable(tf.zeros([32])),
           'b_conv2':tf.Variable(tf.zeros([64])),
           'b_fc':tf.Variable(tf.zeros([1024])),
           'out':tf.Variable(tf.zeros([n_classes]))}

saver = tf.train.Saver()


# Forward pass through the CNN
def convolutional_neural_network(x, keep_rate=KEEP_RATE):
    

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)


    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2,[-1, 54080])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']
    # output = tf.nn.softmax(output)

    return output

print("\n\n\nProgram initialized, now loading lung scans...")


# Load in data
much_data = np.load(TRAINING_DATA_DIR, encoding='latin1')

train_data = much_data

validation_data = much_data[-2:]

# test_data = np.load(TEST_DATA_DIR, encoding='latin1')



print("Done loading data! Starting the training process.\n")

# For training the CNN on training scans
def train_neural_network(x):
    samples_analyzed = 0

    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    test_predict = tf.nn.softmax(convolutional_neural_network(x, keep_rate=1.))

    # saver = tf.train.Saver()
    
    hm_epochs = 5
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        successful_runs = 0
        total_runs = 0
        
        # Get MPL figure ready
        plt.ion()
        fig = plt.figure()
        fig.canvas.set_window_title('CAIS++ Lung Cancer Demo (Data Science Bowl)')
        warnings.filterwarnings("ignore",".*GUI is implemented.*")

        for epoch in range(hm_epochs):

            sample_counter = 0
            
            epoch_loss = 0
            for data in train_data:
                total_runs += 1

                try:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    successful_runs += 1

                    sample_counter += 1

                    # print("Samples analyzed: ", samples_analyzed)
                    # visualize lung slices
                    
                    # if epoch == 0 and sample_counter == 1:
                    #     title = fig.suptitle('Epoch: ' + str(epoch) + ', Sample: ' \
                    #         + str(sample_counter) + ', Label: ' + str(Y))
                    # else:
                    #     title.set_text('Epoch: ' + str(epoch) + ', Sample: ' \
                    #         + str(sample_counter) + ', Label: ' + str(Y))

                    if Y[0] == 0:
                        label = 'NOT Cancer'
                    else:
                        label = 'CANCER'

                    fig.suptitle('Epoch: ' + str(epoch) + ', Sample: ' + str(sample_counter) + '\n' + \
                        'Label: ' + label + ', Cost (Error): ' + str(c))


                    for num,each_slice in enumerate(X):
                        thisisasubplot = fig.add_subplot(4,5,num+1)
                        thisisasubplot.imshow(each_slice, cmap='gray')

                    fig.subplots_adjust(top=0.85)
                    fig.canvas.draw()
                    plt.pause(1)
                    fig.clf()


                except Exception as e:
                    # I am passing for the sake of notebook space, but we are getting 1 shaping issue from one 
                    # input tensor. Not sure why, will have to look into it. Guessing it's
                    # one of the depths that doesn't come to 20.
                    pass
                    print(str(e))
            
            print('\n', 'Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
        # print('Prediction: ', prediction.eval({x:[i[0] for i in train_data]}))
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
            print('\n')

            # saver.save(sess, SAVED_MODEL_DIR)
        
            
        print('Done. Finishing accuracy:')
        print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
        
        print('fitment percent:',successful_runs/total_runs)

        # saver = tf.train.Saver()

        # saver.save(sess, SAVED_MODEL_DIR)
        


        # id_probabilities = []
        # for data in test_data:
        #     X = data[0]
        #     patient_id = data[1]
        #     pred = sess.run(test_predict, feed_dict={x: X})
        #     prob = pred[0][1]
        #     print(prob)
        #     id_probabilities.append([patient_id, prob])

        # np.save(ID_PROBATILITIES_FILE, id_probabilities)




# For making cancer/not-cancer predictions on the test data (not included in repo)
def predict_test_data(x):
    prediction = convolutional_neural_network(x, keep_rate = 1)
    probabilities = tf.nn.softmax(prediction)
    
    with tf.Session() as sess:
        #tf.reset_default_graph()
        
        

        saver.restore(sess,SAVED_MODEL_DIR)
        test_data = np.load(TEST_DATA_DIR)

        id_probabilities = []
        for data in test_data:
            X = data[0]
            patient_id = data[1]
            
        pred = prediction.eval(feed_dict={x: X})[0]
        probs = probabilities.eval(feed_dict={x: X})[0]
            # pred = prediction.eval(feed_dict={x: X, keep_rate: 1.})
            # print('Outputs: ',pred)
        print('Prediction: ', pred)
        print('Probs: ',probs)
        id_probabilities.append([patient_id, probs])
        # print(sol)

    # out_file = open('id_probs_xavier_5.csv', 'w')
    # file_writer = csv.writer(out_file)
    # file_writer.writerow(['id','cancer'])
    # for patient in id_probabilities:
    #     id = patient[0]
    #     prob = patient[1][1]
    #     file_writer.writerow([id, prob])

        # np.save(PREDICTIONS_FILE_DIR, id_probabilities_array)


# # try saving model
# cnn_output = convolutional_neural_network(x)
# tf.add_to_collection("cnn_output", cnn_output)
# saver = tf.train.Saver()
# saver.save(sess, 'team2-model')

# Run this locally:
# train_neural_network(x)

train_neural_network(x)


# test_data = np.load(TEST_DATA_DIR)
# for patient in test_data:
#     patient_id = patient[1]
#     img_data = patient[0]
#     img_shape = img_data.shape

#     img_data_to_predict = img_data.reshape(1, (img_data.shape))
