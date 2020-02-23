#Author: AlexFang, alex.holla@foxmail.com.
import pandas as pd
import numpy as np
import time
from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import sys
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import random
from Alexnet import Network
import matplotlib.pyplot as plt
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#1---------------------
def generate_folder():
    if not os.path.exists("./image"):
        os.makedirs("./image")
    for i in os.listdir("./image"):
        if i=='tfrecord':
            continue
        os.remove(os.path.join("./image",i))
        
    if not os.path.exists("./image/tfrecord"):
        os.makedirs("./image/tfrecord")
    for i in os.listdir("./image/tfrecord"):
        os.remove(os.path.join("./image/tfrecord",i))


# def random_captcha_text(char_set=number,captcha_size=4):
def random_captcha_text(char_set=number, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)
    image.write(captcha_text, './image/' + captcha_text + '.png')  # write it


#2------------------
def generate():
    num = 10000
    for i in range(num):
        gen_captcha_text_and_image()
        sys.stdout.write('\r>>creating images %d/%d' % (i + 1, num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    print('All picture has been generated')


#3--------------------
def save_as_tf():
    _NUM_TEST = 500
    _RANDOM_SEED = 0
    DATASET_DIR = './image/'
    TFRECORD_DIR = './image/tfrecord/'


    def _dataset_exists(dataset_dir):
        for split_name in ['train', 'test']:
            output_filename = os.path.join(dataset_dir, split_name + 'tfrecords')
            if not tf.gfile.Exists(output_filename):
                return False
        return True


    def _get_filenames_and_classes(dataset_dir):
        photo_filenames = []
        for filename in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, filename)
            photo_filenames.append(path)
        return photo_filenames


    def int64_feature(values):
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


    def bytes_feature(values):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


    def image_to_tfexample(image_data, label0, label1, label2, label3):
        return tf.train.Example(features=tf.train.Features(feature={
            'image': bytes_feature(image_data),
            'label0': int64_feature(label0),
            'label1': int64_feature(label1),
            'label2': int64_feature(label2),
            'label3': int64_feature(label3),
        }))


    def _convert_dataset(split_name, filenames, dataset_dir):
        assert split_name in ['train', 'test']

        with tf.Session() as sess:
            output_filename = os.path.join(TFRECORD_DIR, split_name + '.tfrecords')
            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                for i, filename in enumerate(filenames):
                    try:
                        sys.stdout.write('\r>>changing picture %d / %d' % (i + 1, len(filenames)))
                        sys.stdout.flush()

                        image_data = Image.open(filename)
                        image_data = image_data.resize((224, 224))
                        image_data = np.array(image_data.convert('L'))
                        image_data = image_data.tobytes()

                        labels = filename.split('/')[-1][0:4]
                        num_labels = []
                        for j in range(4):
                            num_labels.append(int(labels[j]))

                        example = image_to_tfexample(image_data, num_labels[0], num_labels[1], num_labels[2],
                                                     num_labels[3])
                        tfrecord_writer.write(example.SerializeToString())

                    except IOError as e:
                        print('\n sth wrong:', filename)
                        print('Error:', e)
            sys.stdout.write('\n')  
            sys.stdout.flush() 


    if _dataset_exists(DATASET_DIR):
        print('file already exists')
    else:
        photo_filenames = _get_filenames_and_classes(DATASET_DIR)

        random.seed(_RANDOM_SEED)
        random.shuffle(photo_filenames)
        training_filenames = photo_filenames[_NUM_TEST:]
        testing_filenames = photo_filenames[:_NUM_TEST]

        _convert_dataset('train', training_filenames, DATASET_DIR)
        _convert_dataset('test', testing_filenames, DATASET_DIR)
        print('-------------We have produced all tfrecord file------------------')


#4-------------
def train():
    tf.reset_default_graph()
    CHAR_NUM = 10
    IMAGE_HEIGHT = 60
    IMAGE_WIDTH = 160
    BATCH_SIZE = 10
    TFRECORD_FILE = "./image/tfrecord/train.tfrecords"
    CHECKPOINT_DIR = './ckpt/'

    # placeholder
    x = tf.placeholder(tf.float32, [None, 224, 224])
    y0 = tf.placeholder(tf.float32, [None])
    y1 = tf.placeholder(tf.float32, [None])
    y2 = tf.placeholder(tf.float32, [None])
    y3 = tf.placeholder(tf.float32, [None])

    lr = tf.Variable(0.0003, dtype=tf.float32)

    def read_and_decode(filename):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={'image': tf.FixedLenFeature([], tf.string),
                                                                         'label0': tf.FixedLenFeature([], tf.int64),
                                                                         'label1': tf.FixedLenFeature([], tf.int64),
                                                                         'label2': tf.FixedLenFeature([], tf.int64),
                                                                         'label3': tf.FixedLenFeature([], tf.int64)
                                                                         })
        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.reshape(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        label0 = tf.cast(features['label0'], tf.int32)
        label1 = tf.cast(features['label1'], tf.int32)
        label2 = tf.cast(features['label2'], tf.int32)
        label3 = tf.cast(features['label3'], tf.int32)
        return image, label0, label1, label2, label3

    image, label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)

    image_batch, label_batch0, label_batch1, label_batch2, label_batch3 = tf.train.shuffle_batch(
        [image, label0, label1, label2, label3], batch_size=BATCH_SIZE, \
        capacity=1075, min_after_dequeue=1000, num_threads=128)

    network = Network(num_classes=CHAR_NUM,weight_decay=0.0005, is_training=True)

    # gpu_options = tf.GPUOptions(allow_growth=True)

    # with tf.Session(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True,gpu_options=gpu_options)) as sess:
    #     gpu_options = tf.GPUOptions(allow_growth=True)
    #     tf.Session(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True,gpu_options=gpu_options))

    with tf.Session() as sess:

        X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])

        logits0, logits1, logits2, logits3, end_pintos = network.construct(X)

        one_hot_labels0 = tf.one_hot(indices=tf.cast(y0, tf.int32), depth=CHAR_NUM)
        one_hot_labels1 = tf.one_hot(indices=tf.cast(y1, tf.int32), depth=CHAR_NUM)
        one_hot_labels2 = tf.one_hot(indices=tf.cast(y2, tf.int32), depth=CHAR_NUM)
        one_hot_labels3 = tf.one_hot(indices=tf.cast(y3, tf.int32), depth=CHAR_NUM)

        loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits0, labels=one_hot_labels0))
        loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=one_hot_labels1))
        loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=one_hot_labels2))
        loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits3, labels=one_hot_labels3))
        total_loss = (loss0 + loss1 + loss2 + loss3) / 4.0
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)

        correct_prediction0 = tf.equal(tf.argmax(one_hot_labels0, 1), tf.argmax(logits0, 1))
        accuracy0 = tf.reduce_mean(tf.cast(correct_prediction0, tf.float32))

        correct_prediction1 = tf.equal(tf.argmax(one_hot_labels1, 1), tf.argmax(logits1, 1))
        accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

        correct_prediction2 = tf.equal(tf.argmax(one_hot_labels2, 1), tf.argmax(logits2, 1))
        accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))

        correct_prediction3 = tf.equal(tf.argmax(one_hot_labels3, 1), tf.argmax(logits3, 1))
        accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, './ckpt/crack_captcha-10000.ckpt')
        # sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(10001):
            b_image, b_label0, b_label1, b_label2, b_label3 = sess.run(
                [image_batch, label_batch0, label_batch1, label_batch2, label_batch3])
            sess.run(optimizer, feed_dict={x: b_image, y0: b_label0, y1: b_label1, y2: b_label2, y3: b_label3})

            if i % 100 == 0:
                if i % 5000 == 0:
                    sess.run(tf.assign(lr, lr / 3))
                acc0, acc1, acc2, acc3, loss_ = sess.run([accuracy0, accuracy1, accuracy2, accuracy3, total_loss],
                                                         feed_dict={x: b_image, y0: b_label0,
                                                                    y1: b_label1,
                                                                    y2: b_label2,
                                                                    y3: b_label3})
                learning_rate = sess.run(lr)
                print("Iter: %d , Loss:%.3f , Accuracy:%.3f, %.3f, %.3f, %.3f  Learning_rate:%.7f" % (
                i, loss_, acc0, acc1, acc2, acc3, learning_rate))

                # if acc0 > 0.9 and acc1 > 0.9 and acc2 > 0.9 and acc3 > 0.9 :

                if i % 5000 == 0:
                    # saver.save(sess,'./ckpt/crack_captcha.ckpt', global_step=1)
                    saver.save(sess, CHECKPOINT_DIR + 'crack_captcha-' + str(i) + '.ckpt')
                    print("Save model %s------"%str(i))
                    continue
        coord.request_stop()
        coord.join(threads)




#5-------------
def test():
    CHAR_NUM = 10 # category
    IMAGE_HEIGHT = 60
    IMAGE_WIDTH = 160
    BATCH_SIZE = 1
    TFRECORD_FILE = "./image/tfrecord/test.tfrecords"

    x = tf.placeholder(tf.float32, [None, 224, 224])
    def read_and_decode(filename):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={'image': tf.FixedLenFeature([], tf.string),
                                                                         'label0': tf.FixedLenFeature([], tf.int64),
                                                                         'label1': tf.FixedLenFeature([], tf.int64),
                                                                         'label2': tf.FixedLenFeature([], tf.int64),
                                                                         'label3': tf.FixedLenFeature([], tf.int64)
                                                                         })
        image = tf.decode_raw(features['image'], tf.uint8)
        image_raw = tf.reshape(image, [224, 224])  #raw data

        image = tf.reshape(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0  #standardlize
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)

        label0 = tf.cast(features['label0'], tf.int32)
        label1 = tf.cast(features['label1'], tf.int32)
        label2 = tf.cast(features['label2'], tf.int32)
        label3 = tf.cast(features['label3'], tf.int32)
        return image, image_raw, label0, label1, label2, label3

    # get label
    image, image_raw, label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)
    # print(len(sess.run(image)))
    image_batch, image_raw_batch, label_batch0, label_batch1, label_batch2, label_batch3 = tf.train.shuffle_batch(
        [image, image_raw, label0, label1, label2, label3], \
        batch_size=BATCH_SIZE, \
        capacity=53, min_after_dequeue=50, \
        num_threads=1)

    network = Network(num_classes=CHAR_NUM, weight_decay=0.0005, is_training=True)
    gpu_options = tf.GPUOptions(allow_growth=True)
    # with tf.Session(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True,gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])

        logits0, logits1, logits2, logits3, end_pintos = network.construct(X)

        prediction0 = tf.reshape(logits0, [-1, CHAR_NUM])
        prediction0 = tf.argmax(prediction0, 1)

        prediction1 = tf.reshape(logits1, [-1, CHAR_NUM])
        prediction1 = tf.argmax(prediction1, 1)

        prediction2 = tf.reshape(logits2, [-1, CHAR_NUM])
        prediction2 = tf.argmax(prediction2, 1)

        prediction3 = tf.reshape(logits3, [-1, CHAR_NUM])
        prediction3 = tf.argmax(prediction3, 1)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, './ckpt/crack_captcha-10000.ckpt')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(5):
            b_image, b_image_raw, b_label0, b_label1, b_label2, b_label3 = sess.run([image_batch,
                                                                                     image_raw_batch,
                                                                                     label_batch0,
                                                                                     label_batch1,
                                                                                     label_batch2,
                                                                                     label_batch3])

            # img = np.array(b_image_raw[0],dtype=np.uint8)

            #[1,224,224]
            img = Image.fromarray(b_image_raw[0], 'L')
            '''
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            '''
            print('label:', b_label0, b_label1, b_label2, b_label3)

            label0, label1, label2, label3 = sess.run([prediction0, prediction1, prediction2, prediction3],
                                                      feed_dict={x: b_image})
            print('predict:', label0, label1, label2, label3)

        coord.request_stop()
        coord.join(threads)




if __name__ == '__main__':
    generate_folder()
    generate()
    save_as_tf()
    train()
    test()