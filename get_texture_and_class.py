import numpy as np
import os
import sys
import tensorflow as tf
import scipy.io
import matplotlib.pyplot as plt
from datasets import imagenet
from nets import inception
from preprocessing import inception_preprocessing


############



def get_texture(first_layer, second_layer):
    results = []
    shape_1 = first_layer.shape
    shape_2 = second_layer.shape
    size = shape_1[0]
    for k in range(size):
        result = {}
        first_layer_mag = np.zeros((shape_1[1], shape_1[2]))
        first_layer_ind = np.zeros((shape_1[1], shape_1[2]))
        second_layer_mag = np.zeros((shape_2[1], shape_2[2]))
        second_layer_ind = np.zeros((shape_2[1], shape_2[2]))
        for i in range(shape_1[1]):
            for j in range(shape_1[2]):
                index = np.argmax(first_layer[k, i, j, :])
                first_layer_ind[i, j] = index
                first_layer_mag[i, j] = first_layer[k, i, j, index]
        for i in range(shape_2[1]):
            for j in range(shape_2[2]):
                index = np.argmax(second_layer[k, i, j, :])
                second_layer_ind[i, j] = index
                second_layer_mag[i, j] = second_layer[k, i, j, index]
        result['index_first_layer'] = first_layer_ind
        result['mag_first_layer'] = first_layer_mag
        result['index_second_layerd'] = second_layer_ind
        result['mag_second_layer'] = second_layer_mag
        results.append(result)
    return results


############
checkpoints_dir = '/tmp/checkpoints'
slim = tf.contrib.slim
image_size = inception.inception_v3.default_image_size

# real_A_dic = {}
# fake_B_dic = {}
# real_B_dic = {}
# fake_A_dic = {}
# fake_A_conv0 = {}
# fake_A_conv1 = {}

#####

image_class = 'photo2monet-class'
folder = 'test'
image_attrib = 'fake_B'

#####
class_set = ['horse2zebra', 'horse2zebra-extra', 'horse2zebra-good', 'horse2zebra-bad', 'apple2orange-good',
             'apple2orange-bad', 'photo2monet-class']
#####
orig_dir = ''
dest_dir = ''
orig_dic = {}

#####



if image_class == 'photo2monet':
    if folder == 'test':
        if image_attrib == 'real_A':
            orig_dir = '/Users/Martin/Desktop/CS280/finalproj/photo-to-monet/test/real_A/'
        elif image_attrib == 'real_B':
            orig_dir = '/Users/Martin/Desktop/CS280/finalproj/photo-to-monet/test/real_B/'
        elif image_attrib == 'fake_A':
            orig_dir = '/Users/Martin/Desktop/CS280/finalproj/photo-to-monet/test/fake_A/'
        elif image_attrib == 'fake_B':
            orig_dir = '/Users/Martin/Desktop/CS280/finalproj/photo-to-monet/test/fake_B/'
        else:
            print('wrong attrib')
            sys.exit()
        dest_dir = '/Users/Martin/Desktop/CS280/finalproj/photo-to-monet-dist/test/'+image_attrib+'/'
    else:
        print('wrong set!!!')
        sys.exit()

elif image_class == 'horse2zebra':
    if folder == 'train':
        if image_attrib == 'real_A':
            orig_dir = '/Users/Martin/Desktop/CS280/finalproj/horse-to-zebra-supplemental/train/real_A/'
        elif image_attrib == 'fake_B':
            orig_dir = '/Users/Martin/Desktop/CS280/finalproj/horse-to-zebra-supplemental/train/fake_B/'
        elif image_attrib == 'real_B':
            orig_dir = '/Users/Martin/Desktop/CS280/finalproj/horse-to-zebra-supplemental/train/real_B/'
        elif image_attrib == 'fake_A':
            orig_dir = '/Users/Martin/Desktop/CS280/finalproj/horse-to-zebra-supplemental/train/fake_A/'
        else:
            print('wrong attrib')
            sys.exit()
        dest_dir = '/Users/Martin/Desktop/CS280/finalproj/horse-to-zebra-dist/train/'
    elif folder == 'best':
        if image_attrib == 'real_A':
            orig_dir = '/Users/Martin/Desktop/CS280/finalproj/horse-to-zebra-supplemental/best/real_A/'
        elif image_attrib == 'fake_B':
            orig_dir = '/Users/Martin/Desktop/CS280/finalproj/horse-to-zebra-supplemental/best/fake_B/'
        elif image_attrib == 'real_B':
            orig_dir = '/Users/Martin/Desktop/CS280/finalproj/horse-to-zebra-supplemental/best/real_B/'
        elif image_attrib == 'fake_A':
            orig_dir = '/Users/Martin/Desktop/CS280/finalproj/horse-to-zebra-supplemental/best/fake_A/'
        else:
            print('wrong attrib')
            sys.exit()
        dest_dir = '/Users/Martin/Desktop/CS280/finalproj/horse-to-zebra-dist/best/'
    else:
        print('wrong set!!!')
        sys.exit()

elif image_class == 'horse2zebra-extra':

    if folder == 'best':
        if image_attrib == 'real_A':
            orig_dir = '/Users/Martin/Desktop/CS280/finalproj/horse-to-zebra-extra/best/real_A/'
        elif image_attrib == 'fake_B':
            orig_dir = '/Users/Martin/Desktop/CS280/finalproj/horse-to-zebra-extra/best/fake_B/'
        else:
            print('wrong attrib')
            sys.exit()
        dest_dir = '/Users/Martin/Desktop/CS280/finalproj/horse-to-zebra-extra-dist/best/'
    else:
        print('wrong set!!!')
        sys.exit()

elif image_class == 'horse2zebra-bad':
    orig_dir = '/Users/Martin/Desktop/CS280/finalproj/horse-to-zebra-bad/' + folder + '/' + image_attrib + '/'
    dest_dir = '/Users/Martin/Desktop/CS280/finalproj/horse-to-zebra-bad-dist/' + folder + '/'

elif image_class == 'horse2zebra-good':
    orig_dir = '/Users/Martin/Desktop/CS280/finalproj/horse-to-zebra-good/' + folder + '/' + image_attrib + '/'
    dest_dir = '/Users/Martin/Desktop/CS280/finalproj/horse-to-zebra-good-dist/' + folder + '/'

elif image_class == 'apple2orange-bad':
    orig_dir = '/Users/Martin/Desktop/CS280/finalproj/apple-to-orange-bad/' + folder + '/' + image_attrib + '/'
    dest_dir = '/Users/Martin/Desktop/CS280/finalproj/apple-to-orange-bad-dist/' + folder + '/'

elif image_class == 'apple2orange-good':
    orig_dir = '/Users/Martin/Desktop/CS280/finalproj/apple-to-orange-good/' + folder + '/' + image_attrib + '/'
    dest_dir = '/Users/Martin/Desktop/CS280/finalproj/apple-to-orange-good-dist/' + folder + '/'

elif image_class == 'photo2monet-class':
    orig_dir = '/Users/Martin/Desktop/CS280/finalproj/photo-to-monet/'+folder+'/'+image_attrib + '/'
    dest_dir = '/Users/Martin/Desktop/CS280/finalproj/photo-to-monet-dist/' + folder + '/'

elif image_class == 'apple2orange':
    if folder == 'train':
        orig_dir = '/Users/Martin/Desktop/CS280/finalproj/apple-to-orange-supplemental/train/'
        dest_dir = '/Users/Martin/Desktop/CS280/finalproj/apple-to-orange-dist/train/'
    elif folder == 'test':
        orig_dir = '/Users/Martin/Desktop/CS280/finalproj/apple-to-orange-supplemental/test/'
        dest_dir = '/Users/Martin/Desktop/CS280/finalproj/apple-to-orange-dist/test/'
    elif folder == 'best':
        orig_dir = '/Users/Martin/Desktop/CS280/finalproj/apple-to-orange-supplemental/best/'
        dest_dir = '/Users/Martin/Desktop/CS280/finalproj/apple-to-orange-dist/best/'
    else:
        print('wrong set!!!')
        sys.exit()

else:
    print('wrong class!!!')
    sys.exit()

orig_files = os.listdir(orig_dir)

if image_class == 'photo2monet':
    sequence = []
    texture = None
    with tf.Graph().as_default():
        processed_image_batch = []
        for files in orig_files:
            if files.split('.')[-1] != 'jpg' and files.split('.')[-1] != 'png':
                continue
            if files.split('.')[-1] == 'jpg':
                img_type = 'jpg'
            else:
                img_type = 'png'
            print(files)
            file_name = files[:-4]
            orig_dic[file_name] = []
            sequence.append(file_name)
            image_string = tf.constant(orig_dir + files)
            image_string = tf.read_file(image_string)
            if img_type == 'jpg':
                image = tf.image.decode_jpeg(image_string, channels=3)
            else:
                image = tf.image.decode_png(image_string, channels=3)
            processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
            processed_image_batch.append(processed_image)
        processed_image_batch = tf.stack(processed_image_batch)
        print(processed_image_batch.shape)
        # processed_images = tf.expand_dims(processed_image, 0)
        # Create the model, use the default arg scope to configure the batch norm parameters.

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, endpoints = inception.inception_v3(processed_image_batch, num_classes=1001, is_training=False)
            # print(endpoints['Conv2d_1a_3x3'])
            # print(endpoints['Conv2d_2a_3x3'])
            first_layer = endpoints['Conv2d_1a_3x3']
            second_layer = endpoints['Conv2d_2a_3x3']

            # fake_A_conv0[file_name] = endpoints['Conv2d_1a_3x3']
            # fake_A_conv1[file_name] = endpoints['Conv2d_2a_3x3']
        # probabilities = tf.nn.softmax(logits)

        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'inception_v3.ckpt'),
            slim.get_model_variables('InceptionV3'))

        with tf.Session() as sess:
            init_fn(sess)
            # probabilities = sess.run(probabilities)
            # print(probabilities.shape)
            first_layer = sess.run(first_layer)
            second_layer = sess.run(second_layer)
            print(first_layer)
            texture = get_texture(first_layer, second_layer)

        # for j in range(len(sequence)):
        #     orig_dic[sequence[j]] = probabilities[j, :]

    # with open(dest_dir + image_attrib + '.txt', 'w') as f:
    #     for key in orig_dic:
    #         f.write(key)
    #         for item in orig_dic[key]:
    #             f.write(',' + str(item))
    #         f.write('\n')
    for i, item in enumerate(texture):
        scipy.io.savemat(dest_dir + image_attrib + sequence[i] + '.mat', item)

if image_class in class_set:
    sequence = []
    with tf.Graph().as_default():
        processed_image_batch = []
        for files in orig_files:
            if files.split('.')[-1] != 'jpg' and files.split('.')[-1] != 'png':
                continue
            if files.split('.')[-1] == 'jpg':
                img_type = 'jpg'
            else:
                img_type = 'png'
            print(files)
            if image_class == 'photo2monet-class':
                file_name = files[:-4]
            else:
                file_name = files[:-4].split('_')[1]
            orig_dic[file_name] = []
            sequence.append(file_name)
            image_string = tf.constant(orig_dir + files)
            image_string = tf.read_file(image_string)
            if img_type == 'jpg':
                image = tf.image.decode_jpeg(image_string, channels=3)
            else:
                image = tf.image.decode_png(image_string, channels=3)
            processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
            processed_image_batch.append(processed_image)
        processed_image_batch = tf.stack(processed_image_batch)
        print(processed_image_batch.shape)
        # processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure the batch norm parameters.

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, endpoints = inception.inception_v3(processed_image_batch, num_classes=1001, is_training=False)
            # print(endpoints['Conv2d_1a_3x3'])
            # print(endpoints['Conv2d_2a_3x3'])

            # fake_A_conv0[file_name] = endpoints['Conv2d_1a_3x3']
            # fake_A_conv1[file_name] = endpoints['Conv2d_2a_3x3']
        probabilities = tf.nn.softmax(logits)

        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'inception_v3.ckpt'),
            slim.get_model_variables('InceptionV3'))

        with tf.Session() as sess:
            init_fn(sess)
            probabilities = sess.run(probabilities)
            print(probabilities.shape)

        for j in range(len(sequence)):
            orig_dic[sequence[j]] = probabilities[j, :]

    with open(dest_dir + image_attrib + '.txt', 'w') as f:
        for key in orig_dic:
            f.write(key)
            for item in orig_dic[key]:
                f.write(',' + str(item))
            f.write('\n')

elif image_class == 'apple2orange':
    sequence = []
    with tf.Graph().as_default():
        processed_image_batch = []
        for files in orig_files:
            print('haha')
            attrib = files[:-4].split('_')[4] + '_' + files[:-4].split('_')[5]
            if attrib != image_attrib:
                print('not the right one')
                continue
            file_name = files[:-4].split('_')[2]
            orig_dic[file_name] = []
            sequence.append(file_name)
            # image_string = tf.constant('/Users/Martin/Desktop/CS280/finalproj/horse-to-zebra-supplemental/best/fake_B/horse2zebra_16_50_fake_B.jpg')
            # reader = tf.WholeFileReader()
            # filename_queue = tf.train.string_input_producer(files_real_A)
            # key, value = reader.read(filename_queue)
            # my_img = tf.image.decode_jpeg(value, channels=3)
            #
            # image_string = tf.constant('/Users/Martin/Desktop/CS280/finalproj/apple-to-orange-supplemental/best/monet_johnson_710_latest_real_A.jpg')
            image_string = tf.constant(orig_dir + files)
            image_string = tf.read_file(image_string)
            image = tf.image.decode_jpeg(image_string, channels=3)
            processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
            processed_image_batch.append(processed_image)
        processed_image_batch = tf.stack(processed_image_batch)
        print(processed_image_batch.shape)
        # processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure the batch norm parameters.

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, endpoints = inception.inception_v3(processed_image_batch, num_classes=1001, is_training=False)
            # print(endpoints['Conv2d_1a_3x3'])
            # print(endpoints['Conv2d_2a_3x3'])

            # fake_A_conv0[file_name] = endpoints['Conv2d_1a_3x3']
            # fake_A_conv1[file_name] = endpoints['Conv2d_2a_3x3']
        probabilities = tf.nn.softmax(logits)

        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'inception_v3.ckpt'),
            slim.get_model_variables('InceptionV3'))

        with tf.Session() as sess:
            init_fn(sess)
            probabilities = sess.run(probabilities)
            # probabilities = probabilities[0, 0:]
            # sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]
            # sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

        # plt.figure()
        # plt.imshow(np_image.astype(np.uint8))
        # plt.axis('off')
        # plt.show()

        # names = imagenet.create_readable_names_for_imagenet_labels()
        for j in range(len(sequence)):
            orig_dic[sequence[j]] = probabilities[j, :]
            # for i in range(1001):
            # index = sorted_inds[i]
            # index = i
            # orig_dic[file_name].append(probabilities[index])
            # print('Probability %0.8f%% => [%s]' % (probabilities[index], names[index]))
            # f.write(str(probabilities[index])+','+names[index]+'\n')

    with open(dest_dir + image_attrib + '.txt', 'w') as f:
        for key in orig_dic:
            f.write(key)
            for item in orig_dic[key]:
                f.write(',' + str(item))
            f.write('\n')
