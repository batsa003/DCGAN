import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt

import sys

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/");

# takes an input tensor x_image, returns a scalar tensor
def discriminator(x_image, reuse = False):
    # Reusing the variable for this scope
    if (reuse):
        tf.get_variable_scope().reuse_variables()
    
    # First layer: Use 32 filters of 5x5. 28 x 28 x 1 -> 14 x 14 x 32.
    d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
    d1 = tf.nn.conv2d(input=x_image, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
    d1 = d1 + d_b1
    d1 = tf.nn.relu(d1)
    d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second layer: 14 x 14 x 32 -> 7 x 7 x 64
    d_w2 = tf.get_variable('d_w2', shape = [5, 5, 32, 64], initializer = tf.truncated_normal_initializer(stddev = 0.02))
    d_b2 = tf.get_variable('d_b2', shape = [64], initializer = tf.constant_initializer(0))
    d2 = tf.nn.conv2d(input = d1, filter = d_w2, strides = [1,1,1,1], padding = 'SAME')
    d2 = d2 + d_b2
    d2 = tf.nn.relu(d2)
    d2 = tf.nn.avg_pool(d2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # FC layer 1: 
    d_w3 = tf.get_variable('d_w3', shape = [7*7*64, 1024], initializer = tf.truncated_normal_initializer(stddev = 0.02))
    d_b3 = tf.get_variable('d_b3', shape = [1024], initializer = tf.constant_initializer(0))
    d3 = tf.reshape(d2, shape = [-1, 7*7*64])
    d3 = tf.matmul(d3, d_w3)
    d3 = d3 + d_b3
    d3 = tf.nn.relu(d3)

    # FC layer 2:
    d_w4 = tf.get_variable('d_w4', shape = [1024, 1], initializer = tf.truncated_normal_initializer(stddev = 0.02))
    d_b4 = tf.get_variable('d_b4', shape = [1], initializer = tf.constant_initializer(0))

    d4 = tf.matmul(d3, d_w4) + d_b4

    return d4

# Takes a batch_size, z_dim and generates tensor of shape [batch_size x 28 x 28 x 1]
# z_dim is the latent space dimensionality (maybe 100)
def generator(batch_size, z_dim):
    z = tf.truncated_normal([batch_size, z_dim], mean=0, stddev=1, name='z')
    #first deconv block. z_dim -> [-1, 56, 56, 1]
    g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, g_w1) + g_b1
    g1 = tf.reshape(g1, [-1, 56, 56, 1])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tf.nn.relu(g1)

    # Generate z_dim/2 = 50 features.  [-1, 56, 56, 1] -> [-1,56,56,z_dim/2]
    g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, 56, 56)

    # Generate z_dim/2 = 25 features. [-1, 56, 56, z_dim/2] -> [-1, 56, 56, z_dim/4]
    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, 56, 56)

    # Final convolution with one output channel [-1, 56, 56, z_dim/4] -> [-1, 28, 28, 1]
    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)

    # Dimensions of g4: batch_size x 28 x 28 x 1
    return g4

#with tf.Session() as sess:

batch_size = 50
z_dimensions = 100

# feed image to D
x_placeholder = tf.placeholder('float', shape = [None, 28, 28, 1], name = 'placeholder')

Gz = generator(batch_size, z_dimensions) #(g(z))
Dx = discriminator(x_placeholder) #d(x) - probs of real images
Dg = discriminator(Gz, reuse = True) #d(g(z)) - probs of generated images

# The goal of G is to make Dg close to 1. Use cross_entropy as loss function for all
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, targets=tf.ones_like(Dg)))

# Dx should be close to 1 and Dg should be close to 0
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, targets=tf.fill([batch_size, 1], 0.9)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, targets=tf.zeros_like(Dg)))
d_loss = d_loss_real + d_loss_fake

tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name] # discriminator variables
g_vars = [var for var in tvars if 'g_' in var.name] # generator variables

d_trainer_fake = tf.train.AdamOptimizer(0.0001).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.0001).minimize(d_loss_real, var_list=d_vars)
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

#Outputs a Summary protocol buffer containing a single scalar value.
tf.scalar_summary('Generator_loss', g_loss)
tf.scalar_summary('Discriminator_loss_real', d_loss_real)
tf.scalar_summary('Discriminator_loss_fake', d_loss_fake)
d_real_count_ph = tf.placeholder(tf.float32)
d_fake_count_ph = tf.placeholder(tf.float32)
g_count_ph = tf.placeholder(tf.float32)

tf.scalar_summary('d_real_count', d_real_count_ph)
tf.scalar_summary('d_fake_count', d_fake_count_ph)
tf.scalar_summary('g_count', g_count_ph)

# Sanity check to see how the discriminator evaluates
# generated and real MNIST images
d_on_generated = tf.reduce_mean(discriminator(generator(batch_size, z_dimensions)))
d_on_real = tf.reduce_mean(discriminator(x_placeholder))

tf.scalar_summary('d_on_generated_eval', d_on_generated)
tf.scalar_summary('d_on_real_eval', d_on_real)

images_for_tensorboard = generator(batch_size, z_dimensions)
tf.image_summary('Generated_images', images_for_tensorboard, 10)
merged = tf.merge_all_summaries()
logdir = "/tmp/gan/"
saver = tf.train.Saver()

#During every iteration, update to generator and discriminator:
#sess = tf.Session()
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    writer = tf.train.SummaryWriter(logdir, sess.graph)

    gLoss = 0
    dLossFake, dLossReal = 1, 1
    d_real_count, d_fake_count, g_count = 0, 0, 0
    for i in range(20000):
        real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
        # Some hacks that work for some reason
        if dLossFake > 0.6:
            # Train discriminator on generated images
            _, dLossReal, dLossFake, gLoss = sess.run([d_trainer_fake, d_loss_real, d_loss_fake, g_loss],
                                                        {x_placeholder: real_image_batch})
            d_fake_count += 1

        if gLoss > 0.5:
            # Train the generator
            _, dLossReal, dLossFake, gLoss = sess.run([g_trainer, d_loss_real, d_loss_fake, g_loss],
                                                        {x_placeholder: real_image_batch})
            g_count += 1

        if dLossReal > 0.45:
            # If the discriminator classifies real images as fake,
            # train discriminator on real values
            _, dLossReal, dLossFake, gLoss = sess.run([d_trainer_real, d_loss_real, d_loss_fake, g_loss],
                                                        {x_placeholder: real_image_batch})
            d_real_count += 1

        if i % 10 == 0:
            real_image_batch = mnist.validation.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
            summary = sess.run(merged, {x_placeholder: real_image_batch, d_real_count_ph: d_real_count,
                                        d_fake_count_ph: d_fake_count, g_count_ph: g_count})
            writer.add_summary(summary, i)
            d_real_count, d_fake_count, g_count = 0, 0, 0

        if i % 500 == 0:
            # Periodically display a sample image in the notebook
            # (These are also being sent to TensorBoard every 10 iterations)
            images = sess.run(generator(3, z_dimensions))
            d_result = sess.run(discriminator(x_placeholder), {x_placeholder: images})
            print("TRAINING STEP", i, "AT", datetime.datetime.now())
            for j in range(3):
                print("Discriminator classification", d_result[j])
#                im = images[j, :, :, 0]
#                plt.imshow(im.reshape([28, 28]), cmap='Greys')
#                plt.show()

        # A lot of issues with tensorflow version 0.10
#        if i % 5 == 0:
#            save_path = saver.save(sess = sess, save_path = "models/pretrained_gan.ckpt", global_step=i)
#            save_path = saver.save(sess, "models/pretrained_gan.ckpt")
#            print("saved to %s" % save_path)
        
#test_images = sess.run(generator(10, 100))
#test_eval = sess.run(discriminator(x_placeholder), {x_placeholder: test_images})
#
#real_images = mnist.validation.next_batch(10)[0].reshape([10, 28, 28, 1])
#real_eval = sess.run(discriminator(x_placeholder), {x_placeholder: real_images})
#
## Show discriminator's probabilities for the generated images,
## and display the images
#for i in range(10):
#    print(test_eval[i])
#    plt.imshow(test_images[i, :, :, 0], cmap='Greys')
#    plt.show()
#
## Now do the same for real MNIST images
#for i in range(10):
#    print(real_eval[i])
#    plt.imshow(real_images[i, :, :, 0], cmap='Greys')
#    plt.show()
