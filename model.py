import os
import random

from tqdm import trange
from scipy.misc import imsave
import tensorflow as tf
import numpy as np

from generator import Generator
from discriminator import Discriminator
from utils import logger


class HistoryQueue(object):
    def __init__(self, shape=[128,128,3], size=50):
        self.size = size
        self.shape = shape
        self.count = 0
        self.queue = []

    def query(self, image):
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        if self.size == 0:
            return image
        if self.count < self.size:
            self.count += 1
            self.queue.append(image)
            return image

        p = random.random()
        if p > 0.5:
            idx = random.randrange(0, self.size)
            ret = self.queue[idx]
            self.queue[idx] = image
            return ret
        else:
            return image


class CycleGAN(object):
    def __init__(self, args):
        self.log_step = args.log_step
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.cycle_loss_coeff = args.cycle_loss_coeff

        self.augment_size = self.image_size + (30 if self.image_size == 256 else 15)
        self.image_shape = [self.image_size, self.image_size, 3]

        self.is_train = tf.placeholder(tf.bool, name='is_train')
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)

        image_a = self.image_a = \
            tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='image_a')
        image_b = self.image_b = \
            tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='image_b')
        history_fake_a = self.history_fake_a = \
            tf.placeholder(tf.float32, [None] + self.image_shape, name='history_fake_a')
        history_fake_b = self.history_fake_b = \
            tf.placeholder(tf.float32, [None] + self.image_shape, name='history_fake_b')

        # Data augmentation
        def augment_image(image):
            image = tf.image.resize_images(image, [self.augment_size, self.augment_size])
            image = tf.random_crop(image, [self.batch_size] + self.image_shape)
            image = tf.map_fn(tf.image.random_flip_left_right, image)
            return image

        image_a = tf.cond(self.is_train,
                          lambda: augment_image(image_a),
                          lambda: image_a)
        image_b = tf.cond(self.is_train,
                          lambda: augment_image(image_b),
                          lambda: image_b)

        # Generator
        G_ab = Generator('G_ab', is_train=self.is_train,
                         norm='instance', activation='relu', image_size=self.image_size)
        G_ba = Generator('G_ba', is_train=self.is_train,
                         norm='instance', activation='relu', image_size=self.image_size)

        # Discriminator
        D_a = Discriminator('D_a', is_train=self.is_train,
                            norm='instance', activation='leaky')
        D_b = Discriminator('D_b', is_train=self.is_train,
                            norm='instance', activation='leaky')

        # Generate images (a->b->a and b->a->b)
        image_ab = self.image_ab = G_ab(image_a)
        image_aba = self.image_aba = G_ba(image_ab)
        image_ba = self.image_ba = G_ba(image_b)
        image_bab = self.image_bab = G_ab(image_ba)

        # Discriminate real/fake images
        D_real_a = D_a(image_a)
        D_fake_a = D_a(image_ba)
        D_real_b = D_b(image_b)
        D_fake_b = D_b(image_ab)
        D_history_fake_a = D_a(history_fake_a)
        D_history_fake_b = D_b(history_fake_b)

        # Least squre loss for GAN discriminator
        loss_D_a = (tf.reduce_mean(tf.squared_difference(D_real_a, 0.9)) +
            tf.reduce_mean(tf.square(D_history_fake_a))) * 0.5
        loss_D_b = (tf.reduce_mean(tf.squared_difference(D_real_b, 0.9)) +
            tf.reduce_mean(tf.square(D_history_fake_b))) * 0.5

        # Least squre loss for GAN generator
        loss_G_ab = tf.reduce_mean(tf.squared_difference(D_fake_b, 0.9))
        loss_G_ba = tf.reduce_mean(tf.squared_difference(D_fake_a, 0.9))

        # L1 norm for reconstruction error
        loss_rec_aba = tf.reduce_mean(tf.abs(image_a - image_aba))
        loss_rec_bab = tf.reduce_mean(tf.abs(image_b - image_bab))
        loss_cycle = args.cycle_loss_coeff * (loss_rec_aba + loss_rec_bab)

        loss_G_ab_final = loss_G_ab + loss_cycle
        loss_G_ba_final = loss_G_ba + loss_cycle

        # Optimizer
        self.optimizer_D_a = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
                            .minimize(loss_D_a, var_list=D_a.var_list, global_step=self.global_step)
        self.optimizer_D_b = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
                            .minimize(loss_D_b, var_list=D_b.var_list)
        self.optimizer_G_ab = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
                            .minimize(loss_G_ab_final, var_list=G_ab.var_list)
        self.optimizer_G_ba = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
                            .minimize(loss_G_ba_final, var_list=G_ba.var_list)

        # Summaries
        self.loss_D_a = loss_D_a
        self.loss_D_b = loss_D_b
        self.loss_G_ab = loss_G_ab
        self.loss_G_ba = loss_G_ba
        self.loss_cycle = loss_cycle

        tf.summary.scalar('loss/dis_A', loss_D_a)
        tf.summary.scalar('loss/dis_B', loss_D_b)
        tf.summary.scalar('loss/gen_AB', loss_G_ab)
        tf.summary.scalar('loss/gen_BA', loss_G_ba)
        tf.summary.scalar('loss/cycle', loss_cycle)
        tf.summary.scalar('model/D_a_real', tf.reduce_mean(D_real_a))
        tf.summary.scalar('model/D_a_fake', tf.reduce_mean(D_fake_a))
        tf.summary.scalar('model/D_b_real', tf.reduce_mean(D_real_b))
        tf.summary.scalar('model/D_b_fake', tf.reduce_mean(D_fake_b))
        tf.summary.scalar('model/lr', self.lr)
        tf.summary.image('A/A', image_a[0:1])
        tf.summary.image('A/A-B', image_ab[0:1])
        tf.summary.image('A/A-B-A', image_aba[0:1])
        tf.summary.image('B/B', image_b[0:1])
        tf.summary.image('B/B-A', image_ba[0:1])
        tf.summary.image('B/B-A-B', image_bab[0:1])
        self.summary_op = tf.summary.merge_all()

    def train(self, sess, summary_writer, data_A, data_B):
        logger.info('Start training.')
        logger.info('  {} images from A'.format(len(data_A)))
        logger.info('  {} images from B'.format(len(data_B)))

        data_size = min(len(data_A), len(data_B))
        num_batch = data_size // self.batch_size
        epoch_length = num_batch * self.batch_size

        num_initial_iter = 100
        num_decay_iter = 100
        lr = lr_initial = 0.0002
        lr_decay = lr_initial / num_decay_iter

        history_a = HistoryQueue(shape=self.image_shape, size=50)
        history_b = HistoryQueue(shape=self.image_shape, size=50)

        initial_step = sess.run(self.global_step)
        num_global_step = (num_initial_iter + num_decay_iter) * epoch_length
        t = trange(initial_step, num_global_step,
                   total=num_global_step, initial=initial_step)

        for step in t:
            #TODO: resume training with global_step
            epoch = step // epoch_length
            iter = step % epoch_length

            if epoch > num_initial_iter:
                lr = max(0.0, lr_initial - (epoch - num_initial_iter) * lr_decay)

            if iter == 0:
                random.shuffle(data_A)
                random.shuffle(data_B)

            image_a = np.stack(data_A[iter*self.batch_size:(iter+1)*self.batch_size])
            image_b = np.stack(data_B[iter*self.batch_size:(iter+1)*self.batch_size])
            fake_a, fake_b = sess.run([self.image_ba, self.image_ab],
                                      feed_dict={self.image_a: image_a,
                                                 self.image_b: image_b,
                                                 self.is_train: True})
            fake_a = history_a.query(fake_a)
            fake_b = history_b.query(fake_b)

            fetches = [self.loss_D_a, self.loss_D_b, self.loss_G_ab,
                       self.loss_G_ba, self.loss_cycle,
                       self.optimizer_D_a, self.optimizer_D_b,
                       self.optimizer_G_ab, self.optimizer_G_ba]
            if step % self.log_step == 0:
                fetches += [self.summary_op]

            fetched = sess.run(fetches, feed_dict={self.image_a: image_a,
                                                   self.image_b: image_b,
                                                   self.is_train: True,
                                                   self.lr: lr,
                                                   self.history_fake_a: fake_a,
                                                   self.history_fake_b: fake_b})

            if step % self.log_step == 0:
                summary_writer.add_summary(fetched[-1], step)
                summary_writer.flush()
                t.set_description(
                    'Loss: D_a({:.3f}) D_b({:.3f}) G_ab({:.3f}) G_ba({:.3f}) cycle({:.3f})'.format(
                        fetched[0], fetched[1], fetched[2], fetched[3], fetched[4]))


    def test(self, sess, data_A, data_B, base_dir):
        step = 0
        for data in data_A:
            step += 1
            fetches = [self.image_ab, self.image_aba]
            image_a = np.expand_dims(data, axis=0)
            image_ab, image_aba = sess.run(fetches, feed_dict={self.image_a: image_a,
                                                    self.is_train: False})
            images = np.concatenate((image_a, image_ab, image_aba), axis=2)
            images = np.squeeze(images, axis=0)
            imsave(os.path.join(base_dir, 'a_to_b_{}.jpg'.format(step)), images)

        step = 0
        for data in data_B:
            step += 1
            fetches = [self.image_ba, self.image_bab]
            image_b = np.expand_dims(data, axis=0)
            image_ba, image_bab = sess.run(fetches, feed_dict={self.image_b: image_b,
                                                    self.is_train: False})
            images = np.concatenate((image_b, image_ba, image_bab), axis=2)
            images = np.squeeze(images, axis=0)
            imsave(os.path.join(base_dir, 'b_to_a_{}.jpg'.format(step)), images)
