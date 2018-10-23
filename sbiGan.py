
#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import sys
import matplotlib.gridspec as gridspec


from utils import *

init_kernel = tf.random_normal_initializer(mean=0, stddev=0.05)


class sbiGAN(object):
    model_name = "sbiGAN"     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name,original_dataset_name,checkpoint_dir, result_dir,log_dir,
                 seed,seed_date,learning_rate,freq,training_type):
        self.sess = sess
        self.dataset_name = dataset_name
        self.original_dataset_name = original_dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.seed = seed
        self.seed_data = seed_date
        self.learning_rate = learning_rate
        self.freq = freq
        self.training_type = training_type

        if dataset_name == 'mnist.npz' or dataset_name == 'mnist_r.npz':
            # parameters
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = z_dim         # dimension of noise-vector
            self.c_dim = 1

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 100  # number of generated images to be saved

            # load mnist
            self.data_X = load_mnist(seed, dataset_name)
            self.INPUT_DIM = self.input_height * self.input_width

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
        else:
            raise NotImplementedError

    def gaussian_noise_layer(input_layer, std, deterministic):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
        # if deterministic or std==0:
        #     return input_layer
        # else:
        #     return input_layer + noise
        y = tf.cond(deterministic, lambda: input_layer, lambda: input_layer + noise)
        return y

    def _leakyReLu_impl(self, x, alpha):
        return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))

    def leakyReLu(self, x, alpha=0.2, name=None):
        if name:
            with tf.variable_scope(name):
                return self._leakyReLu_impl(x, alpha)
        else:
            return self._leakyReLu_impl(x, alpha)



    def discriminator_1(self, z, inp, is_training,reuse=False):
        with tf.variable_scope("discriminator_feature",reuse=reuse):
            placeholder_disciminator_input = tf.concat((inp, z), 1)
            with tf.variable_scope('discriminator', reuse=reuse):
                d_net = tf.layers.dense(placeholder_disciminator_input, 1024,activation=tf.nn.relu,reuse=reuse,name='dis1')
                d_net = tf.layers.batch_normalization(d_net, training=is_training)
                d_net = tf.layers.dense(d_net, 1024, activation=tf.nn.relu,reuse=reuse,name='dis2')
                d_net = tf.layers.batch_normalization(d_net, training=is_training)
                d_net = tf.layers.dense(d_net, 1, activation=None, reuse=reuse, name='disout')

        return d_net


    def encoder(self, inp, is_training,reuse=False):
        with tf.variable_scope("encoder",reuse=reuse):
            d_net = tf.layers.dense(inp, 1024, activation=tf.nn.relu, reuse=reuse, name='enc1')
            d_net = tf.layers.batch_normalization(d_net, training=is_training)
            d_net = tf.layers.dense(d_net, 1024, activation=tf.nn.relu, reuse=reuse, name='enc2')
            d_net = tf.layers.batch_normalization(d_net, training=is_training)
            d_net = tf.layers.dense(d_net, self.z_dim, activation=None, reuse=reuse, name='encout')
        return d_net

    def generator_s(self,z, is_training,reuse=False):
        with tf.variable_scope("generator_s",reuse=reuse):
            d_net = tf.layers.dense(z, 1024, activation=tf.nn.relu, reuse=reuse,name='gen1')
            d_net = tf.layers.batch_normalization(d_net, training=is_training)
            d_net = tf.layers.dense(d_net, 1024, activation=tf.nn.relu, reuse=reuse,name='gen2')
            d_net = tf.layers.batch_normalization(d_net, training=is_training)
            d_net = tf.layers.dense(d_net, self.INPUT_DIM, activation=tf.nn.sigmoid, reuse=reuse,name='genout')
        return d_net


    def build_model(self):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        self.nr_batches_train = int(self.data_X.shape[0] / self.batch_size)

        '''//////construct graph //////'''
        print('constructing graph')
        self.inp = tf.placeholder(tf.float32, [self.batch_size, self.input_height * self.input_width], name='unlabeled_data_input_pl')
        self.is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')

        gens = self.generator_s
        enc = self.encoder
        dis = self.discriminator_1

        with tf.variable_scope('encoder_model'):
            z_gen = enc(self.inp, is_training=self.is_training_pl)

        with tf.variable_scope('generator_s_model') as scope:
            zs = tf.random_uniform([self.batch_size, self.z_dim])
            x_gens = gens(zs, is_training=self.is_training_pl)
            scope.reuse_variables()
            self.reconstruct_s = gens(z_gen, is_training=self.is_training_pl,reuse=True) # reconstruction image dataset though bottleneck
            self.l_reconstruct_error = tf.reduce_mean(
                tf.abs(self.inp - self.reconstruct_s) / self.batch_size)
            self.reconstruct_s = tf.reshape(self.reconstruct_s, [-1,28,28,1], name=None)



        with tf.variable_scope('discriminator_feature_model') as scope:
            l_encoder = dis(z_gen, self.inp, is_training=self.is_training_pl)
            scope.reuse_variables()
            l_generator = dis(zs, x_gens, is_training=self.is_training_pl)



        with tf.name_scope('loss_functions'):
            # discriminator_feature
            loss_dis_enc = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l_encoder,labels=tf.ones_like(l_encoder) ))
            loss_dis_gen = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l_generator,labels=tf.zeros_like(l_generator)))
            self.loss_discriminator = loss_dis_gen + loss_dis_enc
            # generator
            self.loss_generator = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=l_generator,labels=tf.ones_like(l_generator))) \
                                                        + self.l_reconstruct_error
            # encoder
            self.loss_encoder = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=l_encoder,labels=tf.zeros_like(l_encoder))) \
                                                + self.l_reconstruct_error

        with tf.name_scope('optimizers'):
            # control op dependencies for batch norm and trainable variables
            train_vars = tf.trainable_variables()
            dvars = [var for var in train_vars if 'discriminator_feature_model' in var.name]
            gsvars = [var for var in train_vars if 'generator_s_model' in var.name]
            evars = [var for var in train_vars if 'encoder_model' in var.name]

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            update_ops_gens = [x for x in update_ops if ('generator_s_model' in x.name)]
            update_ops_enc = [x for x in update_ops if ('encoder_model' in x.name)]
            update_ops_dis = [x for x in update_ops if ('discriminator_feature_model' in x.name)]

            optimizer_dis = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, name='dis_optimizer')
            optimizer_gens = tf.train.AdamOptimizer(learning_rate=self.learning_rate*5, beta1=0.5, name='gens_optimizer')
            optimizer_enc = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, name='enc_optimizer')


            with tf.control_dependencies(update_ops_gens):  # attached op for moving average batch norm
                self.train_gens_op = optimizer_gens.minimize(self.loss_generator, var_list=gsvars)
            with tf.control_dependencies(update_ops_enc):
                self.train_enc_op = optimizer_enc.minimize(self.loss_encoder, var_list=evars)
            with tf.control_dependencies(update_ops_dis):
                self.train_dis_op = optimizer_dis.minimize(self.loss_discriminator, var_list=dvars)


        with tf.name_scope('summary'):
            with tf.name_scope('dis_summary'):
                tf.summary.scalar('loss_discriminator', self.loss_discriminator, ['dis'])
                tf.summary.scalar('loss_encoder', loss_dis_enc, ['dis'])
                tf.summary.scalar('loss_generator', loss_dis_gen, ['dis'])

            with tf.name_scope('gen_summary'):
                tf.summary.scalar('loss_generator', self.loss_generator, ['gen'])
                tf.summary.scalar('loss_encoder', self.loss_encoder, ['gen'])
                tf.summary.scalar('loss_reconstruct', self.l_reconstruct_error, ['gen'])

            with tf.name_scope('image_summary'):
                tf.summary.image('reconstruct', self.reconstruct_s, 20, ['image'])
                tf.summary.image('input_images', tf.reshape(self.inp, [-1, 28, 28, 1]), 20, ['image'])

            self.sum_op_dis = tf.summary.merge_all('dis')
            self.sum_op_gen = tf.summary.merge_all('gen')
            self.sum_op_im = tf.summary.merge_all('image')

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=1)

        self.saver1 = tf.train.Saver([v for v in tf.global_variables() if 'encoder_model' in v.name])
        self.saver2 = tf.train.Saver([v for v in tf.global_variables() if 'generator_s_model' in v.name])
        self.saver3 = tf.train.Saver([v for v in tf.global_variables() if 'discriminator_feature_model' in v.name])



        '''//////perform training //////'''
        print('start training')

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.nr_batches_train)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            train_batch = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            train_batch = 1
            print(" [!] Load failed...")

        for epoch in range(0,self.epoch):
            begin = time.time()

            # construct randomly permuted minibatches
            trainx = self.data_X
            trainx_2 = trainx.copy()
            trainx = trainx[np.random.RandomState(self.seed).permutation(trainx.shape[0])]  # shuffling unl dataset
            trainx_2 = trainx_2[np.random.RandomState(self.seed).permutation(trainx.shape[0])]

            train_loss_dis, train_loss_gen, train_loss_enc = [ 0, 0, 0]
            # training
            for t in range(self.nr_batches_train):
                self.display_progression_epoch(t,self.nr_batches_train)
                ran_from = t * self.batch_size
                ran_to = (t + 1) * self.batch_size

                # train discriminator
                feed_dict = {self.inp: trainx[ran_from:ran_to],self.is_training_pl: True}
                _, ld, sm = self.sess.run([self.train_dis_op, self.loss_discriminator, self.sum_op_dis], feed_dict=feed_dict)
                train_loss_dis += ld
                self.writer.add_summary(sm, train_batch)

                # train generator and encoder
                feed_dict = {self.inp: trainx_2[ran_from:ran_to],self.is_training_pl: True}



                _,_, le, lg, sm = self.sess.run([self.train_gens_op, self.train_enc_op, self.loss_encoder, self.loss_generator, self.sum_op_gen], feed_dict=feed_dict)
                train_loss_gen += lg
                train_loss_enc += le
                self.writer.add_summary(sm, train_batch)

                if t % self.freq == 0:  # inspect reconstruction
                    self.visualize_results(train_batch)

                train_batch += 1

            train_loss_gen /= self.nr_batches_train
            train_loss_enc /= self.nr_batches_train
            train_loss_dis /= self.nr_batches_train

            self.save(self.checkpoint_dir, train_batch)


            print("Epoch %d--Time = %ds | loss gen = %.4f | loss enc = %.4f | loss dis = %.4f  "
                  % (epoch, time.time() - begin, train_loss_gen, train_loss_enc, train_loss_dis))

            self.save(self.checkpoint_dir, train_batch)


    # L1 loss
    def mae(self, true, pred):
        return np.sum(np.abs(true - pred))

    # L2 loss
    def mse(self, true, pred):
        return np.sum(((true - pred) ** 2))

    def visualize_samples(self,step):
        import matplotlib.pyplot as plt

        t = np.random.randint(0, 4000)
        ran_from = t
        ran_to = t + self.batch_size
        # sm = sess.run(sum_op_im, feed_dict={inp: trainx[ran_from:ran_to],is_training_pl: False})
        sm, samples = self.sess.run([self.sum_op_im, tf.reshape(self.reconstruct_s, [-1, 28, 28, 1], name=None)],
                                    feed_dict={self.inp: self.data_X[ran_from:ran_to], self.is_training_pl: False})

        plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(5, 5)
        for j, generated_image in enumerate(samples):
            ax = plt.subplot(gs[j])
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(generated_image.reshape(28, 28), cmap='Greys_r')
        if not os.path.exists('res'):
            os.makedirs('res')
        plt.savefig('res/{}.png'.format(str(step).zfill(7)), bbox_inches='tight')
        self.writer.add_summary(sm, step)
        plt.close()

    def visualize_results(self, train_batch):

        t = np.random.randint(0, 4000)
        ran_from = t
        ran_to = t + self.batch_size
        # sm = sess.run(sum_op_im, feed_dict={inp: trainx[ran_from:ran_to],is_training_pl: False})
        sm, samples = self.sess.run([self.sum_op_im, tf.reshape(self.reconstruct_s, [-1, 28, 28, 1], name=None)],
                                    feed_dict={self.inp: self.data_X[ran_from:ran_to], self.is_training_pl: False})

        tot_num_samples = 100
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
        if (self.training_type == 'source'):
            suffixname = '_test_source_classes.png'
        else:
            suffixname = '_test_target_classes.png'
        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :],
                    [image_frame_dim, image_frame_dim], check_folder(
                self.log_dir + '/' + 'sbiGan_pic') + '/' + '_step%03d' % train_batch + suffixname)
        self.writer.add_summary(sm, train_batch)


    def display_progression_epoch(self, j, id_max):
        batch_progression = int((j / id_max) * 100)
        sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
        _ = sys.stdout.flush

    def model_test(self):
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=1)
        self.saver1 = tf.train.Saver([v for v in tf.global_variables() if 'encoder_model' in v.name])
        self.saver2 = tf.train.Saver([v for v in tf.global_variables() if 'generator_s_model' in v.name])
        self.saver3 = tf.train.Saver([v for v in tf.global_variables() if 'discriminator_feature_model' in v.name])

        self.load(self.checkpoint_dir)
        self.visualize_results(0)


    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        if(checkpoint_dir=='checkpoint'):
            checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:

            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if (self.training_type=='source'):
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            else:
                self.saver1.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                #self.saver2.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                self.saver3.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            start_epoch = 0
            print(" [*] Failed to find a checkpoint")
            return False, 0


