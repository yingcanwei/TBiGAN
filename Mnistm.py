
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


class Mnistm(object):
    model_name = "Mnistm"     # name for checkpoint

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

        if '.npz' in dataset_name :
            # parameters
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = z_dim         # dimension of noise-vector
            self.c_dim = 3

            # train
            self.learning_rate = 0.001
            self.beta1 = 0.5

            # test
            self.sample_num = 100  # number of generated images to be saved

            # load mnist
            self.data_X = load_mnist_3(seed, dataset_name)
            self.INPUT_DIM = self.input_height * self.input_width * self.c_dim
            self.OUTPUT_DIM = self.output_height * self.output_width * self.c_dim

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

            z=tf.reshape(z, [-1, 16, 16, 1])
            x = tf.reshape(inp, [-1, 28, 28, 3])
            with tf.variable_scope('discriminator', reuse=reuse):

                with tf.variable_scope('layer_1'):  ##out_width = ceil(float(in_width) / float(strides[2]))=28
                    x = tf.layers.conv2d(x, 32, [5, 5], padding='SAME')  # 28*28*32
                    x = tf.layers.batch_normalization(x, training=is_training)
                    x = self.leakyReLu(x)
                    x = tf.layers.dropout(x, rate=0.2)

                with tf.variable_scope('Z_layer'):  #7*7*48
                    z = tf.layers.conv2d(z, 48, [3, 3], strides=[2, 2],padding='VALID')
                    z = tf.layers.dropout(z, rate=0.2)

                with tf.variable_scope('pooling_1'):
                    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2]) #[(28-2)/2]+1=14; 14*14

                with tf.variable_scope('layer_2'):
                    x = tf.layers.conv2d(x, 48, [5, 5], padding='SAME')#14*14*48
                    x = self.leakyReLu(x)
                    x = tf.layers.dropout(x, rate=0.2)

                with tf.variable_scope('pooling_2'):
                    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2])#7*7

                x = tf.concat((x, z), 1)

                with tf.variable_scope('fully_1'):
                    x = tf.layers.dense(x, 100,activation=tf.nn.relu,reuse=reuse)
                    x = tf.layers.batch_normalization(x, training=is_training)

                with tf.variable_scope('output'):
                    x = tf.layers.dense(x, 1, activation=None, reuse=reuse)

        return x


    def encoder(self, inp, is_training,reuse=False):

        with tf.variable_scope("encoder",reuse=reuse):
            x = tf.reshape(inp, [-1, 28, 28, 3])

            with tf.variable_scope('layer_1'):  ##out_width = ceil(float(in_width) / float(strides[2]))=10
                x = tf.layers.conv2d(x, 32, [5, 5], padding='SAME')  # 28*28*32
                x = tf.layers.batch_normalization(x, training=is_training)
                x = self.leakyReLu(x)
                x = tf.layers.dropout(x, rate=0.2)

            with tf.variable_scope('pooling_1'):
                x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2])  # [(28-2)/2]+1=14; 14*14

            '''
            with tf.variable_scope('layer_2'):
                x = tf.layers.conv2d(x, 48, [5, 5], padding='SAME')  # 14*14*48
                x = self.leakyReLu(x)
                x = tf.layers.dropout(x, rate=0.2)

            with tf.variable_scope('pooling_2'):
                x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2])  # 7*7
            '''


            x = tf.contrib.layers.flatten(x)

            with tf.variable_scope('fully_1'):
                x = tf.layers.dense(x, 1024, activation=tf.nn.relu, reuse=reuse)
                x = tf.layers.batch_normalization(x, training=is_training)

            with tf.variable_scope('output'):
                x = tf.layers.dense(x, self.z_dim, activation=None, reuse=reuse)

        return x

    def generator_t(self,z, is_training,reuse=False):
        with tf.variable_scope("generator_t",reuse=reuse):

            with tf.variable_scope('dense_1'):
                x = tf.layers.dense(z, units=7*7*48, kernel_initializer=init_kernel)
                x = tf.layers.batch_normalization(x, training=is_training, name='batchnorm_1')
                x = tf.nn.relu(x)

            x = tf.reshape(x, [-1, 7, 7, 48])

            with tf.variable_scope('deconv_1'):
                x = tf.layers.conv2d_transpose(x, 48, [5, 5],strides=[2, 2],padding='SAME')  # 14*14*48  # Padding==Same: H = H1 * stride
                                                                                            # Padding==Valid H = (H1+ HF-1) * stride
                x = self.leakyReLu(x)
                x = tf.layers.dropout(x, rate=0.2)

            '''
            with tf.variable_scope('deconv_2'): 
                x = tf.layers.conv2d_transpose(x, 32, [5, 5],strides=[2, 2], padding='SAME')  # 28*28*3*32
                x = self.leakyReLu(x)
                x = tf.layers.dropout(x, rate=0.2)
            '''

            x = tf.contrib.layers.flatten(x)
            with tf.variable_scope('fully_1'):
                x = tf.layers.dense(x, 1024, activation=tf.nn.relu, reuse=reuse)
                x = tf.layers.batch_normalization(x, training=is_training)

            with tf.variable_scope('output'):
                x = tf.layers.dense(x, self.INPUT_DIM, activation=tf.nn.sigmoid, reuse=reuse)

        return x


    def build_model(self):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        self.nr_batches_train = int(self.data_X.shape[0] / self.batch_size)

        '''//////construct graph //////'''
        print('constructing graph')
        self.inp = tf.placeholder(tf.float32, [self.batch_size, self.INPUT_DIM], name='unlabeled_data_input_pl')
        self.is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')

        gent = self.generator_t
        enc = self.encoder
        dis = self.discriminator_1

        with tf.variable_scope('encoder_model'):
            self.z_gen = enc(self.inp, is_training=self.is_training_pl)

        with tf.variable_scope('generator_t_model') as scope:
            zs = tf.random_uniform([self.batch_size, self.z_dim])
            x_gens = gent(zs, is_training=self.is_training_pl)
            scope.reuse_variables()
            self.reconstruct_t = gent(self.z_gen, is_training=self.is_training_pl,reuse=True) # reconstruction image dataset though bottleneck
            self.l_reconstruct_error = tf.reduce_mean(
                tf.abs(self.inp - tf.reshape(self.reconstruct_t, [-1, self.INPUT_DIM])) / self.batch_size)
            self.reconstruct_t = tf.reshape(self.reconstruct_t, [-1,28,28,3], name=None)



        with tf.variable_scope('discriminator_feature_model') as scope:
            l_encoder = dis(self.z_gen, self.inp, is_training=self.is_training_pl)
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
            gtvars = [var for var in train_vars if 'generator_t_model' in var.name]
            evars = [var for var in train_vars if 'encoder_model' in var.name]

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            update_ops_gent = [x for x in update_ops if ('generator_t_model' in x.name)]
            update_ops_enc = [x for x in update_ops if ('encoder_model' in x.name)]
            update_ops_dis = [x for x in update_ops if ('discriminator_feature_model' in x.name)]

            optimizer_dis = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, name='dis_optimizer')
            optimizer_gent = tf.train.AdamOptimizer(learning_rate=self.learning_rate*5, beta1=0.5, name='gent_optimizer')
            optimizer_enc = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, name='enc_optimizer')


            with tf.control_dependencies(update_ops_gent):  # attached op for moving average batch norm
                self.train_gens_op = optimizer_gent.minimize(self.loss_generator, var_list=gtvars)
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
                tf.summary.image('reconstruct', self.reconstruct_t, 20, ['image'])
                tf.summary.image('input_images', tf.reshape(self.inp, [-1, 28, 28, 3]), 20, ['image'])

            self.sum_op_dis = tf.summary.merge_all('dis')
            self.sum_op_gen = tf.summary.merge_all('gen')
            self.sum_op_im = tf.summary.merge_all('image')

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=1)

        self.saver1 = tf.train.Saver([v for v in tf.global_variables() if 'encoder_model' in v.name])
        #self.saver2 = tf.train.Saver([v for v in tf.global_variables() if 'generator_t_model' in v.name])
        self.saver3 = tf.train.Saver([v for v in tf.global_variables() if 'discriminator_feature_model' in v.name])



        '''//////perform training //////'''
        print('start training')


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
        data = self.data_X[np.random.RandomState(self.seed).permutation(self.data_X.shape[0])]
        # sm = sess.run(sum_op_im, feed_dict={inp: trainx[ran_from:ran_to],is_training_pl: False})
        sm, samples = self.sess.run([self.sum_op_im, tf.reshape(self.reconstruct_t,
                                    [-1, self.output_height, self.output_width, self.c_dim], name=None)],
                                    feed_dict={self.inp: data[ran_from:ran_to], self.is_training_pl: False})

        plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(10, 10)
        for j, generated_image in enumerate(samples):
            ax = plt.subplot(gs[j])
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(generated_image.reshape(self.output_height, self.output_width, self.c_dim), cmap='Greys_r')
        if not os.path.exists('res'):
            os.makedirs('res')
        plt.savefig('res/{}.png'.format(str(step).zfill(7)), bbox_inches='tight')

        plt.close()

    def generate_D_like(self):
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.saver1 = tf.train.Saver([v for v in tf.global_variables() if 'encoder_model' in v.name])
        #self.saver2 = tf.train.Saver([v for v in tf.global_variables() if 'generator_t_model' in v.name])
        self.saver3 = tf.train.Saver([v for v in tf.global_variables() if 'discriminator_feature_model' in v.name])
        # self.convert_s2t(self.checkpoint_dir)
        self.load(self.checkpoint_dir)
        coder = []
        recons = []
        for t in range(self.nr_batches_train):
            self.display_progression_epoch(t, self.nr_batches_train)
            ran_from = t * self.batch_size
            ran_to = (t + 1) * self.batch_size
            data = self.data_X
            sm, ecoder,samples = self.sess.run([self.sum_op_im,self.z_gen, tf.reshape(self.reconstruct_t,
                                                                    [-1, self.output_height, self.output_width,
                                                                     self.c_dim], name=None)],
                                        feed_dict={self.inp: data[ran_from:ran_to], self.is_training_pl: False})
            recons.append(samples.reshape(self.batch_size,self.OUTPUT_DIM))
            coder.append(ecoder)
        array1=np.array(coder)
        array2=np.array(recons)
        np.savez(self.dataset_name+'_source_'+ self.original_dataset_name +'_target_DST.npz',Dlike=array2.reshape(-1,self.OUTPUT_DIM),Dcoder=array1.reshape(-1,self.z_dim))



    def visualize_results(self, train_batch,training_mode='train'):

        t = np.random.randint(0, 4000)
        ran_from = t
        ran_to = t + self.batch_size
        data = self.data_X[np.random.RandomState(self.seed).permutation(self.data_X.shape[0])]
        # sm = sess.run(sum_op_im, feed_dict={inp: trainx[ran_from:ran_to],is_training_pl: False})
        sm, samples = self.sess.run([self.sum_op_im, tf.reshape(self.reconstruct_t,
                                    [-1, self.output_height, self.output_width, self.c_dim], name=None)],
                                    feed_dict={self.inp: data[ran_from:ran_to], self.is_training_pl: False})

        tot_num_samples = 100
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
        if (self.training_type == 'source'):
            suffixname = '_test_source_classes.png'
        else:
            suffixname = '_test_target_classes.png'
        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :],
                    [image_frame_dim, image_frame_dim], check_folder(
                self.log_dir + '/' + 'Mnistm_pic') + '/' + '_step%03d' % train_batch + suffixname)
        if (training_mode=='test'):
            save_images(data[ran_from:ran_to].reshape(-1,28,28,3)[:image_frame_dim * image_frame_dim, :, :, :],
                        [image_frame_dim, image_frame_dim], check_folder(
                    self.log_dir + '/' + 'Mnistm_pic') + '/' + '_sourcestep%03d' % train_batch + suffixname)
        if (training_mode == 'train'):
            self.writer.add_summary(sm, train_batch)


    def display_progression_epoch(self, j, id_max):
        batch_progression = int((j / id_max) * 100)
        sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
        _ = sys.stdout.flush

    def model_test(self):
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=1)
        self.saver1 = tf.train.Saver([v for v in tf.global_variables() if 'encoder_model' in v.name])
        #self.saver2 = tf.train.Saver([v for v in tf.global_variables() if 'generator_t_model' in v.name])
        self.saver3 = tf.train.Saver([v for v in tf.global_variables() if 'discriminator_feature_model' in v.name])
        #self.convert_s2t(self.checkpoint_dir)
        self.load(self.checkpoint_dir)
        self.visualize_results(0,'test')


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

    def convert_s2t(self,checkpoint_dir):
        import re
        if (checkpoint_dir == 'checkpoint'):
            checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        vars = tf.contrib.framework.list_variables(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            with tf.Graph().as_default(), tf.Session().as_default() as sess:
                new_vars = []
                for name, shape in vars:
                    v = tf.contrib.framework.load_variable(checkpoint_dir, name)
                    name = name.replace('generator_s', 'generator_t')
                    name = name.replace('gens', 'gent')
                    new_vars.append(tf.Variable(v, name=name))

                saver = tf.train.Saver(new_vars)
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                step = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
                sess.run(tf.global_variables_initializer())
                saver.save(sess, os.path.join(checkpoint_dir, self.model_name+'.testmodel'), global_step=step)


