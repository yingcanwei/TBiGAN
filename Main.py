#encoding=utf8

import argparse

import tensorflow as tf

from CombinationTest import CombinationTest
from Mnist_USPS import Mnistm_USPS
from Mnist_test import Mnist_test
from Mnistm import Mnistm
from sbiGan import sbiGAN
from SVHN import SVHN
from SVHN_Mnist import SVHN_Mnist
from SVHN_test import SVHN_test
from TestLoading import SVHN_Mnist_test
## GAN Variants
from utils import *

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='GAN',
                        choices=['sbiGAN', 'testGAN','Mnistm','SVHN', 'Mnistm_USPS', 'SVHN_Mnist','SVHN_Mnist_test','Mnist_test', 'SVHN_test', 'CombinationTest'],
                        help='The type of GAN', required=True)
    parser.add_argument('--dataset', type=str, default='mnist.npz',help='The name of dataset')
    parser.add_argument('--training_type', type=str, default='source', choices=['source','target','transform'],
                        help='The type of training')
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--seed', type=int, default=146, help='seed number')
    parser.add_argument('--seed_data', type=int, default=646, help='seed data')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of batch')
    parser.add_argument('--z_dim', type=int, default=256, help='Dimension of noise vector')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='./log',
                        help='Directory name to save training logs')
    parser.add_argument('--freq', type=int, default=500,
                        help='print frequency image tensorboard [20]')
    parser.add_argument('--learning_rate', type=float, default=0.0003,
                        help='learning_rate dis[0.0005]')
    parser.add_argument('--training_mode', type=str, default='train', choices=['train', 'test', 'meta'],
                        help='The type of training')
    parser.add_argument('--original_dataset', type=str, default='mnist.npz', help='The dataset of another pair')



    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    assert args.epoch >= 1, 'number of epochs must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    # --z_dim
    assert args.z_dim >= 1, 'dimension of noise vector must be larger than or equal to one'

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    models = [sbiGAN, Mnistm, SVHN, Mnistm_USPS, SVHN_Mnist,SVHN_Mnist_test, Mnist_test,SVHN_test,CombinationTest]
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # declare instance for GAN

        gan = None
        for model in models:
            if args.gan_type == model.model_name:
                gan = model(sess,
                            training_type=args.training_type,
                            freq=args.freq,
                            seed=args.seed,
                            seed_date=args.seed_data,
                            epoch=args.epoch,
                            batch_size=args.batch_size,
                            z_dim=args.z_dim,
                            dataset_name=args.dataset,
                            original_dataset_name=args.original_dataset,
                            checkpoint_dir=args.checkpoint_dir,
                            result_dir=args.result_dir,
                            log_dir=args.log_dir,
                            learning_rate=args.learning_rate)
        if gan is None:
            raise Exception("[!] There is no option for " + args.gan_type)

        # build graph
        gan.build_model()

        if (args.training_mode=='test'):
            print('Test start!')
            gan.model_test()
            print('Test finish!')
            exit()
        if (args.training_mode=='meta'):
            print('Get paired metadata start!')
            gan.generate_D_like()
            print('Get paired metadata start!')
            exit()

        # show network architecture
        show_all_variables()

        # launch the graph in a session
        gan.train()
        print(" [*] Training finished!")

        # visualize learned generator
        #gan.visualize_results(args.epoch-1)
        print(" [*] Testing finished!")

if __name__ == '__main__':
    main()