import argparse
from detail_train import Trainer
from utils import *
from model_structure import *
from data_feeder import StandardNumpyLoader
import matplotlib.pyplot as plt

# Allows a model to be tested from the terminal.

# You need to input your test data directory


def main():
    dataPath = './data'
    outputDir = "output/"
    parser = argparse.ArgumentParser(description="Train a neural network for causal inference. ")
    parser.add_argument('--datadir', type=str, default=dataPath, help="directory of dataset")
    parser.add_argument('--dataset', type=str, default='ihdp', help="acic or ihdp or twin or simu")  # ACIC IHDP
    parser.add_argument('--outputd', type=str, default=outputDir, help="directory of output")
    parser.add_argument('--batch_size', type=int, default=512, help="batch size")
    parser.add_argument('--epoch', type=int, default=20, help="epoch")
    parser.add_argument("--targeted_regularization", type=int, default=0)
    parser.add_argument("--auto", type=int, default=0)
    parser.add_argument("--network_type", type=remove_space, default='causalvib', help="dragonnet or causalvib or tarnet or nednet or cevae")
    parser.add_argument("--plot_result", type=bool, default=True)
    parser.add_argument("--replication", type=int, default=1)
    parser.add_argument("--list_to_execute", type=str2int_list, default=None, help="switch files to train")  # '1,2'
    parser.add_argument('--folder', type=str, default='scaling', help='which data sub directory')

    args = parser.parse_args()
    train_test_directory = args.datadir + '/' + args.dataset
    saved_model_dir = args.datadir + '/' + args.dataset + "saved_models/"
    output_dir = args.datadir + '/' + args.dataset + '/' + args.outputd + '/' + args.network_type
    log_file_dir = args.datadir + '/' + args.dataset + "saved_models/"

    if args.network_type == 'dragonnet':
        network = dragonnet(args.targeted_regularization)

    elif args.network_type == 'causalvib':
        network = causalvib(args.targeted_regularization)
    elif args.network_type == 'cevae':
        network = CEVAE()
    elif args.network_type == 'nednet':
        network = create_nednet_model()
    elif args.network_type == 'tarnet':
        network = tarnet()
    if args.dataset == 'ihdp':
        standard_loader = StandardNumpyLoader(file_name='./data/ihdp_npci.npz', output_file=output_dir,
                                              shuffle=True, test_size=0.1)
    elif args.dataset == 'twin':
        standard_loader = StandardNumpyLoader(file_name='./data/twins.npz', output_file=output_dir,
                                              shuffle=True, test_size=0.1)
    elif args.dataset == 'simu':
        standard_loader = StandardNumpyLoader(file_name='./data/simu_bias.npz', output_file=output_dir,
                                              shuffle=True, test_size=0.1)
    else:
        standard_loader = None

    trainer = Trainer(network=network,
                      data_loader=standard_loader,
                      network_type=args.network_type,
                      dataset=args.dataset,
                      targeted_regularization=args.targeted_regularization,
                      batch_size=args.batch_size,
                      epoch=args.epoch,
                      directory=train_test_directory,
                      save_model_dir=saved_model_dir,
                      output_dir=output_dir,
                      plot_result=args.plot_result,
                      folder=args.folder,
                      auto=args.auto,
                      rep=args.replication,
                      exelist=args.list_to_execute)
    trainer.train_model()


if __name__ == '__main__':
    main()
