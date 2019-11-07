import argparse
from utils import check_folder
import os
from train import Train
# from data_process import data_process

def parse_args():
    desc = "AKI Prediction Using CNN based Model"
    parser = argparse.ArgumentParser(description=desc)
    
    # parser.add_argument('--phase', type=str, default='train', help='train or test?')
    parser.add_argument('--data_base', type=str, default='mimic', help='mimic or eicu')
    parser.add_argument('--res', type=str, default='SMOTE', help='SMOTE or others')
    parser.add_argument('--model_name', type=str, default='CNN', help='CNN or Resnet or MLP')
    parser.add_argument('--nb_layer', type=int, default=16, help='The number of layers')
    
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    
    return check_args(parser.parse_args())
    
def check_args(args):
    # checkpoint_dir
    check_folder(args.checkpoint_dir)
    
    # result_dir
    check_folder(args.log_dir)
    
    
    return args
    
def main():
    
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    
    if not os.path.exists('./data_set/'+args.data_base+'_processed_1D.npz'):
		dp = data_process(args.data_base)
        dp.run()
    else:
		# if args.phase=='train':
		tr=Train(args)
		tr.run()
		
			

if __name__=='__main__':
    main()