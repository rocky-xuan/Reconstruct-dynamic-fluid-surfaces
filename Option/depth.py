import argparse
import datetime
import os
parser = argparse.ArgumentParser(description = 'Depth_Generator')



"""
    保存
"""
parser.add_argument('--pattern', type=str, default='train', #train or test
                    help = 'when train save more imgs of the progress , when test just save result' )
#save path
curr_time = datetime.datetime.now()
time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d--%H_%M_%S')
save_path = os.path.join('./depth_result/' + time_str)
os.makedirs(save_path)
parser.add_argument('--save_path',type=str, default = save_path,
                    help = 'save_path for each new task')

"""
    训练
"""
# hardware
parser.add_argument('--gpu_id', type=str, default='0')

#iter
parser.add_argument('--total_iter', type=int, default=2000,
                    help='total iter of optimizing image')
parser.add_argument('--iter_print', type=int, default=50,
                    help='iter print')
parser.add_argument('--iter_save', type=int, default=50,
                    help='iter save')
# optimization specifications
parser.add_argument('--lr', type=float, default=8e-4,
                    help='learning rate')
parser.add_argument('--lr-decay', type=int, default=100,
                    help='learning rate decay per N epochs')
parser.add_argument('--gamma', type=float, default=0.95,
                    help='learning rate decay factor for step decay')


# loss_weight
parser.add_argument('--weight_MSE_loss', type=float, default=10.0,
                   )


'''
    网络初始化
'''
parser.add_argument('--module', type=str, default= 'U-NET', #INR or U-NET or VAE or fourier
                    help= 'choose the model for train and test' )
parser.add_argument('--output_channel', type=int, default=1,
                    help='INR output channel')
parser.add_argument('--input_mappings', type=int, default=6,  # Fourier_6
                    help='L represents the number of Fourier domain extensions')
parser.add_argument('--hidden_list', nargs='+', type=int, default=[32, 32, 16, 16],
                    help='')
parser.add_argument('--activation_function', type=str, default='siren',
                    help='relu|siren')

#refract
parser.add_argument('--n1', type=float, default=1.0,
                    help='Refractive index of the first medium')
parser.add_argument('--n2', type=float, default=1.33,
                    help='Refractive index of the second medium')
parser.add_argument('--step', type=float, default=20,
                    help='step')
args = parser.parse_args()