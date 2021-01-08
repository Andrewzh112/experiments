import argparse

from jointclassifier.trainer import Trainer


parser = argparse.ArgumentParser(description='Train JointClassifier')

parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--grad_clip', default=None, type=float, help='gradient clip')
parser.add_argument('--weight_decay', default=0., type=float, help='weight decay')
parser.add_argument('--x_latent_dim', default=64, type=int, help='latent dimension for x')
parser.add_argument('--y_latent_dim', default=16, type=int, help='latent dimension for y')
parser.add_argument('--hidden_dim', default=128, type=int, help='common hidden dim for fcs')
parser.add_argument('--in_channels', default=1, type=int, help='image channels')
parser.add_argument('--multipler', default=32, type=int, help='CNN channels growth multiplier')
parser.add_argument('--img_dim', default=28, type=int, help='initial image height/width')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
parser.add_argument('--data_dir', default='data/mnist', type=str, help='path to data')
parser.add_argument('--checkpoint_dir', default='jointclassifier/check_point/jointclassifier.pth', type=str, help='path to model weights')

args = parser.parse_args()


if __name__ == '__main__':
    trainer = Trainer(args.x_latent_dim, args.y_latent_dim, args.hidden_dim,
                      args.in_channels, args.multipler, args.img_dim, args.num_classes,
                      args.lr, args.epochs, args.batch_size, args.grad_clip,
                      args.weight_decay, args.checkpoint_dir, args.data_dir)
    trainer.fit()
