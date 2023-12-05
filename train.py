import argparse
from src.util import *
from src.trainer import Trainer
from src.gan_trainer import GAN_Trainer
import random
import gc

def main(args):
    gc.collect()

    random_seed = args.seed if args.seed is not None else 42
    random.seed(random_seed)
    np.random.seed(random_seed)

    gray_path = args.gray_path
    ab_path = args.ab_path
    images_gray, images_lab = load_data(gray_path, ab_path, num_images=args.num_images)
    train_data, val_data, test_data = split_data(images_gray, images_lab, args.split_ratios)
    if(args.model.lower() == "gan"):
        trainer = GAN_Trainer(batch_size=args.batch_size, epochs=args.epoch, lr=args.lr, device=args.device)
    else:
        model = get_model(args.model)
        trainer = Trainer(model, batch_size=args.batch_size, epochs=args.epoch, lr=args.lr, device=args.device)
    trainer.train(train_data, val_data, test_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on image data.")
    parser.add_argument("--gray-path", type=str, default="data/gray_scale.npy", help="Path to the gray scale images")
    parser.add_argument("--ab-path", type=str, default="data/ab/ab1.npy", help="Path to the ab channel images")
    parser.add_argument("--num-images", type=int, default=100, help="Number of images to load")
    parser.add_argument("--split-ratios", type=float, nargs=3, default=[0.7, 0.1, 0.2],
                        help="Train, validation, and test split ratios")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--model", type=str, default='cnn',
                        help="The model to use for training")
    parser.add_argument("--epoch", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--device", type=str, default="cpu", help="device")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)
