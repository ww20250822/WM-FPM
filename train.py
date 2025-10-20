"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md

--dataroot ./datasets/maps
--name maps_cyclegan
--model cycle_gan

python -m visdom.server

python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
"""
import time

from tqdm import tqdm

from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.utils import save_image
import os

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    # 添加验证集
    opt_val = TestOptions().parse()
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    opt_val.phase = 'test'
    opt_val.isTrain = False
    dataset_val = create_dataset(opt_val)
    # opt.continue_train = True

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    best_ssim = 0.0
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim = ssim.to(model.device)
    save_dir = os.path.join('results', opt.name, f'epoch')
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        
        # 每5个epoch计算SSIM
        if epoch % 1 == 0:
            model.eval()
            total_ssim = 0.0
            for i, val_data in enumerate(tqdm(dataset_val)):
                model.set_input(val_data)
                model.test()
                visuals = model.get_current_visuals()
                real_A = (visuals['real_A'] + 1) / 2
                fake_B = (visuals['fake_B'] + 1) / 2
                real_B = (visuals['real_B'] + 1) / 2
                save_image(fake_B, os.path.join(save_dir, f'fake_B_{i}.png'))
                save_image(real_B, os.path.join(save_dir, f'real_B_{i}.png'))
                save_image(real_A, os.path.join(save_dir, f'real_A_{i}.png'))
                total_ssim += ssim(visuals['fake_B'], visuals['real_B'])
            avg_ssim = total_ssim / len(dataset_val)
            print(f'Validation SSIM at epoch {epoch}: {avg_ssim:.4f}')
            
            if avg_ssim > best_ssim:
                best_ssim = avg_ssim
                model.save_networks('best_ssim')
                print(f'New best SSIM {avg_ssim:.4f}, saving model...')

    print(f'Best SSIM: {best_ssim:.4f}')
