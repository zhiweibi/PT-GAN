import time
from options.config import load_config
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer


if __name__ == '__main__':
    opt = load_config()
    dataloader = create_dataset(opt)
    dataset_size = len(dataloader)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    print('start training {}'.format(opt.name))

    opt.n_input_modal = dataloader.dataset.n_modal - 1
    opt.modal_names = dataloader.dataset.get_modal_names()
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load networks; create schedulers
    total_iters = 0                # the total number of training iterations
    visualizer = Visualizer(opt, dataset_size)  # create a visualizer that save images

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataloader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.save_visual_freq == 0:
                save_result = total_iters % opt.update_freq == 0
                model.compute_visuals()
                visuals = model.get_current_visuals()
                visualizer.save_current_results(model.get_current_visuals(), total_iters)

            if i == 0:
                loss = model.get_current_losses()
            else:
                temp = model.get_current_losses()
                for k, v in temp.items():
                    loss[k] += v
            if i == dataset_size - 1:
                for k, v in loss.items():
                    loss[k] /= dataset_size
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, loss, t_comp, t_data)

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
