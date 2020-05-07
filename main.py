''' TO CONSIDER:
1) Skip connections
2) Dilated convolutions: https://arxiv.org/abs/1606.00915
3) Multi-Scale Context Module: https://arxiv.org/pdf/1511.07122.pdf
4) Joint Pyramid Upsampling: https://arxiv.org/pdf/1903.11816.pdf (See other links here)
'''












import sys
sys.stdout.flush()
import torch.nn.parallel
import torch.utils.data
import helper
from dataloaders.kitti_dataloader import load_calib, oheight, owidth
from inverse_warp import Intrinsics, homography_from
import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True
from models import ResNet, VGGNet
from metrics import AverageMeter, Result
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo, RandomSampling
import criteria
import utils

device = torch.device("cuda")

args = utils.parse_command()
print(args)

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time']
eval_fieldnames = ['num_samples', 'mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time']

best_result = Result()
best_result.set_to_worst()

cuda = torch.cuda.is_available() and not args.cpu
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("=> using '{}' for computation.".format(device))

# define loss functions
depth_criterion = criteria.MaskedMSELoss() if (
    args.criterion == 'l2') else criteria.MaskedL1Loss()
photometric_criterion = criteria.PhotometricLoss()
smoothness_criterion = criteria.SmoothnessLoss()

if args.use_pose:
    # hard-coded KITTI camera intrinsics
    K = load_calib()
    fu, fv = float(K[0, 0]), float(K[1, 1])
    cu, cv = float(K[0, 2]), float(K[1, 2])
    kitti_intrinsics = Intrinsics(owidth, oheight, fu, fv, cu, cv)
    if cuda:
        kitti_intrinsics = kitti_intrinsics.cuda()

def create_data_loaders(args):
    # Data loading code
    print("=> creating data loaders ...")
    traindir = os.path.join('data', args.data, 'train')
    
    if args.evaluate:
        valdir = os.path.join('data', args.data, 'test')
    else:
        valdir = os.path.join('data', args.data, 'val')
        
    train_loader = None
    val_loader = None

    # sparsifier is a class for generating random sparse depth input from the ground truth
    sparsifier = None
    max_depth = args.max_depth if args.max_depth >= 0.0 else np.inf
    if args.sparsifier == UniformSampling.name:
        sparsifier = UniformSampling(num_samples=args.num_samples, max_depth=max_depth)
    elif args.sparsifier == SimulatedStereo.name:
        sparsifier = SimulatedStereo(num_samples=args.num_samples, max_depth=max_depth)
    elif args.sparsifier == RandomSampling.name:
        sparsifier = RandomSampling(num_samples=args.num_samples, max_depth=max_depth)    

    if args.data == 'nyudepthv2':
        from dataloaders.nyu_dataloader import NYUDataset
        if not args.evaluate:
            train_dataset = NYUDataset(traindir, type='train',
                modality=args.modality, sparsifier=sparsifier)
        val_dataset = NYUDataset(valdir, type='val',
            modality=args.modality, sparsifier=sparsifier)

    elif args.data == 'kitti' or args.data == 'kitti_small':
        from dataloaders.kitti_dataloader import KITTIDataset
        if not args.evaluate:
            train_dataset = KITTIDataset(traindir, type='train',
                modality=args.modality, sparsifier=sparsifier)
        val_dataset = KITTIDataset(valdir, type='val',
            modality=args.modality, sparsifier=sparsifier)

    else:
        raise RuntimeError('Dataset not found.' +
                           'The dataset must be either of nyudepthv2 or kitti or kitti_small.')

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    # put construction of train loader here, for those who are interested in testing only
    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id:np.random.seed(work_id))
            # worker_init_fn ensures different sampling patterns for each data loading thread

    print("=> data loaders created.")
    return train_loader, val_loader

def main():
    global args, best_result, output_directory, train_csv, test_csv, eval_csv

    # evaluation mode
    start_epoch = 0
    if args.evaluate:
        assert os.path.isfile(args.evaluate), \
        "=> no best model found at '{}'".format(args.evaluate)
        print("=> loading best model '{}'".format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        output_directory = os.path.dirname(args.evaluate)
        eval_csv = os.path.join(output_directory, 'eval.csv')
        
        with open(eval_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=eval_fieldnames)
            writer.writeheader()  
            
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        args.evaluate = True
        for num_samples in range(2,9):
            args.num_samples = int(10 ** (num_samples/2))
            _, val_loader = create_data_loaders(args)
            validate(val_loader, model, checkpoint['epoch'], write_to_file=True)
            
        plot_results()    
        return

    # optionally resume from a checkpoint
    elif args.resume:
        chkpt_path = args.resume
        assert os.path.isfile(chkpt_path), \
            "=> no checkpoint found at '{}'".format(chkpt_path)
        print("=> loading checkpoint '{}'".format(chkpt_path))
        checkpoint = torch.load(chkpt_path)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        output_directory = os.path.dirname(os.path.abspath(chkpt_path))
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        train_loader, val_loader = create_data_loaders(args)
        args.resume = True

    # create new model
    else:
        train_loader, val_loader = create_data_loaders(args)
        print("=> creating Model ({}-{}) ...".format(args.arch, args.decoder))
        in_channels = len(args.modality)
        if args.arch == 'resnet50':
            model = ResNet(layers=50, decoder=args.decoder, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet18':
            model = ResNet(layers=18, decoder=args.decoder, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'vgg16':
        	model = VGGNet(layers=16, decoder=args.decoder, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'vgg19':
          model = VGGNet(layers=19, decoder=args.decoder, output_size=train_loader.dataset.output_size,
                in_channels=in_channels, pretrained=args.pretrained)
        print("=> model created.")
        #change here
        optimizer = torch.optim.SGD(model.parameters(), args.lr, \
            momentum=args.momentum, weight_decay=args.weight_decay)

        # model = torch.nn.DataParallel(model).cuda() # for multi-gpu training
        model = model.to(device)

    # create results folder, if not already exists
    output_directory = utils.get_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    train_csv = os.path.join(output_directory, 'train.csv')
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')

    # create new csv files with only header
    if not args.resume:
        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
         

    for epoch in range(start_epoch, args.epochs):
      # epoch=start_epoch
      # print(epoch)
      utils.adjust_learning_rate(optimizer, epoch, args.lr)
      train(train_loader, model, optimizer, epoch) # train for one epoch
      result, img_merge = validate(val_loader, model, epoch) # evaluate on validation set

      # remember best rmse and save checkpoint
      is_best = result.rmse < best_result.rmse
      if is_best:
          best_result = result
          with open(best_txt, 'w') as txtfile:
              txtfile.write("epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                  format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1, result.gpu_time))
          if img_merge is not None:
              img_filename = output_directory + '/comparison_best.png'
              utils.save_image(img_merge, img_filename)

      utils.save_checkpoint({
          'args': args,
          'epoch': epoch,
          'arch': args.arch,
          'model': model,
          'best_result': best_result,
          'optimizer' : optimizer,
      }, is_best, epoch, output_directory)


def train(train_loader, model, optimizer, epoch):
    # print('training....')
    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]
    model.train() # switch to train mode
    end = time.time()
    for i, (input, target,near,r_mat,t_vec) in enumerate(train_loader):

        input, target = input.to(device), target.to(device)
        torch.cuda.synchronize()
        #time to load the data
        data_time = time.time() - end

        # compute pred
        end = time.time()
        pred = model(input)
        depth_loss, photometric_loss, smooth_loss, mask = 0, 0, 0, None
        # loss = criterion(pred, target)
        # Loss 1:depth loss
        depth_loss = depth_criterion(pred,target)
        mask = (target<1e-3).float() #not specified why
        # Loss 2: the self-supervised photometric loss
        if args.use_pose:
            # create multi-scale pyramids
            pred_array = helper.multiscale(pred)
            rgb_curr_array = helper.multiscale(input)
            # how to get the near rgb frame (the next one)
            rgb_near_array = helper.multiscale(near)
            if mask is not None:
                mask_array = helper.multiscale(mask)
            num_scales = len(pred_array)

            # compute photometric loss at multiple scales
            for scale in range(len(pred_array)):
                pred_ = pred_array[scale]
                rgb_curr_ = rgb_curr_array[scale]
                rgb_near_ = rgb_near_array[scale]
                mask_ = None
                if mask is not None:
                    mask_ = mask_array[scale]

                # compute the corresponding intrinsic parameters
                height_, width_ = pred_.size(2), pred_.size(3)
                intrinsics_ = kitti_intrinsics.scale(height_, width_)

                # inverse warp from a nearby frame to the current frame
                warped_ = homography_from(rgb_near_, pred_,
                                            r_mat,
                                            t_vec, intrinsics_)
                photometric_loss += photometric_criterion(
                    rgb_curr_, warped_, mask_) * (2**(scale - num_scales))

        # Loss 3: the depth smoothness loss
        smooth_loss = smoothness_criterion(pred) if args.w2 > 0 else 0

        # backprop
        loss = depth_loss + args.w1 * photometric_loss + args.w2 * smooth_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            #print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                  epoch, i+1, len(train_loader), data_time=data_time,
                  gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
            'gpu_time': avg.gpu_time, 'data_time': avg.data_time})


def validate(val_loader, model, epoch, write_to_file=True):
    
    print("=> Evaluating for no. of samples = ", args.num_samples)
    
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
    img_merge = None
    for i, (input, target) in enumerate(val_loader):
        input, target = input.to(device), target.to(device)
        torch.cuda.synchronize() #This is needed only if we want to compute the time
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        torch.cuda.synchronize() #This is needed only if we want to compute the time
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        
        # save 8 images for visualization
        if not args.evaluate:
            skip = 50
            if args.modality == 'd':
                img_merge = None
            else:
                if args.modality == 'rgb':
                    rgb = input
                elif args.modality == 'rgbd':
                    rgb = input[:,:3,:,:]
                    depth = input[:,3:,:,:]



                if i == 0:
                    if args.modality == 'rgbd':
                        img_merge = utils.merge_into_row_with_gt(rgb, depth, target, pred)
                    else:
                        img_merge = utils.merge_into_row(rgb, target, pred)
                elif (i < 8*skip) and (i % skip == 0):
                    if args.modality == 'rgbd':
                        row = utils.merge_into_row_with_gt(rgb, depth, target, pred)
                    else:
                        row = utils.merge_into_row(rgb, target, pred)
                    img_merge = utils.add_row(img_merge, row)
                elif i == 8*skip:
                    filename = output_directory + '/comparison_' + str(epoch) + '.png'
                    utils.save_image(img_merge, filename)

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        if args.evaluate:
            with open(eval_csv, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=eval_fieldnames)
                writer.writerow({'num_samples': args.num_samples, 'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel,  'lg10': avg.lg10, 'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3, 'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
                
        else:
            with open(test_csv, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                    'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                    'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
            
                
    return avg, img_merge

def plot_results():
    
    df = pd.read_csv( eval_csv)
    for y in ['rmse', 'absrel', 'delta1', 'delta2']:
        fig = plt.figure()
        plt.plot( np.log10(df['num_samples']) , df[y])
        plt.xlabel('Log of no. of samples')
        plt.ylabel(y)
        fig_path = os.path.join(output_directory, y)
        plt.savefig( fig_path)  
    

if __name__ == '__main__':
    main()
