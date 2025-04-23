from monai.metrics import compute_dice
from monai.data import DataLoader
from losses import *
from dataset import *
from model import UTMorph
import datetime
from utils import warp
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='UTMorph Training Script')

    # basic
    parser.add_argument('--datapath', type=str, default=r"D:\github_demo\dataset\pro_mix", help='Path to the dataset directory')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size for training and validation')
    parser.add_argument('--checkpoints', type=str, default=None, help='Checkpoints for training')
    parser.add_argument('--mode', type=str, default="MRtoUS", choices=['MRtoUS', 'UStoMR'], help='MR-to-US registration or US-to-MR registration')
    # model
    parser.add_argument('--base_chan', type=int, default=96, help='Base channel in model')
    parser.add_argument('--reduce_size', type=int, default=8, help='Reduction size for the model')
    parser.add_argument('--block_list', type=str, default="234", help='List of blocks to use in the model')
    parser.add_argument('--projection', type=str, default='interp', choices=['interp', 'maxpool'], help='Projection type')
    parser.add_argument('--attn_drop', type=float, default=0.1, help='Attention dropout rate')
    parser.add_argument('--proj_drop', type=float, default=0.1, help='Projection dropout rate')
    parser.add_argument('--rel_pos', type=bool, default=True, help='Whether to use relative position encoding')
    parser.add_argument('--maxpool', type=bool, default=True, help='Whether to use max pooling')
    parser.add_argument('--feature', type=bool, default=True, help='Whether to use feature extraction')
    parser.add_argument('--proj_att', type=bool, default=True, help='Whether to use projected attention')
    # training
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--alph', type=float, default=1, help='Weight for edge loss')
    parser.add_argument('--beta', type=float, default=1, help='Weight for regularization loss')
    parser.add_argument('--output_dir', type=str, default="./output", help='Directory to save model checkpoints')
    return parser.parse_args()


def forward(batch_data,model):
    fixed_image = batch_data["fixed_image"].to('cuda')
    moving_image = batch_data['moving_image'].to('cuda')
    moving_label = batch_data['moving_label'].to('cuda')
    ddf = model(moving_image, fixed_image)
    pred_image = warp(moving_image,ddf)
    pre_label = warp(moving_label,ddf)
    return pred_image, pre_label, ddf

if __name__ == "__main__":
    args = parse_args()

    train_data = TrainDataset(os.path.join(args.datapath, "train"), mode=args.mode)
    valid_data = TrainDataset(os.path.join(args.datapath, "val"), mode=args.mode)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batchsize, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset=valid_data, batch_size=args.batchsize, shuffle=False, num_workers=0)

    model = UTMorph(args.base_chan, reduce_size=args.reduce_size, block_list=args.block_list, num_blocks=[1,1,1,1], num_heads=[4,4,4,4],
                    projection=args.projection, attn_drop=args.attn_drop,proj_drop=args.proj_drop, rel_pos=args.rel_pos, maxpool=args.maxpool, feature=args.feature,
                       proj_att=args.proj_att).to("cuda")

    edgeloss = EdgeLoss()
    label_loss_fn = DiceLoss()
    label_loss_fn = MultiScaleLoss(label_loss_fn, scales=[0,1,2,4,8,16])
    regularization_fn = BendingEnergyLoss()
    optim = torch.optim.Adam(model.parameters(), args.lr)

    max_dice = 0
    max_epoch=0
    train_loss = []
    valid_loss = []
    metric = []
    train_edge_losses = []
    train_dice_losses = []

    if args.checkpoints is not None:
        print(f"Loading checkpoint from {args.checkpoints}")
        checkpoint = torch.load(args.checkpoints)
        model.load_state_dict(checkpoint)

    # start_time = time.time()
    # init_memory_allocated = torch.cuda.memory_allocated()
    for epoch in range(args.epochs):
        epoch_total_loss = 0
        step = 0
        epoch_edge_loss = 0
        epoch_dice_loss = 0
        model.train()
        for batch_data in train_loader:
            step +=1
            optim.zero_grad()

            pred_image, pred_label, ddf = forward(batch_data, model)
            fixed_image = batch_data['fixed_image'].to('cuda')
            fixed_label = batch_data['fixed_label'].to('cuda')
            edge_loss = edgeloss(pred_label, fixed_label) * args.alph
            label_loss = label_loss_fn(pred_label,fixed_label)    #DICE LOSS
            regularization_loss = regularization_fn(ddf) * args.beta                 #regularization LOSS
            loss = edge_loss+label_loss+regularization_loss               #total  LOSS

            loss.backward()
            optim.step()

            #output loss
            epoch_total_loss = epoch_total_loss +loss
            epoch_edge_loss = epoch_edge_loss+edge_loss
            epoch_dice_loss  = epoch_dice_loss +label_loss

        epoch_mean_loss = ((epoch_total_loss/len(train_data)).cpu()).detach().numpy()
        print(f'The EDGE LOSS of the {epoch+1} epoch in the training set is :',(epoch_edge_loss/len(train_data)).item())
        print(f'The DICE LOSS of the {epoch+1} epoch in the training set is ',(epoch_dice_loss/len(train_data)).item())
        print(f"The TOTAL LOSS of the {epoch+1} epoch in the training set is :", epoch_mean_loss)
        train_loss.append(epoch_mean_loss)
        train_edge_losses.append((epoch_edge_loss/len(train_data)).item())
        train_dice_losses.append((epoch_dice_loss/len(train_data)).item())

        model.eval()
        valid_epoch_loss= 0
        dice = 0
        with torch.no_grad():
            for batch_data1 in valid_loader:
                pred_image,pred_label,ddf = forward(batch_data1,model)
                fixed_image = batch_data1['fixed_image'].to('cuda')
                fixed_label = batch_data1['fixed_label'].to('cuda')

                edge_loss = edgeloss(pred_label, fixed_label) * args.alph
                label_loss = label_loss_fn(pred_label, fixed_label)
                regularization_loss = regularization_fn(ddf) * args.beta
                loss = edge_loss + label_loss + regularization_loss

                valid_epoch_loss = valid_epoch_loss+loss
                dice_epoch = compute_dice(y_pred = pred_label,y = fixed_label)
                dice_epoch = torch.mean(dice_epoch, dim=0)
                dice+=dice_epoch

            valid_mean_loss =((valid_epoch_loss/len(valid_data)).cpu()).detach().numpy()
            dice = (dice / len(valid_data) * args.batchsize).cpu().detach().numpy()
            metric.append(dice)
            print(f'The DICE of the {epoch+1} epoch in the validation set is:', dice)
            print(f"The TOTAL LOSS of the {epoch+1} epoch in the validation set is:", valid_mean_loss)
            valid_loss.append(valid_mean_loss)

        current_date = datetime.datetime.now().strftime("%Y%m%d")
        if dice>max_dice:
            max_dice = dice
            max_epoch=epoch+1
            param = model.state_dict()
            torch.save(param, os.path.join(args.output_dir,f'Epoch_{epoch+1}.pth'))
        print(f"The best performance is {max_epoch} epoch")
    # end_time = time.time()
    # epoch_time = end_time - start_time
    # print(f"One epoch completed in: {epoch_time} seconds")
    # final_memory_allocated = torch.cuda.memory_allocated()
    # memory_increase_mb = round((final_memory_allocated - init_memory_allocated) / (1024 * 1024), 2)
    # print(f"VRAM allocation increased: {memory_increase_mb} MB")