from monai.data import DataLoader
from metics import *
from dataset import *
from datetime import datetime
from model import UTMorph
from utils import warp,save_flow
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='UTMorph Validation Script')
    # basic
    parser.add_argument('--datapath', type=str, default=r"D:\github_demo\dataset\pro_mix", help='Path to the dataset directory')
    parser.add_argument('--batchsize', type=int, default=1, help='Batch size for training and validation')
    parser.add_argument('--checkpoints', type=str, default="./output/UStoMR_.pth", help='Checkpoints for training')
    parser.add_argument('--save', type=bool, default=True, help='Directory to save model checkpoints')
    parser.add_argument('--mode', type=str, default="UStoMR", choices=['MRtoUS', 'UStoMR'], help='MR-to-US registration or US-to-MR registration')
    parser.add_argument('--augment', type=bool, default=False, help='Checkpoints for training')
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

    return parser.parse_args()

if __name__ == "__main__":
    now = datetime.now().strftime('%m-%d_%H-%M-%S')
    args = parse_args()

    valid_data = TrainDataset(os.path.join(args.datapath, "val"), mode=args.mode, augment=args.augment)
    valid_loader = DataLoader(dataset=valid_data, batch_size=args.batchsize, shuffle=False, num_workers=0)

    model = UTMorph(args.base_chan, reduce_size=args.reduce_size, block_list=args.block_list, num_blocks=[1, 1, 1, 1],
                    num_heads=[4, 4, 4, 4],
                    projection=args.projection, attn_drop=args.attn_drop, proj_drop=args.proj_drop,
                    rel_pos=args.rel_pos, maxpool=args.maxpool, feature=args.feature,
                    proj_att=args.proj_att).to("cuda")

    model.load_state_dict(torch.load(args.checkpoints))
    model.eval()
    metric = []

    HD=[]
    MSD=[]
    folding = []
    mag_det_jac = []
    i = 0
    for batch_data in valid_loader:
        i=i+1
        fixed_image = batch_data["fixed_image"].to('cuda')
        fixed_label = batch_data['fixed_label'].to('cuda')
        moving_image = batch_data['moving_image'].to('cuda')
        moving_label = batch_data['moving_label'].to('cuda')

        ddf = model(moving_image, fixed_image)
        pred_image = warp(moving_image,ddf)
        pre_label = warp(moving_label,ddf)

        dice, msd, hd = calculate_metrics(pre_label, fixed_label)
        metric.extend(dice)
        HD.append(hd)
        MSD.append(msd)

        folding_ratio, mag_det_jac_det = calculate_jacobian_metrics(ddf.cpu().detach().numpy())
        folding.append(folding_ratio)
        mag_det_jac.append(mag_det_jac_det)

        if args.save:
            save_dir = os.path.join("./result", now)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, f"{i}.jpg")
            save_flow(fixed_image,moving_image,fixed_label,moving_label,ddf,pred_image,pre_label,save_path)

    print("dice  mean: ",round(np.mean(metric), 3))
    print("dice  std: ",round(np.std(metric), 3))
    print("HD  mean: ",round(np.mean(HD), 3))
    print("HD  std: ",round(np.std(HD), 3))
    print("MSD  mean: ",round(np.mean(MSD), 3))
    print("MSD  std: ",round(np.std(MSD), 3))
    print("JAC  mean: ", round(np.mean(mag_det_jac), 3))
    print("JAC  std: ", round(np.std(mag_det_jac), 3))
    print("Folding Ratio  mean: ", round(np.mean(folding), 3))
    print("Folding Ratio  std: ", round(np.std(folding), 3))
    print('****************************************')
    print('****************************************')
    # pro_mix = []
    # for i in range(52):
    #     pro_mix.append("Shanghai East Hospital")
    # for i in range(830):
    #     pro_mix.append("μ-RegPro")
    # for i in range(790):
    #     pro_mix.append("The Cancer Imaging Archive")
    # methods = "UTMorph"
    #
    # df = pd.DataFrame({
    #     '数据名称': pro_mix,
    #     '方法名字': methods,
    #     'DICE值': metric
    # })
    #
    # # 写入Excel文件
    # output_file = 'result/UTMorph.xlsx'
    # df.to_excel(output_file, index=False)