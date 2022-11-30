import torch
import numpy as np
import os
import itertools
import cv2
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from Load_LIDC_Data import LIDC_IDRI
from model.Net import Net
from tqdm import tqdm
from utils import *

torch.cuda.set_device(0)

#######################################################
BATCH_SIZE = 12
EPOCH = 201
epoch = 1       # initial epoch
Continue_Train = False
data_path = './data/'     # the path for LIDC data pickle file
singleAnnotation = False     # use single or four annotation for training
use_mKL_V = False  # mitigate KL-Vanishing , only work when singleAnnotation = True
checkpoint_dir = './source/model/checkpoint'
result_dir = './source/model/result'
samples_dir = './source/model' # mask sure there is an image for sample (visualize)
KL_beta = 10    # range 1 ~ 10
#######################################################




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = LIDC_IDRI(dataset_location=data_path, anno1=singleAnnotation)
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.2 * dataset_size))
np.random.shuffle(indices)
train_val_indices, test_indices = indices[split:], indices[:split]

split = int(np.floor(0.2 * len(train_val_indices)))
train_indices, val_indices = train_val_indices[split:], train_val_indices[:split]


filters=[32, 64, 128, 192, 192]
net = Net(n_channels=1, device=device, batch_size=BATCH_SIZE, n_sample=1,
                  latent_dim=8, filters=filters, training=True, use_mKL_V=use_mKL_V)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)

GED = []
H_IoU = []
Diversity = []
NCC = []
DICE = []
DICE_S = []

filename = os.path.join(checkpoint_dir, 'model.pth.tar')
besttestname = os.path.join(checkpoint_dir, 'model_test_best.pth.tar')
bestvalname = os.path.join(checkpoint_dir, 'model_val_best.pth.tar')
historyname = os.path.join(checkpoint_dir, 'model.txt')
historyKLname = os.path.join(checkpoint_dir, 'model_KL.txt')
samplepath = samples_dir
if Continue_Train and os.path.isfile(filename):
    print("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    train_indices = checkpoint['train_indices']
    val_indices = checkpoint['val_indices']
    test_indices = checkpoint['test_indices']
    GED = checkpoint['GED']
    H_IoU = checkpoint['H_IoU']
    Diversity = checkpoint['Diversity']
    NCC = checkpoint['NCC']
    DICE = checkpoint['DICE']
    DICE_S = checkpoint['DICE_S']
    print("=> model load success")
    

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, drop_last=True)
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler, drop_last=True)
test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler, drop_last=True)
print("Number of training/val/test patches:", (len(train_indices),len(val_indices),len(test_indices)))


def train():
    try:
        best_GED = np.inf
        is_test_best = False
        best_valLoss = np.inf
        is_val_best = False
        for epo in range(epoch, EPOCH):
            net.train()
            bar = tqdm(train_loader)
            for step, (image, mask, all_masks, index, _) in enumerate(bar):
                bar.set_description('Epoch %i' % epo)
                image = image.to(device)
                mask = mask.to(device)
                
                dist_I, dist_M, rec_I_with_I, rec_I_with_M, rec_M = net(image, mask)
                
                #loss_rec_I = - net.elbo_rec_I(rec_I_with_I, image, dist_I, dist_M)
                rec_loss, kl_loss = net.elbo_rec_M(rec_I_with_M, image, rec_M, mask, dist_I, dist_M)
                
    
                loss = rec_loss + KL_beta * kl_loss
                reg_loss = l2_regularisation(net.Iencoder) + \
                            l2_regularisation(net.Idecoder) + \
                            l2_regularisation(net.Mencoder) + \
                            l2_regularisation(net.Mdecoder) + \
                            l2_regularisation(net.SampleLatent)
                loss = loss + 1e-5 * reg_loss
                
                #optimizer = opt1 if step % 2 == 0 else opt2
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bar.set_postfix(rec_loss=rec_loss.item(),
                                kl_loss=kl_loss.item())
                
            #val(net)
            if (epo) % 30 == 0:
                val_loss, KL = validation(net)
                #recordKL(epo, KL, val_loss)
                #sample(epo)
                ged, h_iou, diversity, ncc, dice, dice_s = test(net)
                
                write_history(epo, ged, h_iou, diversity, ncc, dice, dice_s, val_loss)
                is_test_best = True if best_GED > ged else False
                best_GED = ged if is_test_best else best_GED
                is_val_best = True if best_valLoss > val_loss else False
                best_valLoss = val_loss if is_val_best else best_valLoss
                GED.append(ged)
                H_IoU.append(h_iou)
                Diversity.append(diversity)
                NCC.append(ncc)
                DICE.append(dice)
                DICE_S.append(dice_s)
                print('save checkpoint ...')
                save_checkpoint({
                    'epoch': epo,
                    'state_dict': net.state_dict(),
                    'GED': GED,
                    'H_IoU': H_IoU,
                    'Diversity': Diversity,
                    'NCC': NCC,
                    'DICE': DICE,
                    'DICE_S': DICE_S,
                    'optimizer' : optimizer.state_dict(),
                    'train_indices': train_indices,
                    'val_indices': val_indices,
                    'test_indices': test_indices
                }, is_test_best, is_val_best, filename, besttestname, bestvalname)
                print('save checkpoint success!')
                
    except KeyboardInterrupt:
        pass
    except:
        raise

def recordKL(epoch, KL, val_loss):
    with open(historyKLname, 'a+') as f:
        f.write('#epoch: {0}, KL: {1}, val_loss: {2}\n'.format(
                epoch, KL, val_loss))
        
def write_history(epoch, GED, H_IoU, Diversity, NCC, Dice, singleDice, val_loss):
    with open(historyname, 'a+') as f:
        f.write('#epoch: {0}, GED: {1}, H_IOU: {2}, Diversity: {3} NCC: {4}, Dice: {5}, singleDice: {6}, val_loss: {7}\n'.format(
                epoch, GED, H_IoU, Diversity, NCC, Dice, singleDice, val_loss))

def validation(net):
    net.eval()
    with torch.no_grad():
        bar = tqdm(val_loader)
        total_loss = 0
        KL = 0
        for step, (image, mask, all_masks, index, _) in enumerate(bar):
            bar.set_description('Val')
            image = image.to(device)
            mask = mask.to(device)
            
            dist_I, dist_M, rec_I_with_I, rec_I_with_M, rec_M = net(image, mask)
            
            rec_loss, kl_loss = net.elbo_rec_M(rec_I_with_M, image, rec_M, mask, dist_I, dist_M)
            
            loss = rec_loss + KL_beta * kl_loss
            
            bar.set_postfix(loss=loss.item())
            
            total_loss += loss.item()
            KL += kl_loss.item()
        
        total_loss /= len(bar)
        KL /= len(bar)
    
    net.train()
    return total_loss, KL

def sample(epoch):
    image = os.path.join(samples_dir, 'img.jpg')
    image = cv2.imread(image)
    image = image[:,:,0:1].transpose((2, 0, 1)) / 255
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = torch.unsqueeze(image, 0)
    
    os.mkdir(os.path.join(samplepath, 'sample-' + str(epoch)))
    
    net.eval()
    with torch.no_grad():
        n_sample = 32
        image = image.to(device)
        net.save_tensor_for_sample(image)
        rec_M = net.sample(image, n_sample)
        pred = []
        for i in range(rec_M.size(0)):
            rec_m = rec_M[i:i+1, ...]
            rec_m = torch.squeeze(rec_m)
            pred.append(process(rec_m))
            rec_m = process(rec_m) * 255
            rec_m = rec_m.cpu().numpy()
            cv2.imwrite(os.path.join(samplepath, 'sample-' + str(epoch), 'sample-'+str(i)+'.jpg'), rec_m)
    pred = torch.stack(pred, dim=0)
    pred = torch.squeeze(pred)
    pred = torch.mean(pred, 0)
    
    heatmapshow = None
    heatmapshow = cv2.normalize(pred.cpu().numpy(), heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(samplepath, 'sample-' + str(epoch), 'hotmap.jpg'), heatmapshow)
    

def process(s, is_project=True):
    if is_project:
        s = (torch.sigmoid(s) > 0.5).float()
    else:
        s = torch.sigmoid(s)
    s = torch.squeeze(s)
    return s

def test(net):
    net.eval()
    with torch.no_grad():
        n_sample = 32
        D = 0
        h_IoU = 0
        Diversity = 0
        Ncc = 0
        Dice = 0
        Dice_single = 0
        bar = tqdm(test_loader)
        for step, (image, mask, all_masks, index, _) in enumerate(bar):
            bar.set_description('Test')
            image = image.to(device)
            mask = mask.to(device)
            all_masks = all_masks.to(device)
            
            samples = [] # for compute generalized energy distance
            sig_samples = [] # for compute ncc
            process_samples = [] # for visualize
            all_masks = torch.squeeze(all_masks, 0) # squeeze batch
            mask = torch.squeeze(mask) # squeeze batch and channel
            groundtruths = [all_masks[i] for i in range(all_masks.size(0))]
            net.save_tensor_for_sample(image)
            rec_M = net.sample(image, n_sample)
            for i in range(rec_M.size(0)):
                rec_m = rec_M[i:i+1, ...]
                samples.append(process(rec_m))
                sig_samples.append(process(rec_m, is_project=False))
                if i < 8 and step <= 200:
                    process_samples.append(process(rec_m) * 255)
            ged, diversity = generalized_energy_distance(samples, groundtruths)
            D += ged
            Diversity += diversity
            h_IoU += HM_IoU(samples, groundtruths)
            Ncc += variance_ncc_dist(sig_samples, groundtruths)
            Dice += mean_dice(samples, groundtruths)
            Dice_single += mean_dice(samples, [groundtruths[0]])
            if step <= 200:
                groundtruths = [all_masks[i]*255 for i in range(all_masks.size(0))]
                image = torch.squeeze(image) * 255
                IM = organize_image(process_samples, groundtruths, image, size=(4, 4))
                cv2.imwrite(os.path.join(result_dir, 'IM_{0}.jpg').format(step), IM)
        GED = D / len(test_indices)
        H_IoU = h_IoU / len(test_indices)
        NCC = Ncc / len(test_indices)
        DICE = Dice / len(test_indices)
        DICE_S = Dice_single / len(test_indices)
        print("generalized energy distance: {0}, H_IoU:{1}, Diversity:{2} ncc: {3}, mean dice: {4}, single dice: {5}".format(
                                            GED, H_IoU, Diversity, NCC, DICE, DICE_S))
    net.train()
    return GED, H_IoU, Diversity, NCC, DICE, DICE_S
    

def quanti(net):
    net.eval()
    with torch.no_grad():
        n_sample = 32
        determ_Dice = 0
        ambu_Dice = 0
        
        bar = tqdm(test_loader)
        for step, (image, mask, all_masks, index, _) in enumerate(bar):
            bar.set_description('Quantitative')
            image = image.to(device)
            mask = mask.to(device)
            all_masks = all_masks.to(device)
            
            all_masks = torch.squeeze(all_masks, 0) # squeeze batch
            determ = torch.ones(128, 128).to(device)
            for i in range(4):
                if torch.sum(all_masks[i]) != 0:
                    determ = torch.logical_and(determ, all_masks[i]) * 1.0
            
#            ambu = torch.zeros(128, 128).to(device)
#            for i in range(4):
#                if torch.sum(all_masks[i]) != 0:
#                    ambu = torch.logical_or(ambu, all_masks[i]) * 1.0
#            ambu -= determ
            
            samples = [] # for compute generalized energy distance
            sample_ambu = torch.zeros(128, 128).to(device)
            sample_determ = torch.ones(128, 128).to(device)
            net.save_tensor_for_sample(image)
            rec_M = net.sample(image, n_sample)
            for i in range(rec_M.size(0)):
                rec_m = rec_M[i:i+1, ...]
                rec_m = process(rec_m)
                samples.append(rec_m)
                
                
                if torch.sum(rec_m) != 0:
                    sample_determ = torch.logical_and(sample_determ, rec_m) * 1.0
                    sample_ambu = torch.logical_or(sample_ambu, rec_m) * 1.0
            
            determ_Dice += mean_dice(samples, [determ])
#            ambu_Dice += mean_dice(samples, [ambu])
            sample_ambu -= sample_determ
            ambu_Dice += torch.sum(sample_ambu) / (128*128)
            
        
        determ_Dice /= len(test_loader)
        ambu_Dice /= len(test_loader)
        print(f'determ_Dice: {determ_Dice}; ambu_Dice: {ambu_Dice}')

def sample_new():
    image = os.path.join(samples_dir, 'image.jpg')
    image = cv2.imread(image)
    image = image[:,:,0:1].transpose((2, 0, 1)) / 255
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = torch.unsqueeze(image, 0)
    
    net.eval()
    with torch.no_grad():
        n_sample = 32
        image = image.to(device)
        net.save_tensor_for_sample(image)
        rec_M = net.sample(image, n_sample)
        pred = []
        for i in range(rec_M.size(0)):
            rec_m = rec_M[i:i+1, ...]
            rec_m = torch.squeeze(rec_m)
            rec_m = process(rec_m)
            if torch.sum(rec_m) != 0:
                pred.append(rec_m)
#            rec_m = process(rec_m) * 255
#            rec_m = 255 - rec_m.cpu().numpy()
#            cv2.imwrite(os.path.join(samplepath, 'samples', 'sample-'+str(i)+'.jpg'), rec_m)
    pred = torch.stack(pred, dim=0)
    pred = torch.squeeze(pred)
    pred = torch.std(pred, 0)
    
    heatmapshow = None
    heatmapshow = cv2.normalize(pred.cpu().numpy(), heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(samplepath, 'samples', 'hotmap.jpg'), heatmapshow)

train()








