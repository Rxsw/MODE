from torch.utils.data import DataLoader
import torch
from utils.resnet import *
from torch import nn
from utils.datasets import *
import os
import argparse
from utils.model import *
from utils.data_reader import *
from utils.utils import *
from torchvision.utils import save_image
import torch.fft
from utils.MODE import MODE_A, MODE_F

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--times", default=1, type=int)
    # Datasets and model Settings
    parser.add_argument("--data_path", default="./data", type=str)
    parser.add_argument("--output_path", default="./output", type=str)
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument("--model", default="resnet18", type=str)
    parser.add_argument("--dataset", default="minidomainnet", type=str)
    parser.add_argument("--test_index", default=1, type=int)
    # Adversarial Exploration Settings
    parser.add_argument("--mode", default="A", type=str, help="F or A")
    parser.add_argument("--move_step", default=10, type=int)
    parser.add_argument("--gamma", default=1, type=float)
    parser.add_argument("--num_mix", default=4, type=int)
    parser.add_argument("--mu", default=0.05, type=float)
    parser.add_argument("--beta", default=0.4, type=float)

    # Optim Settings 1
    parser.add_argument("--optim", default="sgd", type=str)
    parser.add_argument("--sche", default="cosin", type=str)
    parser.add_argument("--num_epochs", default=60, type=int)
    parser.add_argument("--lr", default=5e-3, type=float)
    parser.add_argument("--batch_size", default=96, type=int)
    # Optim Settings 2
    parser.add_argument("--save_epoch", default=1, type=int)
    parser.add_argument("--seed", default=-1, type=int, help="-1 means random")
    parser.add_argument("--test_batch_size", default=512, type=int)
    parser.add_argument("--decay_ratio", default=0.8, type=float)
    parser.add_argument("--decay_gamma", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    # Sp Settings
    parser.add_argument("--limit_number_per_class", default=480, type=int, help="just for digit4 dataset,the limit of numbers of sample")
    args = parser.parse_args()
    return args

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    
    for i in range(args.times):
        my_trainer = Trainer(device, args, i)
        my_trainer.train(args)

class Trainer(object):
    def __init__(self, device, args, times):
        self.current_times = times
        self.args = args
        self.device = device
        # -------------------------------------------------------  
        self.data_path = args.data_path
        self.output_path = args.output_path
        self.num_workers = args.num_workers
        if not os.path.exists(self.output_path+'/results'):
            os.makedirs(self.output_path+'/results')
        # -------------------------------------------------------
        self.criterion = nn.CrossEntropyLoss()
        # -------------------------------------------------------
        self.seed_setting(args.seed)
        self.pre_dataset(args)
        self.model_init(args)
    
    def seed_setting(self, seed):
        # -------------------------------------------------------
        if seed == -1:
            seed = np.random.randint(1000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        print("#########")
        print("Seed:", seed)
        print("#########")
        # -------------------------------------------------------
    
    def pre_dataset(self, args):
        test_index = args.test_index
        batch_size = args.batch_size
        
        if args.dataset == "Digit":
            self.num_classes = 10
            self.num_domains_in_train = 3
            train_name = ["mnist", "mnist_m", "svhn", "syn"]
            test_name = train_name.pop(test_index)
            
            norm_mean = [0.5, 0.5, 0.5]
            norm_std = [0.5, 0.5, 0.5]
            train_transform = transforms.Compose([transforms.Resize(32),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(norm_mean, norm_std)])

            test_transform = transforms.Compose([transforms.Resize(32),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(norm_mean, norm_std)])
            self.train_set = []
            for i in range(len(train_name)):
                self.train_set.append(ImageFolder(
                    root=self.data_path+'/digit/{}'.format(train_name[i]), transform=train_transform, limit_number_per_class=args.limit_number_per_class))

            self.train_set = ConcatDataset(self.train_set)
            self.train_loader = DataLoader(
                self.train_set, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
            
            self.val_set = []
            for i in range(len(train_name)):
                self.val_set.append(ImageFolder(
                    root=self.data_path+'/digit/{}'.format(train_name[i]), transform=test_transform, limit_number_per_class=120))

            self.val_set = ConcatDataset(self.val_set)
            self.val_loader = DataLoader(
                self.val_set, batch_size=args.test_batch_size, shuffle=True, num_workers=self.num_workers)


            self.test_set = ImageFolder(
                root=self.data_path+'/digit/{}'.format(test_name), transform=test_transform, limit_number_per_class=600)
            self.test_loader = DataLoader(
                self.test_set, batch_size=args.test_batch_size, shuffle=True, num_workers=self.num_workers)
            
        elif args.dataset == "PACS":
            
            self.num_classes = 7
            self.num_domains_in_train = 3
            source_root = self.data_path+'/PACS'
            train_name = ["photo", "art_painting", "cartoon", "sketch"]
            test_name = train_name.pop(test_index)
            
            train_datalists_root = []
            for i in range(len(train_name)):
                train_datalists_root.append(
                    "./datalists/{}_train.txt".format(train_name[i]))
            
            val_datalists_root = []
            for i in range(len(train_name)):
                val_datalists_root.append(
                    "./datalists/{}_val.txt".format(train_name[i]))

            test_datalists_root = "./datalists/{}_test.txt".format(test_name)
            
            norm_mean = [0.485, 0.456, 0.406]
            norm_std = [0.229, 0.224, 0.225]
            train_transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=[0.8, 1.0]),
                                                  transforms.ColorJitter(
                                                      brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(norm_mean, norm_std)])

            test_transform = transforms.Compose([transforms.Resize((224,224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(norm_mean, norm_std)])
            self.train_set = []
            for i in range(len(train_name)):
                self.train_set.append(ImageFolder(
                    root=self.data_path+'/PACS/{}'.format(train_name[i]), transform=train_transform, datalists_path=train_datalists_root[i], source_root=source_root))

            self.train_set = ConcatDataset(self.train_set)
            self.train_loader = DataLoader(
                self.train_set, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
            
            self.val_set = []
            for i in range(len(train_name)):
                self.val_set.append(ImageFolder(
                    root=self.data_path+'/PACS/{}'.format(train_name[i]), transform=test_transform, datalists_path=val_datalists_root[i], source_root=source_root))

            self.val_set = ConcatDataset(self.val_set)
            self.val_loader = DataLoader(
                self.val_set, batch_size=args.test_batch_size, shuffle=True, num_workers=self.num_workers)
            
            self.test_set = ImageFolder(
                root=self.data_path+'/PACS/{}'.format(test_name), transform=test_transform, datalists_path=test_datalists_root, source_root=source_root)
            self.test_loader = DataLoader(
                self.test_set, batch_size=args.test_batch_size, shuffle=True, num_workers=self.num_workers)
        
        elif args.dataset == "VLCS":
            self.num_classes = 5
            self.num_domains_in_train = 3
            source_root = self.data_path+'/VLCS'
            train_name = ["PASCAL", "LABELME", "CALTECH", "SUN"]
            test_name = train_name.pop(test_index)

            
            norm_mean = [0.485, 0.456, 0.406]
            norm_std = [0.229, 0.224, 0.225]
            train_transform = transforms.Compose([transforms.Resize((224,224)),
                                                  Random2DTranslation(224,224),
                                                  transforms.ColorJitter(
                                                      brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                                                  transforms.RandomHorizontalFlip(),
                                                  RandomGrayscale(p=0.2),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(norm_mean, norm_std)])

            test_transform = transforms.Compose([transforms.Resize((224,224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(norm_mean, norm_std)])
            self.train_set = []
            for i in range(len(train_name)):
                self.train_set.append(ImageFolder(
                    root=self.data_path+'/VLCS/{}/train'.format(train_name[i]), transform=train_transform))

            self.train_set = ConcatDataset(self.train_set)
            self.train_loader = DataLoader(
                self.train_set, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
            
            self.val_set = []
            for i in range(len(train_name)):
                self.val_set.append(ImageFolder(
                    root=self.data_path+'/VLCS/{}/crossval'.format(train_name[i]), transform=test_transform, limit_number_per_class=10))

            self.val_set = ConcatDataset(self.val_set)
            self.val_loader = DataLoader(
                self.val_set, batch_size=args.test_batch_size, shuffle=True, num_workers=self.num_workers)
            
            self.testset = ImageFolder(
                root=self.data_path+'/VLCS/{}/test'.format(test_name), transform=test_transform)
            self.test_loader = DataLoader(
                self.testset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
        
        
        elif args.dataset == "minidomainnet":
            self.num_classes = 126
            self.num_domains_in_train = 3
            source_root = self.data_path+'/minidomainnet'
            train_name = ["clipart", "painting", "real", "sketch"]
            test_name = train_name.pop(test_index)
            
            train_datalists_root = []
            for i in range(len(train_name)):
                train_datalists_root.append(
                    self.data_path+"/minidomainnet/splits_mini/{}_train.txt".format(train_name[i]))
            
            val_datalists_root = []
            for i in range(len(train_name)):
                val_datalists_root.append(
                    self.data_path+"/minidomainnet/splits_mini/{}_test.txt".format(train_name[i]))

            test_datalists_root = self.data_path+"/minidomainnet/splits_mini/{}_test.txt".format(test_name)
            
            norm_mean = [0.485, 0.456, 0.406]
            norm_std = [0.229, 0.224, 0.225]
            train_transform = transforms.Compose([transforms.Resize((96,96)),
                                                  Random2DTranslation(96,96),
                                                  transforms.ColorJitter(
                                                      brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                                                  transforms.RandomHorizontalFlip(),
                                                  RandomGrayscale(p=0.2),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(norm_mean, norm_std)])

            test_transform = transforms.Compose([transforms.Resize((96,96)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(norm_mean, norm_std)])
            self.train_set = []
            for i in range(len(train_name)):
                self.train_set.append(ImageFolder(
                    root=self.data_path+'/minidomainnet/{}'.format(train_name[i]), transform=train_transform, datalists_path=train_datalists_root[i], source_root=source_root))

            self.train_set = ConcatDataset(self.train_set)
            self.train_loader = DataLoader(
                self.train_set, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
            
            self.val_set = []
            for i in range(len(train_name)):
                self.val_set.append(ImageFolder(
                    root=self.data_path+'/minidomainnet/{}'.format(train_name[i]), transform=test_transform, datalists_path=val_datalists_root[i], source_root=source_root))

            self.val_set = ConcatDataset(self.val_set)
            self.val_loader = DataLoader(
                self.val_set, batch_size=args.test_batch_size, shuffle=True, num_workers=self.num_workers)
            
            self.test_set = ImageFolder(
                root=self.data_path+'/minidomainnet/{}'.format(test_name), transform=test_transform, datalists_path=test_datalists_root, source_root=source_root)
            self.test_loader = DataLoader(
                self.test_set, batch_size=args.test_batch_size, shuffle=True, num_workers=self.num_workers)

        self.norm_mean = norm_mean
        self.norm_std = norm_std

        print("#########")
        print("test_dataset:", test_name)
        print("#########")

    def model_init(self, args):
        # ------------------------------------------------------
        # task model
        if args.dataset == "Digit":
            self.task_model = ClassC(num_classes=self.num_classes)
            self.task_model.to(self.device)
            self.task_model_optimizer = torch.optim.SGD(
                self.task_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            self.task_model_scheduler = torch.optim.lr_scheduler.StepLR(
                self.task_model_optimizer, step_size=int(args.decay_ratio*args.num_epochs), gamma=args.decay_gamma, last_epoch=-1)
        
        elif args.dataset == "PACS"  or args.dataset == "VLCS" or args.dataset == "minidomainnet":
            if args.model == "resnet18":
                self.task_model = resnet18(
                    pretrained=False, num_classes=self.num_classes)
                weight = torch.load(self.data_path+"/resnet18-5c106cde.pth")
                weight['fc.weight'] = self.task_model.state_dict()['fc.weight']
                weight['fc.bias'] = self.task_model.state_dict()['fc.bias']
                self.task_model.load_state_dict(weight)          
            elif args.model == "resnet50":
                self.task_model = resnet50(
                    pretrained=False, num_classes=self.num_classes)
                weight = torch.load(self.data_path+"/resnet50-0676ba61.pth")  
                weight['fc.weight'] = self.task_model.state_dict()['fc.weight']
                weight['fc.bias'] = self.task_model.state_dict()['fc.bias']
                self.task_model.load_state_dict(weight)          
            elif args.model == "alexnet":
                self.task_model = alexnet(
                    pretrained=False, model_root=self.data_path+"/", num_classes=self.num_classes)     
                weight = torch.load(self.data_path+"/alexnet-owt-4df8aa71.pth")  
                weight['classifier.6.weight'] = self.task_model.state_dict()['classifier.6.weight']
                weight['classifier.6.bias'] = self.task_model.state_dict()['classifier.6.bias']
                self.task_model.load_state_dict(weight) 
            
            self.task_model.to(self.device)
            
            if args.optim == "sgd":
                self.task_model_optimizer = torch.optim.SGD(
                    self.task_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            elif args.optim == "adam":
                self.task_model_optimizer = torch.optim.Adam(
                    self.task_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            if args.sche == "steplr":        
                self.task_model_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.task_model_optimizer, step_size=int(args.decay_ratio*args.num_epochs), gamma=args.decay_gamma, last_epoch=-1)
            elif args.sche == "cosin":
                self.task_model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = self.task_model_optimizer,
                                                        T_max = args.num_epochs)
            elif args.sche == "constant":
                self.task_model_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.task_model_optimizer, step_size=args.num_epochs+1, gamma=args.decay_gamma, last_epoch=-1)

        # ------------------------------------------------------
        # MODE_A
        if args.mode == "A":
            self.MODE = MODE_A(
                decoder_weights="./decoder.pth",
                vgg_weights="./vgg_normalised.pth",
                device=self.device,
                gamma=args.gamma,
                beta=args.beta,
                num_mix=args.num_mix,
                move_step=args.move_step,
                mu=args.mu,
                criterion=self.criterion,
                norm_mean = [0.485, 0.456, 0.406],
                norm_std = [0.229, 0.224, 0.225]
            )
        # MODE_F
        elif args.mode == "F":
            self.MODE = MODE_F(
                device=self.device,
                gamma=args.gamma,
                num_mix=args.num_mix,
                move_step=args.move_step,
                mu=args.mu,
                criterion=self.criterion,
            )
        print("#########")
        print("Train_mode_based : {}".format(args.mode))
        print("#########") 
        # ------------------------------------------------------

    def train(self, args):
        
        val_acc_best = 0.0
        test_acc_best = 0.0
        val_acc_save = []
        test_acc_save = []
        val_loss_save = []
        test_loss_save = []
        train_loss_save = [[],[],[]]
        avg_loss_epochs = [[],[],[]]

        for e in range(1, args.num_epochs + 1):
            
            torch.cuda.empty_cache()

            avg_loss_real = 0
            avg_loss_final = 0
            avg_loss_mid = torch.zeros((args.move_step), dtype=torch.float)

            for it, ((x, label), domain) in enumerate(self.train_loader):
                
                #########################################################################################################################
                save_flag = (it == len(self.train_loader)-2)

                x = x.to(device=self.device)
                label = label.to(device=self.device)
                domain = domain.to(device=self.device)
                
                if args.beta == 0:
                    self.task_model.train()
                    self.task_model_optimizer.zero_grad()
                    score = self.task_model(x)
                    Loss_real = self.criterion(score, label)
                    Loss_real.backward()
                    self.task_model_optimizer.step()
                    
                    if (it == len(self.train_loader)-2):
                        print("#######")
                        print()
                        print('Task epoch {}'.format(e))
                        print()
                        print('Loss_real:{}'.format(Loss_real.item()))
                        print()
                        print("#######")
                else:
                    self.task_model.eval()
                    x_final, log = self.MODE.main(x.clone(), label, self.task_model, domain, save_flag=save_flag)
                    
                    self.task_model.train()
                    self.task_model_optimizer.zero_grad()

                    input = torch.cat([x, x_final],dim=0).detach()
                    
                    score = self.task_model(input)
                    score_real, score_final = torch.chunk(score, 2, dim=0)
                    
                    Loss_real = self.criterion(score_real, label)
                    
                    Loss_final = self.criterion(score_final, label)
                    
                    Loss_total = (1-args.beta) * Loss_real + args.beta * Loss_final

                    Loss_total.backward()

                    self.task_model_optimizer.step()
                    
                    avg_loss_real += Loss_real.item()
                    avg_loss_final += Loss_final.item()
                    avg_loss_mid += log["loss_temp"]

                    train_loss_save[0].append(Loss_real.item())
                    train_loss_save[2].append(Loss_final.item())

                    if (it == len(self.train_loader)-2):
                        print("#######")
                        print()
                        print('Task epoch {}'.format(e))
                        print()
                        print("change of loss during searching :", log["loss_temp"])
                        print()
                        print("part of final_weight_norm :", log["normalized_weight_temp"][-1][8:10])
                        print()       
                        print('Loss_real:{}'.format(Loss_real.item()))
                        print('Loss_final:{}'.format(Loss_final.item()))
                        
                        print()
                        print("#######")
                        
                        if e == args.num_epochs - 1 or e % 3 == 0:
                            img_temp = torch.cat(torch.chunk(log["save_img_temp"], args.batch_size, dim=0), dim=1).squeeze(0)
                            save_image(denormalize(img_temp, self.norm_mean, self.norm_std), self.output_path +
                                    '/results/{}_epoch_{}.jpg'.format(self.current_times, e), nrow=args.move_step + 2)
                    
            
            self.task_model_scheduler.step()

            avg_loss_epochs[0].append(avg_loss_real / len(self.train_loader))
            avg_loss_epochs[1].append(avg_loss_mid / len(self.train_loader))
            avg_loss_epochs[2].append(avg_loss_final / len(self.train_loader))

            if e % args.save_epoch == 0:
                print("------------------------------------")
                print('Acc of test in epoch : {}'.format(e))
                acc_val, loss_val = self.val(num_batch=1000)
                # acc_val, loss_val = self.test(num_batch=10)
                acc_test, loss_test = self.test(num_batch=1000)  
                val_acc_save.append(acc_val)
                test_acc_save.append(acc_test)
                val_loss_save.append(loss_val)
                test_loss_save.append(loss_test)
                if acc_val > val_acc_best:
                    val_acc_best = acc_val
                    corr_test_acc = acc_test
                    val_best_epoch = e
                    # torch.save(self.task_model.state_dict(),
                    #         f=self.save_path+"/BEST_in_{}.pth".format(t))
                    print("Val New best {} ".format(val_acc_best))
                if acc_test > test_acc_best:
                    test_acc_best = acc_test
                    test_best_epoch = e
                    torch.save(self.task_model.state_dict(),
                            f=self.output_path+"/{}_BEST.pth".format(self.current_times))
                    print("Test New best {} ".format(test_acc_best))
                print("------------------------------------")
            #################################################################################################################################################
        
        print("Best val acc:{}, corr to test acc:{}, in epoch:{}".format(val_acc_best, corr_test_acc, val_best_epoch))
        print("Best test acc:{}, in epoch:{}".format(test_acc_best, test_best_epoch))

        log = {}
        log["train_loss"] = train_loss_save
        log["val_loss"] = val_loss_save
        log["val_acc"] = val_acc_save
        log["test_loss"] = test_loss_save
        log["test_acc"] = test_acc_save
        log["best_val"] = [val_acc_best, corr_test_acc, val_best_epoch]
        log["best_test"] = [test_acc_best, test_best_epoch]
        log["avg"] = avg_loss_epochs
        torch.save(log, f=self.output_path+"/log_{}.pth".format(self.current_times))
        
        return
    
    def val(self, num_batch=100):
        num_correct = 0
        num_samples = 0
        count = 0
        loss = 0
        model = self.task_model
        model.eval()
        with torch.no_grad():
            for it, ((x, y), domain) in enumerate(self.val_loader):
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                out = model(x)
                loss += self.criterion(out, y)
                _, preds = out.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
                count = count+1
                if count > num_batch:
                    break
            acc = float(num_correct) / num_samples
            print('Val Got %d / %d correct (%.2f)' %
                  (num_correct, num_samples, 100 * acc))
            loss = loss/count
            print('avgloss:%.5f' % (loss))
        return acc, loss
    
    def test(self, num_batch=100, loader = None):
        num_correct = 0
        num_samples = 0
        count = 0
        loss = 0
        loader = self.test_loader if loader is None else loader
        model = self.task_model
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                out = model(x)
                loss += self.criterion(out, y)
                _, preds = out.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
                count = count+1
                if count > num_batch:
                    break
            acc = float(num_correct) / num_samples
            print('Test Got %d / %d correct (%.2f)' %
                  (num_correct, num_samples, 100 * acc))
            loss = loss/count
            print('avgloss:%.5f' % (loss))
        return acc, loss

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
