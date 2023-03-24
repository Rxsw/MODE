import torch
import torch.nn as nn
import torch.fft
import random

class MODE_A:
    def __init__(
        self,
        decoder_weights,
        vgg_weights,
        device,
        gamma=1,
        beta=1,
        num_mix=8,
        move_step=5,
        mu=0.05,
        criterion=nn.CrossEntropyLoss(),
        norm_mean=None,
        norm_std=None,
    ):
        self.device = device
        self.gamma = gamma
        self.beta = beta
        self.num_mix = num_mix
        self.move_step = move_step
        self.mu = mu
        self.criterion = criterion
        self.undo_norm = None
        self.norm = None

        if norm_mean is not None and norm_std is not None:
            self.undo_norm = UndoNorm(norm_mean, norm_std)
            self.norm = Norm(norm_mean, norm_std)

        self.build_models(decoder_weights, vgg_weights)

    def build_models(self, decoder_weights, vgg_weights):
        print("Building vgg and decoder for style transfer")

        self.decoder = decoder
        self.vgg = vgg

        self.decoder.eval()
        self.vgg.eval()

        print(f"Loading decoder weights from {decoder_weights}")
        self.decoder.load_state_dict(torch.load(decoder_weights))

        print(f"Loading vgg weights from {vgg_weights}")
        self.vgg.load_state_dict(torch.load(vgg_weights))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])

        self.vgg.to(self.device)
        self.decoder.to(self.device)

        for param in self.vgg.parameters():
            param.requires_grad = False

        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert len(size) == 4
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std
    
    def shuffle_data(self, data):
        num = data.size(0)
        index = torch.randint(0, num, (num,), device=self.device)
        new_data = torch.index_select(data, dim=0, index=index)
        return new_data
    
    def shuffle_data_chose_domain(self, data, domain=None, chosen_idx=2):
        batch_size = data.size(0)
        data_accept = data[domain==chosen_idx]
        num_accept = data_accept.size(0)
        if num_accept == 0:
            return self.shuffle_data(data)
        random_index = torch.randint(0, num_accept, (batch_size,), device=self.device)
        new_data = torch.index_select(data_accept, dim=0, index=random_index)
        return new_data
    
    def prepare(self, x, domain=None, gamma=None, num_mix=None):
        
        # get feature map
        x_featuremap = self.vgg(x)
        featuremap_size = x_featuremap.size()
        # calc mean and std
        batch_mean, batch_std = self.calc_mean_std(x_featuremap)  #N,C,1,1
        # normalized the feature map
        normalized_x_featuremap = (x_featuremap - batch_mean.expand(featuremap_size)) / batch_std.expand(featuremap_size)
        # concate mean and std 
        mean_std = torch.cat([batch_mean.squeeze().unsqueeze(1),batch_std.squeeze().unsqueeze(1)],dim=1)  #N,2,C
        # use num_mix to get mean_std_set
        if num_mix == 4 and domain is not None:
            mean_std_set = torch.cat([mean_std.unsqueeze(1)]+[self.shuffle_data_chose_domain(mean_std, domain, idx).unsqueeze(1) for idx in range(3)], dim=1)
        elif num_mix == 6 and domain is not None:
            mean_std_set = torch.cat([mean_std.unsqueeze(1)]+[self.shuffle_data_chose_domain(mean_std, domain, idx).unsqueeze(1) for idx in range(6)], dim=1)
        elif gamma >= 1:
            mean_std_set = torch.cat([mean_std.unsqueeze(1)]+[self.shuffle_data(mean_std).unsqueeze(1) for _ in range(num_mix-1)], dim=1) #N,num_mix,2,C
        else:
            mean_std_set = torch.cat([self.shuffle_data(mean_std).unsqueeze(1) for _ in range(num_mix)], dim=1)  #N,num_mix,2,C

        return mean_std_set.detach(), x_featuremap.detach(), normalized_x_featuremap.detach()
    
    def init_weight(self, batch_size, num_mix):
        # init weight
        ori_weight = torch.ones((batch_size, num_mix), device=self.device) / num_mix
        # random init
        # ori_weight = torch.randn((batch_size, num_mix), device=self.device)*0.2 + 1 / num_mix
        # ori_weight = torch.clamp(ori_weight, min=0, max=1)
        return ori_weight

    def stylize(self, mix_weight, mean_std_set, x_featuremap, normalized_x_featuremap, gamma):
        featuremap_size = x_featuremap.size()
        mean_std_set_size = mean_std_set.size()
        # # normalized the weight
        weight_norm = mix_weight/(torch.sum(mix_weight, dim=1, keepdim=True)+1e-6)   # N,num_mix

        # calc mean_std mix
        mean_std_mix = torch.bmm(weight_norm.unsqueeze(1), mean_std_set.view(mean_std_set_size[0], mean_std_set_size[1], -1)).view(mean_std_set_size[0],mean_std_set_size[2],mean_std_set_size[3])
        
        # chunk mean_std to mean and std
        mean_mix, std_mix = torch.chunk(mean_std_mix,2,dim=1) # N,1,C

        mean_mix = mean_mix.squeeze(1).unsqueeze(2).unsqueeze(2)  # N,C,1,1
        std_mix = std_mix.squeeze(1).unsqueeze(2).unsqueeze(2)   # N,C,1,1
        # apply mean and std
        x_featuremap_mix = normalized_x_featuremap * std_mix.expand(featuremap_size) + mean_mix.expand(featuremap_size)
        # use gamma calc gamma mix featuremap
        x_featuremap_final = x_featuremap_mix * gamma + x_featuremap * (1 - gamma)
        # get final image
        x_fake = self.decoder(x_featuremap_final)   
        
        return x_fake, weight_norm
    
    def main(self, x, label, current_model, domain=None, classifiers=None, gamma=None, num_mix=None, move_step=None, mu=None, criterion=None, save_flag=False):

        if save_flag:            
            normalized_weight_temp = []
            save_img_temp = []  
            save_img_temp.append(x.clone().unsqueeze(1))
        
        gamma = self.gamma if gamma is None else gamma
        num_mix = self.num_mix if num_mix is None else num_mix
        move_step = self.move_step if move_step is None else move_step
        mu = self.mu if mu is None else mu
        criterion = self.criterion if criterion is None else criterion
        
        loss_temp = torch.zeros((move_step), dtype=torch.float)
        
        if self.undo_norm is not None:
            # Map pixel values to [0, 1]
            x = self.undo_norm(x)
        
        mean_std_set, x_featuremap, normalized_x_featuremap = self.prepare(x, domain, gamma, num_mix)

        ori_weight = self.init_weight(x.size(0), num_mix)
        
        mix_weight = ori_weight.detach_()

        # begin search
        

        if move_step != 0:
            random_idx = random.randint(0, move_step-1)
        else:
            random_idx = 0
        
        random_label = torch.randint_like(label, max(label)+1)
        

        for i in range(move_step):

            mix_weight.requires_grad = True
            
            x_fake, weight_norm = self.stylize(mix_weight, mean_std_set, x_featuremap, normalized_x_featuremap, gamma)
            
            # Normalize pixel values
            if self.norm is not None:
                x_fake = self.norm(x_fake)
            
            # forward
            if classifiers is None:
                score_fake = current_model(x_fake)
                loss = criterion(score_fake, label) # - criterion(score_fake, random_label)
            else:
                features = current_model(x_fake)
                score_fake = classifiers(features)
                loss = criterion(score_fake, label) # - criterion(score_fake, random_label)

            loss.backward()
            
            # update weight
            mix_weight = mix_weight + mu * mix_weight.grad.sign()
            eta = torch.clamp(
                mix_weight - ori_weight, min=-1, max=1)
            mix_weight = torch.clamp(
                ori_weight + eta, min=0, max=1).detach_()
            
            if i == 0:    
                x_ori = x_fake
            if i == random_idx:    
                x_random = x_fake
            
            loss_temp[i] = loss.item()
            # save_flag
            if save_flag:
                
                normalized_weight_temp.append(weight_norm)
                save_img_temp.append(x_fake.unsqueeze(1))

        
        x_fake, final_weight_norm = self.stylize(mix_weight, mean_std_set, x_featuremap, normalized_x_featuremap, gamma)
        # Normalize pixel values
        if self.norm is not None:
            x_fake = self.norm(x_fake)
        if save_flag:
            save_img_temp.append(x_fake.unsqueeze(1))
        log = {}
        if move_step != 0:
            log["x_ori"] = x_ori
            log["x_random"] = x_random
        log["random_idx"] = random_idx
        log["loss_temp"] = loss_temp
        # save_flag
        if save_flag:
            normalized_weight_temp.append(final_weight_norm)
            log["normalized_weight_temp"] = normalized_weight_temp
            log["save_img_temp"] = torch.cat(save_img_temp, dim=1)

        return x_fake, log
      
class MODE_F:
    """FFT
    """
    def __init__(
        self,
        device,
        gamma=1,
        num_mix=8,
        move_step=5,
        mu=0.05,
        criterion=nn.CrossEntropyLoss()
    ):
        """
        """
        assert 0 <= gamma <= 1
        self.device = device
        self.gamma = gamma
        self.num_mix = num_mix
        self.move_step = move_step
        self.mu = mu
        self.criterion = criterion
    
    def shuffle_data(self, data):
        num = data.size(0)
        index = torch.randint(0, num, (num,), device=self.device)
        new_data = torch.index_select(data, dim=0, index=index)
        return new_data
    
    def shuffle_data_chose_domain(self, data, domain=None, chosen_idx=2):
        batch_size = data.size(0)
        data_accept = data[domain==chosen_idx]
        num_accept = data_accept.size(0)
        if num_accept == 0:
            return self.shuffle_data(data)
        random_index = torch.randint(0, num_accept, (batch_size,), device=self.device)
        new_data = torch.index_select(data_accept, dim=0, index=random_index)
        return new_data
    
    def main(self, x, label, current_model, domain=None, classifiers=None, gamma=None, num_mix=None, move_step=None, mu=None, criterion=None, save_flag=False):
        """
        Input:
            gamma (float, optional): interpolation parameter within (0, 1]
        """
        gamma = self.gamma if gamma is None else gamma
        num_mix = self.num_mix if num_mix is None else num_mix
        move_step = self.move_step if move_step is None else move_step
        mu = self.mu if mu is None else mu
        criterion = self.criterion if criterion is None else criterion
        if save_flag:            
            normalized_weight_temp = []
            save_img_temp = []  
            save_img_temp.append(x.clone().unsqueeze(1))
        loss_temp = torch.zeros((move_step), dtype=torch.float)
        
        # calc fft of ori x
        x_fft = torch.fft.fft2(x, dim=[-2, -1])
        # calc abs and pha
        x_fft_abs, x_fft_pha = torch.abs(x_fft).detach(), torch.angle(x_fft).detach()
        # create set of abs and abs of random index, total num_mix
#         if num_mix == 4:
#             fft_set = torch.cat([x_fft_abs.unsqueeze(1)]+[self.shuffle_data_chose_domain(x_fft_abs, domain, idx).unsqueeze(1) for idx in range(3)], dim=1).detach()
#         elif num_mix == 7:
#             fft_set = torch.cat([x_fft_abs.unsqueeze(1)]+[self.shuffle_data_chose_domain(x_fft_abs, domain, idx).unsqueeze(1) for idx in range(3)]+[self.shuffle_data_chose_domain(x_fft_abs, domain, idx).unsqueeze(1) for idx in range(3)], dim=1).detach()
#         else:
        fft_set = torch.cat([x_fft_abs.unsqueeze(1)]+[self.shuffle_data(x_fft_abs).unsqueeze(1) for _ in range(num_mix-1)], dim=1).detach()
        # init weight
        ori_weight = torch.ones((x_fft.size(0), num_mix), device=self.device) / num_mix
        # random init
        # ori_weight = torch.randn((x_fft.size(0), num_mix), device=self.device)*0.2 + 1 / num_mix
        # ori_weight = torch.clamp(ori_weight, min=0, max=1)
        mix_weight = ori_weight.detach_()
        # begin search    
        
        random_idx = random.randint(0, move_step-1)
        
        for i in range(move_step):

            mix_weight.requires_grad = True
            # norm the weight
            weight_norm = mix_weight/torch.sum(mix_weight, dim=1, keepdim=True)
            # use the weight calc the mix_abs 
            x_fft_abs_mix = torch.bmm(weight_norm.unsqueeze(
                1), fft_set.view(x_fft_abs.size(0), num_mix, -1)).reshape_as(x_fft_abs)
            # use gamma 
            x_fake_fft = ((1-gamma) * x_fft_abs + gamma *
                            x_fft_abs_mix) * torch.exp(1j * x_fft_pha)
            # ifft for x_fake
            x_fake = torch.real(
                torch.fft.ifft2(x_fake_fft, dim=[-2, -1]))
            
            # forward
            if classifiers is None:
                score_fake = current_model(x_fake)
                loss = criterion(score_fake, label)
            else:
                features = current_model(x_fake)
                score_fake = classifiers(features)
                loss = criterion(score_fake, label)
            
            # backward
            loss.backward()
            # update weight
            mix_weight = mix_weight + mu * mix_weight.grad.sign()
            eta = torch.clamp(
                mix_weight - ori_weight, min=-1, max=1)
            mix_weight = torch.clamp(
                ori_weight + eta, min=0, max=1).detach_()

            
            if i == 0:    
                x_ori = x_fake
            if i == random_idx:    
                x_random = x_fake
            
            loss_temp[i] = loss.item()
            # save_flag
            if save_flag:
                
                normalized_weight_temp.append(weight_norm)
                save_img_temp.append(x_fake.unsqueeze(1))
        
        # begin calc final x_fake
        final_weight_norm = mix_weight/torch.sum(mix_weight, dim=1, keepdim=True)
        x_fft_abs_mix = torch.bmm(final_weight_norm.unsqueeze(
            1), fft_set.view(x_fft_abs.size(0), num_mix, -1)).reshape_as(x_fft_abs)
        x_fake_fft = ((1-gamma) * x_fft_abs + gamma * x_fft_abs_mix) * torch.exp(1j * x_fft_pha)
        x_fake = torch.real(
            torch.fft.ifft2(x_fake_fft, dim=[-2, -1]))
        if save_flag:
            save_img_temp.append(x_fake.unsqueeze(1))
        log = {}
        if move_step != 0:
            log["x_ori"] = x_ori
            log["x_random"] = x_random
        log["random_idx"] = random_idx
        log["loss_temp"] = loss_temp
        # save_flag
        if save_flag:
            normalized_weight_temp.append(final_weight_norm)
            log["normalized_weight_temp"] = normalized_weight_temp
            log["save_img_temp"] = torch.cat(save_img_temp, dim=1)

        return x_fake, log

class UndoNorm:
    """Denormalize batch images."""

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor(std).cuda().view(1, 3, 1, 1)

    def __call__(self, tensor):
        """
        Input:
            tensor (torch.Tensor): tensor image of size (B, C, H, W)
        """
        tensor *= self.std
        tensor += self.mean
        return tensor

class Norm:
    """Normalize batch images."""

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor(std).cuda().view(1, 3, 1, 1)

    def __call__(self, tensor):
        """
        Input:
            tensor (torch.Tensor): tensor image of size (B, C, H, W)
        """
        tensor -= self.mean
        tensor /= self.std
        return tensor

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-4
)
