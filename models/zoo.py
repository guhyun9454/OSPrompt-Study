import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
import numpy as np
import copy
from timm.models import load_checkpoint
import effvitmodels
import timm
import torchvision.transforms as transforms
from timm.models import vit_base_patch16_224




class OSPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)

            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0, 1, 2, 3, 4]

        # strenth of ortho penalty
        self.qr_loss_weight = prompt_param[2]

    def process_task_count(self):
        self.task_count += 1

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more
        # fair in the spirit of continual learning and has little affect on performance
        #
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self, f'e_k_{e}')
            P = getattr(self, f'e_p_{e}')
            k = self.gram_schmidt(K)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:, k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)

        return torch.nn.Parameter(uu)

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True

            x_querry_ori = x_querry
            x_querry = x_block[:, 0,:]


            K = getattr(self, f'e_k_{l}')
            p = getattr(self, f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)

            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                p = p[0:f]

            n_K = nn.functional.normalize(K, dim=1)
            a_querry_ori = x_querry_ori.unsqueeze(1).repeat(1, n_K.size(0), 1)
            q_ori = nn.functional.normalize(a_querry_ori, dim=2)
            aq_k_ori = torch.einsum('bkd,kd->bk', q_ori, n_K)

            n_K = nn.functional.normalize(K, dim=1)
            a_querry = x_querry.unsqueeze(1).repeat(1, n_K.size(0),1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length / 2)
            Ek = P_[:, :i, :]
            Ev = P_[:, i:, :]

            # loss definition
            # loss = 0
            loss = nn.MSELoss()(x_querry, x_querry_ori) * self.qr_loss_weight
            # print (aq_k.size())
            # print (aq_k.sum(1))
            # loss = nn.MSELoss()(aq_k, aq_k_ori) * self.qr_loss_weight



        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block


# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p    

class ViTZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None, query=None):
        super(ViTZoo, self).__init__()

        # get last layer
        self.last = nn.Linear(512, num_classes)
        self.prompt_flag = prompt_flag
        self.task_id = None
        self.query = query

        # get feature encoder
        if pt:
            if query == 'vit':
                print( "Load vit fine-tuned on in1k ...")
                zoo_model_query = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                            num_heads=12, ckpt_layer=0,
                                            drop_path_rate=0
                                            )
                from timm.models import vit_base_patch16_224
                load_dict = vit_base_patch16_224(pretrained=True).state_dict()
                del load_dict['head.weight']; del load_dict['head.bias']
                zoo_model_query.load_state_dict(load_dict)
            else:
                NotImplementedError


        from timm.models import vit_base_patch16_224
        zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                      num_heads=12, ckpt_layer=0,
                                      drop_path_rate=0
                                      )
        load_dict = vit_base_patch16_224(pretrained=True).state_dict()
        del load_dict['head.weight'];
        del load_dict['head.bias']
        zoo_model.load_state_dict(load_dict)


        # classifier
        self.last = nn.Linear(768, num_classes)

        print ('prompt name: {} / query: {} '.format(self.prompt_flag, query))

        # create prompting module

        if self.prompt_flag == 'os':
            self.prompt = OSPrompt(768, prompt_param[0], prompt_param[1])
        else:
            self.prompt = None
        
        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model
        self.feat_query = zoo_model_query

        self.dset_mean = (0.0, 0.0, 0.0)
        self.dset_std = (1.0, 1.0, 1.0)

        if query == 'vit':
            self.dset_mean_q = (0.0,0.0,0.0)
            self.dset_std_q = (1.0,1.0,1.0)
        else:
            self.dset_mean_q  = timm.data.resolve_model_data_config(zoo_model_query)['mean']
            self.dset_std_q  = timm.data.resolve_model_data_config(zoo_model_query)['std']

        print ('norm for query: {} /{}'.format(self.dset_mean_q, self.dset_std_q ))

        
    # pen: get penultimate features    
    def forward(self, x, pen=False, train=False):

        x_backbone = transforms.Normalize(self.dset_mean, self.dset_std)(x)
        x_query = transforms.Normalize(self.dset_mean_q, self.dset_std_q)(x)

        if self.prompt is not None:
            if self.prompt_flag == 'os':
                # q= None
                with torch.no_grad():
                    if self.query == 'vit':
                        q, _ = self.feat_query(x_query)
                        q = q[:,0,:]
                    elif self.query in ['poolformer', 'swin']:
                        q = self.feat_query(x_query)
                        q = q[-1].mean(-2).mean(-1)
                    else:
                        q = self.feat_query(x_query)



            out, prompt_loss = self.feat(x_backbone, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
            out = out[:,0,:]
        else:
            out, _ = self.feat(x_backbone)
            out = out[:, 0, :]

        out = out.view(out.size(0), -1)
        if not pen:
            out = self.last(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out

def vit_pt_imnet(out_dim, block_division = None, prompt_flag = 'None', prompt_param=None, query = None):
    return ViTZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param, query = query)

