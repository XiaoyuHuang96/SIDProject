from model import U_net
from dataset import OurDataset
import torch.utils.data as Data
from model_utils.common_tools import *
#from utils.data_provider import *
import FCN_solver
import torchvision.transforms as transforms
import sys
from model_utils.dataset import BufferDataLoader

transform_img = transforms.Compose([
                    transforms.ToTensor()
                    ])
models=[]
# models.append(U_net.SeeInDark())
models.append(U_net.U_net())

for i in range(0,len(models)):
    models[i]=nn.DataParallel(models[i],device_ids=[0,1])

lrs=[0.0001]

# torch.cuda.set_device(0)

optimizers=generate_optimizers(models,lrs,optimizer_type="adam",weight_decay=0.001)
function=weights_init(init_type='gaussian')
solver=FCN_solver.FCN_solver(models,model_name="U_net")
solver.set_optimizers(optimizers)

solver.init_models(function)
solver.cuda()
# sony_train_dataset=TrainDataset.TrainDataset(info_path='_train_list.txt',img_path='/home/huangxiaoyu/dataset/',type='Sony',transform=transform_img,patch_size=512)
sony_train_dataset=OurDataset.OurDataset(dir = '/home/hda/nfs_disk/dataset/dark2light/',ps=512,type='train',imgtype='raw',return_ratio=False)
sony_train_dataloader=Data.DataLoader(dataset=sony_train_dataset,batch_size=1,shuffle=True,num_workers=0)
# sony_train_dataloader=BufferDataLoader(dataset=sony_train_dataset,batch_size=1,shuffle=True,num_workers=128,buffer_size=100)
# print(sony_train_dataset.__len__())

#train_dataprovider=data_provider(train_provider_dataset ,batch_size=16, is_cuda=False)
#train_dataloader=Data.DataLoader(train_dataset,batch_size=4,shuffle=True,num_workers=0)
param_dict={}
param_dict["loader"]=sony_train_dataloader
#param_dict["provider"]=train_dataprovider
solver.train_loop(param_dict,epoch=4000)
