import model_utils.solver as solver
import torch
# from utils.timer import Timer
import time
from eval import ImageEval
import scipy.io

class FCN_solver(solver.solver):
    def __init__(self, models, model_name, save_path="checkpoints"):
        super(FCN_solver,self).__init__(models,model_name,save_path)
        self.fcn=self.models[0]

    def train_one_batch(self,input_dict):
        optimizer=self.optimizers[0]
        x=input_dict["x"]
        y=input_dict["y"]
        start_time=time.time()

        out=self.fcn(x)

        # end_time=time.time()#Timer()
        # print("forward duration: %.4f"%(end_time-start_time))
        # start_time=time.time()

        loss=torch.abs(out-y).mean()

        # end_time=time.time()#Timer()
        # print("loss duration: %.4f"%(end_time-start_time))
        # start_time=time.time()

        loss.backward()


        optimizer.step()

        self.zero_grad_for_all()

        total_loss={}
        step=input_dict["step"]
        if(step%20==1):
            total_loss["reconst loss"]=loss.detach().cpu().item()
        return total_loss

    def test_one_batch(self,input_dict):
        x=input_dict["x"]
        y=input_dict["y"]
        out=self.fcn(x)
        image=torch.zeros((y.size(0),y.size(1),y.size(2),y.size(3)*2))
        image[:,:,:,0:y.size(3)]=y.cpu()
        image[:,:,:,y.size(3):y.size(3)*2]=out.cpu()
        image[image<0]=0.0
        image[image>1]=1.0

        s_psnr=0.0
        s_ssim=0.0
        for i in range(y.size(0)):
            img_eval=ImageEval(image[i,:,:,y.size(3):y.size(3)*2],image[i,:,:,0:y.size(3)])
            psnr,ssim=img_eval.eval()
            s_psnr+=psnr
            s_ssim+=ssim
        m_psnr=s_psnr/y.size(0)
        m_ssim=s_ssim/y.size(0)
        return image,m_psnr,m_ssim


    def train_loop(self,param_dict,epoch=10000):
        iteration_count=0
        dataloader=param_dict["loader"]
        for i in range(0,epoch):
            start_time=time.time()#Timer()
            for step,(x,y) in enumerate(dataloader):
                end_time=time.time()
                print('time',str(end_time-start_time))
                # print(x)
                self.train_mode()
                input_dict={}
                # print(type(x))
                input_dict["x"]=x.cuda()
                input_dict["y"]=y.cuda()
                input_dict["step"]=step

                # end_time=time.time()#Timer()
                # print("x,y duration: %.4f"%(end_time-start_time))
                # start_time=time.time()

                loss=self.train_one_batch(input_dict)

                # end_time=time.time()#Timer()
                # print("train_one duration: %.4f"%(end_time-start_time))
                # start_time=time.time()

                iteration_count+=1
                if(iteration_count%1==0):
                    self.write_log(loss,iteration_count)
                    self.output_loss(loss,i,iteration_count)


                if(iteration_count%1000==0):
                    self.eval_mode()
                    out_images,m_psnr,m_ssim=self.test_one_batch(input_dict)
                    images={}
                    images["image"]=out_images
                    self.write_log_image(images,int(iteration_count/1000))
                    self.train_mode()
                # end_time=time.time()#Timer()
                # print("batch duration: %.4f"%(end_time-start_time))
                start_time=time.time()
            if(i%100==0):
                self.save_models(epoch=i)
            if(i==2000):
                n_lr=0.00001
                self.update_learningrate(n_lr)
        self.save_models(epoch=i)

    def test_loop(self,param_dict):
        self.fcn=self.models[0]
        dataloader=param_dict["loader"]
        self.eval_mode()
        s_psnr=0.0
        s_ssim=0.0
        num=0.0
        for step,(x,y) in enumerate(dataloader):
            input_dict={}
            input_dict["x"]=x.cuda()
            input_dict["y"]=y.cuda()


            out_images,m_psnr,m_ssim=self.test_one_batch(input_dict)
            s_psnr+=m_psnr
            s_ssim+=m_ssim
            num+=1

            images={}
            images["image"]=out_images
            self.write_log_image(images,int(step))

        print('test mean PSNR: %.2f'%(s_psnr/num))
        print('test mean SSIM: %.2f'%(s_ssim/num))

    def test_dark_loop(self,param_dict):
        self.fcn=self.models[0]
        dataloader=param_dict["loader"]
        self.result_dir=param_dict["result_dir"]
        self.eval_mode()
        # s_psnr=0.0
        # s_ssim=0.0
        # num=0.0
        with torch.no_grad():
            for step,(x,id,ratio) in enumerate(dataloader):
                input_dict={}
                input_dict["x"]=x.cuda()
                input_dict["id"]=id
                input_dict["ratio"]=ratio
                # input_dict["y"]=y.cuda()

                out_images=self.test_dark_one_batch(input_dict)

                # out_images,m_psnr,m_ssim=self.test_dark_one_batch(input_dict)
                # s_psnr+=m_psnr
                # s_ssim+=m_ssim
                # num+=1

                images={}
                images["image"]=out_images
                # self.write_log_image(images,int(step))

            # print('test mean PSNR: %.2f'%(s_psnr/num))
            # print('test mean SSIM: %.2f'%(s_ssim/num))

    def test_dark_one_batch(self,input_dict):
        x=input_dict["x"]
        id=input_dict["id"]
        ratio=input_dict["ratio"]
        # y=input_dict["y"]
        out=self.fcn(x)
        image=torch.zeros((out.size(0),3,out.size(2),out.size(3)))
        # image[:,:,:,0:y.size(3)]=x.cpu()
        image[:,:,:,:]=out.cpu()
        image[image<0]=0.0
        image[image>1]=1.0
        img = image.permute(0, 2, 3, 1).data.numpy()[0,:,:,:]
        scipy.misc.toimage(img*255,  high=255, low=0, cmin=0, cmax=255).save(self.result_dir + '%05d.jpg'%(id))

        # s_psnr=0.0
        # s_ssim=0.0
        # for i in range(x.size(0)):
        #     img_eval=ImageEval(image[i,:,:,y.size(3):y.size(3)*2],image[i,:,:,0:y.size(3)])
        #     psnr,ssim=img_eval.eval()
        #     s_psnr+=psnr
        #     s_ssim+=ssim
        # m_psnr=s_psnr/y.size(0)
        # m_ssim=s_ssim/y.size(0)
        # return image,m_psnr,m_ssim

    def update_learningrate(self,n_lr):
        for g in self.optimizers[0].param_groups:
            g['lr']=n_lr
