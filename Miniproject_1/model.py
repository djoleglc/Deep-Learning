import torch.nn as nn
import torch
import pickle

#######################################################################################################


class conv_block(nn.Module): 

   #convolution encoder block composed by two convolution and two leaky relu activations functions 


    def __init__(self,ch_in,ch_out):

        super(conv_block,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.LeakyReLU(0.1, inplace=True) ,
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.LeakyReLU(0.1, inplace=True) )

    def forward(self,x):

        # x : Tensor with (B, C, H, W)
        #return as output the ooutput of the convolution encoder block having (B, C_out, H_out, W_out)
 
        x = self.conv(x)

        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):

        super(up_conv,self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
			      nn.LeakyReLU(0.1, inplace=True) )

    def forward(self,x):
         # x : Tensor with (B, C, H, W)
        #return as output the ooutput of the convolution encoder block having (B, C_out, H_out, W_out)

        x = self.up(x)

        return x





###########################################################################################################################




class ShortNet(nn.Module):
 
 #implementation of the Short Model described in the pdf file inspired by the U-Net architecture and by the Noise2Noise architecture 
 #weights initialized using xavier normal method 



    def __init__(self, ch_out1=48, img_ch=3,output_ch=3):

        super(ShortNet,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)         #2*2 Maxpooling
        self.Conv1 = conv_block(ch_in=img_ch,ch_out=ch_out1)        # firt encoder convolution block 
        self.Conv2 = conv_block(ch_in=ch_out1,ch_out=ch_out1*2)      #second encoder convolution block 
        self.Conv3 = conv_block(ch_in=ch_out1*2,ch_out=ch_out1*4)      #third encoder convolution block 
        self.Up3 = up_conv(ch_in=ch_out1*4,ch_out=ch_out1*2)              #decoder blocks 
        self.Up_conv3 = conv_block(ch_in=ch_out1*4, ch_out=ch_out1*2)       #encoder block used in decoding 
        self.Up2 = up_conv(ch_in=ch_out1*2,ch_out=ch_out1)                  #decoder block  
        self.Up_conv2 = conv_block(ch_in=ch_out1*2, ch_out=ch_out1)              #encoder block used in decoding 
 
        self.Conv_1x1 = nn.Sequential(nn.Conv2d(ch_out1,output_ch,kernel_size=1,stride=1,padding=0),       #final convolution layer with kernel 1*1
                                      nn.ReLU())


        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        #using xavier weight initialization 
        #set the bias tensor equal to 0 

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self,x):
        # x : Tensor with (B, C, H, W)
        #return as output Tensor  having (B, C, H, W)

        # encoding path
        #setting the input tensors values between 0 and 1 

          x = x/255
          x1 = self.Conv1(x)

          x2 = self.Maxpool(x1)
          x2 = self.Conv2(x2)
          
          x3 = self.Maxpool(x2)
          x3 = self.Conv3(x3)

      
          # # decoding block with concatenation 

          d3 = self.Up3(x3)
          d3 = torch.cat((x2,d3),dim=1)
          d3 = self.Up_conv3(d3)

          d2 = self.Up2(x2)
          d2 = torch.cat((x1,d2),dim=1)
          d2 = self.Up_conv2(d2)

          d1 = self.Conv_1x1(d2)


          return (d1)




####################################################################################################################




class Model(nn.Module):

    def __init__ (self) -> None:

        ## instantiate model + optimizer + loss function + learning rate + check if cuda is available 
        super(Model, self).__init__()
        self.criterion = nn.MSELoss()
        self.lr = 1e-3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ShortNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
  

    def load_pretrained_model(self) -> None:

        ## This loads the parameters saved in bestmodel .pth into the model

        model_path = "./Proj_343212_343812_342352/Miniproject_1/bestmodel.pth"
        self.model = ShortNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        #model = pickle.load(open('bestmodel.pkl'),'rb')

        

    def train(self, train_input, train_target, num_epochs, mini_batch_size=16, PATH=None) -> None:

        #: train_input: tensor of size (N, C, H, W) containing a noisy version of the images

        #: train_target: tensor of size (N, C, H, W) containing another noisy version of the
        #  same images, which only differs from the input by their noise

        train_input = train_input.to(self.device)                              #training images 
        train_target = train_target.div(255).to(self.device)                   #dividing the target images by 255 

        acc_loss = []    
    
        opt_size = train_input.size(0)//mini_batch_size*mini_batch_size      
        
        for e in range(num_epochs):

            #the train and target tensor are randomly permute at each epoch
             
            index = torch.randperm(opt_size)
            train_input_shuffled = train_input[index]
            train_target_shuffled = train_target[index]

            for b in range(0, opt_size, mini_batch_size):

                output = self.model(train_input_shuffled.narrow(0, b, mini_batch_size))

                loss = self.criterion(output, (train_target_shuffled).narrow(0, b, mini_batch_size))

                acc_loss.append(loss.item())

                print(f"Epoch: {e} - {b},  Loss:  {round(sum(acc_loss)/len(acc_loss), 4)},  {round(loss.item(), 4)}")
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


        #if PATH:
        #    torch.save(self.model.state_dict(), PATH)


    def predict(self, test_input, load_model=False):# -> torch.Tensor:

        #: test_input: tensor of size (N1 , C, H, W) with values in range 0 -255 that has to
        #  be denoised by the trained or the loaded network.
        #  returns a tensor of the size (N1 , C, H, W) with values in range 0 -255.
        
        if load_model:
            self.load_pretrained_model()
        
        predicted = self.model(test_input.to(self.device)) * 255         #return a Tensor between 0 and 255 
        
        return predicted