from torch import device, cuda, save, load, Tensor
from .others.utils import Conv2d, Upsampling, SGD, ReLU, Sigmoid, MSE, Sequential

#######################################################################################################

class Model():
    def __init__(self) -> None:

        #initialisation of the sequential model, optimiyer, loss function and device

        self.model = Sequential(
            Conv2d(3, 128, (2, 2), 2),
            ReLU(),
            Conv2d(128, 256, (2, 2), 2),
            ReLU(),
            Upsampling(256, 128, (2, 2), 2),
            ReLU(),
            Upsampling(128, 3, (2, 2), 2),
            Sigmoid()
        )

        self.optimizer = SGD(parameters=self.model.param(), lr=0.5)
        self.criterion = MSE()
        self.device = device("cuda" if cuda.is_available() else "cpu")


    def load_pretrained_model(self) -> None:
        
        #loading of the pretrained model
        #the loading of each of the layers is done 1 by 1
        #note that it is not necessary to load also the gradients, for in our implementation it is done

        best_model = load('./Proj_343212_343812_342352/Miniproject_2/bestmodel.pth', map_location=self.device)

        for l in range(0, 8, 2):
            self.model.args[l].bias.data = best_model[l][0].to(self.device)
            self.model.args[l].dL_db.data = best_model[l][1].to(self.device)
            self.model.args[l].weight.data = best_model[l+1][0].to(self.device)
            self.model.args[l].dL_dw.data = best_model[l+1][1].to(self.device)

    def train(self, train_input, train_target, num_epochs, mini_batch_size=32):

        #training function that takes as inputs the train input and target sets, the number of epochs and additionally the mini batch siye
        #note that all the tensors are rescaled to 0-1 

        train_input = train_input.div(255)
        train_target = train_target.div(255)
        acc_loss = []

        opt_size = train_input.size(0) // mini_batch_size * mini_batch_size
        for e in range(num_epochs):
            for b in range(0, opt_size, mini_batch_size):
                output = self.model.forward(train_input.narrow(0, b, mini_batch_size))
                loss = self.criterion(output, train_target.narrow(0, b, mini_batch_size))
                acc_loss.append(loss.item())
                
                # note that the loss function also needs to call the backward() step, as oposed to the implementation with pytorch
                self.optimizer.zero_grad()
                grad_output = self.criterion.backward()
                self.model.backward(grad_output)
                self.optimizer.step()

                print(f"Epoch: {e} - {b},  Loss:  {round(sum(acc_loss) / len(acc_loss), 6)},  {round(loss.item(), 6)}")

    def save_model(self) -> None:

        #saving the different parameters of the model

        save(self.model.param(), './Proj_343212_343812_342352/Miniproject_2/bestmodel.pth')

    def predict(self, test_input) -> Tensor:

        #predict new outputs with the trained model

        test_input = test_input.float().div(255.)
        return self.model(test_input.to(self.device)).mul(255)