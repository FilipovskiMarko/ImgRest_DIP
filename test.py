import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch, torch.nn as nn
from numpy.f2py.cb_rules import cb_map
from skimage.transform import resize
from skimage.data import chelsea

#======================Starting Config =============================


torch.manual_seed(0)
plt.rcParams["font.size"] = "14"
plt.rcParams['toolbar'] = 'None'
plt.ion()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
nxd = 128

chelseaimage = chelsea() # cat img
true_object_np = 100.0 * resize(chelseaimage[10:299, 110:399,2], (nxd,nxd), anti_aliasing=False)


fig1, axs1 = plt.subplots(2,3, figsize=(20,12)) # no. rows and columns, figsize 20x12 inches
plt.tight_layout()
#fig1.canvas.manager.window.move(0,0)

axs1[0,2].imshow(true_object_np, cmap='Greys_r')
axs1[0,2].set_axis_off()

fig1.canvas.draw()
fig1.canvas.flush_events()


# ============================== CNN CLASSES ====================================
class CNN(nn.Module): #Inherits from nn.Module (Base pytorch class for modules)
    def __init__(self, num_channels):
        super(CNN, self).__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(num_channels, 1, kernel_size=3, padding=1), nn.PReLU(), # 14 layers
        )
    def forward(self, x): return torch.squeeze(self.CNN(x.unsqueeze(0).unsqueeze(0)))

class CNN_configurable(nn.Module):
    def __init__(self, n_lay, n_chan, ksize):
        super(CNN_configurable, self).__init__()
        pd = int(ksize/2)
        layers = [ nn.Conv2d(1, n_chan, kernel_size=ksize, padding=pd), nn.PReLU() ]
        for _ in range(n_lay):
            layers.append(nn.Conv2d(n_chan, n_chan, kernel_size=ksize, padding=pd)); layers.append(nn.PReLU())
        layers.append(nn.Conv2d(n_chan, 1, kernel_size=ksize, padding=pd)); layers.append(nn.PReLU())

        self.deep_net = nn.Sequential(*layers)

    def forward(self, x): return torch.squeeze(self.deep_net(x.unsqueeze(0).unsqueeze(0)))


cnn = CNN_configurable(32, nxd, 3).to(device)
input_image = torch.rand(nxd,nxd).to(device)


#=======================Torch to NP convertors ==================================
def torch_to_np(torch_array): return np.squeeze(torch_array.detach().cpu().numpy())
def np_to_torch(np_array): return torch.from_numpy(np_array).float()

true_object_torch = np_to_torch(true_object_np).to(device)

# Noise
measured_data = torch.poisson(true_object_torch)
# Gaps
mask_image = torch.ones_like(measured_data)
mask_image[int(0.65 * nxd):int(0.85*nxd),int(0.65 * nxd):int(0.85*nxd)] = 0
mask_image[int(0.15 * nxd):int(0.25*nxd),int(0.15 * nxd):int(0.25*nxd)] = 0
measured_data = measured_data * mask_image

axs1[0,1].imshow(torch_to_np(measured_data), cmap='Greys_r'); axs1[0,1].set_title('MEASURED');axs1[0,1].set_axis_off()
axs1[1,0].imshow(torch_to_np(input_image), cmap='Greys_r'); axs1[1,0].set_title('NOISE');axs1[1,0].set_axis_off()
axs1[0,2].set_title('TRUE')

#plt.show()

def nrmse_fn(recon, reference):
    numerator = (reference - recon)**2; denomerator = reference**2
    return 100.0 * torch.mean(numerator)**0.5 / torch.mean(denomerator)**0.5

#==========================Torch Optimizers===========================================
optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)
train_loss = list(); nrmse_list = list(); best_nrmse = 10e9

print("Starting training")

for epoch in range(10000 + 1):
    optimizer.zero_grad()
    output_image = cnn(input_image)

    loss = nrmse_fn(output_image * mask_image, measured_data * mask_image)

    train_loss.append(loss.item())
    loss.backward() # find the gradients
    optimizer.step() # does the update

    nrmse = nrmse_fn(output_image,true_object_torch)
    nrmse_list.append(nrmse.item())

    if nrmse < best_nrmse or epoch == 0:
        best_recon = output_image; best_nrmse = nrmse; best_ep = epoch
        axs1[1,2].cla(); axs1[1,2].imshow(torch_to_np(best_recon),cmap='Greys_r');
        axs1[1,2].set_title('Best Recon %d, NRMSE = %.2f%%' % (best_ep, best_nrmse))
        axs1[1,2].set_axis_off();


    if epoch % 5 == 0:
        axs1[1,1].cla();axs1[1,1].imshow(torch_to_np(output_image),cmap='Greys_r');
        axs1[1,1].set_title('Output Image %d, NRMSE = %.2f%%' % (epoch, nrmse)); axs1[1,1].set_axis_off()
        axs1[0,0].cla();axs1[0,0].plot(train_loss[-200:-1]);
        axs1[0, 0].plot(nrmse_list[-200:-1]); axs1[0,0].set_title('NRMSE (%%), epoch %d' %epoch);
        axs1[0, 0].legend(['ERROR wrt DATA','Error wrt TRUE']);

        # fig1.canvas.draw()
        # fig1.canvas.flush_events()
        # plt.pause(1)



    #print("Epoch:", epoch)
plt.imsave("output.png", torch_to_np(output_image), cmap='Greys_r')

    #Not updating plt for some reason, stupid shit