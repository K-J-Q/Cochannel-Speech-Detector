import torch
import torch.nn as nn
from torchvision.models.vgg import VGG
from torchvision import models
from torchsummary import summary

class encoder(VGG):
	def __init__(self, pretrained=False, requires_grad=True, batch_norm=False, show_params=False):
		super().__init__(self.make_layers(batch_norm))

		if pretrained:
			exec("self.load_state_dict(models.vgg16(pretrained=True).state_dict())")
		# False if you want to freeze the encoder and avoid to train it.
		if not requires_grad:
			for parameter in super().parameters():
				parameter.requires_grad = False
		# Remove fully connected layers
		del self.classifier
		self.features._modules['0'] = nn.Conv2d(
			1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		# Display the params to see the dimensionallity
		if show_params:
			for name, parameter in self.named_parameters():
				print(name, parameter.size())

	def make_layers(self, batch_norm):
	    cfg = [64, 64, 'M',
	           128, 128, 'M',
	           256, 256, 256, 'M',
	           512, 512, 512, 'M',
	           512, 512, 512, 'M']
	    layers = []
	    in_channels = 3
	    for v in cfg:
	    	if v == 'M':
	    		layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
	    	else:
	    		conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
	    		if batch_norm:
	    			layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
	    		else:
	    			layers += [conv2d, nn.ReLU(inplace=True)]
	    		in_channels = v
	    return nn.Sequential(*layers)

	def forward(self, x):
		# Extracts the output of the maxpooling layers to build skip connections.
		ranges = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31))
		output = {}
		for idx in range(len(ranges)):
			for layer in range(ranges[idx][0], ranges[idx][1]):
				x = self.features[layer](x)
			output["pool%d" % (idx+1)] = x
		return output


class decoder(nn.Module):
	def __init__(self, encoder, n_classes):
		super().__init__()
		self.n_classes = n_classes
		self.encoder = encoder
		self.relu = nn.ReLU(inplace=True)
		self.deconv1 = nn.ConvTranspose2d(512, 512, 3, 2, padding=1, output_padding=1, dilation=1) # 1x1 convolution.
		self.bn1 = nn.BatchNorm2d(512)
		self.deconv2 = nn.ConvTranspose2d(512, 256, 3, 2, padding=1, output_padding=1, dilation=1)
		self.bn2 = nn.BatchNorm2d(256)
		self.deconv3 = nn.ConvTranspose2d(256, 128, 3, 2, padding=1, output_padding=1, dilation=1)
		self.bn3 = nn.BatchNorm2d(128)
		self.deconv4 = nn.ConvTranspose2d(128,  64, 3, 2, padding=1, output_padding=1, dilation=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.deconv5 = nn.ConvTranspose2d( 64,  32, 3, 2, padding=1, output_padding=1, dilation=1)
		self.bn5 = nn.BatchNorm2d(32)
		# Convolutional score. layer (replace the FC layers)
		self.score = nn.Conv2d(32, n_classes, 1)

	def forward(self, x):
	    x = self.encoder(x)
	    # Skip connections (connect the output of one layer to a non-adjacent layer)
	    pool_5 = x['pool5'] # Output of the VGG network.
	    pool_4 = x['pool4']
	    pool_3 = x['pool3']
	    
	    x = self.relu(self.deconv1(pool_5))
	    x = self.bn1(x + pool_4)
	    x = self.relu(self.deconv2(x))
	    x = self.bn2(x + pool_3)
	    x = self.bn3(self.relu(self.deconv3(x)))
	    x = self.bn4(self.relu(self.deconv4(x)))
	    x = self.bn5(self.relu(self.deconv5(x)))
	    x = self.score(x)

	    return x

n_classes = 12
batch_size = 10
# It accepts any image size multiple of 32.
img_width = 512
img_height = 512


if __name__ == "__main__":
    fcn = decoder(encoder(show_params=False), 3)

    # summary(fcn, (3,128, 128), device="cpu")
    print(fcn)
    print(fcn(torch.randn(2, 1, 128, 128)).shape)
    # cnn.eval()
    # print(cnn(torch.randn(2, 1, 16000)).shape)


# 
# output = fcn(torch.randn([batch_size, 3, img_width, img_height]))
# print(output) # [10, 12, 512, 512]