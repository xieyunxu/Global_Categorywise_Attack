import torch
import numpy as np
class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    # FeatureExtractor(self.feature_module, target_layers)

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
       # print('target layer:',self.target_layers)

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)

            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        #print('feature_module name: ',self.feature_module)
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        feature = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:# 跳入FeatureExtractor()正传，得到整层的参数 和target_layers梯度
                target_activations, x = self.feature_extractor(x)
                feature.append(x)
            elif "wh" in name.lower():
                wh = module(x)
            elif "hm" in name.lower():
                hm = module(x)
                #print('hm shape',hm.shape)
            elif "reg" in name.lower():
                reg = module(x)
                #x = x.view(x.size(0),-1)
           # elif "avgpool" in name.lower():
            else:# 其他情况   即在除去self.feature_module的其余三层中正向传播
                x = module(x)
        return target_activations, hm

def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def deprocess_image(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)
