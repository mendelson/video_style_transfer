import torch
import torch.optim as optim
from tqdm import tqdm

def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """
    
    ## Need the layers for the content and style representations of an image
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'}
 
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    
    ## get the batch_size, depth, height, and width of the Tensor
    _, d, h, w = tensor.size()
    ## reshape it, so we're multiplying the features for each channel
    tensor = tensor.view(d, h * w)
    ## calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())
    
    return gram

def transfer_to_frame(content, style, vgg, device):
    # get content and style features only once before forming the target image
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)

    # calculate the gram matrices for each layer of our style representation
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # create a third "target" image and prep it for change
    # it is a good idea to start of with the target as a copy of our *content* image
    # then iteratively change its style
    target = content.clone().requires_grad_(True).to(device)

    # weights for each style layer 
    # weighting earlier layers more will result in *larger* style artifacts
    # notice we are excluding `conv4_2` our content representation
    style_weights = {'conv1_1': 1.,
                     'conv2_1': 0.8,
                     'conv3_1': 0.5,
                     'conv4_1': 0.3,
                     'conv5_1': 0.1}

    content_weight = 1  # alpha
    style_weight = 1e6  # beta

    # for displaying the target image, intermittently
    #show_every = 500

    # iteration hyperparameters
    optimizer = optim.Adam([target], lr=0.003)
    steps = 5000  # decide how many iterations to update your image (5000)

    for ii in tqdm(range(1, steps+1)):
        ## Get the features from the target image
        ## Then calculate the content loss
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        
        # the style loss
        # initialize the style loss to 0
        style_loss = 0
        # iterate through each style layer and add to the style loss
        for layer in style_weights:
            # get the "target" style representation for the layer
            target_feature = target_features[layer]
            _, d, h, w = target_feature.shape
            
            ## Calculate the target gram matrix
            target_gram = gram_matrix(target_feature)
            
            ## get the "style" style representation
            style_gram = style_grams[layer]
            # Calculate the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            
            # add to the style loss
            style_loss += layer_style_loss / (d * h * w)
            
        # calculate the *total* loss
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        # update the target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # display intermediate images and print the loss
        #if ii % show_every == 0:
          #fig, ax = plt.subplots(1, 1, figsize=(20, 10))
          #ax.imshow(im_convert(target), interpolation='nearest')
          #ax.title.set_text(f'Step: {ii} - Total loss: {total_loss.item()}')
          #plt.show()

    return target