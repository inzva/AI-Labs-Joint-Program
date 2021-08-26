import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from IPython import display

import torch
import torchvision.utils as utils
from tools import plot_images


classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def visualize_embeddings(encoder, dataloader, n_samples, device):
    n = 0
    codes, labels = [], []
    with torch.no_grad():
        for b_inputs, b_labels in dataloader:
            batch_size = b_inputs.size(0)
            b_codes = encoder(b_inputs.to(device))
            b_codes, b_labels = b_codes.cpu().data.numpy(), b_labels.cpu().data.numpy()
            if n + batch_size > n_samples:
                codes.append(b_codes[:n_samples-n])
                labels.append(b_labels[:n_samples-n])
                break
            else:
                codes.append(b_codes)
                labels.append(b_labels)
                n += batch_size
    codes = np.vstack(codes)
    if codes.shape[1] > 2:
        print('Use t-SNE')
        codes = TSNE().fit_transform(codes)
    labels = np.hstack(labels)

    colors = [
        'black', 'red', 'gold', 'palegreen', 'blue',
        'lightcoral', 'orange', 'mediumturquoise', 'dodgerblue', 'violet'
    ]
    fig, ax = plt.subplots(1)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height], which='both')
    for iclass in range(min(labels), max(labels)+1):
        ix = labels == iclass
        ax.plot(codes[ix, 0], codes[ix, 1], '.', color=colors[iclass])

    plt.legend(classes, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


def visualize_reconstructions(encoder, decoder, dataloader, device):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    images = images[:8].to(device)

    with torch.no_grad():
        reconstructions = decoder(encoder(images))
        images = images / 2 + 0.5  # inverse normalization
        reconstructions = reconstructions / 2 + 0.5  # inverse normalization
        #plot_images(torch.cat([images, reconstructions]), n_rows=2)
        plot_images(torch.cat([images, reconstructions]), ncol=8)

