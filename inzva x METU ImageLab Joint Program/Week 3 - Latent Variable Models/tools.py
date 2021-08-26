import os
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

import torch
import torchvision.utils as utils
import torch.nn.functional as F
import torch.nn as nn


def select_data_dir(data_dir='../data'):
    data_dir = '/coursedata' if os.path.isdir('/coursedata') else data_dir
    print('The data directory is %s' % data_dir)
    return data_dir


def save_model(model, filename):
    try:
        do_save = input('Do you want to save the model (type yes to confirm)? ').lower()
        if do_save == 'yes':
            torch.save(model.state_dict(), filename)
            print('Model saved to %s.' % (filename))
        else:
            print('Model not saved.')
    except:
        raise Exception('The notebook should be run or validated with skip_training=True.')


def load_model(model, filename, device):
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    print('Model loaded from %s.' % filename)
    model.to(device)
    model.eval()


def plot_images(images, ncol=12, figsize=(8,8), cmap=plt.cm.Greys, clim=[0,1]):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    grid = utils.make_grid(images, nrow=ncol, padding=0, normalize=False).cpu()
    ax.imshow(grid[0], cmap=cmap, clim=clim)
    display.display(fig)
    plt.close(fig)


def plot_generated_samples(samples, ncol=12):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.axis('off')
    ax.imshow(
        np.transpose(
            utils.make_grid(samples, nrow=ncol, padding=0, normalize=True).cpu(),
            (1,2,0)
        )
     )
    display.display(fig)
    plt.close(fig)


def show_proba(proba, r, c, ax):
    """Creates a matshow-style plot representing the probabilites of the nine digits in a cell.
    
    Args:
      proba of shape (9): Probabilities of 9 digits.
    """
    cm = plt.cm.Reds
    ix = proba.argmax()
    if proba[ix] > 0.9:
        px, py = c+0.5, r+0.5
        ax.text(px, py, ix.item(), ha='center', va='center', fontsize=24)
    else:
        for d in range(9):
            dx = dy = 1/6
            px = c + dx + (d // 3)*(2*dx)
            py = r + dy + (d % 3)*(2*dy)
            p = proba[d]
            ax.fill(
                [px-dx, px+dx, px+dx, px-dx, px-dx], [py-dy, py-dy, py+dy, py+dy, py-dy],
                #color=[p, 1-p, 1-p]
                color=cm(int(p*256))
            )
            ax.text(px, py, d, ha='center', va='center', fontsize=8)    


def draw_sudoku(x, logits=False):
    """
    
    Args:
        x of shape (9, 9, 9)
        logits (bool): Indicator what x represents.
                        True: x represents the logits of the solution (along dim=2).
                        False: x represents unsolved puzzle with one-hot coded digits. Missing digits are represented
                        with all zeros.
    """
    fig, ax = plt.subplots(1, figsize=(7,7))
    ax.set(
        xlim=(0, 9), ylim=(9, 0),
        xticks=np.arange(10), xticklabels=[],
        yticks=np.arange(10), yticklabels=[]
    )
    ax.grid(True, which='major', linewidth=2)
    ax.xaxis.set_major_locator(plt.MultipleLocator(3))
    ax.yaxis.set_major_locator(plt.MultipleLocator(3))
    ax.tick_params(which='major', length=0)

    ax.grid(True, which='minor')
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(which='minor', length=0)
    
    if logits:
        with torch.no_grad():
            probs = F.softmax(x, dim=2)
            for r in range(9):
                for c in range(9):
                    show_proba(probs[r, c], r, c, ax)
    else:
        for r in range(9):
            for c in range(9):
                ix = x[r, c].nonzero()
                if ix.numel() > 0:
                    digit = ix.item()
                    px, py = c+0.5, r+0.5
                    ax.text(px, py, digit, ha='center', va='center', fontsize=24)
