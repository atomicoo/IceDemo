import numpy as np
import matplotlib.pylab as plt


def save_figure(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_spectrogram(spectrogram, max_len=None):
    if max_len is not None:
        spectrogram = spectrogram[:max_len, :]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    im = ax.imshow(np.rot90(spectrogram), aspect="auto", origin="lower",
                    interpolation='none')
    # fig.colorbar(mappable=im, shrink=0.65, ax=ax)

    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    fig.canvas.draw()
    data = save_figure(fig)
    plt.close()

    return fig, data
