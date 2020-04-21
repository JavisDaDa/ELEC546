from PIL import Image
from numpy import *
from pylab import *
import os

def process_image(imagename, resultname, params="--edge-thresh 10 --peak-thresh 5"):

    if imagename[-3:] != 'pgm':
        # create a pgm file
        im = Image.open(imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    cmmd = str("D:\\Rice\\ELEC 546\\ELEC546\\Assignment7\\upload\\sift.exe" + imagename + " --output=" + resultname +
               " " + params)
    os.system(cmmd)
    print('processed', imagename, 'to', resultname)


def read_features_from_file(filename):

    f = loadtxt(filename)
    return f[:, :4], f[:, 4:]  # feature locations, descriptors


def plot_features(im, locs, circle=True):

    def draw_circle(c, r):
        t = arange(0, 1.01, .01) * 2 * pi
        x = r * cos(t) + c[0]
        y = r * sin(t) + c[1]
        plot(x, y, 'b', linewidth=2)

    imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2], p[2])
    else:
        plot(locs[:, 0], locs[:, 1], 'ob')
    axis('off')


if __name__ == '__main__':
    imname = ('D:\\Rice\\ELEC 546\\ELEC546\\Assignment7\\upload\\Assignment07_data\\Assignment06_data\\Assignment06_data_reduced\\TrainingDataset\\024.butterfly\\024_0001.jpg')
    im = Image.open(imname)
    process_image(imname, 'test.sift')
    l1, d1 = read_features_from_file('test.sift')
    figure()
    gray()
    plot_features(im, l1, circle=True)
    title('sift-features')
    show()