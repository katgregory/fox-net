import cv2
import glob


Y_MIN = 65
Y_SIZE = 31
X_MIN = 47
X_SIZE = 26

R_MEAN = 88
G_MEAN = 152
B_MEAN = 75
RGB_TOL = 15


def iteration_from_filename(filename):
    name = filename[filename.rfind('/i=') + 3:]
    iteration = name[:name.find('_')]
    return int(iteration)


def filter_image(image):
    image = image[:, :, 0]

    # Filter out pixels falling outside of the target range.
    # image[abs(image - B_MEAN) > RGB_TOL] = 0
    # cv2.imshow('filtered', image)
    # cv2.waitKey(0)

    return image

def load_images(dir, filter_image_flag=True, group_by_image=True):
    '''
    :param group_by_image:
       * If true, returns a tuple for each input file in dir. Each tuple has the format:
         (filename, 100s digit as numpy array, 10s digit as a numpy array, 1s digit as a numpy array)
       * If false, returns a tuple for each digit in each input file in dir. Each tuple has the format:
         (filename, digit as numpy array)
    '''

    image_filenames = [filename for _, filename in
                       sorted([(iteration_from_filename(filename), filename) for filename in glob.glob(dir)])]

    images = []
    for index, image_filename in enumerate(image_filenames):
        image = cv2.imread(image_filename)
        hundreds = image[Y_MIN : Y_MIN + Y_SIZE, X_MIN : X_MIN + X_SIZE]
        tens = image[Y_MIN : Y_MIN + Y_SIZE, X_MIN + X_SIZE : X_MIN + 2 * X_SIZE]
        ones = image[Y_MIN : Y_MIN + Y_SIZE, X_MIN + 2 * X_SIZE : X_MIN + 3 * X_SIZE]

        if filter_image_flag:
            hundreds = filter_image(hundreds)
            tens = filter_image(tens)
            ones = filter_image(ones)

        # cv2.imwrite(str(index) + '_100.png', hundreds)
        # cv2.imwrite(str(index) + '_10.png', tens)
        # cv2.imwrite(str(index) + '_1.png', ones)

        hundreds = hundreds.flatten()
        tens = tens.flatten()
        ones = ones.flatten()

        if group_by_image:
            images.append((image_filename, hundreds, tens, ones))
        else:
            images.append((image_filename, hundreds))
            images.append((image_filename, tens))
            images.append((image_filename, ones))

    return images