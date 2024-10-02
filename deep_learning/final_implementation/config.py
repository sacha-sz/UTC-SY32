INPUT_SIZE = 512
PANNEAU_SIZE = 64
BATCH_SIZE = 32

CLASSES = {
    "frouge": 0,
    "forange": 1,
    "fvert": 2,
    "stop": 3,
    "ceder": 4,
    "interdiction": 5,
    "danger": 6,
    "obligation": 7
}
CLASS_NAMES = ['frouge', 'forange', 'fvert', 'stop', 'ceder', 'interdiction', 'danger', 'obligation']

NB_CLASSES = len(CLASSES)

IMG_WIDTH = INPUT_SIZE
IMG_HEIGHT = INPUT_SIZE
IMG_CHANNELS = 3

