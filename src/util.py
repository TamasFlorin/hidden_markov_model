
BLACK_PIXEL = 0
WHITE_PIXEL = 1


def image_to_array(im, new_size=(28, 28)):
    if im.size != new_size:
        im = im.resize(new_size)

    col, row = im.size
    data = []
    for _ in range(row):
        data.append([0 for _ in range(col)])

    pixels = im.load()
    for i in range(row):
        for j in range(col):
            if isinstance(pixels[i, j], int):
                r, g, b = pixels[i, j], 255, 255
            else:
                r, g, b = pixels[i, j]
            if r == 255 and b == 255 and g == 255:
                data[j][i] = WHITE_PIXEL
            else:
                data[j][i] = BLACK_PIXEL
    return find_inner_rectangle(data)


def find_inner_rectangle(image_pixels):
    height, width = len(image_pixels), len(image_pixels[0])

    min_x, max_x = width, 0
    min_y, max_y = height, 0

    for y in range(0, height):
        for x in range(0, width):
            if image_pixels[y][x] == BLACK_PIXEL:
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                min_x = min(min_x, x)
                min_y = min(min_y, y)

    # scale the image
    new_data = [[WHITE_PIXEL for _ in range(width)] for _ in range(height)]
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            new_data[y - min_y][x - min_x] = image_pixels[y][x]
    return new_data


def get_blocks(image_pixels, block_size=(14, 14)):
    height, width = len(image_pixels), len(image_pixels[0])

    block_height, block_width = block_size
    rows = []
    blocks = []
    for y in range(0, height):
        for x in range(0, width, block_width):
            rows.append(image_pixels[y][x: x + block_width + 1])

        if (y + 1) % block_height == 0:
            blocks.append(rows)
            rows = []

    if len(rows) > 0:
        blocks.append(rows)

    return blocks


def observation_from_block(block, keep=3):
    pixels = []
    for component in block:
        black_pixels = sum(
            [1 for value in component if value == BLACK_PIXEL])
        pixels.append(black_pixels)

    # make sure we have 3 items
    if len(pixels) < 3:
        for _ in range(3 - len(pixels)):
            pixels.append(0)

    pixels = sorted(pixels, reverse=True)
    return (pixels[0], pixels[1], pixels[2])


def map_obs_item_to_class(observation_item, d=10):
    if observation_item == 0:
        return 'None'
    elif observation_item < d:
        return 'Small'
    else:
        return 'Large'


CLASS_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
START_ID = -1
END_ID = 12
VOCABULARY = [START_ID] + CLASS_IDS + [END_ID]


def class_triple_to_id(class_triple):
    m = {
        ('None', 'None', 'None'): 0,
        ('Small', 'None', 'None'): 1,
        ('Small', 'Small', 'None'): 2,
        ('Small', 'Small', 'Small'): 3,
        ('None', 'Small', 'None'): 4,
        ('None', 'None', 'Small'): 5,
        ('Large', 'None', 'None'): 6,
        ('Large', 'Large', 'None'): 7,
        ('Large', 'Small', 'None'): 8,
        ('Large', 'Small', 'Small'): 9,
        ('Large', 'Large', 'Small'): 10,
        ('Large', 'Large', 'Large'): 11
    }

    return m[class_triple]


def observations_from_blocks(blocks):
    observations = []
    for block in blocks:
        observation = observation_from_block(block)
        item_classes = tuple([map_obs_item_to_class(item)
                              for item in observation])
        observation = class_triple_to_id(item_classes)
        observations.append(observation)
    return observations
