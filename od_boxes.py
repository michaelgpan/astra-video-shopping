import cv2
import argparse
import sys
import json
import random
import os.path
import numpy as np


COLORS_COCO = {
    0: [189, 114, 0],    # Blue
    1: [25, 83, 217],    # Orange
    2: [32, 176, 237],   # Yellow
    3: [142, 47, 126],   # Purple
    4: [48, 172, 119],   # Green
    5: [238, 190, 77],   # Light Blue
    6: [47, 20, 162],    # Dark Red
    7: [77, 77, 77],     # Gray
    8: [153, 153, 153],  # Light Gray
    9: [0, 0, 255],      # Red
    10: [0, 128, 255],   # Orange
    11: [0, 191, 191],   # Olive
    12: [0, 255, 0],     # Green
    13: [255, 0, 0],     # Blue
    14: [255, 0, 170],   # Purple 
    15: [0, 85, 85],     # Dark Olive
    16: [0, 170, 85],    # Olive Green
    17: [0, 255, 85],    # Light Green
    18: [0, 85, 170],    # Brown
    19: [0, 170, 170],   # Dark Yellow
    20: [0, 255, 170],   # Lime
    21: [0, 85, 255],    # Orange Red
    22: [0, 170, 255],   # Light Orange
    23: [0, 255, 255],   # Yellow
    24: [128, 85, 0],    # Teal
    25: [128, 170, 0],   # Dark Cyan
    26: [128, 255, 0],   # Aquamarine
    27: [128, 0, 85],    # Indigo
    28: [128, 85, 85],   # Slate Blue
    29: [128, 170, 85],  # Light Sea Green
    30: [128, 255, 85],  # Medium Sea Green
    31: [128, 0, 170],   # Dark Magenta
    32: [128, 85, 170],  # Medium Orchid
    33: [128, 170, 170], # Khaki
    34: [128, 255, 170], # Pale Green
    35: [128, 0, 255],   # Deep Pink
    36: [128, 85, 255],  # Hot Pink
    37: [128, 170, 255], # Light Coral
    38: [128, 255, 255], # Light Yellow
    39: [255, 85, 0],    # Dodger Blue
    40: [255, 170, 0],   # Deep Sky Blue
    41: [255, 255, 0],   # Cyan
    42: [255, 0, 85],    # Medium Blue
    43: [255, 85, 85],   # Slate Blue
    44: [255, 170, 85],  # Steel Blue
    45: [255, 255, 85],  # Light Cyan
    46: [255, 0, 170],   # Purple
    47: [255, 85, 170],  # Medium Purple
    48: [255, 170, 170], # Lavender
    49: [255, 255, 170], # Pale Turquoise
    50: [255, 0, 255],   # Magenta
    51: [255, 85, 255],  # Orchid
    52: [255, 170, 255], # Plum
    53: [255, 255, 255], # White
    54: [0, 0, 85],      # Dark Red
    55: [0, 0, 128],     # Maroon
    56: [0, 0, 170],     # Firebrick
    57: [0, 0, 212],     # Crimson
    58: [0, 0, 255],     # Red
    59: [0, 85, 0],      # Dark Green
    60: [0, 128, 0],     # Green
    61: [0, 170, 0],     # Forest Green
    62: [0, 212, 0],     # Lime Green
    63: [0, 255, 0],     # Lime
    64: [85, 0, 0],      # Midnight Blue
    65: [128, 0, 0],     # Navy
    66: [170, 0, 0],     # Dark Blue
    67: [212, 0, 0],     # Medium Blue
    68: [255, 0, 0],     # Blue
    69: [0, 85, 85],     # Olive Drab
    70: [0, 85, 128],    # Peru
    71: [0, 85, 170],    # Chocolate
    72: [0, 85, 212],    # Dark Orange
    73: [0, 85, 255],    # Orange Red
    74: [0, 128, 85],    # Olive Green
    75: [0, 128, 128],   # Dark Khaki
    76: [0, 128, 170],   # Goldenrod
    77: [0, 128, 212],   # Dark Goldenrod
    78: [0, 128, 255],   # Orange
    79: [0, 170, 85],    # Dark Olive Green
}


def get_colors_from_json(colors_json: str | os.PathLike, verbose: bool = False):
    """
    Parses [R,G,B] colors from a JSON file specified at @param:colors_json.
    R, G, B must be >= 0 and <= 255, and JSON must have format:
    {
        "class_index": [R,G,B],
        ...
    }
    Logs errors if @param:verbose is set to True.
    """

    def is_valid_color(color):
        return (
            isinstance(color, list) and 
            len(color) == 3 and 
            all(isinstance(c, int) and 0 <= c <= 255 for c in color)
        )

    try:
        with open(colors_json, "r") as f:
            mask_colors = json.load(f)
            mask_colors = {int(k): v for k, v in mask_colors.items()}
            if not(all(is_valid_color(color) for color in mask_colors.values())):
                raise ValueError(f"Erroneous color values in \"{colors_json}\"")
            return mask_colors
    except ValueError as e:
        if verbose:
            print(f"ERROR: Couldn't parse mask colors: {e.args[0]}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        if verbose:
            print(f"ERROR: Invalid mask colors file \"{colors_json}\"")
    return None

def get_rand_color():
    return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

def scale_mask(mask: np.ndarray, mask_w: int, mask_h: int, inp_w: int, inp_h: int):
    """Reshape mask into 2D and scale to input image size"""
    mask = mask.reshape((mask_h, mask_w))
    return cv2.resize(mask, (inp_w, inp_h), interpolation=cv2.INTER_LINEAR)

def overlay_mask(mask: np.ndarray, img: np.ndarray, roi: tuple[int, int, int, int], thresh: float = 0, color: list[int] | None = None):
    """Confine mask to bounding box and overlay on img"""
    x, y, dx, dy = roi
    mask[y: y + dy, x: x + dx] = mask[y: y + dy, x: x + dx] > thresh
    colored_mask = np.zeros_like(img)
    colored_mask[mask == 1] = color or get_rand_color()
    return cv2.add(img, colored_mask)

def add_mask(mask: np.ndarray, combined_mask: np.ndarray, roi: tuple[int, int, int, int], thresh: float = 0, color: list[int] | None = None):
    """Confine mask to bounding box and add to combined mask"""
    x, y, dx, dy = roi
    mask[y: y + dy, x: x + dx] = mask[y: y + dy, x: x + dx] > thresh
    combined_mask[mask == 1] = color or get_rand_color()

def image_od(src, dst, json_od_result:str, mask_colors: dict, verbose: bool):
    img = cv2.imread(src)
    inp_h, inp_w, _ = img.shape
    try:
        od_result = json.loads(json_od_result)
    except:
        print("Error: failed to parse JSON data.")
        sys.exit(1)
    if not 'items' in od_result:
        print("Error: object-detection data not found in the input JSON")
        sys.exit(1)
    
    prev_mask = None
    combined_mask = np.zeros_like(img)
    # separate loop to prevent masks from affecting bounding box color
    for i, detection in enumerate(od_result['items']):
        try:
            bb = detection['bounding_box']
            x1 = int(bb['origin']['x'])
            y1 = int(bb['origin']['y'])
            dx = int(bb['size']['x'])
            dy = int(bb['size']['y'])
            ci = detection['class_index']
            if detection['mask']['data']:
                mask_h, mask_w = detection['mask']['height'], detection['mask']['width']
                mask = np.array(detection['mask']['data'], dtype=np.float32)
                if prev_mask is not None and np.array_equal(prev_mask, mask):
                    if verbose:
                        print("Current mask same as previous mask")
                prev_mask = mask
                mask = scale_mask(mask, mask_w, mask_h, inp_w, inp_h)
                mask_color = mask_colors.get(ci, None) if mask_colors else None
                if mask_color is None:
                    if verbose:
                        print(f"WARNING: Using random color for detection {i} as no color defined for class {ci}")
                add_mask(mask, combined_mask, (x1, y1, dx, dy), color=mask_color)
                # individually overlay mask on image and save a copy
                # cv2.imwrite(f'mask_{i} (class {ci}).jpg', overlay_mask(mask, cv2.imread(src), (x1, y1, dx, dy), color=mask_color))
        except KeyError as e:
            if verbose:
                print(f"WARNING: Missing {e} data for detection {i}")
            continue
    # overlay combined mask on image
    combined_mask = np.clip(combined_mask, 0, 255)
    img = cv2.add(img, combined_mask)

    print("#   Score  Class   Position        Size  Description     Landmarks")
    for i, detection in enumerate(od_result['items']):
        try:
            bb = detection['bounding_box']
            x1 = int(bb['origin']['x'])
            y1 = int(bb['origin']['y'])
            dx = int(bb['size']['x'])
            dy = int(bb['size']['y'])
            confidence = detection['confidence']
            ci = detection['class_index']
            lms = detection['landmarks']['points']
            print(f"{i:<5}{confidence:.2f}{ci:>7}  {x1:4},{y1:4}   {dx:4},{dy:4}                  ", end='')
            print(" ".join([f"{lm['x']},{lm['y']}" for lm in lms]))
            color = (0, 255, 128)
            cv2.rectangle(img, (x1, y1), (x1 + dx, y1 + dy), color, 2)
            cv2.putText(img, str(ci) + f": {confidence:.2f}", (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)[0]
            for lm in lms:
                lmx = int(lm['x'])
                lmy = int(lm['y'])
                cv2.rectangle(img, (lmx, lmy), (lmx+2, lmy+2), color, 2)
        except KeyError as e:
            if verbose:
                print(f"WARNING: Missing {e} data for detection {i}")
            continue
    cv2.imwrite(dst, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--src', help='Source image (.png or .jpg)')
    parser.add_argument('-o', '--dst', help='Destination image file')
    parser.add_argument('--mask_colors', help='JSON file containing segmentation mask colors in {"class_index": [B,G,R], ...} format')
    parser.add_argument('--verbose', action="store_true", default=False, help="Enable verbose logging")
    args = parser.parse_args()
    od_result = sys.stdin.read()

    if not os.path.isfile(args.src):
        print(f"Error: file {args.src} not found.")
        sys.exit(1)

    json_begin = od_result.find('{')
    if json_begin < 0:
        print("Error: JSON data not found in the input.")
        sys.exit(1)
    od_result = od_result[json_begin:]

    if args.mask_colors:
        mask_colors = get_colors_from_json(args.mask_colors, args.verbose)
    else:
        mask_colors = COLORS_COCO

    image_od(args.src, args.dst, od_result, mask_colors, args.verbose)

if __name__ == "__main__":
    main()