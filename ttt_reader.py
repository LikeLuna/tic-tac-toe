"""
Tic-Tac-Toe board reader.

Takes an image of a 3x3 tic-tac-toe board (assumed to be roughly cropped
to the board itself) and prints the contents of every cell.

Usage:
    python ttt_reader.py path/to/board.png
    python ttt_reader.py path/to/board.png --draw    # also saves an annotated image
"""

import argparse
import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]        # top-left has smallest x+y
    rect[2] = pts[np.argmax(s)]        # bottom-right has largest x+y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]     # top-right has smallest y-x
    rect[3] = pts[np.argmax(diff)]     # bottom-left has largest y-x
    return rect


def four_point_transform(img: np.ndarray, pts: np.ndarray, size: int = 450) -> np.ndarray:
    """Warp the quadrilateral defined by pts into a flat size x size square."""
    rect = order_points(pts)
    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1],
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (size, size))


def find_board(img: np.ndarray, size: int = 450):
    """
    Locate the tic-tac-toe board inside a (possibly cluttered) photo and
    return a flat, top-down, cropped version of just the board.

    Strategy:
      1. Blur + adaptive threshold / Canny to get clean edges regardless
         of lighting (works far better than a single global threshold
         when there's background clutter or shadows).
      2. Find all external contours, keep the largest ones by area.
      3. For each, approximate its shape with approxPolyDP; the board
         should approximate to a 4-sided polygon (a quadrilateral) that
         is reasonably square (not a long thin sliver) and covers a
         sizeable chunk of the frame -- that combo filters out random
         background objects.
      4. If found, perspective-warp those 4 corners flat. If nothing
         confidently quadrilateral is found, fall back to the original
         image (assume it was already cropped to the board).
    """
    h, w = img.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold copes with uneven lighting/shadows far better
    # than a single global Otsu threshold over the whole photo.
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    best_quad = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.15 * img_area:      # too small to plausibly be the board
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            x, y, bw, bh = cv2.boundingRect(pts)
            aspect = bw / float(bh)
            if 0.7 <= aspect <= 1.3 and area > best_area:   # roughly square
                best_area = area
                best_quad = pts

    if best_quad is None:
        return img  # fall back: assume the photo is already just the board

    return four_point_transform(img, best_quad.astype("float32"), size=size)


def classify_cell(cell_img: np.ndarray) -> str:
    """
    Classify a single cell crop as 'X', 'O', or 'no symbol'.

    Idea:
      - Threshold the cell so the symbol's strokes become white pixels.
      - If almost no ink -> empty cell.
      - Find contours with hierarchy (RETR_CCOMP): an 'O' is a ring, so it
        produces an outer contour with a child contour (the hole) inside it.
        An 'X' is made of solid crossing strokes, so it has no such hole.
    """
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY) if cell_img.ndim == 3 else cell_img
    h, w = gray.shape

    # Trim a margin so grid lines from the board don't get picked up as ink.
    margin = int(0.12 * min(h, w))
    gray = gray[margin:h - margin, margin:w - margin]
    if gray.size == 0:
        return "no symbol"

    # An empty cell (even with a subtle lighting gradient or shadow) is
    # near-flat: pixel values barely vary. A cell with a drawn symbol has
    # real dark strokes against a lighter background, so its standard
    # deviation is much higher. Checking this BEFORE thresholding matters:
    # Otsu's method always forces a split into two groups even when there's
    # no real content, so on a flat/noisy empty cell it can grab half the
    # pixels as "ink" purely from lighting noise. This check filters those
    # false positives out up front.
    if gray.std() < 10:
        return "no symbol"

    # Otsu threshold, symbol assumed darker (or lighter) than background.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ink_ratio = np.count_nonzero(thresh) / thresh.size
    if ink_ratio < 0.02:
        return "no symbol"

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return "no symbol"

    hierarchy = hierarchy[0]
    min_area = 0.01 * thresh.size
    significant = [i for i, c in enumerate(contours) if cv2.contourArea(c) > min_area]
    if not significant:
        return "no symbol"

    has_hole = any(
        hierarchy[i][2] != -1 and cv2.contourArea(contours[hierarchy[i][2]]) > min_area * 0.3
        for i in significant
    )

    return "O" if has_hole else "X"


def process_image(path: str, grid_size: int = 3):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    board = find_board(img)
    h, w = board.shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size

    results = {}
    for row in range(grid_size):
        for col in range(grid_size):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            cell = board[y1:y2, x1:x2]
            results[(row, col)] = classify_cell(cell)

    # Return the cropped/warped board (not the original photo) so drawing
    # results on top of it lines up correctly.
    return results, board, cell_h, cell_w


def process_image_from_array(img: np.ndarray, grid_size: int = 3):
    """Same as process_image but takes an already-loaded image array (used by the web app)."""
    board = find_board(img)
    h, w = board.shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size

    results = {}
    for row in range(grid_size):
        for col in range(grid_size):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            cell = board[y1:y2, x1:x2]
            results[(row, col)] = classify_cell(cell)

    # Return the cropped/warped board so drawing lines up correctly.
    return results, board, cell_h, cell_w


def print_results(results: dict, grid_size: int = 3):
    for row in range(grid_size):
        for col in range(grid_size):
            symbol = results[(row, col)]
            print(f"At position ({row},{col}) there is {symbol}")


def print_results_as_lines(results: dict, grid_size: int = 3):
    """Same info as print_results but returned as a list of strings (used by the web app)."""
    lines = []
    for row in range(grid_size):
        for col in range(grid_size):
            symbol = results[(row, col)]
            lines.append(f"At position ({row},{col}) there is {symbol}")
    return lines


def draw_results(img, results, cell_h, cell_w, grid_size=3, out_path="output.png"):
    annotated = img.copy()
    for row in range(grid_size):
        for col in range(grid_size):
            symbol = results[(row, col)]
            text = symbol if symbol != "no symbol" else "-"
            cx = col * cell_w + cell_w // 2
            cy = row * cell_h + cell_h // 2
            cv2.putText(annotated, text, (cx - 15, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    if out_path:
        cv2.imwrite(out_path, annotated)
    return annotated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read a 3x3 tic-tac-toe board from an image.")
    parser.add_argument("image", help="Path to the tic-tac-toe board image")
    parser.add_argument("--draw", action="store_true", help="Save an annotated output image")
    parser.add_argument("--out", default="output.png", help="Path for the annotated image")
    args = parser.parse_args()

    results, img, cell_h, cell_w = process_image(args.image)
    print_results(results)

    if args.draw:
        draw_results(img, results, cell_h, cell_w, out_path=args.out)
        print(f"\nAnnotated image saved to: {args.out}")