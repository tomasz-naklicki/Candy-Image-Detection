import cv2
import numpy as np
from . import colors


def morph(src: np.ndarray):
    """Function performing 2 morphology transformations, opening and closing the source image in order to fill the empty spots in the masks

    Parameters
    ----------
    src: numpy.ndarray
        Source image

    Returns
    -------
    src: numpy.ndarray
        Source image after transformations
    """
    src = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8))
    src = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel=np.ones((2, 2), np.uint8))
    return src


def findCandy(src: np.ndarray, lower: np.ndarray, upper: np.ndarray):
    """Function creating a mask for a specific color, detecting edges and counting the number of found contours of a specific area

    Parameters
    ----------
    src: numpy.ndarray
        Source image

    lower: np.ndarray
        Array containing the lower bounds of the value of the chosen color

    upper: np.ndarray
        Array containing the upper bounds of the value of the chosen color

    Returns
    -------
    count: int
        Number of found objects

    """
    count = 0
    mask = cv2.inRange(src, lower, upper)
    mask = morph(mask)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask = cv2.Canny(mask, 1000, 0)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        if cv2.contourArea(contour) > 50 and cv2.contourArea(contour) < 1600:
            count = count + 1
    return count


def countCandy(img_path: str, color: colors.Color) -> dict[str, int]:
    """Object detection function, according to the project description.

    Parameters
    ----------
    img_path : str
        Path to processed image.
    color : colors.Color
        Flag describing the colors of candy to be counted.

    Returns
    -------
    dict[str, int]
        Dictionary with quantity of each object.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Converting an image to HSV color scale
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Resizing the image
    img_hsv = cv2.resize(img_hsv, None, fx=0.2, fy=0.2)

    # Finding the number of candy of each color
    red = (
        {"red": findCandy(img_hsv, colors.LRED, colors.URED)}
        if color & colors.Color.Red
        else {}
    )

    yellow = (
        {"yellow": findCandy(img_hsv, colors.LYELLOW, colors.UYELLOW)}
        if color & colors.Color.Yellow
        else {}
    )

    green = (
        {"green": findCandy(img_hsv, colors.LGREEN, colors.UGREEN)}
        if color & colors.Color.Green
        else {}
    )

    purple = (
        {"purple": findCandy(img_hsv, colors.LPURPLE, colors.UPURPLE)}
        if color & colors.Color.Purple
        else {}
    )

    return red | yellow | green | purple
