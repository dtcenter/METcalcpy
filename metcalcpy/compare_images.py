# ============================*
 # ** Copyright UCAR (c) 2020
 # ** University Corporation for Atmospheric Research (UCAR)
 # ** National Center for Atmospheric Research (NCAR)
 # ** Research Applications Lab (RAL)
 # ** P.O.Box 3000, Boulder, Colorado, 80307-3000, USA
 # ============================*
 
 
 
"""
Program Name: compare_images.py
Required packages:
 - pip install scikit-image
 - pip install opencv-python
 - pip install imutils

How to use:
   #initialise the class
   compare = CompareImages('/path/to/image1', '/path/to/image2')
   # get the  mssim value. If vmssim == 1 - images are identical
   mssim = compare.get_ssim()

   compare.save_difference_image(self, '/path/to/image')
   compare.save_difference_image(self, '/path/to/image')
   compare.show_thresh_image(True)
"""
import argparse
import imutils
import cv2
from skimage.metrics import structural_similarity


class CompareImages:
    """A class that performs images comparison using mean structural similarity index over the image (mssim).
        The images are considered equal if mssim = 1
        This class has methods to save or display found differences as an image.
        Currently, the following file formats are supported:
       -   Windows bitmaps - *.bmp, *.dib
       -   JPEG files - *.jpeg, *.jpg, *.jpe
       -   JPEG 2000 files - *.jp2
       -   Portable Network Graphics - *.png
       -   WebP - *.webp
       -   Portable image format - *.pbm, *.pgm, *.ppm *.pxm, *.pnm
       -   Sun rasters - *.sr, *.ras
       -   TIFF files - *.tiff, *.tif
       -   OpenEXR Image files - *.exr
       -   Radiance HDR - *.hdr, *.pic
     """

    def __init__(self, image_path_1, image_path_2):
        """Initialises the class by creating images matrix, mssim
            and the difference image

            Args:
                image_path_1 - path to image 1
                image_path_2 - path to image 2
            Raises:
                an error if on of the images cannot be read
                (because of missing file, improper permissions, unsupported or invalid format)
        """
        self.image_1 = cv2.imread(image_path_1)
        if self.image_1 is None:
            raise KeyError(f'Problems reading the image: {image_path_1}')
        self.image_2 = cv2.imread(image_path_2)
        if self.image_2 is None:
            raise KeyError(f'Problems reading the image: {image_path_2}')

        # convert the images to grayscale
        gray_image_1 = cv2.cvtColor(self.image_1, cv2.COLOR_BGR2GRAY)
        gray_image_2 = cv2.cvtColor(self.image_2, cv2.COLOR_BGR2GRAY)

        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        (self.mssim, self.grad) = structural_similarity(gray_image_1, gray_image_2, full=True)
        self.grad = (self.grad * 255).astype("uint8")
        self.thresh = None
        self.diff_in_image_1 = None
        self.diff_in_image_2 = None

    def get_mssim(self):
        """Returns mean structural similarity index over the image (mssim) as a float
            The images are equal if mssim = 1
        """
        return self.mssim

    def save_difference_image(self, path_to_file):
        """Saves the difference image to the specified file. The image format is chosen based on the
              filename extension
            Args:
                path_to_file - Name of the file
        """
        cv2.imwrite(path_to_file, self.grad)

    def show_difference_image(self, wait_for_key_event=False):
        """ Displays the difference image in the popup window
            Args:
                wait_for_key_event : If True, The function waitKey waits for a key event infinitely
        """
        cv2.imshow("Difference", self.grad)
        if wait_for_key_event:
            cv2.waitKey(0)

    def _init_thresh(self):
        """ The function applies fixed-level thresholding to a multiple-channel array
            to create a threshold image
        """

        # threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        self.thresh = cv2.threshold(self.grad, 0, 255,
                                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    def save_thresh_image(self, path_to_file):
        """Saves the threshold image to the specified file. The image format is chosen based on the
            filename extension
            Args:
                path_to_file - Name of the file
        """
        if not self.thresh:
            self._init_thresh()
        cv2.imwrite(path_to_file, self.thresh)

    def show_thresh_image(self, wait_for_key_event=False):
        """ Displays the threshold image in the popup window
            Args:
                wait_for_key_event : If True, The function waitKey waits for a key event infinitely
        """
        if not self.thresh:
            self._init_thresh()
        cv2.imshow("Threshold", self.thresh)
        if wait_for_key_event:
            cv2.waitKey(0)

    def _init_diff_in_image_1(self):
        """This function initialises an image with the computed the bounding box
            on the first input images to represent where the two images differ
        """
        if self.thresh is None:
            self._init_thresh()
        contours = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        self.diff_in_image_1 = self.image_1.copy()
        # loop over the contours
        for contour in contours:
            # compute the bounding box of the contour and then draw the
            # bounding box on the first input image to represent where the two
            # images differ
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(self.diff_in_image_1, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def _init_diff_in_image_2(self):
        """This function initialises an image with the computed the bounding box
            on the second input images to represent where the two images differ
        """
        if self.thresh is None:
            self._init_thresh()
        contours = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        self.diff_in_image_2 = self.image_2.copy()
        # loop over the contours
        for contour in contours:
            # compute the bounding box of the contour and then draw the
            # bounding box on the the first input image to represent where the two
            # images differ
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(self.diff_in_image_2, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def save_difference_in_image_1(self, path_to_file):
        """Saves the first image with  the bounding box to the specified file.
            The image format is chosen based on the filename extension
            Args:
                path_to_file - Name of the file
        """
        if self.diff_in_image_1 is None:
            self._init_diff_in_image_1()
        cv2.imwrite(path_to_file, self.diff_in_image_1)

    def show_difference_in_image_1(self, wait_for_key_event=False):
        """ Displays the first image with  the bounding box in the popup window
            Args:
                wait_for_key_event : If True, The function waitKey waits for a key event infinitely
        """
        if self.diff_in_image_1 is None:
            self._init_diff_in_image_1()
        cv2.imshow('Difference in image 1', self.diff_in_image_1)
        if wait_for_key_event:
            cv2.waitKey(0)

    def save_difference_in_image_2(self, path_to_file):
        """Saves the second image with  the bounding box to the specified file.
            The image format is chosen based on the filename extension
            Args:
                path_to_file - Name of the file
        """
        if not self.diff_in_image_2:
            self._init_diff_in_image_2()
        cv2.imwrite(path_to_file, self.diff_in_image_2)

    def show_difference_in_image_2(self, wait_for_key_event=False):
        """ Displays the second image with  the bounding box in the popup window
            Args:
                wait_for_key_event : If True, The function waitKey waits for a key event infinitely
        """
        if not self.diff_in_image_2:
            self._init_diff_in_image_2()
        cv2.imshow('Difference in image 2', self.diff_in_image_2)
        if wait_for_key_event:
            cv2.waitKey(0)


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ARG_PARSER = argparse.ArgumentParser()
    ARG_PARSER.add_argument("-f", "--first", required=True,
                            help="first input image")
    ARG_PARSER.add_argument("-s", "--second", required=True,
                            help="second")
    ARGS = vars(ARG_PARSER.parse_args())
    COMPARE = CompareImages(ARGS["first"], ARGS["second"])
    MSSIM = COMPARE.get_mssim()
    if MSSIM == 1.0:
        print("Images are identical")
    else:
        print("Images are different")
        COMPARE.show_difference_image()
        COMPARE.show_thresh_image()
        COMPARE.show_difference_in_image_1()
        COMPARE.show_difference_in_image_2(True)
