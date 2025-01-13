# Import the necessary packages
from helper import FACIAL_LANDMARKS_68_IDXS, FACIAL_LANDMARKS_5_IDXS
from helper import shape_to_np, rect_to_bb
import numpy as np
import cv2
from skimage import transform as tf

def crop_image(image, detector, predictor):
    """
    Detects faces in an image, crops them, and resizes them to a fixed size (163x163).
    
    Parameters:
    - image: The input image (numpy array).
    - detector: The face detector (e.g., dlib.get_frontal_face_detector()).
    - predictor: The facial landmark predictor (e.g., dlib.shape_predictor()).

    Returns:
    - roi: The cropped region of interest containing the face.
    - shape: The updated facial landmarks.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        (x, y, w, h) = rect_to_bb(rect)
        center_x = x + int(0.5 * w)
        center_y = y + int(0.5 * h)
        r = int(0.64 * h)

        new_x = center_x - r
        new_y = center_y - r

        roi = image[new_y:new_y + 2 * r, new_x:new_x + 2 * r]
        roi = cv2.resize(roi, (163, 163), interpolation=cv2.INTER_AREA)

        scale = 163. / (2 * r)
        shape = ((shape - np.array([new_x, new_y])) * scale)

        return roi, shape

class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=256, desiredFaceHeight=None):
        """
        Initializes the FaceAligner class.
        
        Parameters:
        - predictor: The facial landmark predictor (dlib.shape_predictor).
        - desiredLeftEye: The desired position of the left eye in the output face (tuple).
        - desiredFaceWidth: The desired width of the output face.
        - desiredFaceHeight: The desired height of the output face (optional).
        """
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight if desiredFaceHeight else desiredFaceWidth

    def align(self, image, gray, rect, shape, scale=None):
        """
        Aligns a face based on eye positions.

        Parameters:
        - image: The input image.
        - gray: Grayscale version of the input image.
        - rect: The bounding box (rectangle) of the face.
        - shape: Facial landmarks.
        - scale: The scaling factor (optional).

        Returns:
        - output: The aligned face.
        - scale: The scaling factor used.
        """
        shape = shape_to_np(shape)

        if len(shape) == 68:
            (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # Calculate the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # Calculate the angle between the eye centers
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0]) * self.desiredFaceWidth

        if scale is None:
            scale = 1.2 * desiredDist / dist

        # Find the center between the eyes
        eyesCenter = (float(leftEyeCenter[0] + rightEyeCenter[0]) / 2,
                      float(leftEyeCenter[1] + rightEyeCenter[1]) / 2)

        # Get the rotation matrix for the alignment
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # Adjust the translation component
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # Apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        return output, scale

    def get_tform(self, image, shape, mean_shape, scale=None):
        """
        Estimate the similarity transform based on eye and nose positions.

        Parameters:
        - image: Input image.
        - shape: Facial landmarks of the image.
        - mean_shape: Mean landmarks (template).
        """
        left_eye = [40, 39]
        right_eye = [42, 47]
        nose = [30, 31, 32, 33, 34, 35]

        leftEyeCenter = mean_shape[left_eye].mean(axis=0)
        rightEyeCenter = mean_shape[right_eye].mean(axis=0)
        noseCenter = mean_shape[nose].mean(axis=0)

        template_points = np.float32([leftEyeCenter, rightEyeCenter, noseCenter])
        leftEyeCenter = shape[left_eye].mean(axis=0)
        rightEyeCenter = shape[right_eye].mean(axis=0)
        noseCenter = shape[nose].mean(axis=0)

        dst_points = np.float32([leftEyeCenter, rightEyeCenter, noseCenter])
        tform = tf.SimilarityTransform()
        tform.estimate(template_points, dst_points)

        self.tform = tform

    def apply_tform(self, image):
        """
        Apply the similarity transform to the image.

        Parameters:
        - image: Input image.
        
        Returns:
        - output: The transformed image.
        """
        output = tf.warp(image, self.tform, output_shape=(self.desiredFaceWidth, self.desiredFaceHeight))
        output = (output * 255).astype('uint8')

        return output, None

    def align_three_points(self, image, shape, mean_shape, scale=None):
        """
        Align the face using three points: left eye, right eye, and nose.

        Parameters:
        - image: Input image.
        - shape: Facial landmarks.
        - mean_shape: Mean landmarks (template).

        Returns:
        - output: The aligned image.
        """
        left_eye = [40, 39]
        right_eye = [42, 47]
        nose = [30, 31, 32, 33, 34, 35]

        leftEyeCenter = mean_shape[left_eye].mean(axis=0)
        rightEyeCenter = mean_shape[right_eye].mean(axis=0)
        noseCenter = mean_shape[nose].mean(axis=0)

        template_points = np.float32([leftEyeCenter, rightEyeCenter, noseCenter])
        leftEyeCenter = shape[left_eye].mean(axis=0)
        rightEyeCenter = shape[right_eye].mean(axis=0)
        noseCenter = shape[nose].mean(axis=0)

        dst_points = np.float32([leftEyeCenter, rightEyeCenter, noseCenter])
        tform = tf.SimilarityTransform()
        tform.estimate(template_points, dst_points)
        output = tf.warp(image, tform, output_shape=(self.desiredFaceWidth, self.desiredFaceHeight))
        output = (output * 255).astype('uint8')

        return output, None

    def align_box(self, shape, scale=None):
        """
        Calculate the mean point for alignment based on specific facial features.

        Parameters:
        - shape: Facial landmarks.
        
        Returns:
        - mean_pts: The mean point of the selected features.
        """
        left_eye = [40, 39]
        right_eye = [42, 47]
        nose = [30, 31, 32, 33, 34, 35]

        all_pts = nose + left_eye + right_eye
        mean_pts = shape[all_pts, :].mean(axis=0).astype(int)
        return mean_pts
