import tkinter as tk
from tkinter import ttk, filedialog, messagebox as mbox
from PIL import Image, ImageTk, ImageDraw
import os.path
import cv2
import numpy as np
from nn import useNN


XMLDIRECTORY = 'digits_haar_cascade.xml'
HEIGHT_OF_PANEL = 300
PADDING_FACTOR = 1.4
CACHE_LOCATION = './cache/images'
CACHE_UPPER_LOCATION = './cache'
NN_LOCATION = './trainednn.pt'


def predict(num):
    result = useNN(NN_LOCATION, CACHE_UPPER_LOCATION, len(num))
    print(result)
    #print(list(zip(num, result)))


def cancel():
    '''
    Single message box for checking if the person
    wants to exit the application or not
    '''
    yes = mbox.askyesno("Exit", "Do you want to exit?")
    if yes:
        exit()


def detect(im, xml):
    '''
    Returns the origin, width and height
    of the detected rectangles
    '''
    digit_cascade = cv2.CascadeClassifier(xml)
    digits = digit_cascade.detectMultiScale(im)
    return digits


def annotate_detection(im, regions, color=128):
    '''
    Draw the rectangles on the image
    '''
    clone = im.copy()
    draw = ImageDraw.Draw(clone)
    for (x, y, w, h) in regions:
        draw.rectangle((x, y, x + w, y + h), outline=color)
    return clone


def black_white(image, thresh):
    '''
    Generate the Threshold Image
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    retval, image = cv2.threshold(
        image, int(thresh), 255, cv2.THRESH_BINARY_INV)
    return image


def sort_n_process(digits, clone, v_or_h, type_):
    global numbers_auto
    if len(numbers_auto) != 0:
        numbers_auto = []

    numbers = list(digits[digits[:, v_or_h].argsort()])
    for index, rect in enumerate(numbers):
        leng = int(rect[3] * PADDING_FACTOR)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = clone[max(0, pt1):min(pt1 + leng, clone.shape[0]), max(0, pt2):min(pt2 + leng, clone.shape[1])]
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        saveloc = os.path.join(CACHE_LOCATION, f"number{index}.jpg")
        if not cv2.imwrite(saveloc, roi):
            print("Image files cannot be saved: Write permissions may not be provided!")
        numbers_auto.append(roi)
    predict(numbers_auto)


def image_processing(path, thresh):
    global auto, canvas_h_c, label_h_c, haar_or_contour, haar, contour_
    global canvas_v_h, label_v_h, vertical, horizontal, vertical_or_horizontal

    # Get the processed image for further use
    image = cv2.imread(path)
    if thresh != 0:
        image = black_white(image, thresh)
    clone = image.copy()

    # Display the rectangles found from HAAR classifier
    im = Image.fromarray(image)
    haar_image = clone.copy()
    haar_digits = detect(haar_image, XMLDIRECTORY)
    haar_result = annotate_detection(im, haar_digits)
    haar_result.show()

    # Display the rectangles using Contours
    contour_image = clone.copy()
    contours, _ = cv2.findContours(
        contour_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_digits = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(contour_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        contour_digits.append(np.array([x, y, w, h]))

    contour_result = annotate_detection(im, contour_digits)
    contour_result.show()

    # Styling the vertical or horizontal radio button
    if canvas_v_h is None or label_v_h is None or vertical is None or horizontal is None:
        canvas_v_h = tk.Canvas(auto)
        canvas_v_h.pack(expand='yes', fill='x', pady=5, padx=5)
        vertical_or_horizontal = tk.IntVar()
        label_v_h = ttk.Label(
            canvas_v_h, text="Select the alignment of numbers")
        label_v_h.pack(side='left')
        vertical = tk.Radiobutton(
            canvas_v_h, text="VERTICAL", variable=vertical_or_horizontal, value=1)
        vertical.pack(side='left')
        horizontal = tk.Radiobutton(
            canvas_v_h, text="HORIZONTAL", variable=vertical_or_horizontal, value=0)
        horizontal.pack(side='left')

    # Styling the HAAR or Contour radio button
    if canvas_h_c is None or label_h_c is None or haar is None or contour_ is None:
        canvas_h_c = tk.Canvas(auto)
        canvas_h_c.pack(expand='yes', fill='x', pady=5, padx=5)
        haar_or_contour = tk.IntVar()
        label_h_c = ttk.Label(canvas_h_c, text="Select any one to predict:")
        label_h_c.pack(side='left')

        haar = tk.Radiobutton(canvas_h_c, text="HAAR", variable=haar_or_contour, value=1,
                              command=lambda: sort_n_process(haar_digits, clone, vertical_or_horizontal.get(), "HAAR"))
        haar.pack(side='left')
        # If HAAR Classifier is unable to detect any digits, disable the button
        if len(haar_digits) == 0:
            haar.configure(state='disable')

        contour_ = tk.Radiobutton(canvas_h_c, text="CONTOUR", variable=haar_or_contour, value=2, command=lambda: sort_n_process(
            np.array(contour_digits), clone, vertical_or_horizontal.get(), "CONTOUR"))
        contour_.pack(side='left')
        # If no contours are found, disable the button
        if len(contour_digits) == 0:
            contour_.configure(state='disable')
    else:
        # configure is used to reset the attributes
        haar.configure(command=lambda: sort_n_process(
            haar_digits, clone, vertical_or_horizontal.get(), "HAAR"))
        if len(haar_digits) == 0:
            haar.configure(state='disable')
        contour_.configure(command=lambda: sort_n_process(
            haar_digits, clone, vertical_or_horizontal.get(), "CONTOUR"))
        if len(contour_digits) == 0:
            contour_.configure(state='disable')


def threshold_determine(thresh, image):
    '''
    The function responsible for printing the
    threshold image on Panel B - Right side image
    '''
    global panelB
    image = black_white(image, thresh)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    if panelB is None:
        panelB = tk.Label(image=image)
        panelB.image = image
        panelB.pack(padx=10, pady=10)
    else:
        panelB.configure(image=image)
        panelB.image = image


def auto_select():
    path = ''
    '''
    Global access to ensure that changes are
    being made to this variables
    '''
    global panelA, panelB, submit, can_auto, threshold, haar, contour_, image_canvas, canvas_h_c, label_h_c

    # For new selection, we need to reset the radio buttons
    if canvas_h_c is not None:
        canvas_h_c.pack_forget()
        label_h_c.pack_forget()
        haar.pack_forget()
        contour_.pack_forget()
        canvas_h_c = None
        label_h_c = None
        haar = None
        contour_ = None

    # Extensions permitted to be selected for prediction
    filetypes = (("jpeg files", "*.jpg"),
                 ("png files", "*.png"))
    while True:
        # Opens the window to browse to the desired image
        path = filedialog.askopenfilename(
            title="choose your file", filetypes=filetypes)
        # To check for selection of the file
        if len(path) > 0:
            # To check for validity of the file location
            if os.path.isfile(path):
                image = cv2.imread(path, 1)

                # Resizing the image keeping the aspect ratio intact
                width, height, _ = image.shape
                width = (HEIGHT_OF_PANEL * width) // height
                height = HEIGHT_OF_PANEL
                image = cv2.resize(image, (width, height),
                                   interpolation=cv2.INTER_AREA)

                # Convert resized image to Tkinter image
                image_ = Image.fromarray(image)
                image_ = ImageTk.PhotoImage(image_)

                # To add the image to the panel
                if panelA is None or panelB is None:
                    panelA = tk.Label(image_canvas, image=image_)
                    panelA.image = image_
                    panelA.pack(side='left', padx=10, pady=10)
                    panelB = tk.Label(image_canvas, image=image_)
                    panelB.image = image_
                    panelB.pack(side='left', padx=10, pady=10)
                else:
                    # configure is used to change an attribute/reset the panel
                    panelA.configure(image=image_)
                    panelA.image = image_
                    panelB.configure(image=image_)
                    panelB.image = image_

                # To style the buttons
                if submit is None or can_auto is None or threshold is None:
                    threshold_canvas = tk.Canvas(auto)
                    threshold_canvas.pack(
                        expand='yes', fill='x', padx=5, pady=5)
                    label_thresh = ttk.Label(
                        threshold_canvas, text="Threshold")
                    label_thresh.pack(side='left')
                    threshold = tk.Spinbox(threshold_canvas, from_=0, to=255, bd=4,
                                           command=lambda: threshold_determine(threshold.get(), image.copy()))
                    threshold.pack(side="left")

                    button_canvas = tk.Canvas(auto)
                    button_canvas.pack(expand='yes', fill='x', padx=5, pady=5)
                    submit = ttk.Button(button_canvas, text='Submit', command=lambda: image_processing(
                        path, threshold.get()))
                    submit.pack(side=tk.LEFT, expand=tk.YES, fill=tk.X)
                    can_auto = ttk.Button(
                        button_canvas, text="Cancel", command=cancel)
                    can_auto.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
                else:
                    # configure is used to reset the attribute
                    submit.configure(
                        text='Submit', command=lambda: image_processing(path, threshold.get()))
                    threshold.configure(
                        command=lambda: threshold_determine(threshold.get(), image.copy()))
                break
            else:
                mbox.showwarning("Error", "Path doesn't exist")
        else:
            break


def manual_select():
    path = ''
    global numbers_manual, refPt, can_manual, canvas_manual_buttons, predict_button
    if len(numbers_manual) != 0:
        numbers_manual = []
        refPt = []
    filetypes = (("jpeg files", "*.jpg"),
                 ("png files", "*.png"), ("all files", "*.*"))
    while True:
        path = filedialog.askopenfilename(
            title="choose your file", filetypes=filetypes)
        if len(path) > 0:
            if os.path.isfile(path):
                image = cv2.imread(path, 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                width = 400
                height = 600
                image = cv2.resize(image, (width, height),
                                   interpolation=cv2.INTER_AREA)

                def click_and_crop(event, x, y, flags, param):
                    global refPt
                    nonlocal image
                    if event == cv2.EVENT_LBUTTONDOWN:
                        refPt = [(x, y)]
                    elif event == cv2.EVENT_LBUTTONUP:
                        refPt.append((x, y))
                        cv2.rectangle(
                            image, refPt[0], refPt[1], (0, 255, 0), 2)
                        cv2.imshow("image", image)

                clone = image.copy()
                cv2.namedWindow("image")
                cv2.setMouseCallback("image", click_and_crop)

                numbers = []
                while True:
                    cv2.imshow("image", image)
                    if len(refPt) == 2:
                        numbers.append([refPt[0], refPt[1]])
                        refPt = []
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("r"):
                        image = clone.copy()
                        refPt = []
                        numbers = []
                    elif key == ord("q"):
                        break

                for number in numbers:
                    roi = clone[number[0][1]:number[1]
                                [1], number[0][0]:number[1][0]]
                    numbers_manual.append(roi)
                # cv2.destroyAllWindows()
                saveImage = Image.fromarray(numbers_manual[-1])
                saveImage.save("predict.jpg")
                for img in numbers_manual:
                    cv2.imshow("img", img)
                    cv2.waitKey(0)
                cv2.destroyAllWindows()
                if can_manual is None:
                    # predict_button = ttk.Button(
                    #     canvas_manual_buttons, text="Predict", command=lambda: predict(numbers_manual))
                    # predict_button.pack(side="left", expand='yes', fill='x')
                    can_manual = ttk.Button(
                        canvas_manual_buttons, text="Cancel", command=cancel)
                    can_manual.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
                else:
                    # predict_button.configure(
                    #     command=lambda: predict(numbers_manual))
                    can_manual.configure(command=cancel)
                break
            else:
                mbox.showwarning("Error", "Path doesn't exist")
        else:
            break


if __name__ == "__main__":
    win = tk.Tk()

    '''
    Variables for auto-mode
    '''
    panelA = None           # To display original image
    panelB = None           # To display threshold image
    submit = None           # Continue Button
    can_auto = None         # Cancel Button
    threshold = None        # Threshold slider
    numbers_auto = []       # Co-ordinates of the image

    haar_or_contour = None  # Stores the selected button
    canvas_h_c = None       # Canvas for the above
    haar = None             # Radio button - HAAR
    contour_ = None         # Radio button - Contour
    label_h_c = None        # Canvas for the above

    canvas_v_h = None
    vertical_or_horizontal = None
    label_v_h = None
    vertical = None
    horizontal = None

    '''
    Variables for manual-mode
    '''
    predict_button = None           # Predict Button
    can_manual = None               # Cancel Button
    canvas_manual_buttons = None    # Canvas for buttons
    numbers_manual = []             # Co-ordinates of the image
    refPt = []                      # Co-ordinates of a single image

    win.title("Handwritten Digit Recognition")
    # auto.geometry('400x600')
    win.resizable(width=True, height=True)

    tabControl = ttk.Notebook(win)
    auto = ttk.Frame(tabControl)
    tabControl.add(auto, text="Auto")
    manual = ttk.Frame(tabControl)
    tabControl.add(manual, text="Manual")
    tabControl.pack(expand=1, fill="both")

    select_canvas = tk.Canvas(auto)
    select_canvas.pack(expand='yes', fill="x", padx=5, pady=5)
    select_image_btn = ttk.Button(
        select_canvas, text="Select An Image", command=auto_select)
    select_image_btn.pack(side='left', expand='yes', fill='x')

    image_canvas = tk.Canvas(auto)
    image_canvas.pack(expand='yes', fill='x', padx=5, pady=5)

    select_canvas_manual = tk.Canvas(manual)
    select_canvas_manual.pack(side='top', expand='yes',
                              fill="x", padx=5, pady=5)
    select_image_btn = ttk.Button(
        select_canvas_manual, text="Select An Image", command=manual_select)
    select_image_btn.pack(side='left', expand='yes', fill='x')
    canvas_manual_buttons = tk.Canvas(manual)
    canvas_manual_buttons.pack(
        side='bottom', expand='yes', fill='x', padx=5, pady=5)

    win.mainloop()
