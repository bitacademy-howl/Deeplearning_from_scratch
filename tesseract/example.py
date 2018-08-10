from PIL import Image
from pytesseract import *

def OCR_eng(imgfile, lang='eng'):
    im = Image.open(imgfile)
    text = image_to_string(im, lang=lang)

    print('+++ OCR Result +++')
    print(text)

def OCR_kor(imgfile, lang='kor'):
    im = Image.open(imgfile)
    text = image_to_string(im, lang=lang)

    print('+++ OCR Result +++')
    print(text)

OCR_eng('images/ocr_test.jpg')
OCR_eng('images/1.png')

OCR_kor("images/2.png")
OCR_kor("images/손글씨.png")