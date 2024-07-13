# coding: utf-8
from __future__ import print_function
from tkinter import *
from predict import main as pre1
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import datetime
from tkinter import messagebox as tkMessageBox
import tkinter as tk
import PySimpleGUI as sg
import io
from PIL import Image, ImageTk
import cv2
from batch_predict import main as pre2


def get_img_data1(f, maxsize=(550, 350), first=False):
    """Generate image data using PIL
    """
    img1 = cv2.imread(f)
    img2 = cv2.resize(img1, (550, 350))
    cv2.imwrite('output.jpg', img2)
    img = Image.open('output.jpg')
    img.thumbnail(maxsize)
    if first:  # tkinter is inactive the first time
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)

def make_window(theme):
    sg.theme('LightGrey2')

    # 菜单栏
    menu_def = [['Help', ['About...', ['你好']]], ]
    empty = []

    layout1 = [
        [sg.MenubarCustom(menu_def, key='-MENU-', font='Courier 15', tearoff=True)],
        [sg.Text('垃圾识别', size=(40, 1), justification='center', font=("Helvetica", 16),
                 relief=sg.RELIEF_RIDGE, k='-TEXT HEADING-', enable_events=True, expand_x=True)],
        [sg.Text('')],
        # [sg.Multiline(s=(46, 2), key='_INPUT_news_', font=("Helvetica", 15), expand_x=True)],
        [sg.Text('')],

        [sg.Text('',s=8),sg.Text('选择一张图片：', size=12, font=("Helvetica", 16)),
         sg.Input(size=(18, 1), k="path", font=("Helvetica", 16)), sg.FileBrowse('选择', font=("Helvetica", 16)),sg.Text('',s=8)],

        [sg.Text('', s=15), sg.Button('打开', font=("Helvetica", 15)), sg.Text('', s=15),
         sg.Button('识别', font=("Helvetica", 15))],
        [sg.Text('')],

        [sg.Text('')],
        [sg.Text('',s=18),sg.Text('结果: ', size=5, font=("Helvetica", 16)), sg.Text(' ', k="out", size=20, font=("Helvetica", 16))],
        [sg.Text('')],
        [sg.Text('',s=8),sg.Image(s=(550, 350), data=get_img_data1('input.jpg', first=True), key="a"),sg.Text('',s=8),],
        [sg.Text('')],
        [sg.Sizegrip()]

    ]

    layout2 = [
        [sg.Text('')],
        [sg.Text('')],

        [sg.Text('')],
        # [sg.Multiline(s=(46, 2), key='_INPUT_news_', font=("Helvetica", 15), expand_x=True)],

        [sg.Text('', s=10),sg.Text('', size=30, k = "aa",font=("Helvetica", 16))],

        [sg.Text('')],
        [sg.Text('')],



        [sg.Text('', s=30), sg.Button('批量识别', font=("Helvetica", 15))],
        [sg.Text('')],
        [sg.Text('')],

        [sg.Text('')],

        [sg.Text('', s=10), sg.Text('请在image文件夹中存入待批量识别的图片', font=("Helvetica", 15)),],
        [sg.Text('')],
    ]

    layout = [[sg.MenubarCustom(menu_def, key='-MENU-', font='Courier 15', tearoff=True)],
              ]
    layout += [[sg.TabGroup([[sg.Tab(' 单 张 识 别 ', layout1),
                              sg.Tab('                                 ', empty),

                              sg.Tab(' 批 量 识 别 ', layout2),
                              sg.Tab('                                 ', empty),

                              ]], expand_x=True, expand_y=True),

                ]]




    window = sg.Window('垃圾识别系统', layout,
                       right_click_menu_tearoff=True, grab_anywhere=True, resizable=True, margins=(0, 0),
                       use_custom_titlebar=True, finalize=True, keep_on_top=True)
    window.set_min_size(window.size)
    return window

def main_WINDOW():



    window = make_window(sg.theme())
    while True:
        event, values = window.read(timeout=100)
        # print(event)

        if event in (None, 'Exit'):
            print("[LOG] Clicked Exit!")
            break
        elif event == '打开':
            img_path = values["path"]
            # print(img_path)
            window['a'].update(data=get_img_data1(img_path, first=False))


        elif event == '识别':

            #
            image_path  = values["path"]
            a = pre1(image_path)
            window['out'].update(a)
        elif event == '批量识别':
            pre2("./image/")

            window['aa'].update('识别完成，结果请查看image.txt')


            # r_image.save("output.jpg")
            # window['a'].update(data=get_img_data1('output.jpg', first=True))





    window.close()
    exit(0)

if __name__ == '__main__':

    sg.theme('LightGrey2')
    main_WINDOW()


