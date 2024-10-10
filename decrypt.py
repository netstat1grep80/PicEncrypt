import argparse
from argparse import RawTextHelpFormatter
import os
import hashlib
from PIL import Image
import numpy as np


def modify_filename(filename):
    basename = os.path.basename(filename)
    name, ext = os.path.splitext(basename)
    new_filename = f"{name}_output{ext}"
    return new_filename


def produce_logistic_sort(a):
    return a[0]


def produce_logistic(x1, n):
    l = np.zeros((n, 2))
    x = x1
    l[0] = [x, 0]
    for i in range(1, n):
        x = 3.9999999 * x * (1 - x)
        l[i] = [x, i]
    return l


def amess(arrlength, ast):
    arr = list(range(arrlength))
    for i in range(arrlength - 1, 0, -1):
        hash_value = hashlib.md5((ast + str(i)).encode()).hexdigest()[:7]
        rand = int(hash_value, 16) % (i + 1)

        arr[i], arr[rand] = arr[rand], arr[i]
    return arr


def decrypt_B2(img_path, key, sx, sy):
    img = Image.open(img_path)
    img = img.convert("RGBA")
    wid, hit = img.size
    wid1, hit1 = wid, hit

    while wid % sx > 0:
        wid += 1
    while hit % sy > 0:
        hit += 1

    ssx = wid // sx
    ssy = hit // sy

    cv = Image.new("RGBA", (wid, hit))
    cvd = np.array(cv)

    imgdata = np.array(img)
    oimgdata = np.zeros((hit, wid, 4), dtype=np.uint8)

    cvd[0:hit1, 0:wid1] = imgdata
    cvd[hit1:hit, 0:wid1] = imgdata
    cvd[0:hit1, wid1:wid] = imgdata
    cvd[hit1:hit, wid1:wid] = imgdata

    xl = amess(sx, key)
    yl = amess(sy, key)

    for i in range(wid):
        for j in range(hit):
            m = i
            n = j
            m = (xl[((n // ssy) % sx)] * ssx + m) % wid
            m = xl[(m // ssx)] * ssx + m % ssx
            n = (yl[((m // ssx) % sy)] * ssy + n) % hit
            n = yl[(n // ssy)] * ssy + n % ssy

            oimgdata[n, m] = cvd[j, i]

    output_img = Image.fromarray(oimgdata, "RGBA")
    return output_img, wid, hit


def decrypt_C(img_path, key):
    img1 = Image.open(img_path).convert("RGBA")

    wid, hit = img1.size

    oimgdata = Image.new("RGBA", (wid, hit))

    xl = amess(wid, key)
    yl = amess(hit, key)

    imgdata = np.array(img1)

    for i in range(wid):
        for j in range(hit):
            m = i
            n = j

            m = (xl[n % wid] + m) % wid
            m = xl[m]
            n = (yl[m % hit] + n) % hit
            n = yl[n]

            if 0 <= i < wid and 0 <= j < hit:
                oimgdata.putpixel((m, n), tuple(imgdata[j, i]))

    return oimgdata


def decrypt_C2(img_path, key):
    img = Image.open(img_path)
    img = img.convert("RGBA")
    wid, hit = img.size
    imgdata = np.array(img)

    oimgdata = np.zeros_like(imgdata)

    xl = amess(wid, key)
    yl = amess(hit, key)

    for i in range(wid):
        for j in range(hit):
            m = i
            n = j
            m = (xl[n % wid] + m) % wid
            m = xl[m]

            oimgdata[j, m] = imgdata[j, i]

    output_img = Image.fromarray(oimgdata, "RGBA")
    return output_img


def decrypt_PE1(img_path, key):
    img = Image.open(img_path)
    img = img.convert("RGBA")
    wid, hit = img.size
    imgdata = np.array(img)

    oimgdata = np.zeros_like(imgdata)

    arrayaddress = produce_logistic(key, wid)
    arrayaddress = sorted(arrayaddress, key=produce_logistic_sort)
    arrayaddress = [int(x[1]) for x in arrayaddress]

    for i in range(wid):
        for j in range(hit):
            m = arrayaddress[i]
            oimgdata[j, m] = imgdata[j, i]

    output_img = Image.fromarray(oimgdata, "RGBA")

    return output_img


def decrypt_PE2(img_path, key):
    img = Image.open(img_path)
    img = img.convert("RGBA")
    wid, hit = img.size
    imgdata = np.array(img)

    oimgdata = np.zeros_like(imgdata)
    o2imgdata = np.zeros_like(imgdata)

    x = key

    for i in range(wid):
        arrayaddress = produce_logistic(x, hit)
        x = arrayaddress[hit - 1][0]
        arrayaddress = sorted(arrayaddress, key=produce_logistic_sort)
        arrayaddress = [int(a[1]) for a in arrayaddress]
        for j in range(hit):
            n = arrayaddress[j]
            oimgdata[n, i] = imgdata[j, i]

    x = key

    for j in range(hit):
        arrayaddress = produce_logistic(x, wid)
        x = arrayaddress[wid - 1][0]
        arrayaddress = sorted(arrayaddress, key=produce_logistic_sort)
        arrayaddress = [int(a[1]) for a in arrayaddress]
        for i in range(wid):
            m = arrayaddress[i]
            o2imgdata[j, m] = oimgdata[j, i]

    output_img = Image.fromarray(o2imgdata, "RGBA")

    return output_img


def generate2d(x, y, ax, ay, bx, by, coordinates):
    w = abs(ax + ay)
    h = abs(bx + by)

    dax = (ax > 0) - (ax < 0)
    day = (ay > 0) - (ay < 0)
    dbx = (bx > 0) - (bx < 0)
    dby = (by > 0) - (by < 0)

    if h == 1:

        for i in range(w):
            coordinates.append((x, y))
            x += dax
            y += day
        return

    if w == 1:

        for i in range(h):
            coordinates.append((x, y))
            x += dbx
            y += dby
        return

    ax2 = ax // 2
    ay2 = ay // 2
    bx2 = bx // 2
    by2 = by // 2

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2 * w > 3 * h:
        if (w2 % 2) and (w > 2):
            ax2 += dax
            ay2 += day

        generate2d(x, y, ax2, ay2, bx, by, coordinates)
        generate2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by, coordinates)

    else:
        if (h2 % 2) and (h > 2):
            bx2 += dbx
            by2 += dby

        generate2d(x, y, bx2, by2, ax2, ay2, coordinates)
        generate2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2, coordinates)
        generate2d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            -bx2, -by2, -(ax - ax2), -(ay - ay2), coordinates
        )


def gilbert2d(width, height):
    coordinates = []

    if width >= height:
        generate2d(0, 0, width, 0, 0, height, coordinates)
    else:
        generate2d(0, 0, 0, height, width, 0, coordinates)

    return coordinates


def decrypt_tomato(img_path):
    img = Image.open(img_path)
    width, height = img.size
    imgdata = np.array(img)
    imgdata2 = np.zeros_like(imgdata)

    curve = gilbert2d(width, height)
    offset = round((5 ** 0.5 - 1) / 2 * width * height)

    for i in range(width * height):
        old_pos = curve[i]
        new_pos = curve[(i + offset) % (width * height)]
        old_p = (old_pos[0] + old_pos[1] * width) * 4
        new_p = (new_pos[0] + new_pos[1] * width) * 4

        imgdata2[old_pos[1], old_pos[0], :] = imgdata[new_pos[1], new_pos[0], :]

    output_image = Image.fromarray(imgdata2)
    return output_image


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter, description="This is a script to decrypt images. This project is inspired by and references code from https://github.com/picencrypt/PicEncrypt  .\n\nExample : python decrypt.py --src /home/root/abc.jpg --dst /tmp/abc_decrypt.jpg --type pe2")

    parser.add_argument(
        '-s',
        '--src',
        type=str,
        default='',
        required=True,
        help="The file path of the source image to be decrypted."
    )
    parser.add_argument(
        '-d',
        '--dst',
        type=str,
        default='',
        help="The file path where the decrypted image will be saved."
    )

    parser.add_argument(
        '-t',
        '--type',
        type=str,
        required=True,
        help="b,c2,c,pe1,pe2,t\n\nDescription:\nb => 方块混淆 (Block Confusion)\nc2 => 行像素混淆 (Row Pixels Confusion)\nc => 像素混淆 (Pixels Confusion)\npe1 => 行模式(vertical)\npe2 => 行+列模式(disordered)\nt => Tomato(小番茄)\n\n"
    )

    parser.add_argument(
        '-k',
        '--key',
        type=float,
        default=0.666,
        help='default 0.666'
    )

    args = parser.parse_args()
    try:
        if args.dst == "":
            args.dst = "./{}".format(modify_filename(args.src))
        # 方块
        if args.type == 'b':
            output_img = decrypt_PE2(args.src, key=args.key)
        # 行像素
        elif args.type == 'c2':
            output_img = decrypt_C2(args.src, key=str(args.key))
        # 像素
        elif args.type == 'c':
            output_img = decrypt_C(args.src, key='')
        # 兼容PicEncrypt: 行模式
        elif args.type == 'pe1':
            output_img = decrypt_PE1(args.src, key=args.key)
        # 兼容PicEncrypt: 行+列模式
        elif args.type == 'pe2':
            output_img = decrypt_PE2(args.src, key=args.key)
        # 小番茄
        elif args.type == 't':
            output_img = decrypt_tomato(args.src)

        if args.type == 't':
            output_img.save(args.dst, "JPEG", quality=95)
        else:
            output_img = output_img.convert("RGB")  # 转换为 RGB 模式
            output_img.save(args.dst)

    except ValueError as e:
        print(e)