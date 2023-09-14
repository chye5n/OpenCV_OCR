import cv2
from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    jsonify,
)
import numpy as np
from googletrans import Translator
import pytesseract
import os

app = Flask(__name__)
translator = Translator()

UPLOAD_FOLDER = "project_copy5/static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 'uploads' 디렉토리가 존재하지 않으면 생성
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def process_image(image_path):
    # 이미지 파일을 OpenCV로 읽어옴
    path = "C:/Users/zkdlz/OneDrive/Desktop/OpenCV/project_copy5"
    image = cv2.imread(path + image_path)

    # 이미지를 그레이스케일로 변환
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 그레이스케일 이미지를 문자열로 변환
    text = pytesseract.image_to_string(image, lang="eng")
    new_str = text.replace("\n", " ")

    detected = translator.detect(new_str)
    src_lang = detected.lang
    dest_lang = None
    if src_lang == "en":
        dest_lang = "ko"
    elif src_lang == "ko":
        dest_lang = "en"

    # 번역
    translated = translator.translate(new_str, dest_lang, src_lang)
    trans = translated.text
    return new_str, trans


# 이미지 선택 및 변환 관련 코드
def drawROI(img, corners):
    cpy = img.copy()

    c1 = (192, 192, 255)
    c2 = (128, 128, 255)

    for pt in corners:
        cv2.circle(cpy, tuple(pt.astype(int)), 25, c1, -1, cv2.LINE_AA)

    cv2.line(
        cpy,
        tuple(corners[0].astype(int)),
        tuple(corners[1].astype(int)),
        c2,
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        cpy,
        tuple(corners[1].astype(int)),
        tuple(corners[2].astype(int)),
        c2,
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        cpy,
        tuple(corners[2].astype(int)),
        tuple(corners[3].astype(int)),
        c2,
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        cpy,
        tuple(corners[3].astype(int)),
        tuple(corners[0].astype(int)),
        c2,
        2,
        cv2.LINE_AA,
    )

    disp = cv2.addWeighted(img, 0.3, cpy, 0.7, 0)

    return disp


def onMouse(event, x, y, flags, param):
    global srcQuad, dragSrc, ptOld, src

    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(4):
            if cv2.norm(srcQuad[i] - (x, y)) < 25:
                dragSrc[i] = True
                ptOld = (x, y)
                break

    if event == cv2.EVENT_LBUTTONUP:
        for i in range(4):
            dragSrc[i] = False

    if event == cv2.EVENT_MOUSEMOVE:
        for i in range(4):
            if dragSrc[i]:
                dx = x - ptOld[0]
                dy = y - ptOld[1]

                srcQuad[i] += (dx, dy)

                cpy = drawROI(src, srcQuad)
                cv2.imshow("img", cpy)
                ptOld = (x, y)
                break


@app.route("/")
def index():
    result = {"new_str": "", "trans": ""}
    return render_template("index.html", image_url="", result=result)


@app.route("/upload_image", methods=["POST"])
def upload_image():
    # 업로드된 파일을 가져옴
    uploaded_file = request.files["image_upload"]
    global img_url
    # 파일이 업로드되었다면
    if uploaded_file.filename != "":
        # 업로드된 파일을 저장할 경로 설정
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)

        # 파일 저장
        uploaded_file.save(file_path)

        # 업로드한 이미지 표시
        image_url = f"/static/uploads/{uploaded_file.filename}"
        img_url = file_path
        print(img_url)
        # new_str, trans = process_image(file_path)
        new_str = ""
        trans = ""
    else:
        image_url = ""
        new_str = ""
        trans = ""

    result = {"new_str": new_str, "trans": trans}
    return render_template("index.html", image_url=image_url, result=result)


@app.route("/uploads/<filename>")
def get_uploaded_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/process_image_endpoint", methods=["POST"])
def process_image_endpoint():
    print(trans_url)
    if trans_url != "":
        try:
            new_str, trans = process_image(trans_url)
        except cv2.error as e:
            # 이미지 읽기 오류가 발생한 경우
            print(f"Error reading image: {e}")
            new_str = ""
            trans = ""
    else:
        new_str = ""
        trans = ""

    result = {"new_str": new_str, "trans": trans}
    return jsonify(result)  # JSON 형식으로 응답 반환s


@app.route("/select_roi", methods=["POST"])
def select_roi():
    global src, srcQuad, dstQuad, dragSrc, ptOld, h, w

    h, w = None, None

    # 업로드한 이미지를 읽어옵니다 (파일 경로를 수정할 수 있습니다).
    image_path = "C:/Users/zkdlz/OneDrive/Desktop/OpenCV/" + img_url
    src = cv2.imread(image_path)
    print(img_url)
    if src is None:
        return "이미지를 열 수 없습니다!"

    # ROI와 관련된 변수 초기화
    h, w = src.shape[:2]
    dw = round(300 * 297 / 210)
    dh = 300  # A4 용지 크기: 210x297cm

    # 모서리 점들의 좌표, 드래그 상태 여부
    srcQuad = np.array(
        [[30, 30], [30, h - 30], [w - 30, h - 30], [w - 30, 30]], np.float32
    )
    dstQuad = np.array([[0, 0], [0, dh - 1], [dw - 1, dh - 1], [dw - 1, 0]], np.float32)
    dragSrc = [False, False, False, False]
    ptOld = (0, 0)
    # ROIs가 있는 디스플레이 이미지를 생성합니다.
    disp = drawROI(src, srcQuad)

    cv2.imshow("img", disp)
    cv2.setMouseCallback("img", onMouse)

    while True:
        key = cv2.waitKey()
        if key == 13:  # 엔터 키
            break
        elif key == 27:  # ESC 키
            cv2.destroyWindow("img")
            break

    # 투시 변환을 수행합니다.
    pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
    dst = cv2.warpPerspective(src, pers, (dw, dh), flags=cv2.INTER_CUBIC)

    # 이미지 파일 이름을 동적으로 생성하여 저장합니다.
    transformed_image_filename = "transformed_image.jpg"  # 파일 이름을 원하는대로 수정하세요
    transformed_image_path = os.path.join(
        app.config["UPLOAD_FOLDER"], transformed_image_filename
    )
    cv2.imwrite(transformed_image_path, dst)
    global trans_url
    # 변환된 이미지를 화면에 표시합니다.
    transformed_image_url = f"/static/uploads/{transformed_image_filename}"
    trans_url = transformed_image_url
    result = {"new_str": "", "trans": ""}
    return render_template("index.html", image_url=transformed_image_url, result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, threaded=True)
