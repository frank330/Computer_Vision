import os
import torch
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from urt.predict import main as pre

app = Flask(__name__)
CORS(app)  # 解决跨域问题


def save_upload_file(file):
    """保存上传的文件并返回文件路径"""
    try:
        # 创建临时文件
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file.filename)

        # 保存文件
        file.save(temp_path)
        return temp_path
    except Exception as e:
        print(f"保存文件时出错: {str(e)}")
        return None


def get_prediction(image_file):
    try:
        # 保存上传的文件
        image_path = save_upload_file(image_file)
        if image_path is None:
            return {"result": ["文件保存失败"]}

        try:
            # 调用预测函数
            text = pre(image_path)
            # 统一输出为字符串列表
            if isinstance(text, (list, tuple)):
                result_list = [str(x) for x in text]
            else:
                result_list = [str(text)]
            return_info = {"result": result_list}
        finally:
            # 删除临时文件
            try:
                os.remove(image_path)
            except:
                pass

    except Exception as e:
        return_info = {"result": [str(e)]}

    return return_info


@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    if "file" not in request.files:
        return jsonify({"result": ["没有上传文件"]})

    image = request.files["file"]
    if image.filename == '':
        return jsonify({"result": ["未选择文件"]})

    # 检查文件类型
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    file_ext = os.path.splitext(image.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({"result": ["只支持 JPG、JPEG、PNG 和 BMP 格式的图片"]})

    info = get_prediction(image)
    # 确保最终返回始终为 {"result": [str, ...]}
    result = info.get("result") if isinstance(info, dict) else None
    if not isinstance(result, list):
        result = [str(result)] if result is not None else ["未知错误"]
    result = [str(x) for x in result]
    return jsonify({"result": result})


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("up.html")


if __name__ == '__main__':
    # 确保临时文件目录存在
    os.makedirs(tempfile.gettempdir(), exist_ok=True)
    app.run(host="0.0.0.0", port=5000)




