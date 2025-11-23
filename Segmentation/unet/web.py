"""
基于U-Net的肠道息肉检测Web服务
提供图像上传、分割预测和结果展示功能
"""
import os
import io
import json
import torch
from flask import Flask, jsonify, request, render_template, send_file
from flask_cors import CORS
from predict import main as predict_image
import tempfile

app = Flask(__name__)
CORS(app)  # 解决跨域问题

# 结果输出目录
OUTPUT_DIR = "./templates"


def save_upload_file(file):
    """
    保存上传的文件并返回文件路径
    
    Args:
        file: Flask上传的文件对象
        
    Returns:
        str: 保存的文件路径，失败返回None
    """
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
    """
    获取图像的分割预测结果
    
    Args:
        image_file: Flask上传的文件对象
        
    Returns:
        dict: 包含预测结果信息的字典
    """
    try:
        # 保存上传的文件
        image_path = save_upload_file(image_file)
        if image_path is None:
            return {"success": False, "message": "文件保存失败"}

        try:
            # 调用预测函数，结果保存到templates目录
            result = predict_image(image_path, output_dir=OUTPUT_DIR)
            
            # 返回成功信息和图像路径
            return {
                "success": True,
                "message": f"检测完成，推理时间: {result['inference_time']:.3f}秒",
                "original": "original.jpg",
                "mask": "mask.jpg",
                "overlay": "overlay.jpg",
                "inference_time": result['inference_time']
            }
                
        except Exception as e:
            return {"success": False, "message": f"检测过程出错: {str(e)}"}
        finally:
            # 删除临时文件
            try:
                os.remove(image_path)
            except:
                pass
                
    except Exception as e:
        return {"success": False, "message": f"处理文件时出错: {str(e)}"}


@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    """
    预测接口：接收上传的图像，进行分割预测
    
    Returns:
        JSON: 包含预测结果的JSON响应
    """
    if "file" not in request.files:
        return jsonify({"success": False, "message": "没有上传文件"})
    
    image = request.files["file"]
    if image.filename == '':
        return jsonify({"success": False, "message": "未选择文件"})
    
    # 检查文件类型
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    file_ext = os.path.splitext(image.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({"success": False, "message": "只支持 JPG、JPEG、PNG 和 BMP 格式的图片"})

    info = get_prediction(image)
    return jsonify(info)


@app.route("/original.jpg")
def get_original_image():
    """返回原始图像"""
    try:
        return send_file(os.path.join(OUTPUT_DIR, "original.jpg"), mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/mask.jpg")
def get_mask_image():
    """返回分割mask图像"""
    try:
        return send_file(os.path.join(OUTPUT_DIR, "mask.jpg"), mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/overlay.jpg")
def get_overlay_image():
    """返回叠加图像（原图+分割结果）"""
    try:
        return send_file(os.path.join(OUTPUT_DIR, "overlay.jpg"), mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/", methods=["GET", "POST"])
def root():
    """主页面路由"""
    return render_template("index.html")


if __name__ == '__main__':
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 确保临时文件目录存在
    os.makedirs(tempfile.gettempdir(), exist_ok=True)
    app.run(host="0.0.0.0", port=5000)




