# -*- coding: utf-8 -*-
"""
基于UCTransNet的图像分割Web服务
提供图像上传、分割预测和结果展示功能

该服务使用UCTransNet模型进行图像分割，支持多种医学图像分割任务。
"""
import os
import io
import json
import torch
from flask import Flask, jsonify, request, render_template, send_file
from flask_cors import CORS
from predict import predict_single_image as predict_image
import tempfile

app = Flask(__name__)
CORS(app)  # 解决跨域问题

# 结果输出目录（用于Web服务展示）
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
            

            original_filename = os.path.basename(result['original'])
            mask_filename = os.path.basename(result['mask'])
            overlay_filename = os.path.basename(result['overlay'])
            
            # 返回成功信息和图像路径（使用/image/前缀用于路由）
            return {
                "success": True,
                "message": f"检测完成，推理时间: {result['inference_time']:.3f}秒",
                "original": f"/image/{original_filename}",
                "mask": f"/image/{mask_filename}",
                "overlay": f"/image/{overlay_filename}",
                "inference_time": result['inference_time']
            }
                
        except Exception as e:
            import traceback
            error_msg = f"检测过程出错: {str(e)}"
            print(f"预测错误: {error_msg}")
            print(traceback.format_exc())
            return {"success": False, "message": error_msg}
        finally:
            # 删除临时文件
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception as e:
                print(f"删除临时文件失败: {str(e)}")
                
    except Exception as e:
        import traceback
        error_msg = f"处理文件时出错: {str(e)}"
        print(f"文件处理错误: {error_msg}")
        print(traceback.format_exc())
        return {"success": False, "message": error_msg}


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


@app.route("/image/<filename>")
def get_result_image(filename):
    """
    返回预测结果图像（原图、mask或叠加图）
    
    Args:
        filename: 图像文件名（如 test_original.jpg, test_mask.jpg, test_overlay.jpg）
    
    Returns:
        图像文件或错误响应
    """
    try:
        # 安全检查：只允许访问输出目录中的文件
        file_path = os.path.join(OUTPUT_DIR, filename)
        
        # 确保文件在输出目录内（防止路径遍历攻击）
        if not os.path.abspath(file_path).startswith(os.path.abspath(OUTPUT_DIR)):
            return jsonify({"error": "非法文件路径"}), 403
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return jsonify({"error": f"文件不存在: {filename}"}), 404
        
        # 根据文件扩展名确定MIME类型
        ext = os.path.splitext(filename)[1].lower()
        mimetype_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.bmp': 'image/bmp'
        }
        mimetype = mimetype_map.get(ext, 'image/jpeg')
        
        return send_file(file_path, mimetype=mimetype)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/", methods=["GET", "POST"])
def root():
    """主页面路由"""
    return render_template("index.html")


if __name__ == '__main__':
    """
    启动Web服务
    
    使用说明：
        1. 确保已训练好UCTransNet模型
        2. 模型路径配置在Config.py中的test_session或自动查找
        3. 访问 http://localhost:5000 使用Web界面
    """
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 确保临时文件目录存在
    os.makedirs(tempfile.gettempdir(), exist_ok=True)
    
    print("=" * 50)
    print("UCTransNet 图像分割 Web 服务")
    print("=" * 50)
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 50)

    # 启动Flask应用
    app.run(host="0.0.0.0", port=5000, debug=True)




