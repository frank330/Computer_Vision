"""
表情识别Web服务
基于Flask框架提供的表情识别Web API服务
支持图片上传和表情识别功能
"""

import os
import torch
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import sys
import tempfile

# 添加当前目录到系统路径，以便导入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入预测模块
import predict
pre = predict.main  # 获取预测函数


# ==================== Flask应用初始化 ====================
app = Flask(__name__)
CORS(app)  # 启用跨域资源共享，解决跨域问题


def save_upload_file(file, suffix=None):
    """
    保存上传的文件到临时目录
    
    参数:
        file: Flask上传的文件对象
        suffix: 文件后缀名，如果为None则从文件名中提取
    
    返回:
        temp_path: 临时文件路径，如果保存失败返回None
    """
    try:
        # 获取系统临时目录
        temp_dir = tempfile.gettempdir()
        
        # 如果没有指定后缀，从文件名中提取
        if suffix is None:
            suffix = os.path.splitext(file.filename)[1] if file.filename else '.tmp'
        
        # 创建临时文件（使用mkstemp避免文件名冲突）
        temp_fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=temp_dir)
        os.close(temp_fd)  # 关闭文件描述符
        
        # 保存上传的文件到临时路径
        file.save(temp_path)
        return temp_path
    except Exception as e:
        print(f"保存文件时出错: {str(e)}")
        return None


def get_prediction(image_file):
    """
    获取图片预测结果（表情识别）
    
    参数:
        image_file: Flask上传的图片文件对象
    
    返回:
        return_info: 包含预测结果的字典，格式为 {"result": [str, ...]}
    
    功能:
        - 保存上传的图片文件
        - 调用预测函数进行表情识别
        - 处理预测结果并格式化
        - 清理临时文件
    """
    try:
        # 保存上传的文件到临时目录
        image_path = save_upload_file(image_file)
        if image_path is None:
            return {"result": ["文件保存失败"]}

        try:
            # 调用预测函数进行表情识别
            prediction_result = pre(image_path)
            
            # 处理预测结果
            if isinstance(prediction_result, tuple):
                # 如果返回 (label, confidence) 元组
                label = str(prediction_result[0])  # 表情类别名称
                confidence = prediction_result[1] if len(prediction_result) > 1 else 1.0  # 置信度
                
                # 格式化结果：表情名称 + 置信度
                result_str = f"{label}：置信度 {confidence:.4f}"
                return_info = {"result": [result_str]}
            elif isinstance(prediction_result, list):
                # 如果是列表，转换为字符串列表
                result_list = [str(x) for x in prediction_result]
                return_info = {"result": result_list}
            else:
                # 其他情况，直接转换为字符串
                return_info = {"result": [str(prediction_result)]}
        finally:
            # 无论成功与否，都删除临时文件
            try:
                os.remove(image_path)
            except:
                pass

    except Exception as e:
        # 捕获异常并返回错误信息
        print(f"预测时出错: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细的错误堆栈信息
        return_info = {"result": [f"识别失败: {str(e)}"]}

    return return_info


@app.route("/predict", methods=["POST"])
@torch.no_grad()  # 禁用梯度计算，节省内存和加速推理
def predict():
    """
    图片识别API接口
    
    请求方法: POST
    请求参数: file (multipart/form-data格式的图片文件)
    
    返回:
        JSON格式的响应，包含识别结果
        格式: {"result": ["识别结果字符串"]}
    
    功能:
        - 接收上传的图片文件
        - 验证文件类型
        - 调用预测函数进行表情识别
        - 返回识别结果
    """
    # 检查请求中是否包含文件
    if "file" not in request.files:
        return jsonify({"result": ["没有上传文件"]})

    # 获取上传的文件
    image = request.files["file"]
    
    # 检查是否选择了文件
    if image.filename == '':
        return jsonify({"result": ["未选择文件"]})

    # 检查文件类型
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}  # 允许的文件扩展名
    file_ext = os.path.splitext(image.filename)[1].lower()  # 获取文件扩展名（转换为小写）
    
    if file_ext not in allowed_extensions:
        return jsonify({"result": ["只支持 JPG、JPEG、PNG 和 BMP 格式的图片"]})

    # 调用预测函数获取识别结果
    info = get_prediction(image)
    
    # 确保最终返回始终为 {"result": [str, ...]} 格式
    result = info.get("result") if isinstance(info, dict) else None
    if not isinstance(result, list):
        result = [str(result)] if result is not None else ["未知错误"]
    
    # 确保所有结果都是字符串类型
    result = [str(x) for x in result]
    
    # 返回JSON格式的响应
    return jsonify({"result": result})


@app.route("/", methods=["GET", "POST"])
def root():
    """
    主页路由
    
    返回:
        up.html模板渲染的页面
    
    功能:
        - 渲染表情识别系统的前端页面
    """
    return render_template("up.html")


# ==================== 主函数 ====================
if __name__ == '__main__':
    # 确保临时文件目录存在
    os.makedirs(tempfile.gettempdir(), exist_ok=True)
    
    # 启动Flask Web服务器
    # host="0.0.0.0" 表示监听所有网络接口
    # port=5000 指定端口号
    app.run(host="0.0.0.0", port=5000)




