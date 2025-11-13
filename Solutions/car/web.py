import json
import logging
import math
import os
import uuid
from datetime import datetime
from typing import List, Optional

from flask import Flask, jsonify, render_template, request, url_for

from Car_recognition import detect_Recognition_plate11

"""
轻量级车牌识别后台服务
=======================
 
该模块提供进/出库的图片识别接口，核心流程：
1. 读取上传图片并交给 `detect_Recognition_plate11` 识别车牌；
2. 进库：创建一条本地记录，标记车辆已在库内；
3. 出库：匹配当天尚未结算的进库记录，计算停留时长与费用；
4. 所有记录保存至 `records.jsonl`，图片保存在 `static/captures/`。
"""


# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 常量配置
# --------------------------------------------------------------------------- #
# 所有抓拍/上传的图片都落在 `static/captures/`，便于前端直接访问。
CAPTURE_DIR = os.path.join("static", "captures")
# 所有识别记录写入 JSON Lines 文件，避免数据库依赖。
RECORD_FILE = "records.jsonl"
# 计费单位：30 分钟为一个计费块；不足一个块也按一个块收费。
BILLING_UNIT_MINUTES = 30
# 单价（元/每 30 分钟）
UNIT_PRICE = 1


def ensure_storage():
    """初始化本地存储目录与记录文件。"""
    # 图片目录不存在则创建（多次调用也安全）。
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    # 若记录文件不存在，创建一个空文件，便于后续追加写入。
    if not os.path.exists(RECORD_FILE):
        with open(RECORD_FILE, "w", encoding="utf-8") as file:
            file.write("")


ensure_storage()

def process_image(image_path):
    """
    处理图片并识别车牌
    """
    try:
        logger.info(f"开始处理图片: {image_path}")

        # Step 1: 检查文件是否存在，避免识别器抛异常。
        if not os.path.exists(image_path):
            logger.error("文件不存在")
            return {
                'number': '文件不存在',
                'color': '识别失败'
            }

        # Step 2: 调用外部核心识别函数，得到原始结果。
        logger.info("开始调用车牌识别函数")
        result = detect_Recognition_plate11(image_path)
        logger.info(f"识别结果: {result}")

        if result:
            try:
                parts = result.split()
                plate_number = parts[0]
                plate_color = parts[1] if len(parts) > 1 else "未知"

                logger.info(f"识别成功 - 车牌号: {plate_number}, 颜色: {plate_color}")
                # Step 3: 返回结构化车牌信息。
                return {
                    "number": plate_number,
                    "color": plate_color,
                }
            except (IndexError, AttributeError) as error:
                logger.error(f"结果格式解析错误: {str(error)}")
                # 识别函数若返回格式异常则提示用户重新上传。
                return {
                    "number": "结果格式错误",
                    "color": "识别失败",
                }
        else:
            logger.warning("未能识别到车牌")
            # 未识别到车牌视作失败，由上层决定是否保留图片。
            return {
                'number': '未识别到车牌',
                'color': '未知'
            }

    except Exception as e:
        logger.error(f"处理过程中出现错误: {str(e)}", exc_info=True)
        # 兜底：识别过程发生异常时返回统一错误信息。
        return {
            'number': '识别过程出错',
            'color': '识别失败'
        }


def is_successful_result(result: dict) -> bool:
    """判断识别结果是否成功。"""
    number = result.get("number", "")
    if not number:
        return False
    # 统一使用关键字判定失败，方便后续扩展。
    keywords = ["未识别", "错误", "失败", "不存在"]
    return not any(keyword in number for keyword in keywords)


def build_record(
    result: dict, camera_type: str, image_path: str, extra: Optional[dict] = None
) -> dict:
    """根据识别结果构建记录对象。"""
    timestamp = datetime.now()
    # 统一的基础字段，保证列表展示时信息齐全。
    record = {
        "id": uuid.uuid4().hex,
        "camera_type": camera_type,
        "plate_number": result.get("number", ""),
        "plate_color": result.get("color", ""),
        "image_path": image_path.replace("\\", "/"),
        "timestamp": timestamp.isoformat(),
        "display_time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if camera_type == "entry":
        record["entry_time"] = record["display_time"]
        record["status"] = "open"

    if extra:
        record.update(extra)

    return record


def save_record(record: dict) -> None:
    """将识别记录追加保存到本地文件。"""
    # 注意：使用 JSON Lines 追加模式，天然支持 append。
    with open(RECORD_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_records(plate_number: Optional[str] = None) -> List[dict]:
    """从本地文件读取识别记录，可按车牌号过滤。"""
    records = []
    if not os.path.exists(RECORD_FILE):
        return records

    with open(RECORD_FILE, "r", encoding="utf-8") as file:
        for line in file:
            data = line.strip()
            if not data:
                continue
            try:
                record = json.loads(data)
                # 根据需要过滤车牌号；若未指定则返回全部。
                if plate_number:
                    if record.get("plate_number") == plate_number:
                        records.append(record)
                else:
                    records.append(record)
            except json.JSONDecodeError as error:
                logger.warning(f"跳过无法解析的记录: {error}")

    records.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
    return records


def update_record(record_id: str, updates: dict) -> Optional[dict]:
    """根据 ID 更新单条记录。"""
    if not os.path.exists(RECORD_FILE):
        return None

    updated_record = None
    records: List[dict] = []

    with open(RECORD_FILE, "r", encoding="utf-8") as file:
        for line in file:
            data = line.strip()
            if not data:
                continue
            try:
                record = json.loads(data)
            except json.JSONDecodeError:
                continue

            if record.get("id") == record_id and updated_record is None:
                # 找到目标记录后更新字段，但仍需写回整个文件。
                record.update(updates)
                updated_record = record

            records.append(record)

    if updated_record is not None:
        with open(RECORD_FILE, "w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return updated_record


def find_open_entry(plate_number: str) -> Optional[dict]:
    """查找尚未匹配出库记录的最新进库记录。"""
    if not plate_number:
        return None

    for record in load_records():
        if (
            record.get("plate_number") == plate_number
            and record.get("camera_type") == "entry"
            and not record.get("exit_time")
        ):
            # 返回第一条仍处于 open 状态的进库记录。
            return record
    return None


def calculate_fee(entry_time: datetime, exit_time: datetime) -> dict:
    """根据停留时间计算费用。"""
    total_minutes = max(
        1, math.ceil((exit_time - entry_time).total_seconds() / 60)
    )
    # 根据总分钟数折算出需要计费的 30 分钟块数。
    billable_blocks = max(1, math.ceil(total_minutes / BILLING_UNIT_MINUTES))
    fee = billable_blocks * UNIT_PRICE
    return {
        "duration_minutes": total_minutes,
        "billable_blocks": billable_blocks,
        "fee": fee,
    }


def delete_record_by_id(record_id: str) -> bool:
    """删除指定 ID 的识别记录。"""
    if not os.path.exists(RECORD_FILE):
        return False

    records: List[dict] = []
    deleted_record: Optional[dict] = None

    with open(RECORD_FILE, "r", encoding="utf-8") as file:
        for line in file:
            data = line.strip()
            if not data:
                continue
            try:
                record = json.loads(data)
            except json.JSONDecodeError:
                continue

            if record.get("id") == record_id and deleted_record is None:
                deleted_record = record
                continue

            records.append(record)

    if deleted_record is None:
        return False

    entry_to_reopen = None
    if (
        deleted_record.get("camera_type") == "exit"
        and deleted_record.get("entry_record_id")
    ):
        entry_to_reopen = deleted_record["entry_record_id"]

    if entry_to_reopen:
        for record in records:
            if record.get("id") == entry_to_reopen:
                # 删除出库记录时，撤销关联进库记录的结算字段。
                for key in (
                    "exit_time",
                    "exit_record_id",
                    "fee",
                    "status",
                    "duration_minutes",
                    "billable_blocks",
                ):
                    record.pop(key, None)
                record["status"] = "open"
                break

    with open(RECORD_FILE, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return True

@app.route("/", methods=["GET"])
def index():
    """渲染前端单页应用入口。"""
    return render_template("index.html")

def make_record_response(record: dict) -> dict:
    """补充可直接用于前端展示的字段。"""
    record_response = dict(record)
    image_path = record.get("image_path")
    if image_path:
        # 将图片相对路径转换成可直接访问的静态资源 URL。
        record_response["image_url"] = url_for("static", filename=image_path)
    record_response["recognition_time"] = record.get(
        "display_time", record.get("timestamp")
    )
    if record_response.get("camera_type") == "entry":
        record_response.setdefault("entry_time", record_response["recognition_time"])
    if record_response.get("camera_type") == "exit":
        record_response.setdefault("exit_time", record_response["recognition_time"])
    return record_response


@app.route("/predict", methods=["POST"])
def predict():
    """
    处理进/出库识别请求。

    前端通过 multipart/form-data 上传图片并指定 camera_type：
    - entry：写入一条新的进库记录；
    - exit：匹配当天未结算的进库记录，计算费用并返回结算信息。
    """
    try:
        logger.info("收到图片上传识别请求")
        # 1. 校验识别类型，限制在 entry/exit 两种。
        camera_type = request.form.get("camera_type", "").strip().lower()
        if camera_type not in {"entry", "exit"}:
            logger.error(f"识别类型无效: {camera_type}")
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "识别类型无效，仅支持进库或出库识别",
                        "result": {
                            "number": "识别类型无效",
                            "color": "识别失败",
                        },
                    }
                ),
                400,
            )

        # 2. 校验文件是否上传。
        if "file" not in request.files:
            logger.error("没有文件上传")
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "没有文件上传",
                        "result": {"number": "没有文件上传", "color": "识别失败"},
                    }
                ),
                400,
            )

        file = request.files["file"]
        if file.filename == "":
            # 客户端可能选择后又取消，这里提示重新选择。
            logger.error("没有选择文件")
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "没有选择文件",
                        "result": {"number": "没有选择文件", "color": "识别失败"},
                    }
                ),
                400,
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{camera_type}_{timestamp}.jpg"
        # 3. 将上传图片落盘，供识别和后续回溯使用。
        file_path = os.path.join(CAPTURE_DIR, filename)
        file.save(file_path)
        logger.info(f"文件已保存到: {file_path}")

        # 4. 调用识别算法，得到车牌号与颜色。
        result = process_image(file_path)
        logger.info(f"处理结果: {result}")

        # 识别失败时删除临时文件，避免堆积垃圾数据。
        if not is_successful_result(result):
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify(
                {
                    "status": "error",
                    "message": "未识别到有效车牌",
                    "result": result,
                }
            )

        plate_number = result.get("number", "")

        if camera_type == "entry":
            # 5a. 进库：直接创建一条 open 状态的记录。
            record = build_record(result, "entry", f"captures/{filename}")
            save_record(record)
            return jsonify(
                {
                    "status": "success",
                    "result": result,
                    "record": make_record_response(record),
                }
            )

        entry_record = find_open_entry(plate_number)
        if not entry_record:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify(
                {
                    "status": "error",
                    "message": "未找到当天的进库记录，请联系工作人员处理",
                    "result": result,
                }
            )

        try:
            entry_time_iso = entry_record.get("timestamp") or ""
            entry_time = (
                datetime.fromisoformat(entry_time_iso)
                if entry_time_iso
                else datetime.strptime(
                    entry_record.get("entry_time"), "%Y-%m-%d %H:%M:%S"
                )
            )
        except Exception:
            entry_time = datetime.now()

        exit_time = datetime.now()

        # 出库只接受“同一天”内的进库记录，跨天需人工处理。
        if entry_time.date() != exit_time.date():
            logger.warning(
                "跨天出库，需人工处理: plate=%s entry=%s exit=%s",
                plate_number,
                entry_time,
                exit_time,
            )
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify(
                {
                    "status": "error",
                    "message": "该车辆进出时间已跨天，请联系工作人员处理",
                    "result": {
                        "number": plate_number,
                        "color": result.get("color", ""),
                    },
                }
            )

        # 6. 计算停留时长与应缴费用。
        fee_info = calculate_fee(entry_time, exit_time)
        entry_time_display = entry_record.get(
            "entry_time", entry_record.get("display_time")
        )

        record = build_record(
            result,
            "exit",
            f"captures/{filename}",
            extra={
                "entry_time": entry_time_display,
                "exit_time": exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_minutes": fee_info["duration_minutes"],
                "billable_blocks": fee_info["billable_blocks"],
                "fee": fee_info["fee"],
                "entry_record_id": entry_record.get("id"),
            },
        )

        save_record(record)
        update_record(
            entry_record["id"],
            {
                # 将进库记录标记为 closed，写入出库结算数据。
                "exit_time": record.get("exit_time", record.get("display_time")),
                "exit_record_id": record["id"],
                "fee": fee_info["fee"],
                "status": "closed",
                "duration_minutes": fee_info["duration_minutes"],
                "billable_blocks": fee_info["billable_blocks"],
            },
        )

        return jsonify(
            {
                "status": "success",
                "result": result,
                "record": make_record_response(record),
            }
        )

    except Exception as error:
        logger.error(f"预测过程中出现错误: {str(error)}", exc_info=True)
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "服务器处理失败",
                    "result": {"number": "服务器处理失败", "color": "识别失败"},
                }
            ),
            500,
        )


@app.route("/history/<plate_number>", methods=["GET"])
def get_history(plate_number: str):
    """按车牌号返回历史识别记录，默认倒序排列。"""
    try:
        plate_number = plate_number.strip()
        logger.info(f"查询历史记录: plate_number={plate_number}")

        records = load_records(plate_number if plate_number else None)
        response_records = [make_record_response(record) for record in records]

        return jsonify({"status": "success", "records": response_records})

    except Exception as error:
        logger.error(f"查询历史记录时出错: {str(error)}")
        return (
            jsonify({"status": "error", "message": "查询历史记录失败"}),
            500,
        )

@app.route("/delete_record/<record_id>", methods=["DELETE"])
def delete_record(record_id: str):
    """删除指定记录；若删除出库记录会自动恢复对应进库状态。"""
    try:
        if delete_record_by_id(record_id):
            return jsonify({"status": "success", "message": "记录删除成功"})
        return (
            jsonify({"status": "error", "message": "未找到对应的记录"}),
            404,
        )
    except Exception as error:
        logger.error(f"删除记录时出错: {str(error)}")
        return (
            jsonify({"status": "error", "message": "删除记录失败"}),
            500,
        )

if __name__ == '__main__':
    app.run(debug=True, port=5000)