# HuaLi_garbage_system

> 面向社区场景的垃圾与火情识别预警系统。当前仓库主线已切换为 `FastAPI + Jinja2 + SQLite + YOLO/ONNX`，支持图片检测、摄像头 Base64 检测、视频任务处理、预警记录留存与统计分析。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Celery](https://img.shields.io/badge/Celery-5.4%2B-37814A?logo=celery)](https://docs.celeryq.dev/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite)](https://www.sqlite.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.20%2B-grey?logo=onnx)](https://onnxruntime.ai/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-FF9F00)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 项目概览

当前代码已经具备以下主功能：

- 图片上传检测：`POST /api/detect/image`
- Base64 图像检测，适合摄像头抓拍：`POST /api/detect/base64`
- 视频上传检测与任务状态查询：`POST /api/detect/video`、`GET /api/tasks/{task_id}`
- 预警图片留存、历史预警记录查询、统计分析
- ONNX Runtime 优先推理，失败时回退到 Ultralytics `.pt` 权重
- 视频检测支持 Celery 异步任务，也支持在无可用 worker 时退回本地线程处理
- 视频链路集成了升级版跟踪/时序告警流水线，能给检测结果附加 `track_id` 并累计时序告警次数

当前推荐入口是 [app/main.py](app/main.py)。

## 当前实现状态

- Web 框架：`FastAPI`
- 页面模板：`Jinja2`
- 数据库：`SQLite`
- 异步任务：`Celery`
- 检测后端：`ONNX Runtime` / `Ultralytics`
- 一键启动脚本：`start_queue.bat`

代码层面当前真正接入主检测链路的是：

- 垃圾桶
- 垃圾满溢
- 散落垃圾
- 火情

仓库里仍保留了烟雾相关权重与类别定义，但 `app/services/detection_service.py` 当前只构建了 `garbage` 和 `fire` 两组检测器，烟雾尚未正式接入主链路。

## 目录结构

```text
garbage_system/
├── app/
│   ├── api/                # 页面路由与 API 路由
│   ├── models/             # 模型权重与导出的 ONNX 文件
│   ├── services/           # 检测、视频、记录服务
│   ├── templates/          # Jinja2 前端页面
│   ├── upgrade/            # 跟踪与时序告警升级流水线
│   ├── bootstrap.py        # 启动初始化
│   ├── celery_app.py       # Celery 应用
│   ├── config.py           # 项目配置
│   ├── constants.py        # 类别常量
│   ├── database.py         # 数据库连接
│   ├── db_models.py        # ORM 模型
│   ├── main.py             # FastAPI 入口
│   ├── schemas.py          # Pydantic 响应模型
│   └── tasks.py            # 视频异步任务
├── dataset/                # 当前仓库内保留的数据集目录
├── start_queue.bat         # Windows 一键启动脚本
├── requirements.txt
└── README.md
```

## 主要代码对应关系

### Web 与接口

- [app/main.py](app/main.py)：FastAPI 应用入口
- [app/api/pages.py](app/api/pages.py)：首页、检测页、预警页、统计页、数据集页、视频页
- [app/api/routes.py](app/api/routes.py)：图片检测、Base64 检测、视频任务、预警、统计、状态接口
- [app/templates/detection.html](app/templates/detection.html)：综合检测页
- [app/templates/video.html](app/templates/video.html)：独立视频检测页
- [app/templates/alerts.html](app/templates/alerts.html)：预警记录页
- [app/templates/statistics.html](app/templates/statistics.html)：统计页
- [app/templates/dataset.html](app/templates/dataset.html)：数据集说明页

### 检测与视频处理

- [app/services/inference.py](app/services/inference.py)：ONNX / Ultralytics 双后端推理封装
- [app/services/detection_service.py](app/services/detection_service.py)：图片检测、场景分析、框绘制
- [app/services/video_service.py](app/services/video_service.py)：视频逐帧检测、告警冷却、升级流水线接入
- [app/tasks.py](app/tasks.py)：Celery 视频任务封装

### 数据与记录

- [app/database.py](app/database.py)：数据库引擎与会话
- [app/db_models.py](app/db_models.py)：预警记录、检测记录、视频任务记录
- [app/services/record_service.py](app/services/record_service.py)：预警与任务数据读写
- [app/bootstrap.py](app/bootstrap.py)：启动时自动建表与初始化上传目录

### 升级流水线

- [app/upgrade/pipeline.py](app/upgrade/pipeline.py)：检测 -> 跟踪 -> 时序告警
- [app/upgrade/tracker.py](app/upgrade/tracker.py)：当前为轻量跟踪占位实现，逐个分配 `track_id`
- [app/upgrade/alarm.py](app/upgrade/alarm.py)：连续帧告警规则
- [app/upgrade/detection.py](app/upgrade/detection.py)：原始检测结果适配器

## 模型加载规则

配置以 [app/config.py](app/config.py) 为准，当前加载优先级如下：

- 垃圾模型：`app/models/garbege.onnx` -> `app/models/garbege.pt`
- 火情模型：`app/models/only_fire.onnx` -> `app/models/only_fire.pt`
- 烟雾模型路径已在配置中保留：`app/models/fire_smoke.onnx` / `app/models/fire_smoke.pt`

注意：

- `garbege` 是仓库沿用的历史命名，当前代码就是按这个文件名读取的，不要直接重命名权重文件，除非同步修改配置。
- 仓库中还保留 `fire_smoke.*`、`smoke_yolov8.pt`、`fire_recall2.*` 等文件，但它们目前不属于 FastAPI 主检测入口默认加载集。

## 安装依赖

### 1. 克隆项目

```bash
git clone https://github.com/Nyzeep/HuaLi_garbage_system.git
cd HuaLi_garbage_system
```

### 2. 创建并激活虚拟环境

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

## 可选环境变量

项目会自动读取根目录 `.env`。没有 `.env` 时也可以按默认配置直接运行。

```env
APP_NAME=垃圾识别预警系统
APP_VERSION=2.0.0
DEBUG=false
DATABASE_URL=sqlite:///garbage_system.db
REDIS_URL=redis://localhost:6379/0
VIDEO_DEFAULT_SKIP_FRAMES=1
CELERY_TASK_ALWAYS_EAGER=false
```

## 启动方式

### 方式一：使用 `start_queue.bat`

适合 Windows 本地快速启动：

```bat
start_queue.bat
```

脚本会自动执行这些步骤：

1. 检查当前已激活或仓库内可用的虚拟环境
2. 若不存在可用环境，则自动创建 `.venv`
3. 若依赖缺失，则自动执行 `pip install -r requirements.txt`
4. 启动 Celery Worker
5. 启动 FastAPI Web 并打开浏览器

### 方式二：手动启动

先启动 Web：

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

如需使用 Celery 队列，再额外启动 Worker：

```bash
python -m celery -A app.celery_app worker --loglevel=info --pool=solo
```

说明：

- 视频任务优先尝试走 Celery 队列。
- 如果没有可用 worker，或任务分发失败，`app/api/routes.py` 会退回到本地后台线程处理视频。
- 如果你想稳定使用 Celery 队列，仍建议本地准备好 Redis 并保证 `REDIS_URL` 可连通。

## 页面入口

服务启动后可访问：

- `http://127.0.0.1:8000/`：首页
- `http://127.0.0.1:8000/detection`：综合检测页
- `http://127.0.0.1:8000/video`：独立视频检测页
- `http://127.0.0.1:8000/alerts`：预警记录页
- `http://127.0.0.1:8000/statistics`：统计分析页
- `http://127.0.0.1:8000/dataset`：数据集说明页
- `http://127.0.0.1:8000/docs`：FastAPI 接口文档

## 常用接口

### 图片检测

```http
POST /api/detect/image
```

表单字段：

- `file`：图片文件

### Base64 图像检测

```http
POST /api/detect/base64
```

请求体示例：

```json
{
  "image": "data:image/jpeg;base64,..."
}
```

### 视频检测

```http
POST /api/detect/video
```

表单字段：

- `file`：视频文件
- `skip_frames`：跳帧数，默认值来自 `VIDEO_DEFAULT_SKIP_FRAMES`

任务查询：

```http
GET /api/tasks/{task_id}
```

### 记录与状态

```http
GET /api/alerts
GET /api/alerts/{record_uid}/image
GET /api/statistics
GET /api/status
GET /api/classes
```

## 数据存储位置

- SQLite 数据库：`garbage_system.db`
- 预警截图：`app/uploads/alerts/`
- 视频上传与结果视频：`app/uploads/videos/`
- 静态访问前缀：`/uploads/...`

`app/bootstrap.py` 会在应用启动时自动创建数据表和上传目录。

## 已知说明

- 当前 Web 主入口是 `app/main.py`，不是旧版 Flask 项目结构。
- `requirements.txt` 里仍保留了 `flask`、`werkzeug` 等历史依赖，但当前前端页面并不依赖 Node.js 构建流程。
- `package.json` 只保留了 `docx` 依赖，不参与当前 Web 页面运行。
- `app/test_video.py` 更偏向历史调试脚本，不属于主运行链路。

## 许可证

本项目采用 [MIT License](LICENSE)。
