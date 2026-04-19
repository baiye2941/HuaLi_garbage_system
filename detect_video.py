

import os
import cv2
import argparse
import sys
from pathlib import Path


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.detector import MyDetector


def detect_video(input_path, output_path=None, skip_frames=1,
                 garbage_model=None, fire_model=None, smoke_model=None):


    if not os.path.exists(input_path):
        print(f"[错误] 文件不存在: {input_path}")
        return


    if output_path is None:
        input_dir = os.path.dirname(input_path)
        input_name = os.path.basename(input_path)
        name, ext = os.path.splitext(input_name)
        output_path = os.path.join(input_dir, f"{name}_detected{ext}")

    print(f"[视频检测] 输入: {input_path}")
    print(f"[视频检测] 输出: {output_path}")


    print("[视频检测] 加载模型中...")
    detector = MyDetector(
        garbage_model_path=garbage_model,
        fire_model_path=fire_model,
        smoke_model_path=smoke_model
    )
    print(f"[视频检测] 模型状态: {detector.models_loaded}")


    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {input_path}")
        return


    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')

    print(f"[视频检测] 视频信息: {width}x{height}, {fps:.1f}fps, 共{total_frames}帧")


    out = cv2.VideoWriter(output_path, codec, fps, (width, height))
    if not out.isOpened():
        print("[错误] 无法创建输出视频")
        cap.release()
        return


    frame_count = 0
    detected_count = 0
    alert_count = 0

    print("[视频检测] 开始处理...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1


        if frame_count % skip_frames != 1:

            out.write(frame)
            continue


        results = detector.detect(frame)


        annotated_frame = detector.draw_boxes(frame, results)


        alert_in_frame = sum(1 for r in results if r.get("alert", False))
        if results:
            info_text = f"Frame {frame_count}: {len(results)} detected, {alert_in_frame} alerts"
        else:
            info_text = f"Frame {frame_count}: No detection"

        cv2.putText(annotated_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        out.write(annotated_frame)

        detected_count += len(results)
        alert_count += alert_in_frame


        progress = frame_count / total_frames * 100
        if frame_count % 30 == 0 or frame_count == total_frames:
            print(f"[进度] {progress:.1f}% ({frame_count}/{total_frames})")


    cap.release()
    out.release()

    print("\n" + "="*50)
    print("[视频检测] 处理完成!")
    print(f"[统计] 总帧数: {frame_count}")
    print(f"[统计] 检测目标数: {detected_count}")
    print(f"[统计] 预警次数: {alert_count}")
    print(f"[输出] 保存至: {output_path}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="视频目标检测工具")
    parser.add_argument("--input", "-i", required=True, help="输入视频路径")
    parser.add_argument("--output", "-o", default=None, help="输出视频路径")
    parser.add_argument("--skip", "-s", type=int, default=1, help="跳帧数，每N帧检测1次（默认1）")
    parser.add_argument("--garbage", "-g", default=None, help="垃圾分类模型路径")
    parser.add_argument("--fire", "-f", default=None, help="火焰检测模型路径")
    parser.add_argument("--smoke", "-sm", default=None, help="烟雾检测模型路径")

    args = parser.parse_args()


    if args.garbage is None:
        args.garbage = "app/models/garbege.pt"
    if args.fire is None:
        args.fire = "app/models/fire_smoke.pt"
    if args.smoke is None:
        args.smoke = "app/models/fire_smoke.pt"

    detect_video(
        input_path=args.input,
        output_path=args.output,
        skip_frames=args.skip,
        garbage_model=args.garbage,
        fire_model=args.fire,
        smoke_model=args.smoke
    )


if __name__ == "__main__":
    main()
