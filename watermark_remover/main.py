#!/usr/bin/env python3
"""去图片和视频水印/字幕 - CLI入口

用法:
    # 去除图片字幕
    python -m watermark_remover remove-subtitle image.jpg

    # 去除图片水印
    python -m watermark_remover remove-watermark image.jpg

    # 同时去除字幕和水印
    python -m watermark_remover remove-both image.jpg

    # 去除视频字幕
    python -m watermark_remover remove-subtitle video.mp4

    # 预览检测到的区域
    python -m watermark_remover preview image.jpg

    # 使用手动掩码去除水印
    python -m watermark_remover remove-watermark image.jpg --mask mask.png
"""

import argparse
import sys
from pathlib import Path

from .image_processor import ImageProcessor
from .video_processor import VideoProcessor


def get_processor(args):
    """根据参数创建处理器"""
    kwargs = {
        "inpaint_method": args.method,
        "inpaint_radius": args.radius,
        "subtitle_bottom_ratio": args.bottom_ratio,
        "subtitle_sensitivity": args.sensitivity,
    }

    # 根据文件类型选择处理器
    ext = Path(args.input).suffix.lower()
    is_video = ext in (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm")

    if is_video:
        return VideoProcessor(**kwargs), True
    else:
        return ImageProcessor(**kwargs), False


def cmd_remove_subtitle(args):
    processor, is_video = get_processor(args)
    if is_video:
        result = processor.remove_subtitle(args.input, args.output, quality=args.quality)
    else:
        result = processor.remove_subtitle(args.input, args.output)
    print(f"完成! 输出: {result}")


def cmd_remove_watermark(args):
    processor, is_video = get_processor(args)
    if is_video:
        result = processor.remove_watermark(args.input, args.output, quality=args.quality)
    else:
        mask_path = args.mask if hasattr(args, "mask") and args.mask else None
        result = processor.remove_watermark(args.input, args.output, manual_mask_path=mask_path)
    print(f"完成! 输出: {result}")


def cmd_remove_both(args):
    processor, is_video = get_processor(args)
    if is_video:
        result = processor.remove_both(args.input, args.output, quality=args.quality)
    else:
        result = processor.remove_both(args.input, args.output)
    print(f"完成! 输出: {result}")


def cmd_preview(args):
    processor, is_video = get_processor(args)
    if is_video:
        print("提示: 预览功能仅支持图片文件")
        sys.exit(1)

    detect_type = args.detect_type if hasattr(args, "detect_type") else "both"
    result = processor.preview_mask(args.input, args.output, detect_type)
    print(f"掩码预览已保存: {result}")


def main():
    parser = argparse.ArgumentParser(
        description="去图片和视频水印/字幕工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 公共参数
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("input", help="输入文件路径")
    common.add_argument("-o", "--output", help="输出文件路径 (默认自动生成)")
    common.add_argument(
        "--method", choices=["telea", "ns"], default="ns",
        help="修复算法 (默认: ns)"
    )
    common.add_argument("--radius", type=int, default=5, help="修复半径 (默认: 5)")
    common.add_argument(
        "--bottom-ratio", type=float, default=0.25,
        help="字幕检测的底部区域比例 (默认: 0.25)"
    )
    common.add_argument(
        "--sensitivity", type=float, default=0.7,
        help="检测灵敏度 0.0-1.0 (默认: 0.7)"
    )

    # remove-subtitle
    p_sub = subparsers.add_parser("remove-subtitle", parents=[common], help="去除字幕")
    p_sub.add_argument("--quality", choices=["low", "medium", "high"], default="medium", help="视频输出质量")
    p_sub.set_defaults(func=cmd_remove_subtitle)

    # remove-watermark
    p_wm = subparsers.add_parser("remove-watermark", parents=[common], help="去除水印")
    p_wm.add_argument("--mask", help="手动水印掩码路径 (白色区域为水印)")
    p_wm.add_argument("--quality", choices=["low", "medium", "high"], default="medium", help="视频输出质量")
    p_wm.set_defaults(func=cmd_remove_watermark)

    # remove-both
    p_both = subparsers.add_parser("remove-both", parents=[common], help="同时去除字幕和水印")
    p_both.add_argument("--quality", choices=["low", "medium", "high"], default="medium", help="视频输出质量")
    p_both.set_defaults(func=cmd_remove_both)

    # preview
    p_preview = subparsers.add_parser("preview", parents=[common], help="预览检测到的区域")
    p_preview.add_argument("--detect-type", choices=["subtitle", "watermark", "both"], default="both", help="检测类型")
    p_preview.set_defaults(func=cmd_preview)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
