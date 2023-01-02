import subprocess
import argparse
from pathlib import Path

# ffmpeg -an -i "$INPUT" -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 "$OUTFILE_NO_EXT.mp4"
# ffmpeg -i "$INPUT" -vcodec libvpx-vp9 -b:v 1M -acodec libvorbis "$OUTFILE_NO_EXT.webm"


def process_video(video_path: Path):
    video_dir = video_path.parent
    video_name = video_path.stem
    output_path_mp4 = video_dir / f"{video_name}_processed.mp4"
    output_path_webm = video_dir / f"{video_name}_processed.webm"

    # subprocess.run(
    #     [
    #         "ffmpeg",
    #         "-i",
    #         str(video_path),
    #         "-vcodec",
    #         "libx264",
    #         "-crf",
    #         "0",
    #         "-preset",
    #         "ultrafast",
    #         str(output_path_mp4),
    #     ]
    # )

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(video_path),
            "-c:v",
            "-vcodec",
            "libaom-av1",
            # "-lossless",
            # "1",
            "-crf",
            "0",
            "-acodec",
            "libvorbis",
            str(output_path_webm),
        ]
    )

    # file_size_old = video_path.stat().st_size
    # file_size_new_mp4 = output_path_mp4.stat().st_size
    # file_size_new_webm = output_path_webm.stat().st_size

    # print(f"Old file size: {file_size_old / 1024 / 1024:.2f} MB")
    # print(f"New file size (mp4): {file_size_new_mp4 / 1024 / 1024:.2f} MB")
    # print(f"New file size (webm): {file_size_new_webm / 1024 / 1024:.2f} MB")

    # percent_compression_mp4 = (file_size_old - file_size_new_mp4) / file_size_old * 100
    # percent_compression_webm = (file_size_old - file_size_new_webm) / file_size_old * 100

    # print(f"Percent compression (mp4): {percent_compression_mp4:.2f}%")
    # print(f"Percent compression (webm): {percent_compression_webm:.2f}%")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("video_path", type=Path)
    args = cli.parse_args()
    process_video(args.video_path)

