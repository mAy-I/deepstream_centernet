import sys

sys.path.insert(0, "apps")

import os
import time
import argparse
import json
import numpy as np

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    TimeElapsedColumn,
)

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import pyds

from common import bus_call, create_source_bin
from common import get_video_info, get_affine_transform, transform_preds
from mmcv.tensorrt import load_tensorrt_plugin

load_tensorrt_plugin()

streams_info = {}
detection_result = {}

INPUT_RES = 512

progress = Progress(
    TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
    SpinnerColumn(),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    TimeElapsedColumn(),
    "•",
    TimeRemainingColumn(),
)


### video information ###
def sinkpad_frame_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list

        detection_result[f"stream_{frame_meta.pad_index}"][frame_number] = []

        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            rect_params = obj_meta.rect_params
            x1 = (
                rect_params.left
                / streams_info[f"stream_{frame_meta.pad_index}"]["scaling_factor"]
            )
            y1 = (
                rect_params.top
                / streams_info[f"stream_{frame_meta.pad_index}"]["scaling_factor"]
            )
            x2 = (rect_params.left + rect_params.width) / streams_info[
                f"stream_{frame_meta.pad_index}"
            ]["scaling_factor"]
            y2 = (rect_params.top + rect_params.height) / streams_info[
                f"stream_{frame_meta.pad_index}"
            ]["scaling_factor"]

            score = obj_meta.confidence
            bbox = [x1, y1, x2, y2]

            detection_result[f"stream_{frame_meta.pad_index}"][frame_number].append(
                {"bbox": bbox, "score": score}
            )

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        progress.update(
            streams_info[f"stream_{frame_meta.pad_index}"]["pbar"], advance=1
        )

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def main(args):

    Gst.init()
    print("Creating Pipeline ...")
    pipeline = Gst.Pipeline()

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    pipeline.add(streammux)

    video_list = args.video_file.split(" ")
    num_sources = len(video_list)

    tmp_stream_width_list = []
    tmp_stream_height_list = []

    for i, video_path in enumerate(video_list):
        video_file_name = video_path.split(os.sep)[-1].split(".")[0]
        streams_info[f"stream_{i}"] = get_video_info(video_path)
        tmp_stream_width_list.append(streams_info[f"stream_{i}"]["width"])
        tmp_stream_height_list.append(streams_info[f"stream_{i}"]["height"])
        detection_result[f"stream_{i}"] = {}
        streams_info[f"stream_{i}"]["pbar"] = progress.add_task(
            f"stream_{i}",
            filename=video_file_name,
            total=streams_info[f"stream_{i}"]["frame_count"],
        )

        ## Create source bin ##
        print(f"Creating source bin {i}")
        uri_name = video_path
        source_bin = create_source_bin(i, uri_name)
        pipeline.add(source_bin)
        padname = f"sink_{i}"
        sinkpad = streammux.get_request_pad(padname)
        srcpad = source_bin.get_static_pad("src")
        srcpad.link(sinkpad)

    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    print("Creating filter")
    caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter = Gst.ElementFactory.make("capsfilter", "filter")

    # Use nvinfer to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    print("Creating Sink")
    sink = Gst.ElementFactory.make("fakesink", "fake-renderer")
    print(f"Playing file {args.video_file}")
    streammux.set_property("width", max(tmp_stream_width_list))
    streammux.set_property("height", max(tmp_stream_height_list))
    streammux.set_property("batch-size", num_sources)
    streammux.set_property("batched-push-timeout", 4000000)
    pgie.set_property(
        "config-file-path",
        "apps/centernet/centernet_config.txt",
    )
    pgie.set_property(
        "batch-size",
        args.batch_size,
    )
    filter.set_property("caps", caps)

    print("Adding elements to Pipeline")
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(filter)
    pipeline.add(sink)

    print("Linking elements in the Pipeline ...")
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(filter)
    filter.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    sinkpad = sink.get_static_pad("sink")

    sinkpad.add_probe(Gst.PadProbeType.BUFFER, sinkpad_frame_buffer_probe, 0)

    print("Starting pipeline")
    pipeline.set_state(Gst.State.PLAYING)
    start_time = time.time()
    for stream, info in streams_info.items():
        streams_info[stream]["scaling_factor"] = (
            max(info["width"], info["height"]) / INPUT_RES
        )
    with progress:
        try:
            loop.run()
        except:
            pass
        finally:
            finish_time = time.time()
            pipeline.set_state(Gst.State.NULL)

            total_num_frames = 0
            for _, stream_info in streams_info.items():
                total_num_frames += stream_info["frame_count"]

            print(f"\nFPS: {total_num_frames / (finish_time - start_time):.2f}")
            print(f"DURATION: {(finish_time - start_time):.2f}")

            if not os.path.isdir(args.save_folder):
                os.makedirs(args.save_folder)

            source_max_width = max(tmp_stream_width_list)
            source_max_height = max(tmp_stream_height_list)
            c = np.array([source_max_width, source_max_height], dtype=np.float32) / 2
            s = float(max(source_max_width, source_max_height))
            affine_transform = get_affine_transform(c, s, 0, (128, 128), inv=1)
            for stream_idx in range(num_sources):
                stream_detection_result = detection_result[f"stream_{stream_idx}"]
                for fid, detected_people in stream_detection_result.items():
                    bbox_list = []
                    for detected_person in detected_people:
                        bbox = detected_person["bbox"]
                        bbox_list.append(bbox)
                    bbox_array = np.array(bbox_list)
                    bbox_transform = transform_preds(
                        bbox_array.reshape(-1, 2), affine_transform
                    )
                    bbox_transform = bbox_transform.reshape(-1, 4)
                    bbox_transform[:, 1::2] += (
                        max(source_max_width, source_max_height)
                        - min(source_max_width, source_max_height)
                    ) / 2

                    bbox_list = bbox_transform.tolist()
                    for i in range(len(detected_people)):
                        detection_result[f"stream_{stream_idx}"][fid][i][
                            "bbox"
                        ] = bbox_list[i]

            for stream in detection_result.keys():
                with open(
                    os.path.join(args.save_folder, f"{stream}_result.json"), "w"
                ) as make_file:
                    json.dump(detection_result[stream], make_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tutorial: inference")
    parser.add_argument(
        "--video_file",
        type=str,
        default="file:///test1.mp4 file:///test2.mp4",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )
    parser.add_argument("--save_folder", type=str, default="result")
    args = parser.parse_args()
    main(args)
