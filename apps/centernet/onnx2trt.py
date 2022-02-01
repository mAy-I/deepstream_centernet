import argparse

from mmcv.tensorrt import (
    onnx2trt,
    save_trt_engine,
)


def main(args):
    batch_size = args.batch_size
    input_res = args.input_res

    opt_shape_dict = {
        "input": [
            [1, 3, input_res, input_res],
            [
                int(batch_size / 2),
                3,
                input_res,
                input_res,
            ],
            [
                batch_size,
                3,
                input_res,
                input_res,
            ],
        ]
    }

    print(f"build onnx to tensorrt...{args.save_model}")
    max_workspace_size = 1 << 30
    trt_engine = onnx2trt(
        args.load_model,
        opt_shape_dict,
        fp16_mode=args.fp16,
        max_workspace_size=max_workspace_size,
    )

    save_trt_engine(trt_engine, args.save_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Convertor")
    parser.add_argument("--load_model", type=str, default="", help="ONNX model")
    parser.add_argument("--save_model", type=str, default="", help="export TensorRT")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="model config: batch size"
    )
    parser.add_argument(
        "--input_res", type=int, default=512, help="model config: input resloution"
    )
    parser.add_argument("--fp16", action="store_true", help="model config: FP16")
    args = parser.parse_args()
    main(args)
