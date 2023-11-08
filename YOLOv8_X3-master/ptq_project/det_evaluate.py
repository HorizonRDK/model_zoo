# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import logging
import multiprocessing
import os
import sys
sys.path.append("./python/data/")
from transformer import *
from dataloader import *

import click

from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.utils.tool_utils import init_root_logger, on_exception_exit

from postprocess import eval_postprocess, calc_accuracy, gen_report

sess = None

MULTI_PROCESS_WARNING_DICT = {
    "CPUExecutionProvider": {
        "origin": "onnxruntime infering float model "
                  "does not work well with multiprocess, "
                  "the program may be blocked. "
                  "It is recommended to use single process operation",
        "quanti": ""
    },
    "CUDAExecutionProvider": {
        "origin": "GPU does not work well with multiprocess, "
                  "the program may prompt errors. "
                  "It is recommended to use single process operation",
        "quanti": "GPU does not work well with multiprocess, "
                  "the program may prompt errors. "
                  "It is recommended to use single process operation"
    }
}

SINGLE_PROCESS_WARNING_DICT = {
    "CPUExecutionProvider": {
        "origin": "",
        "quanti": "Infering with single process may take a long time."
                  " It is recommended to use multi process running"
    },
    "CUDAExecutionProvider": {
        "origin": "",
        "quanti": ""
    }
}

DEFAULT_PARALLEL_NUM = {
    "CPUExecutionProvider": {
        "origin": 1,
        "quanti": int(os.environ.get('PARALLEL_PROCESS_NUM', 10)),
    },
    "CUDAExecutionProvider": {
        "origin": 1,
        "quanti": 1,
    }
}


def eval_image_preprocess(image_path, annotation_path, input_shape, input_layout, load_transformer):
    assert load_transformer in ("origin", "quanti")
    if load_transformer == "origin":
        from test_original_float_model import infer_transformers
        transformers = infer_transformers(input_shape, input_layout)
        data_loader = COCODataLoader(transformers,
                                     image_path,
                                     annotation_path,
                                     imread_mode='opencv')
        return data_loader
    else:
        from test_quantized_model import infer_transformers
        transformers = infer_transformers(input_shape, input_layout)
        data_loader = COCODataLoader(transformers,
                                     image_path,
                                     annotation_path,
                                     imread_mode='opencv')
        return data_loader


def init_sess(model):
    global sess
    sess = HB_ONNXRuntime(model_file=model)
    # (Optional) GPU acceleration
    sess.set_providers(["CUDAExecutionProvider"])


class ParallelExector(object):
    def __init__(self, input_layout, input_offset, parallel_num):
        self._results = []
        self.input_layout = input_layout
        self.input_offset = input_offset

        self.parallel_num = parallel_num
        self.validate()

        if self.parallel_num != 1:
            logging.info(f"Init {self.parallel_num} processes")
            self._pool = multiprocessing.Pool(self.parallel_num)
            self._queue = multiprocessing.Manager().Queue(self.parallel_num)

    def get_accuracy(self, annotation_path):
        return calc_accuracy(annotation_path, self._results)

    def infer(self, val_data, batch_id, total_batch):
        if self.parallel_num != 1:
            self.feed(val_data, batch_id, total_batch)
        else:
            logging.info(f"Feed batch {batch_id + 1}/{total_batch}")
            eval_result = run(val_data, self.input_layout, self.input_offset)
            self._results.append(eval_result)

    def feed(self, val_data, batch_id, total_batch):
        self._pool.apply(func=product,
                         args=(self._queue, val_data, batch_id, total_batch))
        r = self._pool.apply_async(func=consumer,
                                   args=(self._queue, self.input_layout,
                                         self.input_offset),
                                   error_callback=logging.error)
        self._results.append(r)

    def close(self):
        if hasattr(self, "_pool"):
            self._pool.close()
            self._pool.join()

    def validate(self):
        provider = sess.get_provider()
        if sess.get_input_type() == 3:
            model_type = "quanti"
        else:
            model_type = "origin"

        if self.parallel_num == 0:
            self.parallel_num = DEFAULT_PARALLEL_NUM[provider][model_type]

        if self.parallel_num == 1:
            warning_message = SINGLE_PROCESS_WARNING_DICT[provider][model_type]
        else:
            warning_message = MULTI_PROCESS_WARNING_DICT[provider][model_type]

        if warning_message:
            logging.warning(warning_message)


def product(queue, val_data, batch_id, total_batch):
    logging.info("Feed batch {}/{}".format(batch_id + 1, total_batch))
    queue.put(val_data)


def consumer(queue, input_layout, input_offset):
    return run(queue.get(), input_layout, input_offset)


def run(val_data, input_layout, input_offset):
    input_name = sess.input_names[0]
    output_name = sess.output_names
    model_hw_shape = sess.get_hw()
    image, entry_dict = val_data
    # make sure pre-process logic is the same with runtime
    output = sess.run(output_name, {input_name: image},
                      input_offset=input_offset)
    eval_result = eval_postprocess(output, model_hw_shape, entry_dict)
    return eval_result


def evaluate(image_path, annotation_path, val_txt_path, input_layout,
             input_offset, total_image_number, parallel_num, load_transformer):
    if not input_layout:
        logging.warning(f"input_layout not provided. Using {sess.layout[0]}")
        input_layout = sess.layout[0]
    if val_txt_path:
        data_loader, total_image_number = eval_image_preprocess(
            image_path, annotation_path, val_txt_path, sess.get_hw(),
            input_layout, load_transformer)
    else:
        data_loader = eval_image_preprocess(image_path, annotation_path,
                                            sess.get_hw(), input_layout, load_transformer)

    val_exe = ParallelExector(input_layout, input_offset, parallel_num)

    for batch_id, val_data in enumerate(data_loader):
        if batch_id >= total_image_number:
            break
        val_exe.infer(val_data, batch_id, total_image_number)
    val_exe.close()
    metric_result = val_exe.get_accuracy(annotation_path)
    gen_report(metric_result)


@click.version_option(version="1.0.0")
@click.command()
@click.option('-m', '--model', type=str, help='Input onnx model(.onnx) file')
@click.option('-i',
              '--image_path',
              type=str,
              help='Evaluation image directory.')
@click.option('-l',
              '--annotation_path',
              type=str,
              help='Evaluate image label path')
@click.option('-v',
              '--val_txt_path',
              type=str,
              default=None,
              help='Val txt file path')
@click.option('-y',
              '--input_layout',
              type=str,
              default="",
              help='Model input layout')
@click.option('-o',
              '--input_offset',
              type=str,
              default=128,
              help='input inference offset.')
@click.option('-p',
              '--parallel_num',
              type=int,
              default=0,
              help='Parallel eval process number. '
              'The default value of evaluating fixed-point model using CPU is 10, '
              'and other defaults are 1')
@click.option('-c',
              '--color_sequence',
              type=str,
              default=None,
              help='Color sequence')
@click.option('-t',
              '--total_image_number',
              type=int,
              default=50,
              help='Total Image Number')
@click.option('-lt',
              '--load_transformer',
              type=str,
              default='quanti',
              help='Load transformer of origin/quanti model, default is quanti')
@on_exception_exit
def main(model, image_path, annotation_path, val_txt_path, input_layout,
         input_offset, parallel_num, color_sequence, total_image_number, load_transformer):
    init_root_logger("evaluation",
                     console_level=logging.INFO,
                     file_level=logging.DEBUG)
    if color_sequence:
        logging.warning("option color_sequence is deprecated.")
    init_sess(model)
    evaluate(image_path, annotation_path, val_txt_path, input_layout,
             input_offset, total_image_number, parallel_num, load_transformer)


if __name__ == '__main__':
    main()