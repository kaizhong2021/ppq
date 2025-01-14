import os
from typing import List

from ppq.core import (DataType, PPQ_CONFIG, NetworkFramework, QuantizationProperty,
                      QuantizationStates)
from ppq.IR import BaseGraph, GraphExporter, QuantableOperation
from ppq.IR.morph import GraphDeviceSwitcher

from .caffe_exporter import CaffeExporter
from .onnx_exporter import OnnxExporter
from .util import convert_value
import json

class TensorrtExporter(GraphExporter):
    def export_quantization_config(self, config_path: str, graph: BaseGraph):
        quant_info = {}
        act_quant_info = {}
        quant_info["act_quant_info"] = act_quant_info

        topo_order =  graph.topological_sort()

        for index, op in enumerate(topo_order):
            
            if op.type in {"Shape", "Gather", "Unsqueeze", "Concat", "Reshape"}:
               continue
            
            if index == 0:
                assert graph.inputs.__contains__(op.inputs[0].name)
                input_cfg = op.config.input_quantization_config[0]
                assert input_cfg.state == QuantizationStates.ACTIVATED and\
                    input_cfg.policy.has_property(QuantizationProperty.PER_TENSOR)
                trt_range_input = input_cfg.scale.item() * (input_cfg.quant_max - input_cfg.quant_min) / 2
                act_quant_info[op.inputs[0].name] = trt_range_input
                output_cfg = op.config.output_quantization_config[0]
                trt_range_output = output_cfg.scale.item() * (output_cfg.quant_max - output_cfg.quant_min) / 2
                act_quant_info[op.outputs[0].name] = trt_range_input

            else:
                output_cfg = op.config.output_quantization_config[0]
                trt_range_output = output_cfg.scale.item() * (output_cfg.quant_max - output_cfg.quant_min) / 2
                act_quant_info[op.outputs[0].name] = trt_range_output

        json_qparams_str = json.dumps(quant_info, indent=4)
        with open(config_path, "w") as json_file:
            json_file.write(json_qparams_str)


    def export(self, file_path: str, graph: BaseGraph, config_path: str = None, input_shapes: List[List[int]] = [[1, 3, 224, 224]]):
        if not PPQ_CONFIG.EXPORT_DEVICE_SWITCHER:
            processor = GraphDeviceSwitcher(graph)
            processor.remove_switcher()
        if config_path is not None:
            self.export_quantization_config(config_path, graph)

        _, ext = os.path.splitext(file_path)
        if ext == '.onnx':
            exporter = OnnxExporter()
            exporter.export(file_path=file_path, graph=graph, config_path=None)
        elif ext in {'.prototxt', '.caffemodel'}:
            exporter = CaffeExporter()
            exporter.export(file_path=file_path, graph=graph, config_path=None, input_shapes=input_shapes)
        
        # no pre-determined export format, we export according to the
        # original model format
        elif graph._built_from == NetworkFramework.CAFFE:
            exporter = CaffeExporter()
            exporter.export(file_path=file_path, graph=graph, config_path=None, input_shapes=input_shapes)

        elif graph._built_from == NetworkFramework.ONNX:
            exporter = OnnxExporter()
            exporter.export(file_path=file_path, graph=graph, config_path=None)
