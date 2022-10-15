import sys
import os
import os.path as osp
import subprocess
from unittest import result
import mmcv

CURRENT_DIR = osp.dirname(__file__)
sys.path.insert(0, CURRENT_DIR)
from dataset import build_mmseg_dataloader, evaluate_model
from qs_config import QS, TARGET_PLATFORM


from ppq.api import ENABLE_CUDA_KERNEL, export_ppq_graph, TorchExecutor, quantize_onnx_model, ppq_warning, TargetPlatform
from ppq.quantization.analyse import parameter_analyse, variable_analyse, layerwise_error_analyse, statistical_analyse, graphwise_error_analyse


def run_cmd(cmd_lines, log_path):
    """
    Args:
        cmd_lines: (list[str]): A command in multiple line style.
        log_path (str): Path to log file.

    Returns:
        int: error code.
    """
    sep = '\\'
    cmd_for_run = f' {sep}\n'.join(cmd_lines) + '\n'
    parent_path = osp.split(log_path)[0]
    os.makedirs(parent_path, exist_ok=True)

    print(100 * '-')
    print(f'Start running cmd\n{cmd_for_run}')
    print(f'Logging log to \n{log_path}')

    with open(log_path, 'w', encoding='utf-8') as file_handler:
        # write cmd
        file_handler.write(f'Command:\n{cmd_for_run}\n')
        file_handler.flush()
        process_res = subprocess.Popen(
            cmd_for_run,
            cwd=os.getcwd(),
            shell=True,
            stdout=file_handler,
            stderr=file_handler)
        process_res.wait()
        return_code = process_res.returncode

    if return_code != 0:
        print(f'Got shell return code={return_code}')
    with open(log_path, 'r') as f:
        content = f.read()
        print(f'Log message\n{content}')
    return return_code


ROOT_DIR = '../work-dir'
MMDEPLOY_DIR = '../mmdeploy'
MMSEG_DIR = '../mmsegmentation'
DEPLOY_CFG_PATH = osp.join(MMDEPLOY_DIR, 'configs/mmseg/segmentation_tensorrt_static-1024x2048.py')
MODEL_CFG_PATH = osp.join(MMSEG_DIR, 'configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py')
PYTORCH_CHECKPOINT = '../mmdeploy_checkpoints/mmseg/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
TEST_IMAGE = osp.join(MMSEG_DIR, 'demo/demo.png')
MODEL_NAME = osp.splitext(osp.split(MODEL_CFG_PATH)[1])[0]
WORKING_DIRECTORY = osp.join(ROOT_DIR, MODEL_NAME)
#trt onnx
TRT_FP32_ONNX_FILE = osp.join(WORKING_DIRECTORY, 'end2end.onnx')
#trt engine
TRT_ENGINE_FILE = osp.join(MMDEPLOY_DIR,'result')
os.makedirs(WORKING_DIRECTORY, exist_ok=True)

IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 2048
NETWORK_INPUTSHAPE = [1, 3, IMAGE_HEIGHT, IMAGE_WIDTH]  # input shape of your network
CALIBRATION_BATCHSIZE = 1  # batchsize of calibration dataset
EXECUTING_DEVICE = 'cuda'  # 'cuda' or 'cpu'.
REQUIRE_ANALYSE = True
DUMP_RESULT = False
TORCH2ONNX = True
TEST_TRT_FP32 = False

# torch2onnx
if TORCH2ONNX:
    cmd_lines = ['python', osp.join(MMDEPLOY_DIR, 'tools/deploy.py'),
                 DEPLOY_CFG_PATH,
                 MODEL_CFG_PATH,
                 PYTORCH_CHECKPOINT,
                 TEST_IMAGE,
                 '--device cuda:0',
                 f'--work-dir {WORKING_DIRECTORY}',
                 ]
    log_path = osp.join(WORKING_DIRECTORY, 'torch2onnx.log')
    run_cmd(cmd_lines, log_path)

# test trt fp32
if TEST_TRT_FP32:
    cmd_lines = ['python', osp.join(MMDEPLOY_DIR, 'tools/test.py'),
                 DEPLOY_CFG_PATH,
                 MODEL_CFG_PATH,
                 '--device cuda:0',
                 f'--model {osp.join(WORKING_DIRECTORY, "end2end.engine")}',
                 '--metrics mIoU'
                 ]
    log_path = osp.join(WORKING_DIRECTORY, 'test_trt_fp32.log')
    run_cmd(cmd_lines, log_path)



print('正准备量化你的网络，检查下列设置:')
print(f'WORKING DIRECTORY    : {WORKING_DIRECTORY}')
print(f'TARGET PLATFORM      : {TARGET_PLATFORM.name}')
print(f'NETWORK INPUTSHAPE   : {NETWORK_INPUTSHAPE}')
print(f'CALIBRATION BATCHSIZE: {CALIBRATION_BATCHSIZE}')

calib_txt = osp.join(CURRENT_DIR, 'data/Quant32FromTrainImages.txt')
calib_dataloader = build_mmseg_dataloader(MODEL_CFG_PATH, 'train', calib_txt)
collate_fn = lambda x: x.to(EXECUTING_DEVICE)
ONNX_MODEL_FILE = os.path.join(WORKING_DIRECTORY, 'end2end.onnx')
PPQ_ONNX_INT8_FILE = os.path.join(WORKING_DIRECTORY, 'ppq-int8.onnx')
PPQ_TRT_INT8_FILE = os.path.join(WORKING_DIRECTORY, 'ppq-int8.engine')
# ENABLE CUDA KERNEL 会加速量化效率 3x ~ 10x，但是你如果没有装相应编译环境的话是编译不了的
# 你可以尝试安装编译环境，或者在不启动 CUDA KERNEL 的情况下完成量化：移除 with ENABLE_CUDA_KERNEL(): 即可
with ENABLE_CUDA_KERNEL():
    print('网络正量化中，根据你的量化配置，这将需要一段时间:')
    quantized = quantize_onnx_model(
        onnx_import_file=ONNX_MODEL_FILE, calib_dataloader=calib_dataloader, calib_steps=32, setting=QS,
        input_shape=NETWORK_INPUTSHAPE, collate_fn=collate_fn, device=EXECUTING_DEVICE,
        platform=TARGET_PLATFORM, do_quantize=True)
    # -------------------------------------------------------------------
    # 如果你需要执行量化后的神经网络并得到结果，则需要创建一个 executor
    # 这个 executor 的行为和 torch.Module 是类似的，你可以利用这个东西来获取执行结果
    # 请注意，必须在 export 之前执行此操作。
    # -------------------------------------------------------------------
    executor = TorchExecutor(graph=quantized, device=EXECUTING_DEVICE)
    val_dataloader = build_mmseg_dataloader(MODEL_CFG_PATH, 'val')
    json_file = osp.join(WORKING_DIRECTORY, 'ppq_executor_val.json')
    print(100 * '--')
    print('evaluate val dataset')
    evaluate_model(executor, val_dataloader, json_file)

    # -------------------------------------------------------------------
    # PPQ 计算量化误差时，使用信噪比的倒数作为指标，即噪声能量 / 信号能量
    # 量化误差 0.1 表示在整体信号中，量化噪声的能量约为 10%
    # 你应当注意，在 graphwise_error_analyse 分析中，我们衡量的是累计误差
    # 网络的最后一层往往都具有较大的累计误差，这些误差是其前面的所有层所共同造成的
    # 你需要使用 layerwise_error_analyse 逐层分析误差的来源
    # -------------------------------------------------------------------
    print('正计算网络量化误差(SNR)，最后一层的误差应小于 0.1 以保证量化精度:')
    reports = graphwise_error_analyse(
        graph=quantized, running_device=EXECUTING_DEVICE, steps=32,
        dataloader=calib_dataloader, collate_fn=collate_fn)
    for op, snr in reports.items():
        if snr > 0.1: ppq_warning(f'层 {op} 的累计量化误差显著，请考虑进行优化')

    if REQUIRE_ANALYSE:
        print('正计算逐层量化误差(SNR)，每一层的独立量化误差应小于 0.1 以保证量化精度:')
        layerwise_error_analyse(graph=quantized, running_device=EXECUTING_DEVICE,
                                interested_outputs=None,
                                dataloader=calib_dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))
    print('--- parameter_analyse')
    # parameter_analyse(graph=quantized)
    # print('--- variable_analyse')
    # variable_analyse(quantized,
    #     dataloader=calibration_dataloader,
    #     interested_outputs=[],
    #     collate_fn=collate_fn,
    #     running_device = DEVICE,
    #     samples_per_step = 65536,
    #     steps = 8,
    #     dequantize = False)
    # records = statistical_analyse(
    #     quantized, running_device=EXECUTING_DEVICE,
    #     dataloader=calib_dataloader, collate_fn=collate_fn, steps= 8)
    print('网络量化结束，正在生成目标文件:')
    export_ppq_graph(
        graph=quantized, platform=TARGET_PLATFORM,
        graph_save_to=PPQ_ONNX_INT8_FILE)

TEST_PPQ_TRT_INT8 = True
if TEST_PPQ_TRT_INT8:
    MODEL_CFG_PATH_INT8 = osp.join(MMDEPLOY_DIR, 'configs/mmseg/segmentation_tensorrt-int8_static-1024x2048.py')

    cmd_lines = ['python', osp.join(MMDEPLOY_DIR, 'tools/onnx2tensorrt.py'),
                 DEPLOY_CFG_PATH,
                 PPQ_ONNX_INT8_FILE,
                 osp.splitext(PPQ_TRT_INT8_FILE)[0],
                 ]
    log_path = osp.join(WORKING_DIRECTORY, 'ppq_onnx2tensorrt.log')
    run_cmd(cmd_lines, log_path)

    cmd_lines = ['python', osp.join(MMDEPLOY_DIR, 'tools/test.py'),
                 MODEL_CFG_PATH_INT8,
                 MODEL_CFG_PATH,
                 '--device cuda:0',
                 f'--model {PPQ_TRT_INT8_FILE}',
                 '--metrics mIoU'
                 ]
    log_path = osp.join(WORKING_DIRECTORY, 'test_ppq_trt_int8.log')
    run_cmd(cmd_lines, log_path)

##onnx2tensorrt
ENGINE_NAME = 'result'
ONNX2TENSORRT = True
if ONNX2TENSORRT:
    MODEL_CFG_PATH_INT8 = osp.join(MMDEPLOY_DIR, 'configs/mmseg/segmentation_tensorrt-int8_static-1024x2048.py')

    cmd_lines = ['python', osp.join(MMDEPLOY_DIR, 'tools/onnx2tensorrt.py'),
                 DEPLOY_CFG_PATH,
                 TRT_FP32_ONNX_FILE,
                 ENGINE_NAME,
                 '--device cuda:0',
                 '--log-level INFO',
                 f'--calib-file {calib_dataloader}',
                ]
    log_path = osp.join(WORKING_DIRECTORY,'ONNX2TENSORRT.log')
    run_cmd(cmd_lines, log_path)

# test trt int8
TEST_TRT_INT8 = True
if TEST_TRT_INT8:
    cmd_lines = ['python', osp.join(MMDEPLOY_DIR, 'tools/test.py'),
                 DEPLOY_CFG_PATH,
                 MODEL_CFG_PATH,
                 '--device cuda:0',
                 f'--model {osp.join(MMDEPLOY_DIR, "result.engine")}',
                 '--metrics mIoU'
                 ]
    log_path = osp.join(WORKING_DIRECTORY, 'test_trt_fp32.log')
    run_cmd(cmd_lines, log_path)



