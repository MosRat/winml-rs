import onnxruntime as ort

# 加载ONNX模型
model_path = 'SqueezeNet.onnx'  # 模型文件路径
session = ort.InferenceSession(model_path)

# 获取模型的输入信息
input_meta = session.get_inputs()
for meta in input_meta:
    print(f'输入名称: {meta.name}')
    print(f'输入形状: {meta.shape}')
    print(f'输入类型: {meta.type}')

# 获取模型的输出信息
output_meta = session.get_outputs()
for meta in output_meta:
    print(f'输出名称: {meta.name}')
    print(f'输出形状: {meta.shape}')
    print(f'输出类型: {meta.type}')
