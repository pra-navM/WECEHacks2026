import torch as th
from stable_baselines3 import PPO
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# 1. Load model
MODEL_PATH = "rl/checkpoints/scheduler_pro_v1.zip"
model = PPO.load(MODEL_PATH)

# 2. Wrapper optimized for Legacy Export
class OnnxablePolicy(th.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor_policy = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net

    def forward(self, observation):
        # We use the explicit sub-modules to ensure no "Value Head" data leaks in
        features = self.features_extractor(observation)
        latent_pi = self.mlp_extractor_policy(features)
        return self.action_net(latent_pi)

onnxable_model = OnnxablePolicy(model.policy)
onnxable_model.eval()

# 3. Export using the Legacy "TorchScript" engine
obs_size = model.observation_space.shape[0] 
dummy_input = th.randn(1, obs_size)

print(f"🚀 Exporting model via Legacy Path (Input: {obs_size})...")

# By wrapping in th.no_grad() and using the standard export call, 
# we trigger the legacy path which is often better for quantization tools.
with th.no_grad():
    th.onnx.export(
        onnxable_model, 
        dummy_input, 
        "rl/governor_fp32.onnx", 
        opset_version=17,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False   # <-- forces the legacy TorchScript path explicitly
    )

# 4. Perform INT8 Quantization
print("⚡ Quantizing to INT8...")
quantize_dynamic(
    model_input="rl/governor_fp32.onnx",
    model_output="rl/governor_int8.onnx",
    weight_type=QuantType.QInt8
)

print("✅ Success! 'rl/governor_int8.onnx' created.")