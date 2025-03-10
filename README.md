# Calculate LLM GPU RAM usage

## Usage

`poetry python run.py <parameter> <quantization>`

If you are an ollama user, you can find the parameter and quantization by running
`ollama show <model>`

If you use other llm platforms, you can find the quantization detail as well.

However, huggingface provides the "tensor type" on the model card instead of quantization, which is also supported.

You can find the parameter from the model card, and "tensor type"; pass "tensor type" to the quantization parameter.

To compile this program to a native executable, take a look at `build.bat` (only runnable on Windows 64).

## References

inspired by:
<https://www.substratus.ai/blog/calculating-gpu-memory-for-llm>

another article talking about RAM calculation:
<[how much GPU
memory](https://masteringllm.medium.com/how-much-gpu-memory-is-needed-to-serve-a-large-languagemodel-llm-b1899bb2ab5d)>
