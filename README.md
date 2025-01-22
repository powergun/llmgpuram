# Calculate LLM GPU RAM usage

## Usage

`poetry python run.py <parameter> <quantization>`

If you are an ollama user, you can find the parameter and quantization by running
`ollama show <model>`

If you use other llm platforms, you can find the quantization detail as well.

However, huggingface provides the "tensor type" on the model card instead of quantization, which is also supported.

You can find the parameter from the model card, and "tensor type"; pass "tensor type" to the quantization parameter.

To compile this program to a native executable, take a look at `build.bat` (only runnable on Windows 64).
