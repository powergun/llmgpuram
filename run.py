import re

import click

# 1) A mapping from quantization “tags” to approximate bits per parameter.
#    Feel free to expand or tweak the values below for more exact calculations.
#    The ones marked ~ are approximate due to overhead.
QUANTIZATION_BITS = {
    # llama.cpp-ish
    "Q2_K": 4,  # ~2-bit data + overhead => ~4 bits effectively
    "Q4_0": 8,  # nominally 4 bits + overhead => ~8 bits
    "Q4_1": 8,
    "Q4_K": 8,
    "Q4_K_S": 8,
    "Q4_K_M": 8,
    "Q5_0": 10,
    "Q5_1": 10,
    "Q6_K": 12,
    "Q8_0": 16,
    "Q8_K": 16,
    # GPTQ-ish / huggingface bitsandbytes
    "int4": 4,
    "int8": 8,
    "nf4": 4,  # GPTQ "nf4" is effectively 4 bits
    # Standard floats
    "fp16": 16,
    "bf16": 16,
    "fp32": 32,
}


def parse_num_params(parameter):
    """
    Parse a string like '7B', '13B', '2.7B', '65M' into an integer count of parameters.
    'B' -> billion, 'M' -> million.
    """
    # e.g. "9.2B" => 9.2 * 1e9
    pattern = r"([\d\.]+)([BM])"
    match = re.match(pattern, parameter.upper().strip())
    if not match:
        raise ValueError(f"Cannot parse number of parameters from '{parameter}'.")

    value_str, unit = match.groups()
    value = float(value_str)

    if unit == "B":
        return int(value * 1e9)
    elif unit == "M":
        return int(value * 1e6)
    else:
        raise ValueError(f"Unexpected unit '{unit}' in param string '{parameter}'.")


def calc_theoretical_gpu_ram(num_params_str, quant_tag):
    """
    Given a param string (e.g. '9.2B') and a quantization tag (e.g. 'Q4_0'),
    return an approximate memory usage in bytes, plus a human-readable string.
    """
    # 1. Convert param string to int
    num_params = parse_num_params(num_params_str)

    # 2. Find the approximate bits from our dictionary
    #    If missing, you might default or raise an error
    bits = QUANTIZATION_BITS.get(quant_tag)
    if bits is None:
        raise ValueError(f"Quantization tag '{quant_tag}' not recognized.")

    # 3. Theoretical memory: (#params * bits) / 8 bytes
    total_bytes = num_params * bits / 8.0

    # 4. Optionally convert to larger units, e.g., MiB or GiB
    total_megabytes = total_bytes / (1024 ** 2)
    total_gigabytes = total_bytes / (1024 ** 3)

    return total_bytes, total_megabytes, total_gigabytes


@click.command()
@click.argument("parameter")
@click.argument("quantization")
def cli(parameter: str, quantization: str):
    b, mb, gb = calc_theoretical_gpu_ram(parameter, quantization)
    print(f"Model: {parameter} params, quant = {quantization}")
    print(f"Approx memory usage: {b:,.0f} bytes ({mb:,.2f} MiB) = {gb:,.2f} GiB")


# Example usage:
def test():
    param_str = "9.2B"
    quant = "Q4_0"
    b, mb, gb = calc_theoretical_gpu_ram(param_str, quant)
    print(f"Model: {param_str} params, quant = {quant}")
    print(f"Approx memory usage: {b:,.0f} bytes ({mb:,.2f} MiB) = {gb:,.2f} GiB")


if __name__ == "__main__":
    cli()
