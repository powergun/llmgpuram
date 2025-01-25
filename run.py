import re
from typing import Optional, NamedTuple

import click

# A mapping from quantization "tags" to approximate bits per parameter.
# Feel free to expand or tweak the values below for more exact calculations.
# The ones commented ~ are approximate due to overhead.
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

    # GPTQ-ish / huggingface bits-and-bytes
    "int4": 4,
    "int8": 8,
    "nf4": 4,  # GPTQ "nf4" is effectively 4 bits

    # Standard floats
    "fp16": 16,
    "bf16": 16,
    "fp32": 32,
}


def parse_num_params(parameter: str) -> tuple[Optional[Exception], int]:
    """
    Parse a string like '7B', '13B', '2.7B', '65M' into an integer count of parameters.
    'B' -> billion, 'M' -> million.

    If fail to parse, return a tuple of (exception, -1).
    """
    # e.g. "9.2B" => 9.2 * 1e9
    pattern = r"([\d\.]+)([BM])"
    match = re.match(pattern, parameter.upper().strip())
    if not match:
        return ValueError(f"Cannot parse number of parameters from '{parameter}'."), -1

    value_str, unit = match.groups()
    value = float(value_str)

    if unit == "B":
        return None, int(value * 1e9)
    elif unit == "M":
        return None, int(value * 1e6)
    else:
        return ValueError(f"Unexpected unit '{unit}' in param string '{parameter}'."), -1


class SizeTuple(NamedTuple):
    bytes: float
    megabytes: float
    gigabytes: float


def calc_theoretical_gpu_ram(num_params_str, quant_tag) -> tuple[Optional[Exception], Optional[SizeTuple]]:
    """
    Given a param string (e.g. '9.2B') and a quantization tag (e.g. 'Q4_0'),
    return an approximate memory usage in SizeTuple (continues the size in bytes, megabytes, and gigabytes).
    """
    # Convert param string to int
    err, num_params = parse_num_params(num_params_str)
    if err:
        return err, None

    # Find the approximate bits from our dictionary
    bits = QUANTIZATION_BITS.get(quant_tag)
    if bits is None:
        return ValueError(f"Quantization tag '{quant_tag}' not recognized."), None

    # Theoretical memory: (params * bits) / 8 bytes
    total_bytes = num_params * bits / 8.0
    total_megabytes = total_bytes / (1024 ** 2)
    total_gigabytes = total_bytes / (1024 ** 3)
    return None, SizeTuple(total_bytes, total_megabytes, total_gigabytes)


@click.command()
@click.argument("parameter")
@click.argument("quantization")
def cli(parameter: str, quantization: str):
    err, size = calc_theoretical_gpu_ram(parameter, quantization)
    if err:
        print(f"Error: {err}")
        return
    print(f"Model: {parameter} params, quant = {quantization}")
    print(f"Approx memory usage: {size.bytes:,.0f} bytes ({size.megabytes:,.2f} MiB) = {size.gigabytes:,.2f} GiB")


# Example usage:
def test():
    parameter = "9.2B"
    quantization = "Q4_0"
    err, size = calc_theoretical_gpu_ram(parameter, quantization)
    if err:
        print(f"Error: {err}")
        return
    print(f"Model: {parameter} params, quant = {quantization}")
    print(f"Approx memory usage: {size.bytes:,.0f} bytes ({size.megabytes:,.2f} MiB) = {size.gigabytes:,.2f} GiB")


if __name__ == "__main__":
    cli()
