# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, Timer


@cocotb.test()
async def test_project(dut):
    dut._log.info("Start")

    # Set the clock period to 1 us (1 MHz), a reasonable low PDM clock.
    clock = Clock(dut.clk, 1, unit="us")
    cocotb.start_soon(clock.start())

    # Reset
    dut._log.info("Reset")
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    dut._log.info("Test PDM VAD behavior")
    await Timer(1, unit="ns")

    assert dut.uo_out.value == 0

    async def feed_pdm_window(bits):
        for bit in bits:
            dut.ui_in.value = 0b10 | bit
            await ClockCycles(dut.clk, 1)
        await Timer(1, unit="ns")

    async def feed_frame(window_bits):
        for _ in range(160):
            await feed_pdm_window(window_bits)

    async def feed_frame_windows(window_list):
        for window_bits in window_list:
            await feed_pdm_window(window_bits)

    # A balanced PDM stream looks like silence around 50 percent density.
    await feed_frame([0, 1] * 32)
    assert dut.uo_out.value == 0

    # A strongly biased 10 ms frame creates a score.
    await feed_frame([1] * 64)
    assert dut.uo_out.value == 255

    # A second biased frame remains saturated, but without the transition delta.
    await feed_frame([1] * 64)
    assert dut.uo_out.value == 255

    # Hold the detector state when sample_enable is low.
    held_score = int(dut.uo_out.value)
    dut.ui_in.value = 0
    await ClockCycles(dut.clk, 80)
    await Timer(1, unit="ns")
    assert dut.uo_out.value == held_score

    # The first quiet frame after a loud frame carries the raw-delta kick.
    await feed_frame([0, 1] * 32)
    assert dut.uo_out.value == 255

    # Then quiet frames decay the score by score / 64.
    await feed_frame([0, 1] * 32)
    assert dut.uo_out.value == 252

    # Alternating positive and negative density windows exercise the mixer
    # bands and saturate the score.
    alternating_windows = ([[1] * 64, [0] * 64] * 80)
    await feed_frame_windows(alternating_windows)
    assert dut.uo_out.value == 255

    assert dut.uio_out.value == 0
    assert dut.uio_oe.value == 0
