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

    async def feed_window(bits):
        for bit in bits:
            dut.ui_in.value = 0b10 | bit
            await ClockCycles(dut.clk, 1)
        await Timer(1, unit="ns")

    # A balanced PDM stream looks like silence around 50 percent density.
    await feed_window([0, 1] * 32)
    assert dut.uo_out.value == 0

    # A strongly biased window creates energy.
    await feed_window([1] * 64)
    assert dut.uo_out.value == 32

    # Hold the detector state when sample_enable is low.
    held_energy = int(dut.uo_out.value)
    dut.ui_in.value = 0
    await ClockCycles(dut.clk, 80)
    await Timer(1, unit="ns")
    assert dut.uo_out.value == held_energy

    # Several quiet windows decay the smoothed energy.
    for _ in range(9):
        await feed_window([0, 1] * 32)

    assert int(dut.uo_out.value) <= 4

    assert dut.uio_out.value == 0
    assert dut.uio_oe.value == 0
