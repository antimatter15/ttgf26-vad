/*
 * Copyright (c) 2026 Kevin
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module tt_um_gf26b_startup_demo (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (active high: 0=input, 1=output)
    input  wire       ena,      // always 1 when the design is powered, so you can ignore it
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);

  reg [5:0] sample_index;
  reg [6:0] ones_count;
  reg [7:0] energy;

  wire       sample_enable = ena && ui_in[1];
  wire [6:0] ones_next = ones_count + {6'd0, ui_in[0]};
  wire [6:0] magnitude = (ones_next >= 7'd32) ? (ones_next - 7'd32) : (7'd32 - ones_next);
  wire [8:0] energy_update = {1'b0, energy} - {3'b000, energy[7:2]} + {2'b00, magnitude};
  wire [7:0] energy_next = energy_update[7:0];

  always @(posedge clk) begin
    if (!rst_n) begin
      sample_index <= 6'd0;
      ones_count   <= 7'd0;
      energy       <= 8'd0;
    end else if (sample_enable) begin
      if (sample_index == 6'd63) begin
        sample_index <= 6'd0;
        ones_count   <= 7'd0;
        energy       <= energy_next;
      end else begin
        sample_index <= sample_index + 6'd1;
        ones_count   <= ones_next;
      end
    end
  end

  // All output pins must be assigned. If not used, assign to 0.
  assign uo_out  = energy;
  assign uio_out = 0;
  assign uio_oe  = 0;

  // List all unused inputs to prevent warnings
  wire _unused = &{uio_in, 1'b0};

endmodule
