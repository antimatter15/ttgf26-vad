/*
 * Copyright (c) 2026 Kevin
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module tt_um_antimatter15_pdm_vad (
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
  reg [7:0] frame_index;
  reg [12:0] raw_sum;
  reg [12:0] prev_raw_sum;
  reg signed [13:0] band_500_sum;
  reg signed [13:0] band_562_sum;
  reg signed [13:0] band_1500_sum;
  reg [7:0] phase_500;
  reg [7:0] phase_562;
  reg [7:0] phase_1500;
  reg [7:0] score;
  reg [7:0] activity_score;

  wire       sample_enable = ena && ui_in[1];
  wire [6:0] ones_next = ones_count + {6'd0, ui_in[0]};
  wire signed [7:0] density = $signed({1'b0, ones_next}) - 8'sd32;
  wire signed [13:0] density_ext = {{6{density[7]}}, density};
  wire [6:0] magnitude = (ones_next >= 7'd32) ? (ones_next - 7'd32) : (7'd32 - ones_next);
  wire [12:0] raw_sum_next = raw_sum + {6'd0, magnitude};

  wire signed [13:0] band_500_next = phase_500[7] ? (band_500_sum - density_ext) : (band_500_sum + density_ext);
  wire signed [13:0] band_562_next = phase_562[7] ? (band_562_sum - density_ext) : (band_562_sum + density_ext);
  wire signed [13:0] band_1500_next = phase_1500[7] ? (band_1500_sum - density_ext) : (band_1500_sum + density_ext);
  wire [13:0] band_500_abs = band_500_next[13] ? (~band_500_next + 14'd1) : band_500_next;
  wire [13:0] band_562_abs = band_562_next[13] ? (~band_562_next + 14'd1) : band_562_next;
  wire [13:0] band_1500_abs = band_1500_next[13] ? (~band_1500_next + 14'd1) : band_1500_next;
  wire [15:0] band_sum = {2'd0, band_500_abs} + {2'd0, band_562_abs} + {2'd0, band_1500_abs};
  wire [12:0] raw_delta = (raw_sum_next >= prev_raw_sum) ? (raw_sum_next - prev_raw_sum) : (prev_raw_sum - raw_sum_next);
  wire [9:0] combined_feature = {7'd0, raw_sum_next[12:10]} +
                                {1'd0, raw_delta[12:4]} +
                                {3'd0, band_sum[13:7]};
  wire [7:0] frame_feature = |combined_feature[9:8] ? 8'hff : combined_feature[7:0];
  wire [7:0] decayed_score = score - {6'd0, score[7:6]};
  wire [8:0] score_update = {1'b0, decayed_score} + {1'b0, frame_feature};
  wire [7:0] score_next = score_update[8] ? 8'hff : score_update[7:0];

  always @(posedge clk) begin
    if (!rst_n) begin
      sample_index <= 6'd0;
      ones_count   <= 7'd0;
      frame_index  <= 8'd0;
      raw_sum      <= 13'd0;
      prev_raw_sum <= 13'd0;
      band_500_sum  <= 14'sd0;
      band_562_sum  <= 14'sd0;
      band_1500_sum <= 14'sd0;
      phase_500     <= 8'd0;
      phase_562     <= 8'd0;
      phase_1500    <= 8'd0;
      score          <= 8'd0;
      activity_score <= 8'd0;
    end else if (sample_enable) begin
      if (sample_index == 6'd63) begin
        sample_index <= 6'd0;
        ones_count   <= 7'd0;
        if (frame_index == 8'd159) begin
          frame_index  <= 8'd0;
          raw_sum      <= 13'd0;
          prev_raw_sum <= raw_sum_next;
          band_500_sum  <= 14'sd0;
          band_562_sum  <= 14'sd0;
          band_1500_sum <= 14'sd0;
          phase_500     <= 8'd0;
          phase_562     <= 8'd0;
          phase_1500    <= 8'd0;
          score          <= score_next;
          activity_score <= score_next;
        end else begin
          frame_index   <= frame_index + 8'd1;
          raw_sum       <= raw_sum_next;
          band_500_sum  <= band_500_next;
          band_562_sum  <= band_562_next;
          band_1500_sum <= band_1500_next;
          phase_500     <= phase_500 + 8'd8;
          phase_562     <= phase_562 + 8'd9;
          phase_1500    <= phase_1500 + 8'd24;
        end
      end else begin
        sample_index <= sample_index + 6'd1;
        ones_count   <= ones_next;
      end
    end
  end

  // All output pins must be assigned. If not used, assign to 0.
  assign uo_out  = activity_score;
  assign uio_out = 0;
  assign uio_oe  = 0;

  // List all unused inputs to prevent warnings
  wire _unused = &{uio_in, ui_in[7:2], 1'b0};

endmodule
