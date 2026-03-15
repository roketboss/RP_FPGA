`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 15.03.2026 15:03:19
// Design Name: 
// Module Name: softmax_layer
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module exp_lut (
    input clk,
    input [15:0] x_in,          // 16-bit fixed point input
    output reg [15:0] exp_out   // 16-bit fixed point output (e^x)
);
    // 256-entry LUT for e^x
    reg [15:0] rom [0:255];

initial begin
    $readmemh("exp_values.mem", rom);
end
    always @(posedge clk) begin
        // Using the 8 middle bits as the address for the LUT
        exp_out <= rom[x_in[11:4]]; 
    end
endmodule

module softmax_layer (
    input clk,
    input rst,
    input [159:0] in_z_flat,      // 10 inputs * 16 bits each
    output [159:0] out_p_flat,    // 10 outputs * 16 bits each
    output reg done
);
    // Internal Arrays
    wire [15:0] in_z [0:9];
    reg [15:0] out_p [0:9];
    wire [15:0] exp_values [0:9];

    // Pipeline Registers
    reg [19:0] sum_stage1_0, sum_stage1_1, sum_stage1_2, sum_stage1_3, sum_stage1_4;
    reg [20:0] sum_stage2_0, sum_stage2_1;
    reg [21:0] final_sum;
    reg [3:0] done_shifter;
    integer i;

    // 1. Unflatten inputs and Flatten outputs
    genvar k;
    generate
        for (k = 0; k < 10; k = k + 1) begin : bit_mapping
            assign in_z[k] = in_z_flat[(k*16) +: 16];
            assign out_p_flat[(k*16) +: 16] = out_p[k];
            
            // Instantiate LUTs
            exp_lut lut_inst (
                .clk(clk),
                .x_in(in_z[k]),
                .exp_out(exp_values[k])
            );
        end
    endgenerate

    // 2. Adder Tree + Normalization Logic
    always @(posedge clk) begin
        if (rst) begin
            final_sum <= 0;
            done <= 0;
            done_shifter <= 0;
            for(i=0; i<10; i=i+1) out_p[i] <= 0;
        end else begin
            // Stage 1: Parallel Addition
            sum_stage1_0 <= exp_values[0] + exp_values[1];
            sum_stage1_1 <= exp_values[2] + exp_values[3];
            sum_stage1_2 <= exp_values[4] + exp_values[5];
            sum_stage1_3 <= exp_values[6] + exp_values[7];
            sum_stage1_4 <= exp_values[8] + exp_values[9];

            // Stage 2: Intermediate Sums
            sum_stage2_0 <= sum_stage1_0 + sum_stage1_1;
            sum_stage2_1 <= sum_stage1_2 + sum_stage1_3;

            // Stage 3: Final Denominator
            final_sum <= sum_stage2_0 + sum_stage2_1 + sum_stage1_4;

            // Shift '1' through to track pipeline depth
            done_shifter <= {done_shifter[2:0], 1'b1};
            
            if (done_shifter[3]) begin
                for (i = 0; i < 10; i = i + 1) begin
                    // (e^x * 2^8) / sum
                    out_p[i] <= (exp_values[i] << 8) / final_sum[15:0];
                end
                done <= 1;
            end else begin
                done <= 0;
            end
        end
    end
endmodule