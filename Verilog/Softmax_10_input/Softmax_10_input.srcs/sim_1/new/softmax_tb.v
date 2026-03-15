`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 15.03.2026 15:11:51
// Design Name: 
// Module Name: softmax_tb
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

module softmax_tb;
    reg clk;
    reg rst;
    reg [15:0] test_z [0:9]; // Array for easy handling in TB
    wire [159:0] in_z_flat;  // Flattened wire for port
    wire [159:0] out_p_flat;
    wire [15:0] out_p [0:9];
    wire done;

    // Packing the array into the flat wire
    genvar j;
    generate
        for (j = 0; j < 10; j = j + 1) begin : packing
            assign in_z_flat[(j*16) +: 16] = test_z[j];
            assign out_p[j] = out_p_flat[(j*16) +: 16];
        end
    endgenerate

    softmax_layer uut (
        .clk(clk), .rst(rst),
        .in_z_flat(in_z_flat),
        .out_p_flat(out_p_flat),
        .done(done)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0; rst = 1;
        // Initialize array
        for(integer m=0; m<10; m=m+1) test_z[m] = 16'h0000;
        
        #20 rst = 0;

        // Test Case: Digit 3 is the winner
        test_z[0]=16'h001A; test_z[1]=16'h001A; test_z[2]=16'h001A;
        test_z[3]=16'h0500; // Winner
        test_z[4]=16'h001A; test_z[5]=16'h001A; test_z[6]=16'h001A;
        test_z[7]=16'h001A; test_z[8]=16'h001A; test_z[9]=16'h001A;

        wait(done);
        #10;
        $display("Digit 3 Probability (Hex): %h", out_p[3]);
        
        #100 $finish;
    end
endmodule