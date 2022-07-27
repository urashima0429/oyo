module bnn(clk, xrst, pix, pred);
	input clk;
	input xrst;
	input [7:0] pix;
	output [9:0] pred;
	
	parameter COUNT_BIT = 16;
	parameter WIDTH_IN = 784;
	parameter WIDTH_MID = 16; // W
	parameter WIDTH_OUT = 10; 
	parameter DEPTH = 2;      // D
	
	reg [35:0] cnt = 36'd0;
	reg       img[WIDTH_IN-1:0];
	
	reg                 W_in[WIDTH_MID*WIDTH_IN-1:0];
	reg [COUNT_BIT-1:0] b_in[WIDTH_MID-1:0];
	reg [COUNT_BIT-1:0] t_in[WIDTH_MID-1:0];

	reg                 W[DEPTH*WIDTH_MID*WIDTH_MID-1:0];
	reg [COUNT_BIT-1:0] b[DEPTH*WIDTH_MID-1:0];
	reg [COUNT_BIT-1:0] t[DEPTH*WIDTH_MID-1:0];
	reg                 z[DEPTH*WIDTH_MID-1:0];	
	
	reg                 W_out[WIDTH_OUT*WIDTH_MID-1:0];
	reg [COUNT_BIT-1:0] b_out[WIDTH_OUT-1:0];
	reg [COUNT_BIT-1:0] t_out[WIDTH_OUT-1:0];
	reg                 z_out[WIDTH_OUT-1:0];
	
	assign pred = {z_out[0], z_out[1], z_out[2], z_out[3], z_out[4], z_out[5], z_out[6], z_out[7], z_out[8], z_out[9]};
	
	integer itr_in, jtr_in, i, j, k, itr_out, jtr_out;
	
	always @(posedge clk) begin
	
		if (clk == 1'b1) begin 
			if (xrst == 1'b0) begin
				
				for (itr_out = 0; itr_out < WIDTH_OUT; itr_out = itr_out + 1) begin
					for (jtr_out = 0; jtr_out < WIDTH_MID; jtr_out = jtr_out + 1) begin
						W_out[itr_out * WIDTH_MID + jtr_out] = 1;
					end
					b_out[itr_out] = 1;
					t_out[itr_out] = 1;
					z_out[itr_out] = 1;
				end
				
				for (i = 0; i < DEPTH; i = i + 1) begin 
					for (j = 0; j < WIDTH_MID; j = j + 1) begin 
						for (k = 0; k < WIDTH_MID; k = k + 1) begin
							W[i * WIDTH_MID*WIDTH_MID + j * WIDTH_MID + k] = 1;
						end
						b[i * WIDTH_MID + j] = 1;
						t[i * WIDTH_MID + j] = 1;
						z[i * WIDTH_MID + j] = 1;
					end
				end
				
				for (itr_in = 0; itr_in < WIDTH_MID; itr_in = itr_in + 1) begin 
					for (jtr_in = 0; jtr_in < WIDTH_MID; jtr_in = jtr_in + 1) begin
						W_in[itr_in * WIDTH_MID + jtr_in] = 1;
					end
					b_in[itr_in] = 1;
					t_in[itr_in] = 1;
				end		

			end else begin 
				for (itr_out = 0; itr_out < WIDTH_OUT; itr_out = itr_out + 1) begin
					z_out[itr_out] = t_out[itr_out] + b_out[itr_out] > WIDTH_MID/2;
					for (jtr_out = 0; jtr_out < WIDTH_MID; jtr_out = jtr_out + 1) begin
						t_out[itr_out] = t_out[itr_out] + (W_out[itr_out * WIDTH_MID + jtr_out] == z[(DEPTH-1) * WIDTH_MID + jtr_out] ? 8'b1 : 8'b0);
					end
					t_out[itr_out] = 0;
				end	
				
				for (i = DEPTH-1; i > 0; i = i - 1) begin 
					for (j = 0; j < WIDTH_MID; j = j + 1) begin 
						z[i * WIDTH_MID + j] = t[i * WIDTH_MID + j] + b[i * WIDTH_MID + j] > WIDTH_MID/2;
						for (k = 0; k < WIDTH_MID; k = k + 1) begin
							t[i * WIDTH_MID + j] = t[i * WIDTH_MID + j] + (W[i * WIDTH_MID*WIDTH_MID + j * WIDTH_MID + k] == z[i * WIDTH_MID + k] ? 8'b1 : 8'b0);
						end
						t[i * WIDTH_MID + j] = 0;
					end
				end	
				
				for (itr_in = 0; itr_in < WIDTH_MID; itr_in = itr_in + 1) begin 
					z[0 * WIDTH_MID + itr_in] = t_in[itr_in] + b_in[itr_in] > WIDTH_IN/2;	
					for (jtr_in = 0; jtr_in < WIDTH_IN; jtr_in = jtr_in + 1) begin
						t_in[itr_in] = t_in[itr_in] + (W_in[itr_in * WIDTH_IN + jtr_in] == img[jtr_in] ? 8'b1 : 8'b0);
					end
					t_in[itr_in] = 0;
				end
				
				img[cnt % 784] = (pix > 8'd64);	
				cnt = cnt + 10'd1;
				
			end
		end
	end
endmodule