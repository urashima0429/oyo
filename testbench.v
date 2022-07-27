`timescale 1ps / 1ps

module testbench();
	parameter CLK = 1000000/10; // 10MHZ
	
	// for input 
	reg clk; 
	reg xrst;
	reg [7:0] pix;
	
	// for output 
	wire [9:0] pred;

	// module 
	bnn bnn0( 
		.clk (clk), 
		.xrst (xrst),
		.pix (pix),
		.pred (pred)
	);

	// clock generation 
	always begin
		clk = 1'b1; 
		#(CLK/2); 
		clk = 1'b0; 
		#(CLK/2);
	end

	// test senario 
	initial begin
		#(CLK); xrst = 0;
		#(CLK); xrst = 1;
		#(CLK); xrst = 0;
		#(CLK); xrst = 1;
		#(CLK); xrst = 0;
		#(CLK); xrst = 1;
		#(CLK); xrst = 0;
		#(CLK); xrst = 1;

		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd0; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		#(CLK); pix = 8'd255; 
		
		
		$finish; 
	end
endmodule
