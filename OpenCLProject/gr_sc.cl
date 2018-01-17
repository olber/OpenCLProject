__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void grayScale(__read_only image2d_t image, __write_only image2d_t greyscaleImage) 
	{
		const int2 pos = {get_global_id(0), get_global_id(1)};
		float4 temp = read_imagef(image, sampler, pos);
		float greyPixel = 0.2989*temp.x + 0.5870*temp.y + 0.1140*temp.z;
		write_imagef(greyscaleImage, pos, greyPixel);
	}